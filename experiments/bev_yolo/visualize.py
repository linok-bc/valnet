#!/usr/bin/env python
"""Draw every arm's prediction on the same frames, in both views.

Two panels per frame:

    left    the source image, carrying the ground truth, the two PERSPECTIVE
            arms' predictions, and -- the point of this script -- the two BEV
            arms' predictions UNWARPED back through the homography into source
            pixels, so all four are visible in one coordinate frame.

    zoom    the same source image zoomed on the runway. At 1.5 km the runway is
            about 20 px across in the full frame, where every arm's outline looks
            identically correct; the disagreements are only visible zoomed.

    bev     the BEV canvas, carrying the ground truth and the two BEV arms'
            predictions in the frame they were actually made in, plus its own
            zoom -- the canvas is 1:3, so at row height it is otherwise a sliver.

Reading it: in the left panel a BEV arm's outline is a quadrilateral that was a
rectangle before unwarping, so any perspective-looking distortion you see is the
warp faithfully undoing itself. What to look for is whether the outlines hug the
runway edges, and -- at long range -- whether the BEV outlines LEAN differently
from the truth, since that lean is the heading error that the range lever turns
into metres of cross-track.

    python -m experiments.bev_yolo.visualize --n 6
    python -m experiments.bev_yolo.visualize --split val --n 8
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from experiments.bev.camera import CameraModel
from experiments.bev.transform import BEVGrid, bev_from_image
from experiments.bev_yolo.arms import load_config
from experiments.bev_yolo.eval import (
    estimate, ground_from_bev, label_polygon, polygon_to_ground, predicted_polygon,
)

# BGR. Ground truth is deliberately the brightest thing on the image; arms are
# coloured in config order, so adding an arm does not reshuffle the others.
TRUTH = (255, 255, 255)
PALETTE = [(235, 200, 60), (220, 90, 220), (90, 230, 110), (60, 150, 255),
           (120, 255, 235), (200, 130, 255), (255, 180, 120), (140, 200, 90)]


def draw_poly(img, poly, colour, thickness=2):
    if poly is None or len(poly) < 3:
        return
    cv2.polylines(img, [poly.astype(np.int32)], True, colour, thickness, cv2.LINE_AA)


def put(img, text, xy, colour=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def unwarp(poly_bev, m_bev):
    """BEV pixels -> source pixels. Drops points the homography sends behind the camera."""
    if poly_bev is None or len(poly_bev) < 3:
        return None
    inv = np.linalg.inv(m_bev)
    p = np.column_stack([poly_bev, np.ones(len(poly_bev))]) @ inv.T
    ok = p[:, 2] > 1e-9
    if ok.sum() < 3:
        return None
    return p[ok, :2] / p[ok, 2:3]


def near_edge(poly, frac=0.15):
    """The part of a BEV polygon nearest the aircraft, i.e. the threshold end.

    BEV rows increase toward the aircraft, so "near" is the bottom of the canvas.
    """
    if poly is None or len(poly) < 3:
        return poly
    v = poly[:, 1]
    return poly[v >= v.max() - frac * (v.max() - v.min() + 1e-9)]


def crop_around(img, poly, margin):
    """A square crop centred on the runway, `margin` x its larger dimension."""
    if poly is None or len(poly) < 3:
        return img.copy()
    cx, cy = poly.mean(axis=0)
    half = max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1])) * margin / 2
    half = float(np.clip(half, 40.0, max(img.shape[:2]) / 2))
    x0 = int(np.clip(cx - half, 0, img.shape[1] - 1))
    x1 = int(np.clip(cx + half, x0 + 1, img.shape[1]))
    y0 = int(np.clip(cy - half, 0, img.shape[0] - 1))
    y1 = int(np.clip(cy + half, y0 + 1, img.shape[0]))
    return img[y0:y1, x0:x1].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6, help="frames, spread over range")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so this does not fight a training run")
    ap.add_argument("--panel-height", type=int, default=460)
    ap.add_argument("--zoom-margin", type=float, default=2.5,
                    help="zoom panel half-size, in multiples of the runway bbox")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg, arms, _ = load_config(args.config)
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(arms)}
    root = Path(cfg["dataset"]["out"])
    project = Path(cfg["train"]["project"]).resolve()
    conf = args.conf if args.conf is not None else cfg["eval"]["conf"]
    out = Path(args.out or (project.parent / f"predictions_{root.name}.jpg"))

    cam = CameraModel(**json.loads((root / "meta" / "camera.json").read_text())["camera"])
    recs = [json.loads(l)
            for l in (root / "meta" / f"{args.split}.jsonl").read_text().splitlines()]
    recs.sort(key=lambda r: r["range_m"])
    picks = [recs[i] for i in np.linspace(0, len(recs) - 1, args.n).round().astype(int)]

    models = {}
    for name, arm in arms.items():
        w = project / name / "weights" / "best.pt"
        if w.exists():
            models[name] = YOLO(str(w))
        else:
            print(f"  (no weights for {name}, skipping it)")
    if not models:
        raise SystemExit(f"no trained arms under {project}")

    # any grid defines the same ground window for the perspective arms, so the
    # panels are drawn against the FIRST bev grid in the config
    ref_grid = next((a.grid for a in arms.values() if a.is_bev), None)
    if ref_grid is None:
        raise SystemExit("this config has no bev arm to visualise")

    rows = []
    for rec in picks:
        g = rec["grids"][ref_grid]
        grid = BEVGrid(g["width"], g["height"], g["mpp"], g["x_near"])
        m_bev = bev_from_image(cam, grid, rec["roll_deg"], rec["pitch_deg"],
                               rec["height_m"])

        persp_arm = next(a.name for a in arms.values() if not a.is_bev)
        src = cv2.imread(str(root / "arms" / persp_arm / "images" / args.split /
                             f"{rec['name']}.jpg"))
        bev = cv2.imread(str(root / "bev" / ref_grid / args.split / f"{rec['name']}.jpg"))
        if src is None or bev is None:
            continue

        # ground truth in both frames
        gt_src = label_polygon(root / "arms" / persp_arm / "labels" / args.split /
                               f"{rec['name']}.txt", cam.width, cam.height)
        bev_arm = next(a.name for a in arms.values()
                       if a.is_bev and a.grid == ref_grid and a.target == "segment")
        gt_bev = label_polygon(root / "arms" / bev_arm / "labels" / args.split /
                               f"{rec['name']}.txt", grid.width, grid.height)
        # The BEV canvas covers a bounded patch of ground -- in `threshold` mode
        # only 617 m of runway at ANY range, about a quarter of a typical one. A
        # BEV arm's outline therefore STOPS at the canvas edge, which looks
        # identical to the model losing the far end of the runway. Drawing the
        # canvas footprint in the source frame tells the two apart.
        draw_poly(src, unwarp(np.array([[0, 0], [grid.width - 1, 0],
                                        [grid.width - 1, grid.height - 1],
                                        [0, grid.height - 1]], float), m_bev),
                  (120, 120, 120), 1)
        draw_poly(src, gt_src, TRUTH, 2)
        draw_poly(bev, gt_bev, TRUTH, 2)

        lines = []
        for name, model in models.items():
            arm = arms[name]
            img_path = root / "arms" / name / "images" / args.split / f"{rec['name']}.jpg"
            res = model.predict([str(img_path)], conf=conf, verbose=False,
                                device=args.device,
                                retina_masks=(arm.target == "segment"))[0]
            poly = predicted_polygon(res, arm.target)
            if poly is None:
                lines.append((name, "no detection"))
                continue

            if arm.is_bev:
                # a bev arm on ANOTHER grid still unwarps into the source frame,
                # but only its own grid's canvas is the one being displayed
                if arm.grid == ref_grid:
                    draw_poly(bev, poly, colours[name], 2)
                gb = rec["grids"][arm.grid]
                own = BEVGrid(gb["width"], gb["height"], gb["mpp"], gb["x_near"])
                m_own = bev_from_image(cam, own, rec["roll_deg"], rec["pitch_deg"],
                                       rec["height_m"])
                draw_poly(src, unwarp(poly, m_own), colours[name], 2)
            else:
                draw_poly(src, poly, colours[name], 2)

            est = estimate(polygon_to_ground(poly, rec, cam, arm))
            lines.append((name, f"xtrack {est['right_m']:+7.2f} m "
                               f"(err {est['right_m'] - rec['right_m']:+6.2f})   "
                               f"yaw {est['yaw_deg']:+5.2f} "
                               f"(err {est['yaw_deg'] - rec['yaw_deg']:+5.2f})"
                          if est else "degenerate fit"))

        put(src, f"{rec['episode']}/{rec['stem']}  {rec['airport']} {rec['runway']}", (10, 26))
        put(src, f"range {rec['range_m']:6.0f} m   h {rec['height_m']:5.1f} m   "
                 f"TRUE xtrack {rec['right_m']:+6.2f} m   yaw {rec['yaw_deg']:+5.2f} deg",
            (10, 48))
        for i, (name, text) in enumerate(lines):
            put(src, f"{name:12s} {text}", (10, 74 + i * 22), colours[name], 0.5)
        put(bev, f"{ref_grid} {grid.mpp:.2f} m/px", (8, 20), scale=0.45)

        h = args.panel_height
        def fit(im):
            k = h / im.shape[0]
            return cv2.resize(im, (max(int(round(im.shape[1] * k)), 1), h),
                              interpolation=cv2.INTER_NEAREST if k > 2 else cv2.INTER_AREA)

        src_zoom = crop_around(src, gt_src, args.zoom_margin)
        # the BEV zoom goes on the THRESHOLD, not the whole runway: the runway
        # fills the canvas, so a bbox crop of it is the canvas again, and the
        # near edge is the only part the cross-track estimate is read from
        bev_zoom = crop_around(bev, near_edge(gt_bev), args.zoom_margin * 1.6)
        panels = [fit(src), fit(src_zoom), fit(bev), fit(bev_zoom)]
        # labels go on the RESIZED panels, or a 40 px crop blown up 10x carries
        # 10x-high lettering
        for panel, text in zip(panels, [
                "source", f"source zoom x{src.shape[1] / max(src_zoom.shape[1], 1):.1f}",
                "BEV canvas", f"BEV zoom x{bev.shape[1] / max(bev_zoom.shape[1], 1):.1f}"]):
            put(panel, text, (8, panel.shape[0] - 12), scale=0.5)
        rows.append(np.hstack(panels))

    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]

    legend = np.zeros((34, width, 3), np.uint8)
    put(legend, "ground truth", (10, 23), TRUTH, 0.55)
    x = 160
    for name in models:
        put(legend, name, (x, 23), colours[name], 0.55)
        x += 11 * len(name) + 30
    sheet = np.vstack([legend] + rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(rows)} frames)")


if __name__ == "__main__":
    main()
