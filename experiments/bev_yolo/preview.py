#!/usr/bin/env python
"""Render a few frames side by side: source, BEV, and the fitted oriented box.

Look at this before spending a training run. The numbers in calibrate.py can
come out fine while the warp still points at the wrong patch of ground, and a
single contact sheet catches it instantly. What you want to see:

  - the runway is a straight-sided rectangle in the BEV panel, not a wedge
  - it leans by the yaw error printed in the corner, and by nothing else
  - the green box (fitted to the warped label) hugs the runway with no slack
  - the measured width matches the true width from runways.dat

Slack between the box and the runway in the BEV panel is the part of the
experiment's premise that would be false.

    python -m experiments.bev_yolo.preview --data /path/to/xp12_dataset -n 8
    python -m experiments.bev_yolo.preview --grid altitude -n 8
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from experiments.bev.camera import CameraModel
from experiments.bev.episodes import find_episodes, load_runways
from experiments.bev.grids import resolution_report
from experiments.bev_yolo.arms import load_config
from experiments.bev.labels import clip_to_canvas, obb_geometry, warp_points
from experiments.bev.transform import bev_from_image, warp_image

GREEN, YELLOW, RED = (80, 230, 120), (60, 220, 240), (70, 70, 235)


def draw_poly(img, poly, colour, thickness=2):
    cv2.polylines(img, [poly.astype(np.int32)], True, colour, thickness, cv2.LINE_AA)


def put_lines(img, lines, x=10, y0=26, colour=(255, 255, 255)):
    for i, text in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1, cv2.LINE_AA)


def panel(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * scale)), height))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--data", default=None, help="override source.root")
    ap.add_argument("-n", type=int, default=8, help="frames, spread over range")
    ap.add_argument("--episode", default=None, help="restrict to one episode dir name")
    ap.add_argument("--out", default="experiments/bev_yolo/preview.jpg")
    ap.add_argument("--panel-height", type=int, default=520)
    ap.add_argument("--grid", default=None,
                    help="which named grid from the config (default: the first)")
    args = ap.parse_args()

    cfg, _, grids = load_config(args.config)
    root = Path(args.data or cfg["source"]["root"])

    fit = Path(cfg["camera"]["fit"]) if cfg["camera"].get("fit") else None
    if fit and fit.exists():
        cam = CameraModel(**json.loads(fit.read_text())["camera"])
    else:
        c = cfg["camera"]
        cam = CameraModel.from_fov(c["width"], c["height"], c["fov_x_deg"],
                                   c["fov_y_deg"], c.get("tilt_deg", 0.0))
    grid_name = args.grid or next(iter(grids))
    if grid_name not in grids:
        raise SystemExit(f"unknown grid {grid_name!r}; config has {', '.join(grids)}")
    policy = grids[grid_name]
    print(f"grid: {grid_name} ({policy.mode})")
    runways = load_runways(cfg["source"].get("runways_dat") or None)

    episodes = find_episodes(root)
    if args.episode:
        episodes = [e for e in episodes if e.path.name == args.episode]
    if not episodes:
        raise SystemExit(f"no episodes under {root}")

    # spread the samples across range rather than clustering at one altitude
    ep = episodes[0] if args.episode else episodes[len(episodes) // 2]
    rw = runways.get((ep.airport, ep.runway_name))
    if rw is None:
        raise SystemExit(f"{ep.airport} {ep.runway_name} not in runways.dat")
    frames = [f for f in ep.frames() if f.labelled]
    idx = np.linspace(0, len(frames) - 1, args.n).round().astype(int)

    rows = []
    for i in idx:
        fr = frames[i]
        img = cv2.imread(str(fr.image_path))
        poly_src = fr.polygon_px(cam.width, cam.height)
        left = img.copy()
        draw_poly(left, poly_src, YELLOW)
        put_lines(left, [
            f"{ep.path.name}/{fr.stem}  {ep.airport} {ep.runway_name}",
            f"range {fr.range_m:6.0f} m   h {fr.height_m:5.1f} m",
            f"yaw {fr.yaw_deg:+.2f}  pitch {fr.pitch_deg:+.2f}  roll {fr.roll_deg:+.2f}",
            f"xtrack {fr.right_m:+.2f} m",
        ])

        grid = policy.for_frame(cam, fr.roll_deg, fr.pitch_deg, fr.height_m,
                                range_m=fr.range_m)
        m_bev = bev_from_image(cam, grid, fr.roll_deg, fr.pitch_deg, fr.height_m)
        bev = warp_image(img, m_bev, grid)
        pts, valid = warp_points(poly_src, m_bev)
        right = bev.copy()
        lines = [f"BEV  {grid_name} ({policy.mode})  {grid.mpp:.2f} m/px"]
        if valid.all():
            poly_bev = clip_to_canvas(pts, grid.width, grid.height)
            if len(poly_bev) >= 3:
                draw_poly(right, poly_bev, YELLOW)
                box = cv2.boxPoints(cv2.minAreaRect(poly_bev.astype(np.float32)))
                draw_poly(right, box, GREEN, 1)
                g = obb_geometry(poly_bev, grid.mpp, grid.width, grid.height, grid.x_near)
                rr = resolution_report(cam, grid, fr.roll_deg, fr.pitch_deg,
                                       fr.height_m, fr.range_m, rw.width_m)
                lines += [
                    f"width {g['width_m']:6.2f} m  (true {rw.width_m:.1f})",
                    f"box heading {g['heading_deg']:+.2f} deg  (-yaw {-fr.yaw_deg:+.2f})",
                    f"res {rr['source_px']:.0f} px src -> {rr['bev_px']:.0f} px bev",
                ]
        else:
            lines.append("CORNER ABOVE THE HORIZON -- frame would be dropped")
        put_lines(right, lines, colour=GREEN if valid.all() else RED)

        h = args.panel_height
        rows.append(np.hstack([panel(left, h), panel(right, h)]))

    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(rows)} frames "
          f"from {ep.path.name})")


if __name__ == "__main__":
    main()
