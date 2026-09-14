#!/usr/bin/env python
"""Warp one frame into a BEV big enough to hold the ENTIRE runway, and draw the
working canvases on it for scale.

This exists to make one trade-off visible rather than argued. The camera's
footprint on the ground is a wedge running from a hundred-odd metres out to the
horizon; a BEV canvas is a bounded rectangle of square metric pixels. Asking for
"the whole runway in BEV" is asking for a rectangle that contains a 3 km-long
slice of that wedge, and this script builds exactly that so you can see what you
get: the runway entire, at a metre-per-pixel the far end cannot support, inside
a canvas that is mostly ground the camera never saw.

    python -m experiments.bev_yolo.whole_runway_bev --range 800
    python -m experiments.bev_yolo.whole_runway_bev --episode 000131 --stem 000265
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from experiments.bev.camera import CameraModel
from experiments.bev.episodes import find_episodes, load_runways, runway_to_ground
from experiments.bev_yolo.arms import load_config
from experiments.bev.transform import (
    BEVGrid, bev_from_image, depression_deg, ground_to_image, horizon_v, warp_image,
)

YELLOW = (60, 220, 240)
# one colour per configured grid, in config order
GRID_COLOURS = [(240, 200, 80), (60, 150, 255), (120, 255, 235), (200, 130, 255)]


def put(img, text, xy, colour=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def ground_to_canvas(pts, grid):
    inv = np.linalg.inv(grid.A)
    p = np.column_stack([np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))]) @ inv.T
    return p[:, :2] / p[:, 2:3]


def near_visible_range(cam, pitch_deg, height_m):
    """Ground range at the bottom image row: the closest ground the camera sees."""
    d = math.radians(depression_deg(pitch_deg, cam.tilt_deg))
    ang = d + math.atan((cam.height - 1 - cam.cy) / cam.fy)
    return height_m / math.tan(ang) if ang > 1e-6 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--range", type=float, default=800.0,
                    help="pick the frame closest to this range to the threshold")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--max-px", type=int, default=2600,
                    help="cap on the canvas's larger side; mpp is derived from it")
    ap.add_argument("--full-fov", action="store_true",
                    help="cover the camera's whole lateral wedge, not just the runway")
    ap.add_argument("--out", default="experiments/bev_yolo/whole_runway_bev.jpg")
    args = ap.parse_args()

    cfg, _, grids = load_config(args.config)
    fit = Path(cfg["camera"]["fit"])
    cam = CameraModel(**json.loads(fit.read_text())["camera"])
    runways = load_runways(cfg["source"].get("runways_dat") or None)

    episodes = find_episodes(cfg["source"]["root"])
    if args.episode:
        episodes = [e for e in episodes if e.path.name == args.episode]
    best = None
    for ep in episodes:
        rw = runways.get((ep.airport, ep.runway_name))
        if rw is None:
            continue
        for fr in ep.frames(stride=1 if args.stem else 5):
            if not fr.labelled:
                continue
            if args.stem and fr.stem != args.stem:
                continue
            score = abs(fr.range_m - args.range) - rw.length_m / 1000.0
            if best is None or score < best[0]:
                best = (score, fr, rw)
        if args.episode:
            break
    if best is None:
        raise SystemExit("no usable frame found")
    _, fr, rw = best

    # the ground the canvas must hold: from the closest visible ground out past
    # the far threshold, and wide enough for the runway (or the whole wedge)
    x_near = max(near_visible_range(cam, fr.pitch_deg, fr.height_m), 1.0)
    corners = runway_to_ground(rw.corners_runway_frame(), fr.along_m, fr.right_m,
                               fr.yaw_deg)
    x_far = corners[:, 0].max() * 1.03
    half = (x_far * cam.width / 2 / cam.fx if args.full_fov
            else max(abs(corners[:, 1]).max() * 1.6, 120.0))

    mpp = max((x_far - x_near) / args.max_px, 2 * half / args.max_px)
    grid = BEVGrid(int(2 * half / mpp), int((x_far - x_near) / mpp), mpp, x_near)

    grid_labels = []
    img = cv2.imread(str(fr.image_path))
    m_bev = bev_from_image(cam, grid, fr.roll_deg, fr.pitch_deg, fr.height_m)
    bev = warp_image(img, m_bev, grid)

    # the runway, and the two working canvases, drawn in this frame
    cv2.polylines(bev, [ground_to_canvas(corners, grid).astype(np.int32)], True,
                  YELLOW, 3, cv2.LINE_AA)
    for i, (gname, pol) in enumerate(grids.items()):
        colour = GRID_COLOURS[i % len(GRID_COLOURS)]
        w = pol.for_frame(cam, fr.roll_deg, fr.pitch_deg, fr.height_m, range_m=fr.range_m)
        a, b, hw = w.extent_m()
        rect = np.array([[a, -hw], [b, -hw], [b, hw], [a, hw]])
        cv2.polylines(bev, [ground_to_canvas(rect, grid).astype(np.int32)], True,
                      colour, 3, cv2.LINE_AA)
        grid_labels.append((gname, colour))

    far_src_px = cam.fx * rw.width_m / math.hypot(x_far, fr.height_m)
    legend = "   ".join(f"{n} canvas" for n, _ in grid_labels)
    lines = [
        f"{fr.episode.path.name}/{fr.stem}  {fr.episode.airport} {fr.episode.runway_name}",
        f"range to threshold {fr.range_m:.0f} m   h {fr.height_m:.1f} m   "
        f"runway {rw.length_m:.0f} x {rw.width_m:.0f} m",
        f"canvas {grid.width} x {grid.height} px at {mpp:.2f} m/px   "
        f"covers {x_near:.0f}-{x_far:.0f} m forward, +-{half:.0f} m lateral",
        f"at 0.75 m/px (the working scale) this canvas would be "
        f"{int((2*half)/0.75)} x {int((x_far-x_near)/0.75)} px = "
        f"{(2*half/0.75)*((x_far-x_near)/0.75)/1e6:.0f} Mpx",
        f"runway width at the FAR threshold: {far_src_px:.1f} px in the source, "
        f"{rw.width_m/mpp:.0f} px here",
        "",
        f"yellow = whole runway   then, in config order: {legend}",
        "black = ground outside the camera's field of view",
    ]
    for i, t in enumerate(lines):
        put(bev, t, (14, 30 + i * 26), scale=0.62)

    # side by side rather than overlaid: a runway-only canvas can be narrower
    # than the source thumbnail would be
    src = img.copy()
    cv2.polylines(src, [fr.polygon_px(cam.width, cam.height).astype(np.int32)], True,
                  YELLOW, 2, cv2.LINE_AA)
    hs = min(bev.shape[0], 760)
    src = cv2.resize(src, (int(round(src.shape[1] * hs / src.shape[0])), hs))
    put(src, "source", (10, hs - 14), scale=0.6)

    H = max(bev.shape[0], src.shape[0])
    sheet = np.zeros((H, src.shape[1] + 20 + bev.shape[1], 3), np.uint8)
    sheet[:src.shape[0], :src.shape[1]] = src
    sheet[:bev.shape[0], src.shape[1] + 20:] = bev
    bev = sheet

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), bev, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {args.out}  ({bev.shape[1]}x{bev.shape[0]} px, {mpp:.2f} m/px)")
    print(f"  frame: {fr.episode.path.name}/{fr.stem}, range {fr.range_m:.0f} m, "
          f"runway {rw.length_m:.0f} m")


if __name__ == "__main__":
    main()
