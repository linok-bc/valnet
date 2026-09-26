#!/usr/bin/env python
"""One-page diagram of how the edge-line cross-track estimator works.

Top row: exact synthetic geometry -- a runway projected through a zoomed pinhole
camera (f = 5000 px, so the runway fills the view) on the centreline, 8 m right,
and 8 m right with a 3 deg heading error. Every estimate printed is computed by
`geometry.crosstrack` from the projected edges. Bottom: a real test frame with
the real YOLOv8 mask (1280 px), row scan and line fits.

    python -m experiments.edge_lines.diagram
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, load_runways, runway_to_ground
from experiments.bev.transform import ground_to_image
from experiments.edge_lines.eval import yolo_masks
from experiments.edge_lines.geometry import (crosstrack, edge_points, fit_line,
                                             line_from_fit, line_through)

INK, INK2, MUTED, RULE = "#14202b", "#475563", "#7d8a96", "#d5dde3"
SKY, GROUND, RUNWAY = "#e8f0f6", "#dfe7d6", "#9aa3aa"
EDGE, VP, OFF, LEFT_C, RIGHT_C, MASK = "#2a78d6", "#c77800", "#b23a2e", "#eb6834", "#1b9e6e", "#f0a33a"

W_RWY, L_RWY = 45.0, 3000.0
CAM = CameraModel(1280, 720, 5000.0, 5000.0, 640.0, 360.0, 0.0)   # zoomed pinhole
PITCH, ROLL = -2.5, 0.0                                            # nose slightly down: horizon near the top


def project(range_m, right_m, yaw_deg, h):
    rwy = np.array([[0, -W_RWY / 2], [L_RWY, -W_RWY / 2], [L_RWY, W_RWY / 2], [0, W_RWY / 2]])
    g = runway_to_ground(rwy, -range_m, right_m, yaw_deg)
    H = ground_to_image(CAM, ROLL, PITCH, h)
    p = np.column_stack([g, np.ones(4)]) @ H.T
    return p[:, :2] / p[:, 2:3]


def u_at(l, v):
    return -(l[1] * v + l[2]) / l[0]


def camera_panel(ax, range_m, right_m, yaw_deg, ghost=None, xlim=(170, 1110)):
    """Synthetic camera view, identical crop in every panel so shifts are visible."""
    h = range_m * math.tan(math.radians(3.0))
    q = project(range_m, right_m, yaw_deg, h)
    lL, lR = line_through(q[0], q[1]), line_through(q[3], q[2])
    vp = np.cross(lL, lR); vp = vp[:2] / vp[2]
    v_near = max(q[0][1], q[3][1])
    top, bot = vp[1] - 45, v_near + 45
    ax.add_patch(Rectangle((0, top), 1280, vp[1] - top, color=SKY, zorder=0))
    ax.add_patch(Rectangle((0, vp[1]), 1280, bot - vp[1], color=GROUND, zorder=0))
    if ghost is not None:
        ax.add_patch(Polygon(ghost, closed=True, fill=False, ec=MUTED, lw=1.4,
                             ls=(0, (4, 3)), zorder=2))
    ax.add_patch(Polygon(q, closed=True, color=RUNWAY, zorder=2))
    for l in (lL, lR):
        ax.plot([vp[0], u_at(l, v_near)], [vp[1], v_near], color=EDGE, lw=3, zorder=3,
                solid_capstyle="round")
    ax.plot(vp[0], vp[1], "o", color=INK, ms=7, zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(bot, top)
    ax.set_aspect("equal"); ax.axis("off")
    out = crosstrack(lL, lR, CAM, ROLL, PITCH, h, W_RWY, yaw_deg)
    return q, out, h


def position_based(q, h):
    """What a position-based estimator reports if it assumes the heading is zero."""
    H = np.linalg.inv(ground_to_image(CAM, ROLL, PITCH, h))
    t = (q[0] + q[3]) / 2
    g = H @ np.array([t[0], t[1], 1.0]); g = g[:2] / g[2]
    return -g[1]


def label(ax, title, caption):
    ax.set_title(title, fontsize=17, fontweight="bold", color=INK, loc="left", pad=10)
    ax.text(0.0, -0.1, caption, transform=ax.transAxes, fontsize=14, color=INK2,
            va="top", linespacing=1.45)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="000135")
    ap.add_argument("--range", type=float, default=800)
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--weights", default="baseline/runs/stock/weights/best.pt")
    ap.add_argument("--out", default="experiments/edge_lines/media/method_diagram.png")
    args = ap.parse_args()

    R, Y, PSI = 800.0, 8.0, 3.0
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.text(0.04, 0.965, "Cross-track from the lean of the runway edges",
             fontsize=26, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.925, "Needs only the image, roll, pitch and the runway width. No heading.",
             fontsize=16, color=INK2, va="top")

    # ---- top row: the idea, on exact synthetic geometry
    axes = [fig.add_axes([0.04 + i * 0.32, 0.64, 0.29, 0.2]) for i in range(3)]
    q0, out0, h = camera_panel(axes[0], R, 0.0, 0.0)
    label(axes[0], "On the centreline", "Edges lean equally")
    q1, out1, _ = camera_panel(axes[1], R, Y, 0.0, ghost=q0)
    label(axes[1], f"{Y:.0f} m right", f"Edges lean left\nestimate {out1['free_width']:+.1f} m")
    q2, out2, _ = camera_panel(axes[2], R, Y, PSI, ghost=q1)
    pos = position_based(q2, h)
    label(axes[2], f"{Y:.0f} m right, nose {PSI:.0f}° off",
          f"Runway slides, lean unchanged\nestimate still {out2['free_width']:+.1f} m")

    # ---- bottom: a real frame
    cam = CameraModel(**json.loads(Path(args.camera).read_text())["camera"])
    ep = Episode(Path("datasets/xp12_dataset") / args.episode)
    rw = load_runways()[(ep.airport, ep.runway_name)]
    f = min((ep.frame(s) for s in ep.stems), key=lambda f: abs(f.range_m - args.range))
    mask = yolo_masks(args.weights, [f], 0.25, None, imgsz=1280)[0]
    left, right, D = edge_points(mask, cam, f.roll_deg, f.pitch_deg)
    back = lambda p: (np.column_stack([p, np.ones(len(p))]) @ np.linalg.inv(D).T)[:, :2]
    fl, fr = fit_line(left), fit_line(right)
    lL, lR = line_from_fit(fl[0], fl[1], D), line_from_fit(fr[0], fr[1], D)
    est = crosstrack(lL, lR, cam, f.roll_deg, f.pitch_deg, f.height_m, rw.width_m,
                     f.yaw_deg)["free_width"]
    vp = np.cross(lL, lR); vp = vp[:2] / vp[2]
    img = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2RGB)
    vs, us = np.nonzero(mask)
    v_bot = vs.max() + 3
    top, bot = vp[1] - 10, v_bot + 10
    half_w = (bot - top) * 1.05
    cx = (us.min() + us.max()) / 2

    ax = fig.add_axes([0.04, 0.12, 0.64, 0.4])
    ax.imshow(img, interpolation="nearest")
    tint = np.zeros((*mask.shape, 4)); tint[mask > 0] = matplotlib.colors.to_rgba(MASK, 0.45)
    ax.imshow(tint, interpolation="nearest")
    pts = np.vstack([back(left), back(right)])
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=4, color="white", mec=INK, mew=0.6, zorder=4)
    for l in (lL, lR):
        ax.plot([vp[0], u_at(l, v_bot)], [vp[1], v_bot], color=EDGE, lw=3, zorder=3,
                solid_capstyle="round")
    ax.plot(vp[0], vp[1], "o", color=INK, ms=8, zorder=5)
    ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(bot, top)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"On a real frame  ·  {ep.airport} {ep.runway_name}, {f.range_m:.0f} m out",
                 fontsize=17, fontweight="bold", color=INK, loc="left", pad=10)

    # legend, one line each, to the right of the image
    lx, ly = 0.72, 0.43
    items = [("mask", dict(marker="s", ms=13, color=MASK, alpha=0.7, lw=0)),
             ("edge of each row", dict(marker="o", ms=7, color="white", mec=INK, mew=0.8, lw=0)),
             ("fitted edges", dict(color=EDGE, lw=3.5)),
             ("where they meet", dict(marker="o", ms=9, color=INK, lw=0))]
    for i, (name, kw) in enumerate(items):
        y = ly - i * 0.04
        la = fig.add_axes([lx, y - 0.012, 0.03, 0.024]); la.axis("off")
        la.set_xlim(0, 1); la.set_ylim(-1, 1)
        if "marker" in kw:
            la.plot([0.5], [0], **kw)
        else:
            la.plot([0.05, 0.95], [0, 0], **kw)
        fig.text(lx + 0.04, y, name, fontsize=14, color=INK, va="center")
    fig.text(lx, ly - 0.2, f"estimate  {est:+.1f} m\ntruth        {f.right_m:+.1f} m",
             fontsize=16, color=INK, va="top", fontweight="bold", linespacing=1.6)

    # ---- the one formula
    fig.text(0.04, 0.05, r"cross-track $= -\dfrac{W}{2}\cdot\dfrac{s_L + s_R}{s_R - s_L}$",
             fontsize=20, color=INK, va="center")
    fig.text(0.36, 0.05, "s = each edge's sideways offset, read from its lean      W = runway width",
             fontsize=14, color=INK2, va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"8 m right: {out1['free_width']:+.3f}; with 3 deg heading {out2['free_width']:+.3f} "
          f"(position-based {pos:+.1f}); real frame {est:+.2f} vs {f.right_m:+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
