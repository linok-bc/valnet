#!/usr/bin/env python
"""Replay video per test flight: partner's heading + our edge-line cross-track.

Layout (1920 x 1080):

    +--------------------------------+----------------+
    | camera 1280x720                | heading        |  from 1500 m
    |  YOLO mask (violet tint)       |  truth / est   |
    |  partner's fitted runway (org) +----------------+
    |  our two edge lines (blue)     | cross-track    |  from 1000 m
    |  truth runway (aqua)           |  truth / est   |
    +--------------------------------+----------------+
    | top view of the whole approach | errors         |
    | experimental zone shaded       |                |
    +--------------------------------+----------------+

    python -m experiments.joint_demo.render                 # all flights
    python -m experiments.joint_demo.render --episodes 000135
"""

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.joint_demo import data  # noqa: E402

sys.path.insert(0, str(data.ROOT / "Restart"))
from evaluation.video import _quad  # noqa: E402
from pose_estimation.g_dof import GDofParams  # noqa: E402
from runway_detection.xp12.calibration import (load_calibrations,  # noqa: E402
                                               pose_from_frame,
                                               scene_from_calibration)
from runway_detection.xp12.loader import load_sequence  # noqa: E402

W_OUT, H_OUT = 1920, 1080
# reference categorical palette, first three slots (all-pairs validated)
HEX = {"truth": "#1baf7a", "heading": "#eb6834", "xtrack": "#2a78d6",
       "mask": "#4a3aa7", "ink": "#0b0b0b", "ink2": "#52514e", "surface": "#fcfcfb",
       "zone": "#e9e8e4"}
RGB = {k: tuple(int(v[i:i + 2], 16) / 255 for i in (1, 3, 5)) for k, v in HEX.items()}
BGR = {k: tuple(int(round(255 * c)) for c in v[::-1]) for k, v in RGB.items()}


def _fig_to_bgr(fig):
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


class Panel:
    def __init__(self, fig, ax, img):
        fig.canvas.draw()
        self.t, self.h = ax.transData, img.shape[0]

    def px(self, x, y):
        u, v = self.t.transform((x, y))
        return int(round(u)), int(round(self.h - v))


def _style(ax):
    ax.set_facecolor(HEX["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HEX["ink2"])
    ax.tick_params(colors=HEX["ink2"], labelsize=9)
    ax.grid(alpha=0.25, lw=0.6)


def series(df, w, h):
    """Static right column: heading, cross-track, and both errors vs distance."""
    fig, axes = plt.subplots(4, 1, figsize=(w / 100, h / 100), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5]})
    fig.patch.set_facecolor(HEX["surface"])
    x = df.range_m.to_numpy()
    hd, xt = df[df.heading_on], df[df.xtrack_on]
    ax = axes[0]
    ax.plot(hd.range_m, hd.yaw_gt, color=RGB["truth"], lw=2, label="truth")
    ax.plot(hd.range_m, hd.yaw_est, color=RGB["heading"], lw=2, label="partner: joint fit + EKF")
    if hd.sigma_yaw.notna().any():
        ax.fill_between(hd.range_m, hd.yaw_est - 2 * hd.sigma_yaw, hd.yaw_est + 2 * hd.sigma_yaw,
                        color=RGB["heading"], alpha=0.15, lw=0, label="±2σ")
    ax.set_title("Heading relative to runway (deg, + nose right)", loc="left",
                 fontsize=11, color=HEX["ink"])
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax = axes[1]
    ax.axvspan(data.HEADING_FROM_M, data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    ax.text(0.5 * (data.HEADING_FROM_M + data.XTRACK_FROM_M), 0.5, "cross-track\nstarts at 1 km",
            transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=9,
            color=HEX["ink2"])
    ax.plot(xt.range_m, xt.truth_m, color=RGB["truth"], lw=2, label="truth")
    ax.plot(xt.range_m, xt.xt_est, color=RGB["xtrack"], lw=2, label="ours: edge-line tilt")
    lo, hi = np.nanpercentile(np.r_[xt.truth_m, xt.xt_est], [1, 99])
    pad = max(0.2 * (hi - lo), 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Cross-track (m, + right of centreline)", loc="left", fontsize=11, color=HEX["ink"])
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for ax, d, col, key, unit in ((axes[2], hd, "heading", "yaw_err", "deg"),
                                  (axes[3], xt, "xtrack", "xt_err", "m")):
        ax.plot(d.range_m, d[key], color=RGB[col], lw=1.5)
        ax.axhline(0, color=RGB["truth"], lw=1)
        v = max(np.nanpercentile(np.abs(d[key]), 98), 0.05)
        ax.set_ylim(-1.3 * v, 1.3 * v)
        mae = np.nanmean(np.abs(d[key]))
        ax.set_title(f"{'heading' if col == 'heading' else 'cross-track'} error ({unit}),"
                     f"  MAE {mae:.2f}", loc="left", fontsize=10, color=HEX["ink"])
    axes[1].axvspan(data.HEADING_FROM_M, data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    axes[3].axvspan(data.HEADING_FROM_M, data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    axes[-1].set_xlim(data.HEADING_FROM_M + 20, max(0, x.min() - 20))
    axes[-1].set_xlabel("distance to threshold (m)", fontsize=10, color=HEX["ink2"])
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    img = _fig_to_bgr(fig)
    panels = [Panel(fig, a, img) for a in axes]
    plt.close(fig)
    return img, panels


def top_view(flight, calib, w, h):
    """Whole approach from the logs; lateral axis exaggerated, + right is down."""
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(HEX["surface"])
    W, L = calib.width_m, calib.length_m
    ax.axvspan(-data.HEADING_FROM_M, -data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    ax.axvspan(-data.XTRACK_FROM_M, 0, color=RGB["zone"], alpha=0.55, lw=0)
    ax.fill([0, L, L, 0], [-W / 2, -W / 2, W / 2, W / 2], color="0.45", zorder=1)
    ax.plot([-flight.range_m.max() - 100, L], [0, 0], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.plot(-flight.range_m, flight.right_m, color=RGB["truth"], lw=2, zorder=3,
            label="flight path (truth)")
    lim = max(40.0, np.abs(flight.right_m).max() * 1.2)
    ax.set_ylim(lim, -lim)  # + right plotted downward: the aircraft flies left to right
    ax.set_xlim(-flight.range_m.max() - 100, 700)
    for x, t, yt in ((-data.HEADING_FROM_M, "heading from 1500 m", 0.92),
                     (-data.XTRACK_FROM_M, "+ cross-track from 1000 m", 0.75)):
        ax.axvline(x, color=HEX["ink2"], lw=0.8)
        ax.text(x + 20, -lim * yt, t, fontsize=9, color=HEX["ink2"], va="top")
    ax.text(0.01, 0.04, "localizer capture (not scored)", transform=ax.transAxes,
            fontsize=9, color=HEX["ink2"])
    ax.set_xlabel("along runway axis (m, threshold at 0)", fontsize=10, color=HEX["ink2"])
    ax.set_ylabel("cross-track (m, + right)", fontsize=10, color=HEX["ink2"])
    ax.set_title("Top view, whole approach (lateral scale exaggerated)", loc="left",
                 fontsize=11, color=HEX["ink"])
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _style(ax)
    fig.tight_layout()
    img = _fig_to_bgr(fig)
    panel = Panel(fig, ax, img)
    plt.close(fig)
    return img, panel


def _line_segment(l, v0, v1):
    """Pixel endpoints of homogeneous line l between image rows v0 and v1."""
    a, b, c = l
    if abs(a) < 1e-9:
        return None
    return [(int(round(-(b * v + c) / a)), int(round(v))) for v in (v0, v1)]


def _poly(img, quad, color, t):
    if quad is None or not np.all(np.isfinite(quad)) or np.abs(quad).max() > 1e5:
        return
    cv2.polylines(img, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], True, color, t,
                  cv2.LINE_AA)


def _put(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _zoom(canvas, cam, pts, x0, y0, size):
    c = np.nanmean(pts, axis=0)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 40) * 1.5
    half = int(min(span, 340) / 2)
    cx = int(np.clip(c[0], half, cam.shape[1] - half))
    cy = int(np.clip(c[1], half, cam.shape[0] - half))
    crop = cv2.resize(cam[cy - half:cy + half, cx - half:cx + half], (size, size),
                      interpolation=cv2.INTER_LINEAR)
    canvas[y0:y0 + size, x0:x0 + size] = crop
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
    _put(canvas, f"zoom x{size / (2 * half):.1f}", (x0 + 8, y0 + size - 10), scale=0.5)


def render(ep, out_dir, fps):
    df, soft, lines = data.load(ep)
    calib = load_calibrations(data.CALIB)[ep]
    seq = load_sequence(data.ROOT / "datasets/xp12_dataset", ep)
    frames = {f.index: f for f in seq.frames}
    scene = scene_from_calibration(calib)
    flight = data.full_flight(ep)
    airport, rwy = data.meta(ep)

    right_w = W_OUT - 1280
    s_img, s_pan = series(df, right_w, H_OUT)
    t_img, t_pan = top_view(flight, calib, 1280, H_OUT - 720)
    mask_bgr = np.array(BGR["mask"], np.float32)

    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.mp4"
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, H_OUT))
    for _, r in df.iterrows():
        f = frames[int(r.frame)]
        cam = seq.read_image(f).astype(np.float32)
        m = soft[int(r.frame)].astype(np.float32)[..., None] / 255
        cam = np.clip(cam * (1 - 0.5 * m) + mask_bgr * 0.5 * m, 0, 255).astype(np.uint8)
        truth_q = np.array([f.corners_px[n] for n in ("TR", "TL", "BL", "BR")])
        params = GDofParams.from_pose(pose_from_frame(f, calib), scene)
        est_q = _quad(scene, params, r.yaw_est, r.lat_est) if np.isfinite(r.yaw_est) else None
        _poly(cam, truth_q, BGR["truth"], 1)
        _poly(cam, est_q, BGR["heading"], 1)
        pts = [truth_q]
        if r.xtrack_on:
            ys = np.nonzero(soft[int(r.frame)] > 64)[0]
            if len(ys):
                for l in lines[int(r.frame)]:
                    seg = None if l is None else _line_segment(l, ys.min(), ys.max())
                    if seg:
                        cv2.line(cam, seg[0], seg[1], BGR["xtrack"], 1, cv2.LINE_AA)

        canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
        canvas[:720, :1280] = cam
        _zoom(canvas, cam, np.vstack(pts), 10, 10, 380)

        x0, y = 410, 36
        _put(canvas, f"Flight {ep}   {airport} rwy {rwy}   (held-out test flight)", (x0, y), scale=0.7, thick=2)
        y += 32
        _put(canvas, f"distance to threshold {r.range_m:5.0f} m    height {r.height_m:4.0f} m", (x0, y))
        y += 30
        _put(canvas, f"HEADING   truth {r.yaw_gt:+5.2f}   est {r.yaw_est:+5.2f}   err {r.yaw_err:+.2f} deg",
             (x0, y), BGR["heading"], 0.6, 2)
        y += 30
        if r.xtrack_on:
            _put(canvas, f"CROSS-TRACK   truth {r.truth_m:+6.1f}   est {r.xt_est:+6.1f}   err {r.xt_err:+.1f} m",
                 (x0, y), BGR["xtrack"], 0.6, 2)
        else:
            _put(canvas, f"CROSS-TRACK   starts at 1000 m  ({r.range_m - data.XTRACK_FROM_M:.0f} m to go)",
                 (x0, y), (200, 200, 200), 0.6, 1)
        ly = 610
        for text, col in (("truth runway", BGR["truth"]), ("partner's fitted runway", BGR["heading"]),
                          ("our edge lines", BGR["xtrack"]), ("YOLO mask", BGR["mask"])):
            cv2.line(canvas, (1020, ly - 6), (1050, ly - 6), col, 3)
            _put(canvas, text, (1058, ly), scale=0.5)
            ly += 24

        tv = t_img.copy()
        p0 = t_pan.px(-r.range_m, r.truth_m)
        # arrows in data coordinates, so they share the axes' lateral exaggeration
        for yaw, col, t in ((r.yaw_gt, BGR["truth"], 4), (r.yaw_est, BGR["heading"], 2)):
            p1 = t_pan.px(-r.range_m + 400 * math.cos(math.radians(yaw)),
                          r.truth_m + 400 * math.sin(math.radians(yaw)))
            cv2.arrowedLine(tv, p0, p1, col, t, cv2.LINE_AA, tipLength=0.2)
        if r.xtrack_on and np.isfinite(r.xt_est):
            cv2.circle(tv, t_pan.px(-r.range_m, r.xt_est), 6, BGR["xtrack"], 2, cv2.LINE_AA)
        cv2.circle(tv, p0, 5, BGR["truth"], -1, cv2.LINE_AA)
        canvas[720:, :1280] = tv[:H_OUT - 720, :1280]

        sv = s_img.copy()
        for k, pan in enumerate(s_pan):
            xp, _ = pan.px(r.range_m, 0)
            cv2.line(sv, (xp, 0), (xp, sv.shape[0]), (150, 150, 150), 1)
        cv2.circle(sv, s_pan[0].px(r.range_m, r.yaw_est), 5, BGR["heading"], -1, cv2.LINE_AA)
        if r.xtrack_on and np.isfinite(r.xt_est):
            cv2.circle(sv, s_pan[1].px(r.range_m, r.xt_est), 5, BGR["xtrack"], -1, cv2.LINE_AA)
        canvas[:, 1280:] = sv[:H_OUT, :right_w]
        wr.write(canvas)
    wr.release()
    out = Path(out_dir) / f"flight_{ep}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--out", default="experiments/joint_demo/media")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for ep in args.episodes or data.episodes():
        print(render(ep, args.out, args.fps), flush=True)


if __name__ == "__main__":
    main()
