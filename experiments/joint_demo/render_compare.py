#!/usr/bin/env python
"""Four cross-track methods, one video each, on the flights unseen by every model.

    samuel   Samuel's joint (heading, cross-track) mask fit + EKF   (Restart/)
    tilt     our edge-line tilt, free_width                          (edge_lines)
    bev      Ryu et al. on the bev_seg BEV warp, INS yaw + Kalman    (bev_yolo)
    fused    Samuel's EKF with tilt as a second cross-track measurement (fuse.py)

Each video plays flight 000145 then 000146 -- the only split_test flights that
neither the stock YOLO (samuel, tilt, fused) nor bev_seg (bev) trained on. Cross-track
is scored from 1000 m against ONE truth for all three (the rotation-fixed logs;
see data.py); Samuel's heading from 1500 m.

Layout (1920 x 1080):

    +--------------------------------+----------------+
    | camera: mask, truth runway,    | cross-track    |
    | what the method reads          +----------------+
    | (+ BEV canvas for bev)         | cross-track err|
    +--------------------------------+----------------+
    | top view of the whole approach | heading, samuel|
    +--------------------------------+----------------+

    python -m experiments.joint_demo.render_compare                # all three
    python -m experiments.joint_demo.render_compare --methods bev
"""

import argparse
import json
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
import pandas as pd  # noqa: E402

from experiments.joint_demo import data  # noqa: E402
from experiments.joint_demo.render import (Panel, _fig_to_bgr, _line_segment,  # noqa: E402
                                           _poly, _put, _zoom)

sys.path.insert(0, str(data.ROOT / "Restart"))
from evaluation.video import _quad  # noqa: E402
from pose_estimation.g_dof import GDofParams  # noqa: E402
from runway_detection.xp12.calibration import (load_calibrations,  # noqa: E402
                                               pose_from_frame,
                                               scene_from_calibration)
from runway_detection.xp12.loader import load_sequence  # noqa: E402

FLIGHTS = ("000145", "000146")
W_OUT, H_OUT = 1920, 1080
# reference categorical palette. Slots 1-3 are all-pairs validated; fused takes
# violet (slot 7) and is drawn DASHED + direct-labelled in the shared charts so
# identity never rests on hue alone. Truth takes the ink colour; the mask tint
# takes yellow, which is never a series here.
HEX = {"samuel": "#eb6834", "tilt": "#2a78d6", "bev": "#1baf7a", "fused": "#4a3aa7",
       "truth": "#0b0b0b", "mask": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e", "surface": "#fcfcfb",
       "zone": "#e9e8e4"}
RGB = {k: tuple(int(v[i:i + 2], 16) / 255 for i in (1, 3, 5)) for k, v in HEX.items()}
BGR = {k: tuple(int(round(255 * c)) for c in v[::-1]) for k, v in RGB.items()}
WHITE = (255, 255, 255)
# on-camera text: dark violet vanishes on the outlined HUD, so it takes the
# palette's dark-mode violet step there (charts keep the light-mode one)
HUD_BGR = {**BGR, "fused": (0xE9, 0x85, 0x90)}

METHODS = {
    "samuel": dict(name="Samuel: joint heading + cross-track fit",
                   what="render-and-compare mask fit + kinematic EKF; estimates heading AND cross-track",
                   inputs="YOLO mask, pitch, roll, height, along-track, runway width"),
    "tilt": dict(name="Edge-line tilt",
                 what="cross-track from how the two runway edges lean; no heading needed",
                 inputs="YOLO mask, pitch, roll, runway width"),
    "bev": dict(name="Bird's-eye view (Ryu et al.)",
                what="warp to BEV, read the runway's front edge, Kalman filter",
                inputs="BEV YOLO mask, pitch, roll, height, runway width, INS heading"),
    "fused": dict(name="Fused: Samuel's EKF + edge-line tilt",
                  what="tilt cross-track added to Samuel's EKF as a second measurement, noise fitted on val",
                  inputs="YOLO mask, pitch, roll, height, along-track, runway width"),
}
HEADING = {"samuel": "yaw_est", "fused": "fused_yaw"}   # methods that estimate heading


def _style(ax):
    ax.set_facecolor(HEX["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HEX["ink2"])
    ax.tick_params(colors=HEX["ink2"], labelsize=9)
    ax.grid(alpha=0.25, lw=0.6)


def load_flight(ep):
    """Per-frame table with all four estimates (m, + right) and the shared truth."""
    df, soft, lines = data.load(ep)
    df["samuel"] = -df.lat_est          # Restart's lateral is + left
    df["tilt"] = df.xt_est
    fu = pd.read_csv(data.HERE / "cache/fusion" / f"{ep}.csv")[["frame", "fused_yaw", "fused_lat"]]
    df = df.merge(fu, on="frame", how="left")
    df["fused"] = -df.fused_lat
    b = pd.DataFrame(json.loads((data.HERE / "cache/bev" / f"{ep}.json").read_text()))
    b = b.rename(columns={"bev_kf": "bev"}).drop(columns="range_m")
    df = df.merge(b, on="frame", how="left")
    return df, soft, lines


def series(df, key, w, h):
    """Right column: cross-track + error (all), heading + error (samuel, fused)."""
    heading = key in HEADING
    n = 4 if heading else 2
    ratios = [3, 1.6, 3, 1.6] if heading else [3, 1.6]
    fig, axes = plt.subplots(n, 1, figsize=(w / 100, h / 100), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    fig.patch.set_facecolor(HEX["surface"])
    xt = df[df.xtrack_on]
    col = RGB[key]

    ax = axes[0]
    ax.axvspan(data.HEADING_FROM_M + 20, data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    ax.text(0.5 * (data.HEADING_FROM_M + data.XTRACK_FROM_M), 0.5, "cross-track\nscored from 1 km",
            transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=9,
            color=HEX["ink2"])
    ax.plot(xt.range_m, xt.truth_m, color=RGB["truth"], lw=2, label="truth")
    if key == "bev":
        ax.plot(xt.range_m, xt.bev_raw, ".", color=col, ms=3, alpha=0.45, label="per frame")
        ax.plot(xt.range_m, xt.bev, color=col, lw=2, label="+ Kalman (reported)")
    else:
        ax.plot(xt.range_m, xt[key], color=col, lw=2, label="estimate")
    lo, hi = np.nanpercentile(np.r_[xt.truth_m, xt[key]], [1, 99])
    pad = max(0.2 * (hi - lo), 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Cross-track (m, + right of centreline)", loc="left", fontsize=11,
                 color=HEX["ink"])
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = axes[1]
    e = xt[key] - xt.truth_m
    ax.axvspan(data.HEADING_FROM_M + 20, data.XTRACK_FROM_M, color=RGB["zone"], lw=0)
    ax.plot(xt.range_m, e, color=col, lw=1.5)
    ax.axhline(0, color=RGB["truth"], lw=1)
    ax.set_ylim(-5, 5)  # shared by all three videos, so they compare directly
    ax.set_title(f"cross-track error (m),  MAE {np.nanmean(np.abs(e)):.2f}", loc="left",
                 fontsize=10, color=HEX["ink"])

    if heading:
        hd = df[df.heading_on]
        yaw = hd[HEADING[key]]
        yerr = yaw - hd.yaw_gt
        ax = axes[2]
        ax.plot(hd.range_m, hd.yaw_gt, color=RGB["truth"], lw=2, label="truth")
        ax.plot(hd.range_m, yaw, color=col, lw=2, label="estimate")
        if key == "samuel":
            ax.fill_between(hd.range_m, yaw - 2 * hd.sigma_yaw, yaw + 2 * hd.sigma_yaw,
                            color=col, alpha=0.15, lw=0, label="±2σ")
        ax.set_title("Heading relative to runway (deg, + nose right)", loc="left", fontsize=11,
                     color=HEX["ink"])
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax = axes[3]
        ax.plot(hd.range_m, yerr, color=col, lw=1.5)
        ax.axhline(0, color=RGB["truth"], lw=1)
        v = max(np.nanpercentile(np.abs(yerr), 98), 0.05)
        ax.set_ylim(-1.3 * v, 1.3 * v)
        ax.set_title(f"heading error (deg),  MAE {yerr.abs().mean():.3f}", loc="left",
                     fontsize=10, color=HEX["ink"])

    axes[-1].set_xlim(data.HEADING_FROM_M + 20, max(0, df.range_m.min() - 20))
    axes[-1].set_xlabel("distance to threshold (m)", fontsize=10, color=HEX["ink2"])
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    img = _fig_to_bgr(fig)
    panels = [Panel(fig, a, img) for a in axes]
    plt.close(fig)
    return img, panels


def top_view(flight, calib, w, h):
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
    ax.set_ylim(lim, -lim)
    ax.set_xlim(-flight.range_m.max() - 100, 700)
    for x, t, yt in ((-data.HEADING_FROM_M, "experiment starts 1500 m", 0.92),
                     (-data.XTRACK_FROM_M, "cross-track scored from 1000 m", 0.75)):
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


def _pts(a):
    return None if a is None else np.round(np.asarray(a)).astype(np.int32)


def _bev_inset(canvas, ep, r, x1, y0, h):
    """The BEV canvas the detector saw, with its mask, front edge and threshold point."""
    bev = cv2.imread(str(data.HERE / "cache/bev" / ep / f"{int(r.frame):06d}.jpg"))
    if bev is None:
        return
    if isinstance(r.poly_bev, list):
        cv2.polylines(bev, [_pts(r.poly_bev).reshape(-1, 1, 2)], True, BGR["bev"], 2, cv2.LINE_AA)
    if isinstance(r.edge_bev, list):
        e = _pts(r.edge_bev)
        cv2.line(bev, tuple(e[0]), tuple(e[1]), (255, 255, 255), 4, cv2.LINE_AA)
    if isinstance(r.point_bev, list):
        cv2.circle(bev, tuple(_pts(r.point_bev)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    s = h / bev.shape[0]
    bev = cv2.resize(bev, (int(bev.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    x0 = x1 - bev.shape[1]
    canvas[y0:y0 + h, x0:x1] = bev
    cv2.rectangle(canvas, (x0, y0), (x1, y0 + h), WHITE, 1)
    _put(canvas, "BEV canvas", (x0 + 6, y0 + 20), scale=0.5)
    _put(canvas, "white = front edge read", (x0 + 6, y0 + h - 10), scale=0.42)


def render_flight(key, ep, writer, fps):
    df, soft, lines = load_flight(ep)
    calib = load_calibrations(data.CALIB)[ep]
    seq = load_sequence(data.ROOT / "datasets/xp12_dataset", ep)
    frames = {f.index: f for f in seq.frames}
    scene = scene_from_calibration(calib)
    flight = data.full_flight(ep)
    airport, rwy = data.meta(ep)
    meta = METHODS[key]

    right_w = W_OUT - 1280
    s_img, s_pan = series(df, key, right_w, H_OUT)
    t_img, t_pan = top_view(flight, calib, 1280, H_OUT - 720)
    mask_bgr = np.array(BGR["mask"], np.float32)

    # a title card, so the two flights read as two runs
    card = np.full((H_OUT, W_OUT, 3), 250, np.uint8)
    for text, y, sc, th in ((meta["name"], 420, 1.4, 3), (f"Flight {ep}   {airport} rwy {rwy}", 490, 1.0, 2),
                            (meta["what"], 560, 0.7, 1), (f"inputs: {meta['inputs']}", 600, 0.7, 1),
                            ("held-out flight: unseen by every model in this comparison", 660, 0.7, 1)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        cv2.putText(card, text, ((W_OUT - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                    BGR["ink"] if y != 420 else BGR[key], th, cv2.LINE_AA)
    for _ in range(int(fps * 2)):
        writer.write(card)

    for _, r in df.iterrows():
        f = frames[int(r.frame)]
        cam = seq.read_image(f).astype(np.float32)
        if key == "bev":
            m = np.zeros(cam.shape[:2], np.float32)
            if isinstance(r.poly_src, list):
                cv2.fillPoly(m, [_pts(r.poly_src).reshape(-1, 1, 2)], 1.0)
            m = m[..., None]
        else:
            m = soft[int(r.frame)].astype(np.float32)[..., None] / 255
        cam = np.clip(cam * (1 - 0.5 * m) + mask_bgr * 0.5 * m, 0, 255).astype(np.uint8)
        truth_q = np.array([f.corners_px[n] for n in ("TR", "TL", "BL", "BR")])
        _poly(cam, truth_q, WHITE, 1)
        if key in HEADING and np.isfinite(r[HEADING[key]]):
            params = GDofParams.from_pose(pose_from_frame(f, calib), scene)
            lat = r.lat_est if key == "samuel" else r.fused_lat
            _poly(cam, _quad(scene, params, r[HEADING[key]], lat), BGR[key], 1)
        if key in ("tilt", "fused") and r.xtrack_on:
            ys = np.nonzero(soft[int(r.frame)] > 64)[0]
            if len(ys):
                for ln in lines[int(r.frame)]:
                    seg = None if ln is None else _line_segment(ln, ys.min(), ys.max())
                    if seg:
                        cv2.line(cam, seg[0], seg[1], BGR["tilt"], 1, cv2.LINE_AA)
        elif key == "bev" and isinstance(r.poly_src, list):
            cv2.polylines(cam, [_pts(r.poly_src).reshape(-1, 1, 2)], True, BGR["bev"], 1,
                          cv2.LINE_AA)

        canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
        canvas[:720, :1280] = cam
        _zoom(canvas, cam, truth_q, 10, 10, 380)
        if key == "bev":
            _bev_inset(canvas, ep, r, 1270, 150, 420)

        x0, y = 410, 36
        _put(canvas, f"{meta['name']}", (x0, y), HUD_BGR[key], 0.7, 2)
        y += 30
        _put(canvas, f"Flight {ep}  {airport} rwy {rwy}   distance {r.range_m:5.0f} m   "
                     f"height {r.height_m:4.0f} m", (x0, y))
        y += 30
        est = r[key]
        if r.xtrack_on:
            _put(canvas, f"CROSS-TRACK  truth {r.truth_m:+6.1f}  est {est:+6.1f}  "
                         f"err {est - r.truth_m:+.1f} m", (x0, y), HUD_BGR[key], 0.6, 2)
        else:
            _put(canvas, f"CROSS-TRACK  scored from 1000 m  ({r.range_m - data.XTRACK_FROM_M:.0f} m to go)",
                 (x0, y), (200, 200, 200))
        if key in HEADING:
            yh = r[HEADING[key]]
            y += 30
            _put(canvas, f"HEADING  truth {r.yaw_gt:+5.2f}  est {yh:+5.2f}  "
                         f"err {yh - r.yaw_gt:+.2f} deg", (x0, y), HUD_BGR[key], 0.6, 2)

        legend = [("truth runway", WHITE)]
        legend += {"samuel": [("Samuel's fitted runway", BGR["samuel"]), ("YOLO mask", BGR["mask"])],
                   "tilt": [("our edge lines", BGR["tilt"]), ("YOLO mask", BGR["mask"])],
                   "bev": [("BEV mask, back-projected", BGR["bev"])],
                   "fused": [("fused runway (heading + cross-track)", BGR["fused"]),
                             ("tilt edge lines (2nd measurement)", BGR["tilt"]),
                             ("YOLO mask", BGR["mask"])]}[key]
        ly = 700 - 24 * len(legend)
        for text, c in legend:
            cv2.line(canvas, (20, ly - 6), (50, ly - 6), c, 3)
            _put(canvas, text, (58, ly), scale=0.5)
            ly += 24

        tv = t_img.copy()
        p0 = t_pan.px(-r.range_m, r.truth_m)
        if key in HEADING:
            for yaw, c, t in ((r.yaw_gt, BGR["truth"], 4), (r[HEADING[key]], BGR[key], 2)):
                p1 = t_pan.px(-r.range_m + 400 * math.cos(math.radians(yaw)),
                              r.truth_m + 400 * math.sin(math.radians(yaw)))
                cv2.arrowedLine(tv, p0, p1, c, t, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(tv, p0, 5, BGR["truth"], -1, cv2.LINE_AA)
        if r.xtrack_on and np.isfinite(est):
            cv2.circle(tv, t_pan.px(-r.range_m, est), 7, BGR[key], 2, cv2.LINE_AA)
        canvas[720:, :1280] = tv[:H_OUT - 720, :1280]

        sv = s_img.copy()
        for pan in s_pan:
            xp, _ = pan.px(r.range_m, 0)
            cv2.line(sv, (xp, 0), (xp, sv.shape[0]), (150, 150, 150), 1)
        if r.xtrack_on and np.isfinite(est):
            cv2.circle(sv, s_pan[0].px(r.range_m, est), 5, BGR[key], -1, cv2.LINE_AA)
        if key in HEADING:
            cv2.circle(sv, s_pan[2].px(r.range_m, r[HEADING[key]]), 5, BGR[key], -1, cv2.LINE_AA)
        canvas[:, 1280:] = sv[:H_OUT, :right_w]
        writer.write(canvas)
    for _ in range(int(fps * 1.5)):   # hold the last frame
        writer.write(canvas)


def render(key, out_dir, fps):
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.mp4"
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, H_OUT))
    for ep in FLIGHTS:
        render_flight(key, ep, wr, fps)
    wr.release()
    out = Path(out_dir) / f"compare_{key}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def summary(out_dir):
    """Figure + table: all four cross-track methods on both flights."""
    rows, fig = [], plt.figure(figsize=(14, 7.5), dpi=150)
    fig.patch.set_facecolor(HEX["surface"])
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.3], hspace=0.45, wspace=0.18)
    for j, ep in enumerate(FLIGHTS):
        df, _, _ = load_flight(ep)
        xt = df[df.xtrack_on]
        a0, a1 = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        a0.plot(xt.range_m, xt.truth_m, color=RGB["truth"], lw=2.2, label="truth")
        for k in METHODS:
            e = xt[k] - xt.truth_m
            ls = "--" if k == "fused" else "-"
            a0.plot(xt.range_m, xt[k], color=RGB[k], lw=1.6, ls=ls, label=METHODS[k]["name"])
            a1.plot(xt.range_m, e.abs(), color=RGB[k], lw=1.4, ls=ls)
            for lo, hi in ((150, 300), (300, 600), (600, 1000), (150, 1000)):
                s = e[(xt.range_m > lo) & (xt.range_m <= hi)].abs()
                rows.append({"flight": ep, "method": k, "bin": f"{lo}-{hi}",
                             "MAE": s.mean(), "p95": s.quantile(0.95), "max": s.max(),
                             "bias": e[(xt.range_m > lo) & (xt.range_m <= hi)].mean()})
        airport, rwy = data.meta(ep)
        a0.set_title(f"Flight {ep}  {airport} rwy {rwy}: cross-track (m, + right)", loc="left",
                     fontsize=11, color=HEX["ink"])
        a1.set_title("|error| (m)", loc="left", fontsize=10, color=HEX["ink"])
        for a in (a0, a1):
            a.set_xlim(1010, 140)
            _style(a)
        a1.set_ylim(0, 4)
        a1.set_xlabel("distance to threshold (m)", color=HEX["ink2"])
        if j == 0:
            a0.legend(fontsize=8, frameon=False, loc="best")
    fig.savefig(out_dir / "compare_summary.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    t = pd.DataFrame(rows)
    t.round(3).to_csv(out_dir / "compare_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(t[t.bin == "150-1000"].pivot(index="method", columns="flight",
                                       values=["MAE", "p95", "bias"]).round(2))
    both = t[t.bin != "150-1000"].groupby(["method", "bin"]).MAE.mean().unstack().round(2)
    print(both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--out", default="experiments/joint_demo/media")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_summary:
        summary(out)
    for k in args.methods:
        print(render(k, out, args.fps), flush=True)


if __name__ == "__main__":
    main()
