#!/usr/bin/env python
"""Replay one test flight: YOLO mask, fitted edge lines, and the live estimate.

Each output frame shows

    top left     the camera frame, with the mask tinted and the two fitted edge
                 lines drawn up to where they meet (the measured vanishing point)
    bottom left  the runway zoomed, same overlay -- at 1.5 km the runway is ~40
                 rows tall, so without the zoom there is nothing to see
    right        cross-track against range: truth, the per-frame estimate and
                 the gated Kalman output, drawn up to the current frame, with a
                 readout of the numbers

It runs the same pipeline as eval.py (YOLO at `--imgsz`, `estimate_from_mask`,
`free_width`, gated `kalman_track`), so the numbers match the report. Heading is
never used. Output is H.264 so it plays in a browser.

    python -m experiments.edge_lines.video --episode 000135
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, load_runways
from experiments.bev.ryu import kalman_track
from experiments.edge_lines.eval import yolo_masks
from experiments.edge_lines.geometry import (crosstrack, edge_points, fit_line,
                                             line_from_fit)

W_OUT, H_OUT = 1280, 720
LEFT_W = 768                      # camera 768x432 on top, zoom 768x288 below
CAM_H, ZOOM_H = 432, 288
TINT = np.array([255, 170, 40], np.uint8)     # mask tint, RGB
LINE = (40, 220, 255)                         # fitted edges, RGB


def fit(mask, cam, f, width_m):
    """Everything the overlay needs: two image lines and the estimate, or None."""
    left, right, D = edge_points(mask, cam, f.roll_deg, f.pitch_deg)
    if D is None or len(left) < 5 or len(right) < 5:
        return None
    fl, fr = fit_line(left), fit_line(right)
    if fl is None or fr is None:
        return None
    lL, lR = line_from_fit(fl[0], fl[1], D), line_from_fit(fr[0], fr[1], D)
    out = crosstrack(lL, lR, cam, f.roll_deg, f.pitch_deg, f.height_m, width_m, f.yaw_deg)
    return {"lL": lL, "lR": lR, "est": out["free_width"]}


def u_at(l, v):
    return -(l[1] * v + l[2]) / l[0]


def draw_overlay(img, mask, lines, x0, y0, scale):
    """Tint the mask and draw both edges, in an image that is a crop at (x0, y0) x scale."""
    out = img.copy()
    if mask is not None:
        m = cv2.resize(mask[y0:y0 + int(round(img.shape[0] / scale)),
                            x0:x0 + int(round(img.shape[1] / scale))],
                       (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (0.55 * out[m] + 0.45 * TINT).astype(np.uint8)
    if lines is not None:
        vp = np.cross(lines["lL"], lines["lR"])
        vs = np.nonzero(mask.any(axis=1))[0]
        v_bot = vs.max() + 4
        v_top = vp[1] / vp[2] if abs(vp[2]) > 1e-9 else vs.min()
        v_top = max(v_top, vs.min() - 40)
        th = int(np.clip(round(scale * 0.4), 1, 3))
        for l in (lines["lL"], lines["lR"]):
            p = [(u_at(l, v) - x0) * scale for v in (v_top, v_bot)]
            q = [(v - y0) * scale for v in (v_top, v_bot)]
            cv2.line(out, (int(p[0]), int(q[0])), (int(p[1]), int(q[1])), LINE, th, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--weights", default="baseline/runs/stock/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--range", default="150,1500")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gate", type=float, default=9.0)
    ap.add_argument("--out-dir", default="experiments/edge_lines/media/video")
    args = ap.parse_args()

    cam = CameraModel(**json.loads(Path(args.camera).read_text())["camera"])
    ep = Episode(Path(args.root) / args.episode)
    rw = load_runways()[(ep.airport, ep.runway_name)]
    lo, hi = map(float, args.range.split(","))
    frames = [f for f in ep.frames() if f.labelled and lo <= f.range_m <= hi]
    frames.sort(key=lambda f: f.along_m)
    print(f"{ep}: {len(frames)} frames, YOLO at {args.imgsz}")

    masks = yolo_masks(args.weights, frames, 0.25, None, imgsz=args.imgsz)
    fits = [None if m is None else fit(m, cam, f, rw.width_m) for m, f in zip(masks, frames)]
    rng = np.array([f.range_m for f in frames])
    truth = np.array([f.right_m for f in frames])
    est = np.array([np.nan if x is None else x["est"] for x in fits])
    kf = kalman_track(est, np.array([f.along_m for f in frames]), r=4.0, gate=args.gate)
    mae_raw, mae_kf = np.nanmean(np.abs(est - truth)), np.nanmean(np.abs(kf - truth))
    print(f"  MAE per-frame {mae_raw:.2f} m, gated Kalman {mae_kf:.2f} m")

    # ---- the plot, redrawn per frame into a fixed figure
    dpi = 100
    fig = plt.figure(figsize=((W_OUT - LEFT_W) / dpi, H_OUT / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0f1720")
    ax = fig.add_axes([0.15, 0.36, 0.8, 0.54])
    ax.set_facecolor("#0f1720")
    for s in ax.spines.values():
        s.set_color("#44505c")
    ax.tick_params(colors="#b9c3cc", labelsize=9)
    ax.grid(color="#26313b", lw=0.8)
    ax.set_xlim(hi, lo)
    vals = np.r_[truth, est[np.isfinite(est)], kf[np.isfinite(kf)]]
    lo_y, hi_y = np.percentile(vals, 1) - 3, np.percentile(vals, 99) + 3
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("range to threshold (m)", color="#b9c3cc", fontsize=10)
    ax.set_ylabel("cross-track, m right of centreline", color="#b9c3cc", fontsize=10)
    ax.set_title(f"{ep.airport} {ep.runway_name}  ·  flight {args.episode}",
                 color="#e8edf2", fontsize=12, loc="left", pad=10)
    l_truth, = ax.plot([], [], color="#e8edf2", lw=2.2, label="truth")
    l_raw, = ax.plot([], [], ".", color="#f4a23b", ms=4, label="per-frame estimate")
    l_kf, = ax.plot([], [], color="#3db8ff", lw=1.8, label="gated Kalman")
    cursor = ax.axvline(hi, color="#6b7885", lw=0.8)
    ax.legend(loc="upper left", fontsize=8.5, facecolor="#0f1720", edgecolor="#44505c",
              labelcolor="#e8edf2")
    readout = fig.text(0.06, 0.03, "", color="#e8edf2", fontsize=11, family="monospace",
                       va="bottom", linespacing=1.55)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{args.episode}_{ep.airport}_{ep.runway_name}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{W_OUT}x{H_OUT}", "-r", str(args.fps),
                           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                          stdin=subprocess.PIPE)
    zoom_box = None
    for i, (f, m, x) in enumerate(zip(frames, masks, fits)):
        img = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2RGB)
        canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
        canvas[:] = (15, 23, 32)

        s_cam = LEFT_W / cam.width
        small = cv2.resize(img, (LEFT_W, CAM_H), interpolation=cv2.INTER_AREA)
        canvas[:CAM_H, :LEFT_W] = draw_overlay(small, m, x, 0, 0, s_cam) if m is not None else small

        # zoom box around the runway, 8:3, smoothed so the view does not jitter
        q = f.polygon_px(cam.width, cam.height)
        cx, cy = q[:, 0].mean(), q[:, 1].mean()
        # fit the whole runway: wide enough for its span, tall enough for its depth
        half_w = max(np.ptp(q[:, 0]) * 0.75, np.ptp(q[:, 1]) * 0.7 * LEFT_W / ZOOM_H, 48)
        box = np.array([cx, cy, half_w])
        zoom_box = box if zoom_box is None else 0.8 * zoom_box + 0.2 * box
        cx, cy, half_w = zoom_box
        half_h = half_w * ZOOM_H / LEFT_W
        x0 = int(np.clip(cx - half_w, 0, cam.width - 2 * half_w))
        y0 = int(np.clip(cy - half_h, 0, cam.height - 2 * half_h))
        crop = img[y0:y0 + int(2 * half_h), x0:x0 + int(2 * half_w)]
        s_zoom = LEFT_W / crop.shape[1]
        zoom = cv2.resize(crop, (LEFT_W, ZOOM_H), interpolation=cv2.INTER_CUBIC)
        canvas[CAM_H:, :LEFT_W] = draw_overlay(zoom, m, x, x0, y0, s_zoom) if m is not None else zoom
        # where the zoom comes from, on the camera view
        cv2.rectangle(canvas, (int(x0 * s_cam), int(y0 * s_cam)),
                      (int((x0 + 2 * half_w) * s_cam), int((y0 + 2 * half_h) * s_cam)),
                      (200, 210, 220), 1)
        cv2.putText(canvas, f"zoom x{s_zoom:.1f}", (10, CAM_H + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 236, 242), 1, cv2.LINE_AA)

        k = slice(0, i + 1)
        l_truth.set_data(rng[k], truth[k])
        l_raw.set_data(rng[k], est[k])
        l_kf.set_data(rng[k], kf[k])
        cursor.set_xdata([f.range_m, f.range_m])
        e_raw = est[i] - truth[i] if np.isfinite(est[i]) else np.nan
        readout.set_text(
            f"range      {f.range_m:7.0f} m      height {f.height_m:5.1f} m\n"
            f"truth      {truth[i]:+7.2f} m\n"
            f"estimate   {est[i]:+7.2f} m   error {e_raw:+6.2f}\n"
            f"filtered   {kf[i]:+7.2f} m   error {kf[i] - truth[i]:+6.2f}\n"
            f"heading used: none   flight MAE {mae_kf:.2f} m")
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        canvas[:, LEFT_W:] = plot[:H_OUT, :W_OUT - LEFT_W]
        ff.stdin.write(canvas.tobytes())
        if i == len(frames) // 2:
            cv2.imwrite(str(mp4.with_suffix(".jpg")), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    ff.stdin.close()
    ff.wait()
    print(f"  wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    (out_dir / f"{args.episode}.json").write_text(json.dumps({
        "episode": args.episode, "airport": ep.airport, "runway": ep.runway_name,
        "frames": len(frames), "mae_raw_m": float(mae_raw), "mae_kf_m": float(mae_kf),
        "video": mp4.name, "poster": mp4.with_suffix(".jpg").name}))


if __name__ == "__main__":
    main()
