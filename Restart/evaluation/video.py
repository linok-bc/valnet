"""
Replay video: camera view with GT vs predicted runway, DOF time series, top view.

Layout (1920 × 1080):

    ┌──────────────────────────────┬──────────────┐
    │ camera 1280×720              │ yaw  (GT/est)│
    │  - mask overlay (blue)       ├──────────────┤
    │  - GT runway quad (green)    │ lat  (GT/est)│
    │  - predicted quad (red)      ├──────────────┤
    ├──────────────────────────────┤ errors       │
    │ top view: runway + GT/est    │              │
    │ trajectory, heading arrows   │              │
    └──────────────────────────────┴──────────────┘

Static curves are drawn once with matplotlib; per-frame cursors are drawn
with OpenCV (fast).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose_estimation.g_dof import GDofParams, GDofState, corners_dict_to_polygon, g_dof_corners
from runway_detection.xp12.calibration import XP12Calibration, pose_from_frame, scene_from_calibration
from runway_detection.xp12.loader import XP12Sequence

W_OUT, H_OUT = 1920, 1080
# Validated categorical palette (dataviz reference): aqua = GT, orange = estimate, blue = measurement/mask
GT_BGR = (0x7A, 0xAF, 0x1B)
EST_BGR = (0x34, 0x68, 0xEB)
MEAS_BGR = (0xD6, 0x78, 0x2A)
MASK_BGR = np.array([0xD6, 0x78, 0x2A], dtype=np.float32)
GT_RGB = tuple(c / 255 for c in GT_BGR[::-1])
EST_RGB = tuple(c / 255 for c in EST_BGR[::-1])
MEAS_RGB = tuple(c / 255 for c in MEAS_BGR[::-1])


def _fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


class _Panel:
    """A matplotlib axes rendered once; maps data coords → panel pixels."""

    def __init__(self, fig, ax, img: np.ndarray):
        self.img = img
        fig.canvas.draw()
        self._trans = ax.transData
        self._h = img.shape[0]

    def px(self, x: float, y: float) -> tuple[int, int]:
        u, v = self._trans.transform((x, y))
        return int(round(u)), int(round(self._h - v))


def _series_panels(df: pd.DataFrame, width: int, height: int) -> tuple[np.ndarray, list[_Panel]]:
    dpi = 100
    fig, axes = plt.subplots(
        4, 1, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.6, 1.6]},
    )
    x = df["frame"].to_numpy()
    specs = [
        ("yaw_gt", "yaw_est", "yaw_meas", "sigma_yaw", "yaw (deg, + nose right)"),
        ("lat_gt", "lat_est", "lat_meas", "sigma_lat", "lateral offset (m, + left)"),
    ]
    for ax, (g, e, m, s, title) in zip(axes[:2], specs):
        ax.plot(x, df[g], color=GT_RGB, lw=2.0, label="GT")
        if df[m].notna().any():
            ax.plot(x, df[m], ".", color=MEAS_RGB, ms=2.5, alpha=0.6, label="measurement")
        ax.plot(x, df[e], color=EST_RGB, lw=1.4, label="estimate")
        if df[s].notna().any():
            lo = df[e] - 2 * df[s]
            hi = df[e] + 2 * df[s]
            ax.fill_between(x, lo, hi, color=EST_RGB, alpha=0.15, lw=0, label="±2σ")
        lo_y = np.nanpercentile(np.r_[df[g], df[e]], 1)
        hi_y = np.nanpercentile(np.r_[df[g], df[e]], 99)
        pad = 0.15 * max(hi_y - lo_y, 1e-3)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right", ncol=4)
    for ax, col, unit in ((axes[2], "yaw_err", "deg"), (axes[3], "lat_err", "m")):
        ax.plot(x, df[col], color=EST_RGB, lw=1.2)
        ax.axhline(0, color=GT_RGB, lw=1.0)
        v = np.nanpercentile(np.abs(df[col]), 98) if df[col].notna().any() else 1.0
        ax.set_ylim(-1.2 * v - 1e-3, 1.2 * v + 1e-3)
        ax.set_title(f"{col.split('_')[0]} error, estimate − GT ({unit})", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[3].set_xlabel("frame", fontsize=9)
    fig.tight_layout()
    img = _fig_to_bgr(fig)
    panels = [_Panel(fig, a, img) for a in axes]
    plt.close(fig)
    return img, panels


def _top_view(df: pd.DataFrame, calib: XP12Calibration, width: int, height: int) -> tuple[np.ndarray, _Panel]:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    W, L = calib.width_m, calib.length_m
    ax.fill([0, L, L, 0], [-W / 2, -W / 2, W / 2, W / 2], color="0.55", zorder=1)
    ax.plot([0, L], [0, 0], color="w", lw=0.8, ls="--", zorder=2)
    ax.plot(-df["along_m"], df["lat_gt"], color=GT_RGB, lw=2, label="GT track", zorder=3)
    ax.plot(-df["along_m"], df["lat_est"], color=EST_RGB, lw=1.2, label="estimated", zorder=4)
    x_min = -df["along_m"].max() - 100
    ax.set_xlim(x_min, min(L, 600))
    lat_all = np.r_[df["lat_gt"], df["lat_est"]]
    lim = max(60.0, np.nanpercentile(np.abs(lat_all), 99) * 1.3)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("along runway axis (m, threshold at 0)", fontsize=9)
    ax.set_ylabel("lateral (m, + left)", fontsize=9)
    ax.set_title("top view — runway frame", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    img = _fig_to_bgr(fig)
    panel = _Panel(fig, ax, img)
    plt.close(fig)
    return img, panel


def _quad(scene, params, yaw, lat) -> np.ndarray | None:
    try:
        return corners_dict_to_polygon(g_dof_corners(scene, params, GDofState(float(yaw), float(lat))))
    except ValueError:
        return None


def _draw_quad(img, quad, color, thickness=2, centerline=False):
    if quad is None or not np.all(np.isfinite(quad)) or np.abs(quad).max() > 1e5:
        return
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    if centerline:
        far = 0.5 * (quad[0] + quad[1])
        near = 0.5 * (quad[2] + quad[3])
        d = near - far
        ext = near + 2.0 * d  # extend toward the camera
        cv2.line(img, tuple(np.round(far).astype(int)), tuple(np.round(ext).astype(int)), color, 1, cv2.LINE_AA)


def _put(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _shade(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _zoom_inset(canvas, cam, quads, x0=8, y0=8, size=300):
    """Magnified crop around the runway (it is only ~20 px tall at long range)."""
    pts = np.vstack([q for q in quads if q is not None])
    c = np.nanmean(pts, axis=0)
    span = max(np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0]), np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1]), 40) * 1.6
    half = int(min(span, 360) / 2)
    cx, cy = int(np.clip(c[0], half, cam.shape[1] - half)), int(np.clip(c[1], half, cam.shape[0] - half))
    crop = cam[cy - half : cy + half, cx - half : cx + half]
    if crop.size == 0:
        return
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    canvas[y0 : y0 + size, x0 : x0 + size] = crop
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
    _put(canvas, f"zoom x{size / (2 * half):.1f}", (x0 + 6, y0 + size - 8), scale=0.45)


def render_video(
    seq: XP12Sequence,
    calib: XP12Calibration,
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    masks: dict[int, np.ndarray] | None = None,
    title: str = "",
    fps: float = 15.0,
) -> Path:
    """``df``: per-frame rows from ``evaluation.runner.run_sequence`` for this sequence."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("frame").reset_index(drop=True)
    scene = scene_from_calibration(calib)
    frames = {f.index: f for f in seq.frames}

    right_w = W_OUT - 1280
    series_img, series_panels = _series_panels(df, right_w, H_OUT)
    top_img, top_panel = _top_view(df, calib, 1280, H_OUT - 720)

    tmp_dir = Path(tempfile.mkdtemp())
    raw = tmp_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, H_OUT))
    try:
        for _, r in df.iterrows():
            f = frames[int(r.frame)]
            pose = pose_from_frame(f, calib)
            params = GDofParams.from_pose(pose, scene)
            cam = seq.read_image(f).astype(np.float32)
            if masks is not None and int(r.frame) in masks:
                m = masks[int(r.frame)][..., None]
                cam = cam * (1 - 0.45 * m) + MASK_BGR * 0.45 * m
            cam = np.clip(cam, 0, 255).astype(np.uint8)
            gt_quad = np.array([f.corners_px[n] for n in ("TR", "TL", "BL", "BR")])
            est_quad = _quad(scene, params, r.yaw_est, r.lat_est) if np.isfinite(r.yaw_est) else None
            _draw_quad(cam, gt_quad, GT_BGR, 2, centerline=True)
            _draw_quad(cam, est_quad, EST_BGR, 2, centerline=True)

            canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
            canvas[:720, :1280] = cam
            _zoom_inset(canvas, cam, [gt_quad, est_quad])
            _shade(canvas, 318, 8, 1000, 150)
            _shade(canvas, 1090, 8, 1272, 94)
            y = 30
            lines = [
                (title, (255, 255, 255)),
                (f"{seq.airport} {seq.runway}  frame {int(r.frame)}  along {r.along_m:6.0f} m  height {r.height_m:5.0f} m", (255, 255, 255)),
                (f"yaw  GT {r.yaw_gt:+6.2f}  est {r.yaw_est:+6.2f}  err {r.yaw_err:+.2f} deg", (255, 255, 255)),
                (f"lat  GT {r.lat_gt:+6.1f}  est {r.lat_est:+6.1f}  err {r.lat_err:+.1f} m", (255, 255, 255)),
                (f"status {r.status}   {r.runtime_ms:.0f} ms", (200, 200, 200)),
            ]
            for text, col in lines:
                if text:
                    _put(canvas, text, (330, y), col)
                    y += 26
            if r.status == "fail":
                _put(canvas, "NO VISION MEASUREMENT (coasting)", (330, y + 4), (0, 200, 255), 0.6, 2)
            _put(canvas, "GT runway", (1100, 30), GT_BGR)
            _put(canvas, "predicted", (1100, 56), EST_BGR)
            if masks is not None:
                _put(canvas, "seg mask", (1100, 82), tuple(int(c) for c in MASK_BGR))

            # top view with current positions + heading arrows
            tv = top_img.copy()
            for yaw, lat, col in ((r.yaw_gt, r.lat_gt, GT_BGR), (r.yaw_est, r.lat_est, EST_BGR)):
                if not np.isfinite(yaw):
                    continue
                p0 = top_panel.px(-r.along_m, lat)
                # heading arrow 250 m long (+yaw = nose right = toward −lat)
                p1 = top_panel.px(-r.along_m + 250 * math.cos(math.radians(yaw)), lat - 250 * math.sin(math.radians(yaw)))
                cv2.arrowedLine(tv, p0, p1, col, 2, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(tv, p0, 5, col, -1, cv2.LINE_AA)
            canvas[720:, :1280] = tv[: H_OUT - 720, :1280]

            sv = series_img.copy()
            for panel, (g, e) in zip(series_panels, (("yaw_gt", "yaw_est"), ("lat_gt", "lat_est"), (None, None), (None, None))):
                x_px, _ = panel.px(r.frame, 0)
                cv2.line(sv, (x_px, 0), (x_px, sv.shape[0]), (90, 90, 90), 1)
                if g is not None:
                    cv2.circle(sv, panel.px(r.frame, r[g]), 5, GT_BGR, -1, cv2.LINE_AA)
                    if np.isfinite(r[e]):
                        cv2.circle(sv, panel.px(r.frame, r[e]), 5, EST_BGR, -1, cv2.LINE_AA)
            canvas[:, 1280:] = sv[:H_OUT, :right_w]
            writer.write(canvas)
    finally:
        writer.release()

    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", str(out_path)],
            check=True,
        )
    else:
        shutil.move(str(raw), out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
