"""Interactive frame-by-frame viewer for sequential g_dof mask fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from PIL import Image

from pose_estimation.g_dof import (
    GDofParams,
    GDofState,
    corners_dict_to_polygon,
    g_dof,
    g_dof_corners,
    mask_iou,
)
from pose_estimation.runway_model import RunwayScene
from runway_detection.lard.loader import LardSample
from runway_detection.lard.projection import CORNER_NAMES


@dataclass
class SequenceFrameResult:
    """One frame of sequential tracking with poses for visualization."""

    sample: LardSample
    scene: RunwayScene
    params: GDofParams
    observed: np.ndarray
    state_gt: GDofState
    state_init: GDofState
    state_est: GDofState
    dice: float
    mask_iou_yolo_gt: float
    yaw_err_deg: float
    lat_err_m: float
    seq_idx: int


def _mask_rgba(mask: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    h, w = mask.shape
    layer = np.zeros((h, w, 4), dtype=float)
    m = mask.astype(bool)
    layer[m, 0] = rgb[0]
    layer[m, 1] = rgb[1]
    layer[m, 2] = rgb[2]
    layer[m, 3] = alpha
    return layer


def _draw_quad(
    ax,
    corners: dict[str, tuple[float, float]],
    *,
    color: str,
    label: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    marker: str = "o",
    markersize: float = 5.0,
) -> None:
    pts = corners_dict_to_polygon(corners)
    order = [0, 1, 2, 3, 0]
    ax.plot(
        pts[order, 0],
        pts[order, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        label=label,
    )


def render_frame_figure(frame: SequenceFrameResult) -> tuple[plt.Figure, np.ndarray]:
    """Build figure for one sequence frame. Returns (fig, rgb image array)."""
    sample = frame.sample
    rgb = np.array(Image.open(sample.image_path).convert("RGB"))
    h, w = rgb.shape[:2]

    mask_gt = g_dof(frame.scene, frame.params, frame.state_gt)
    mask_init = g_dof(frame.scene, frame.params, frame.state_init)
    mask_est = g_dof(frame.scene, frame.params, frame.state_est)
    corners_gt = g_dof_corners(frame.scene, frame.params, frame.state_gt)
    corners_init = g_dof_corners(frame.scene, frame.params, frame.state_init)
    corners_est = g_dof_corners(frame.scene, frame.params, frame.state_est)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: image + mask overlays + quads
    ax = axes[0]
    ax.imshow(rgb)
    ax.imshow(_mask_rgba(frame.observed > 0.25, (0.2, 0.5, 1.0), 0.22), label="YOLO")
    ax.imshow(_mask_rgba(mask_gt, (0.0, 1.0, 0.2), 0.18))
    ax.imshow(_mask_rgba(mask_est, (1.0, 0.15, 0.15), 0.22))
    _draw_quad(ax, corners_gt, color="lime", label="GT g_dof", linewidth=2.5)
    _draw_quad(
        ax,
        corners_init,
        color="gold",
        label="Init (t−1)",
        linewidth=2.0,
        linestyle="--",
        marker="s",
    )
    _draw_quad(
        ax,
        corners_est,
        color="red",
        label="Est g_dof",
        linewidth=2.5,
        linestyle="-",
        marker="x",
        markersize=6.0,
    )
    ax.set_title(
        f"{sample.airport} rwy {sample.runway}  row {sample.row_index}  "
        f"[{frame.seq_idx + 1}]",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()

    # Right: mask comparison panels (GT | init | est | YOLO)
    ax2 = axes[1]
    tiles = np.zeros((h, w * 4, 3), dtype=np.uint8)
    panels = [
        (mask_gt, "GT"),
        (mask_init, "Init"),
        (mask_est, "Est"),
        (frame.observed > 0.25, "YOLO"),
    ]
    colors = [(0, 180, 0), (220, 180, 0), (220, 40, 40), (60, 120, 220)]
    for i, (m, title) in enumerate(panels):
        tile = np.zeros((h, w, 3), dtype=np.uint8)
        c = colors[i]
        tile[m.astype(bool)] = c
        tiles[:, i * w : (i + 1) * w] = tile
        ax2.text(
            i * w + w * 0.5,
            18,
            title,
            color="white",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )
    ax2.imshow(tiles)
    ax2.set_axis_off()
    ax2.set_title("Mask templates (GT · Init · Est · YOLO)", fontsize=10)

    info = (
        f"GT   yaw={frame.state_gt.yaw_cam_deg:+.2f}°  lat={frame.state_gt.lateral_offset_m:+.1f}m\n"
        f"Init yaw={frame.state_init.yaw_cam_deg:+.2f}°  lat={frame.state_init.lateral_offset_m:+.1f}m\n"
        f"Est  yaw={frame.state_est.yaw_cam_deg:+.2f}°  lat={frame.state_est.lateral_offset_m:+.1f}m  "
        f"(Δyaw={frame.yaw_err_deg:+.2f}° Δlat={frame.lat_err_m:+.1f}m)\n"
        f"dice={frame.dice:.3f}  maskIoU(YOLO,GT)={frame.mask_iou_yolo_gt:.3f}"
    )
    fig.text(0.5, 0.02, info, ha="center", fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, rgb


def save_frame_png(frame: SequenceFrameResult, path: Path) -> None:
    fig, _ = render_frame_figure(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def interactive_sequence_viewer(
    frames: list[SequenceFrameResult],
    *,
    title: str = "g_dof sequential tracking",
) -> None:
    """Scroll frames with slider or ← / → keys."""
    if not frames:
        raise ValueError("No frames to display")

    state = {"idx": 0, "fig": None, "slider": None, "updating": False}

    def _draw(idx: int) -> None:
        idx = int(np.clip(idx, 0, len(frames) - 1))
        state["idx"] = idx
        if state["fig"] is not None:
            plt.close(state["fig"])
        state["fig"], _ = render_frame_figure(frames[idx])
        state["fig"].suptitle(
            f"{title}  —  frame {idx + 1}/{len(frames)}  "
            f"(←/→ or slider)",
            fontsize=11,
        )
        if state["slider"] is not None:
            state["updating"] = True
            state["slider"].eventson = False
            state["slider"].set_val(idx)
            state["slider"].eventson = True
            state["updating"] = False
        state["fig"].canvas.draw_idle()
        state["fig"].canvas.flush_events()

    # Control figure with slider
    ctrl = plt.figure(figsize=(8, 1.5))
    ctrl.suptitle(title, fontsize=10)
    ax_slider = ctrl.add_axes([0.12, 0.35, 0.76, 0.18])
    slider = Slider(ax_slider, "Frame", 0, len(frames) - 1, valinit=0, valstep=1)
    state["slider"] = slider

    def _on_slider(val: float) -> None:
        if state["updating"]:
            return
        _draw(int(val))

    slider.on_changed(_on_slider)

    def _on_key(event) -> None:
        if event.key in ("right", "down"):
            _draw(state["idx"] + 1)
        elif event.key in ("left", "up"):
            _draw(state["idx"] - 1)

    ctrl.canvas.mpl_connect("key_press_event", _on_key)
    _draw(0)
    plt.show()
