"""Visualize homography rectification."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from runway_detection.homography.measure import PlanePoseEstimate
from runway_detection.homography.plane_homography import (
    PlaneHomography,
    image_points_to_plane,
    warp_image_to_plane,
)
from runway_detection.lard.projection import CORNER_NAMES


def draw_corners(ax, corners: dict[str, tuple[float, float]], *, color: str, label: str):
    xs = [corners[n][0] for n in CORNER_NAMES]
    ys = [corners[n][1] for n in CORNER_NAMES]
    order = [0, 1, 2, 3, 0]
    ax.plot(
        [xs[i] for i in order],
        [ys[i] for i in order],
        color=color,
        linewidth=2,
        marker="o",
        label=label,
    )


def plot_rectification(
    image_path: Path,
    *,
    homography: PlaneHomography,
    corners_px: dict[str, tuple[float, float]],
    estimate: PlanePoseEstimate,
    gt_heading_deg: float,
    gt_lateral_m: float,
    plane_roi_xy: np.ndarray | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corner_pts = np.array([corners_px[n] for n in CORNER_NAMES], dtype=np.float64)
    roi_xy = plane_roi_xy
    if roi_xy is None:
        roi_xy = image_points_to_plane(corner_pts, homography)
    warped, _ = warp_image_to_plane(
        bgr,
        homography,
        plane_roi_xy=roi_xy,
        margin_m=200.0,
        meters_per_pixel=1.5,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    draw_corners(axes[0], corners_px, color="lime", label="corners")
    axes[0].set_title("Image + corners")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Rectified (4-DOF H)\n"
        f"meas HE={estimate.heading_error_deg:.2f}° lat={estimate.lateral_offset_m:.1f}m\n"
        f"GT HE={gt_heading_deg:.2f}° lat={gt_lateral_m:.1f}m"
    )
    axes[1].axis("off")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
