"""Visualize g_dof rendered masks against LARD annotations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pose_estimation.g_dof import corners_dict_to_polygon
from runway_detection.lard.loader import LardSample
from runway_detection.lard.projection import CORNER_NAMES


def plot_g_dof_overlay(
    sample: LardSample,
    mask: np.ndarray,
    projected_corners: dict[str, tuple[float, float]],
    *,
    iou: float | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    image = Image.open(sample.image_path).convert("RGB")
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[..., 1] = mask.astype(float)
    overlay[..., 3] = mask.astype(float) * 0.45

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(overlay)

    ann = corners_dict_to_polygon(sample.annotated_corners)
    proj = corners_dict_to_polygon(projected_corners)
    order = [0, 1, 2, 3, 0]
    ax.plot(ann[order, 0], ann[order, 1], "c-", linewidth=2, marker="o", label="annotated")
    ax.plot(proj[order, 0], proj[order, 1], "r--", linewidth=2, marker="x", label="g_dof")
    for name in CORNER_NAMES:
        x, y = sample.annotated_corners[name]
        ax.text(x + 4, y + 4, name, color="cyan", fontsize=8)

    if title is None:
        title = f"{sample.airport} rwy {sample.runway}"
    if iou is not None:
        title = f"{title}\nmask IoU @ GT = {iou:.4f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right")
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
