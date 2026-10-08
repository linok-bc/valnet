"""Visualize annotated vs reprojected runway corners."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from runway_detection.lard.loader import LardSample
from runway_detection.lard.projection import CORNER_NAMES


def _draw_corners(ax, corners: dict[str, tuple[float, float]], *, color: str, label: str):
    xs = [corners[name][0] for name in CORNER_NAMES]
    ys = [corners[name][1] for name in CORNER_NAMES]
    order = [0, 1, 2, 3, 0]
    ax.plot(
        [xs[i] for i in order],
        [ys[i] for i in order],
        color=color,
        linewidth=2,
        marker="o",
        markersize=6,
        label=label,
    )
    for name in CORNER_NAMES:
        x, y = corners[name]
        ax.text(x + 6, y + 6, name, color=color, fontsize=9, weight="bold")


def plot_corner_overlay(
    sample: LardSample,
    projected: dict[str, tuple[float, float]],
    *,
    errors: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    image = Image.open(sample.image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    _draw_corners(ax, sample.annotated_corners, color="lime", label="annotated")
    _draw_corners(ax, projected, color="red", label="projected")

    if errors is not None:
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        subtitle = f"mean={mean_err:.2f}px  max={max_err:.2f}px"
    else:
        subtitle = ""

    if title is None:
        title = (
            f"{sample.airport} rwy {sample.runway} | "
            f"yaw={sample.yaw:.1f} pitch={sample.pitch:.1f} roll={sample.roll:.1f}"
        )
    if subtitle:
        title = f"{title}\n{subtitle}"
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
