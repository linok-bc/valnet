"""Region-of-interest helpers for mask-space pose optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROIBox:
    """Axis-aligned crop [y0:y1, x0:x1] with optional full-image fallback."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def crop(self, mask: np.ndarray) -> np.ndarray:
        return mask[self.y0 : self.y1, self.x0 : self.x1]

    def paste(self, full_shape: tuple[int, int], patch: np.ndarray) -> np.ndarray:
        out = np.zeros(full_shape, dtype=patch.dtype)
        out[self.y0 : self.y1, self.x0 : self.x1] = patch
        return out


def roi_from_mask(
    mask: np.ndarray,
    *,
    margin_px: int = 48,
    min_size: int = 64,
) -> ROIBox:
    """Bounding box of foreground pixels, expanded and clamped to image bounds."""
    h, w = mask.shape
    fg = np.argwhere(mask.astype(bool))
    if fg.size == 0:
        side = min(min_size, h, w)
        cy, cx = h // 2, w // 2
        half = side // 2
        return ROIBox(
            max(0, cy - half),
            min(h, cy - half + side),
            max(0, cx - half),
            min(w, cx - half + side),
        )

    y_min, x_min = fg.min(axis=0)
    y_max, x_max = fg.max(axis=0)
    y0 = max(0, int(y_min) - margin_px)
    x0 = max(0, int(x_min) - margin_px)
    y1 = min(h, int(y_max) + margin_px + 1)
    x1 = min(w, int(x_max) + margin_px + 1)

    if (y1 - y0) < min_size:
        pad = (min_size - (y1 - y0) + 1) // 2
        y0 = max(0, y0 - pad)
        y1 = min(h, y1 + pad)
    if (x1 - x0) < min_size:
        pad = (min_size - (x1 - x0) + 1) // 2
        x0 = max(0, x0 - pad)
        x1 = min(w, x1 + pad)
    return ROIBox(y0, y1, x0, x1)


def downsample_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize mask to (size, size) with area interpolation for soft/binary masks."""
    import cv2

    return cv2.resize(
        mask.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
