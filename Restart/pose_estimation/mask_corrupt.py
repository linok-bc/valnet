"""
Procedural soft-mask corruption for synthetic pose-estimation tests (spec §6.1).

Uncalibrated first pass — parameters are hand-picked placeholders until
Stage-1 segmentation statistics are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CorruptionConfig:
    boundary_blur_sigma: float = 1.5
    erosion_dilate_prob: float = 0.5
    erosion_dilate_px: int = 1
    hole_prob: float = 0.25
    hole_count: int = 2
    hole_radius_px: int = 4
    blob_prob: float = 0.15
    blob_count: int = 1
    blob_radius_px: int = 6
    interior_beta_alpha: float = 22.0
    interior_beta_beta: float = 2.0
    noise_sigma: float = 0.02


def corrupt_mask(
    mask: np.ndarray,
    *,
    config: CorruptionConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Turn a hard binary mask into a soft observed mask in [0, 1].

    Steps: optional morphological jitter → holes/blobs → boundary blur →
    interior Beta noise.
    """
    config = config or CorruptionConfig()
    rng = rng or np.random.default_rng()
    h, w = mask.shape
    m = mask.astype(np.uint8)

    if rng.random() < config.erosion_dilate_prob:
        k = config.erosion_dilate_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        if rng.random() < 0.5:
            m = cv2.erode(m, kernel)
        else:
            m = cv2.dilate(m, kernel)

    if rng.random() < config.hole_prob and m.any():
        ys, xs = np.where(m > 0)
        for _ in range(config.hole_count):
            idx = rng.integers(0, len(xs))
            cv2.circle(m, (int(xs[idx]), int(ys[idx])), config.hole_radius_px, 0, -1)

    if rng.random() < config.blob_prob:
        for _ in range(config.blob_count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            cv2.circle(m, (x, y), config.blob_radius_px, 1, -1)

    soft = m.astype(np.float32)
    if config.boundary_blur_sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), config.boundary_blur_sigma)

    interior = rng.beta(config.interior_beta_alpha, config.interior_beta_beta, size=(h, w)).astype(
        np.float32
    )
    soft = np.where(m > 0, np.maximum(soft, interior * m), soft)
    soft = np.clip(soft + rng.normal(0.0, config.noise_sigma, size=(h, w)), 0.0, 1.0).astype(
        np.float32
    )
    return soft
