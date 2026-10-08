"""Mask-space distance metrics for render-and-compare (spec §4.3)."""

from __future__ import annotations

from enum import Enum

import numpy as np

from pose_estimation.roi import ROIBox


class MaskMetric(str, Enum):
    DICE = "dice"
    L2 = "l2"


def _as_float(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32)


def soft_dice(observed: np.ndarray, predicted: np.ndarray, *, eps: float = 1e-6) -> float:
    """Soft Dice in [0, 1]; higher is better."""
    y = _as_float(observed).ravel()
    p = _as_float(predicted).ravel()
    inter = float(np.dot(y, p))
    denom = float(y.sum() + p.sum())
    if denom < eps:
        return 1.0 if inter < eps else 0.0
    return (2.0 * inter + eps) / (denom + eps)


def dice_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Minimize 1 - Dice."""
    return 1.0 - soft_dice(observed, predicted)


def l2_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    diff = _as_float(observed) - _as_float(predicted)
    return float(np.mean(diff * diff))


def residuals_l2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (_as_float(observed) - _as_float(predicted)).ravel()


def metric_value(
    metric: MaskMetric,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if metric == MaskMetric.DICE:
        return dice_loss(observed, predicted)
    if metric == MaskMetric.L2:
        return l2_loss(observed, predicted)
    raise ValueError(f"Unknown metric: {metric}")


def evaluate_in_roi(
    observed: np.ndarray,
    predicted: np.ndarray,
    roi: ROIBox,
    metric: MaskMetric,
) -> float:
    return metric_value(metric, roi.crop(observed), roi.crop(predicted))


def residuals_in_roi(
    observed: np.ndarray,
    predicted: np.ndarray,
    roi: ROIBox,
) -> np.ndarray:
    return residuals_l2(roi.crop(observed), roi.crop(predicted))
