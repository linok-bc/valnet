"""
Observed runway masks for evaluation, and quad (corner) extraction from a mask.

Mask sources:
- ``gt``      : label quad filled (oracle segmentation → isolates the pose solver)
- ``corrupt`` : ``gt`` + procedural corruption (``pose_estimation.mask_corrupt``)
- ``yolo``    : cached YOLO-seg soft masks (``scripts/infer_yolo_xp12.py``)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose_estimation.g_dof import render_runway_mask
from pose_estimation.mask_corrupt import CorruptionConfig, corrupt_mask
from runway_detection.lard.projection import CORNER_NAMES
from runway_detection.xp12.loader import XP12_HEIGHT, XP12_WIDTH, XP12Frame

MASK_SOURCES = ("gt", "corrupt", "yolo")


def gt_corners_array(frame: XP12Frame) -> np.ndarray:
    return np.array([frame.corners_px[n] for n in CORNER_NAMES], dtype=float)


def gt_mask(frame: XP12Frame) -> np.ndarray:
    return render_runway_mask(gt_corners_array(frame), XP12_WIDTH, XP12_HEIGHT, dtype="float")


class YoloMaskCache:
    """Lazy per-sequence access to ``<dir>/<seq>.npz`` (indices, uint8 masks)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._seq: str | None = None
        self._index: dict[int, int] = {}
        self._masks: np.ndarray | None = None

    def available_indices(self, seq_id: str) -> list[int]:
        path = self.directory / f"{seq_id}.npz"
        if not path.exists():
            return []
        with np.load(path) as z:
            return [int(i) for i in z["indices"]]

    def get(self, seq_id: str, index: int) -> np.ndarray | None:
        if self._seq != seq_id:
            path = self.directory / f"{seq_id}.npz"
            if not path.exists():
                return None
            with np.load(path) as z:
                self._masks = z["masks"]
                self._index = {int(i): k for k, i in enumerate(z["indices"])}
            self._seq = seq_id
        k = self._index.get(index)
        if k is None or self._masks is None:
            return None
        return self._masks[k].astype(np.float32) / 255.0


def observed_mask(
    source: str,
    frame: XP12Frame,
    *,
    yolo: YoloMaskCache | None = None,
    rng: np.random.Generator | None = None,
    corruption: CorruptionConfig | None = None,
) -> np.ndarray | None:
    if source == "gt":
        return gt_mask(frame)
    if source == "corrupt":
        return corrupt_mask(gt_mask(frame) > 0, config=corruption, rng=rng)
    if source == "yolo":
        if yolo is None:
            raise ValueError("yolo source requires a YoloMaskCache")
        return yolo.get(frame.seq_id, frame.index)
    raise ValueError(f"Unknown mask source {source!r}")


def mask_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    a = a > thr
    b = b > thr
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# --------------------------------------------------------------------- quads


def extract_quad(mask: np.ndarray, *, thr: float = 0.5, min_area_px: float = 12.0) -> np.ndarray | None:
    """
    Four vertices (cyclic order) approximating the largest blob of ``mask``.

    Convex hull → ``approxPolyDP`` with growing epsilon until 4 vertices; falls
    back to the 4 hull points farthest apart (min-area rectangle as last resort).
    """
    binary = (np.asarray(mask) > thr).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for frac in np.geomspace(0.005, 0.2, 24):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(float)
        if len(approx) < 4:
            break
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return np.asarray(box, dtype=float)


def order_quad_like(quad: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Reorder a cyclic quad (4, 2) to best match ``reference`` (4, 2, TR/TL/BL/BR order).

    Tries the 4 rotations × 2 directions after removing both centroids, so a
    rough reference (e.g. prior pose, or yaw = lat = 0 with known pitch/roll)
    is enough to label near/far and left/right.
    """
    q = np.asarray(quad, dtype=float)
    r = np.asarray(reference, dtype=float)
    rc = r - r.mean(axis=0)
    best, best_cost = q, np.inf
    for seq in (q, q[::-1]):
        seqc = seq - seq.mean(axis=0)
        for k in range(4):
            cand = np.roll(seqc, k, axis=0)
            cost = float(np.sum((cand - rc) ** 2))
            if cost < best_cost:
                best_cost = cost
                best = np.roll(seq, k, axis=0)
    return best


def corners_dict(arr: np.ndarray) -> dict[str, tuple[float, float]]:
    return {n: (float(p[0]), float(p[1])) for n, p in zip(CORNER_NAMES, arr)}
