"""Ultralytics YOLO-seg inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


def load_yolo_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def union_soft_mask_for_class(
    result: Results,
    cls_id: int,
    orig_shape: tuple[int, int],
    *,
    combine: str = "max",
) -> np.ndarray:
    """
    Build a soft mask (H, W) in [0, 1] for one class from a YOLO-seg result.

    ``orig_shape`` is (height, width). Multiple instances are merged with ``max``.
    """
    h, w = orig_shape
    out = np.zeros((h, w), dtype=np.float32)
    if result.masks is None or result.boxes is None or result.boxes.cls is None:
        return out

    import torch

    cls = result.boxes.cls
    conf = result.boxes.conf
    if isinstance(cls, torch.Tensor):
        cls_np = cls.detach().cpu().numpy().astype(int)
        conf_np = conf.detach().cpu().numpy().astype(np.float32)
    else:
        cls_np = np.asarray(cls, dtype=int)
        conf_np = np.asarray(conf, dtype=np.float32)

    idxs = np.where(cls_np == int(cls_id))[0]
    if idxs.size == 0:
        return out

    masks_data = result.masks.data
    for i in idxs:
        m = masks_data[int(i)]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m, dtype=np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        weighted = np.clip(m * float(conf_np[int(i)]), 0.0, 1.0)
        if combine == "max":
            out = np.maximum(out, weighted)
        else:
            out = np.clip(out + weighted, 0.0, 1.0)
    return out


def predict_soft_mask(
    model,
    image_path: str | Path,
    *,
    cls_id: int = 0,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | int | None = None,
) -> tuple[np.ndarray, Results | None]:
    """
    Run segmentation on one image; return (soft_mask, result).

    Soft mask shape matches the input image (H, W), values in [0, 1].
    """
    image_path = Path(image_path)
    import cv2

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]

    kwargs: dict = {"source": bgr, "conf": conf, "imgsz": imgsz, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    mask = union_soft_mask_for_class(result, cls_id, (h, w))
    return mask, result


def mask_has_foreground(mask: np.ndarray, threshold: float = 0.1) -> bool:
    return bool((mask > threshold).any())
