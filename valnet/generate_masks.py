"""
VALNet Test Inference — Overlay predicted segmentation on original images.

One output PNG per input frame:
  - Original image in background
  - Predicted mask drawn as translucent overlay (one color per instance)
"""

import os
import numpy as np
import torch
from PIL import Image

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask


INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]
MASK_ALPHA = 0.45


@torch.no_grad()
def generate_test_masks(
    model,
    dataloader,
    output_dir: str = "test_output",
    nc: int = 80,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
    device: str = "cuda",
):
    """Run inference and save per-frame mask overlays."""
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    img_counter = 0

    for batch in dataloader:
        imgs = batch["img"].to(device).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        bs = imgs.shape[0]
        imgsz = imgs.shape[2:]

        # Forward: VALNetModel returns seg_out
        seg_out = model(imgs)
        (decoded, proto), _ = seg_out

        # NMS for segmentation
        nms_out = non_max_suppression(
            decoded, conf_thres=conf_thres, iou_thres=iou_thres,
            nc=80, classes=[0], multi_label=False, max_det=max_det,
        )

        for si in range(bs):
            det = nms_out[si]
            h, w = imgsz

            # Base image: undo the 0-1 normalization back to uint8 RGB
            base = imgs[si].cpu().numpy()       # [3, H, W]
            base = (base * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            # Ultralytics loads BGR by default; convert to RGB
            base = base[..., ::-1].copy()
            canvas = Image.fromarray(base).convert("RGBA")

            # Mask overlay
            if det.shape[0] > 0:
                pred_boxes = det[:, :4]
                pred_conf = det[:, 4]
                mask_coeffs = det[:, 6:]

                pred_masks = process_mask(
                    proto[si], mask_coeffs, pred_boxes,
                    shape=imgsz, upsample=True,
                ).cpu().numpy()   # [n_det, H, W]

                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                order = pred_conf.argsort(descending=False).cpu().numpy()
                for draw_idx, det_idx in enumerate(order):
                    m = pred_masks[det_idx] > 0
                    color = INSTANCE_COLORS[draw_idx % len(INSTANCE_COLORS)]
                    overlay[m] = (*color, int(255 * MASK_ALPHA))
                canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, "RGBA"))

            # Output filename
            if "im_file" in batch:
                fname = os.path.splitext(os.path.basename(batch["im_file"][si]))[0]
            else:
                fname = f"{img_counter:06d}"
            canvas.convert("RGB").save(os.path.join(output_dir, f"{fname}.png"))
            img_counter += 1

    print(f"Saved {img_counter} overlays to {output_dir}/")
