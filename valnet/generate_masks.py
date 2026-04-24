"""
VALNet Test Inference — Overlay predicted segmentation + pose on original images.

One output PNG per input frame:
  - Original image in background
  - Predicted mask drawn as translucent overlay (one color per instance)
  - Pose (predicted vs GT) printed as text in the top-left corner

Quaternion convention: xyzw (matches the rest of the codebase).
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask

from valnet.pose_loss import denormalize_position


INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]
MASK_ALPHA = 0.45


def _quat_angle_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> float:
    """Geodesic angle between two xyzw quaternions, in degrees."""
    q_pred = F.normalize(q_pred, dim=-1, eps=1e-8)
    dot = (q_pred * q_gt).sum().abs().clamp(max=1.0 - 1e-7)
    return float(2.0 * torch.acos(dot) * 180.0 / math.pi)


def _draw_pose_text(
    img_pil: Image.Image,
    pos_pred_m: torch.Tensor,
    pos_gt_m: torch.Tensor,
    quat_pred: torch.Tensor,
    quat_gt: torch.Tensor,
) -> None:
    """Draw pred/GT pose numbers in top-left, in-place on img_pil."""
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    pos_err = (pos_pred_m - pos_gt_m).abs()
    rot_err = _quat_angle_deg(quat_pred, quat_gt)

    lines = [
        "            x       y       z",
        f"pred:  {pos_pred_m[0]:7.1f} {pos_pred_m[1]:7.1f} {pos_pred_m[2]:7.1f} m",
        f"gt:    {pos_gt_m[0]:7.1f} {pos_gt_m[1]:7.1f} {pos_gt_m[2]:7.1f} m",
        f"|err|: {pos_err[0]:7.1f} {pos_err[1]:7.1f} {pos_err[2]:7.1f} m",
        f"rot err: {rot_err:.2f} deg",
    ]

    # Dark background box for legibility
    pad = 4
    line_h = 16
    box_h = line_h * len(lines) + 2 * pad
    box_w = 330
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 180))

    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255), font=font)


@torch.no_grad()
def generate_test_masks(
    model,
    dataloader,
    output_dir: str = "test_output",
    nc: int = 80,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    max_det: int = 300,
    device: str = "cuda",
    yz_scale: float = 100.0,
):
    """Run inference and save per-frame overlays (mask + pose text)."""
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

        # Forward: VALNetModel returns (seg_out, pose_out)
        seg_out, pose_pred = model(imgs)
        (decoded, proto), _ = seg_out

        # NMS for segmentation
        nms_out = non_max_suppression(
            decoded, conf_thres=conf_thres, iou_thres=iou_thres,
            nc=80, classes=[0], multi_label=False, max_det=max_det,
        )

        # Denormalize pose to meters
        pos_pred_m_batch = denormalize_position(pose_pred[:, :3], yz_scale=yz_scale).cpu()
        pose_gt = batch["pose"].to(device).float()
        pos_gt_m_batch = denormalize_position(pose_gt[:, :3], yz_scale=yz_scale).cpu()
        quat_pred_batch = pose_pred[:, 3:].cpu()
        quat_gt_batch = pose_gt[:, 3:].cpu()

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

            # Pose text
            _draw_pose_text(
                canvas,
                pos_pred_m_batch[si], pos_gt_m_batch[si],
                quat_pred_batch[si], quat_gt_batch[si],
            )

            # Output filename
            if "im_file" in batch:
                fname = os.path.splitext(os.path.basename(batch["im_file"][si]))[0]
            else:
                fname = f"{img_counter:06d}"
            canvas.convert("RGB").save(os.path.join(output_dir, f"{fname}.png"))
            img_counter += 1

    print(f"Saved {img_counter} overlays to {output_dir}/")
