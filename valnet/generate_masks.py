"""
VALNet Test Inference — Generate PNG Segmentation Maps
======================================================

Runs the model on a test dataloader and saves per-image PNG masks.

Outputs per image:
  - Binary mask:    {output_dir}/{image_id}_mask.png       (0/255 single-channel, all instances merged)
  - Instance mask:  {output_dir}/{image_id}_instances.png  (colored, each instance a different color)

Usage:
    from generate_masks import generate_test_masks
    generate_test_masks(model, test_loader, output_dir="test_output")
"""

import os
import torch
import numpy as np
from PIL import Image

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask


# Distinct colors for up to 20 instances (R, G, B)
INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (200, 200, 200), (100, 100, 100),
]


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
    save_binary: bool = True,
    save_instances: bool = True,
):
    """
    Run inference on test set and save PNG segmentation masks.

    Args:
        model:          VALNetModel or patched YOLO model
        dataloader:     test dataloader (same format as val)
        output_dir:     directory to save output PNGs
        nc:             number of classes as seen by the head (80 for COCO-pretrained)
        conf_thres:     confidence threshold for NMS
        iou_thres:      IoU threshold for NMS
        max_det:        max detections per image
        device:         device string
        save_binary:    save merged binary mask (all instances = white)
        save_instances: save colored per-instance mask
    """
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    img_counter = 0

    for batch in dataloader:
        imgs = batch["img"].to(device).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        bs = imgs.shape[0]
        imgsz = imgs.shape[2:]  # (H, W)

        # Forward (eval mode)
        output = model(imgs)
        (decoded, proto), _ = output

        # NMS
        nms_out = non_max_suppression(
            decoded,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=nc,
            multi_label=False,
            max_det=max_det,
        )

        for si in range(bs):
            det = nms_out[si]
            n_det = det.shape[0]

            # Determine image ID from filename if available, else use counter
            if "im_file" in batch:
                fname = os.path.splitext(os.path.basename(batch["im_file"][si]))[0]
            else:
                fname = f"{img_counter:06d}"

            h, w = imgsz

            if n_det == 0:
                # No detections — save blank masks
                if save_binary:
                    blank = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
                    blank.save(os.path.join(output_dir, f"{fname}_mask.png"))
                if save_instances:
                    blank_rgb = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
                    blank_rgb.save(os.path.join(output_dir, f"{fname}_instances.png"))
                img_counter += 1
                continue

            pred_boxes = det[:, :4]
            pred_conf = det[:, 4]
            pred_cls = det[:, 5]
            mask_coeffs = det[:, 6:]

            # Assemble binary masks at proto resolution, then upsample
            pred_masks = process_mask(
                proto[si],
                mask_coeffs,
                pred_boxes,
                shape=imgsz,
                upsample=True,
            )  # [n_det, H, W] uint8 (0 or 1)

            masks_np = pred_masks.cpu().numpy()  # [n_det, H, W]

            # --- Binary mask: merge all instances ---
            if save_binary:
                merged = np.any(masks_np > 0, axis=0).astype(np.uint8) * 255  # [H, W]
                Image.fromarray(merged).save(
                    os.path.join(output_dir, f"{fname}_mask.png")
                )

            # --- Instance mask: each instance a different color ---
            if save_instances:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

                # Sort by confidence so highest-confidence instance is drawn last (on top)
                order = pred_conf.argsort(descending=False).cpu().numpy()

                for draw_idx, det_idx in enumerate(order):
                    mask_i = masks_np[det_idx] > 0
                    color = INSTANCE_COLORS[draw_idx % len(INSTANCE_COLORS)]
                    canvas[mask_i] = color

                Image.fromarray(canvas).save(
                    os.path.join(output_dir, f"{fname}_instances.png")
                )

            img_counter += 1

    print(f"Saved {img_counter} mask images to {output_dir}/")
