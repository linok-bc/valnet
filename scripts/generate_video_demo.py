"""
VALNet Inference on Raw Images
==============================

Runs VALNet on a directory of images and saves per-instance segmentation masks.

Usage:
    python infer.py --images path/to/images --checkpoint path/to/valnet.pt --output path/to/output

Outputs per image:
    {name}_mask.png       — binary mask (all instances merged, 0/255)
    {name}_instances.png  — colored instance mask (each instance a different color)
    {name}_overlay.png    — original image with colored mask overlay
"""

import os
import argparse
import glob
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask

from valnet.valnet import VALNetModel

INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (200, 200, 200), (100, 100, 100),
]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


def load_image(path, imgsz=640):
    """
    Load and preprocess a single image for VALNet inference.

    Returns:
        img_tensor: [1, 3, imgsz, imgsz] normalized float32 tensor
        orig_size:  (orig_h, orig_w) for rescaling masks back
    """
    img = Image.open(path).convert("RGB")
    orig_size = (img.height, img.width)

    img_resized = img.resize((imgsz, imgsz), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0  # [H, W, 3]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    return img_tensor, orig_size


@torch.no_grad()
def infer_single(model, img_tensor, device, conf_thres=0.25, iou_thres=0.7, max_det=300):
    """
    Run inference on a single image tensor.

    Returns:
        masks_np:   [n_det, H, W] binary masks (numpy, at model resolution)
        confs:      [n_det] confidence scores (numpy)
        boxes:      [n_det, 4] xyxy boxes (numpy)
        n_det:      number of detections
    """
    img_tensor = img_tensor.to(device)
    imgsz = img_tensor.shape[2:]

    output = model(img_tensor)
    (decoded, proto), _ = output

    nms_out = non_max_suppression(
        decoded,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=80,
        classes=[0],
        multi_label=False,
        max_det=max_det,
    )

    det = nms_out[0]
    n_det = det.shape[0]

    if n_det == 0:
        return np.zeros((0, imgsz[0], imgsz[1])), np.array([]), np.array([]), 0

    pred_boxes = det[:, :4]
    pred_conf = det[:, 4]
    mask_coeffs = det[:, 6:]

    pred_masks = process_mask(
        proto[0],
        mask_coeffs,
        pred_boxes,
        shape=imgsz,
        upsample=True,
    )

    return (
        pred_masks.cpu().numpy(),
        pred_conf.cpu().numpy(),
        pred_boxes.cpu().numpy(),
        n_det,
    )


def save_masks(masks_np, confs, orig_size, output_dir, name, save_binary, save_instances, save_overlay, orig_img_path):
    """Save binary, instance, and overlay masks for one image."""
    h, w = masks_np.shape[1], masks_np.shape[2]
    n_det = masks_np.shape[0]

    # Resize masks to original image size
    if (orig_size[0] != h) or (orig_size[1] != w):
        masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()  # [1, n_det, H, W]
        masks_resized = F.interpolate(masks_tensor, size=orig_size, mode='bilinear', align_corners=False)
        masks_np = (masks_resized[0] > 0.5).numpy()

    oh, ow = orig_size

    if save_binary:
        merged = np.any(masks_np > 0, axis=0).astype(np.uint8) * 255
        Image.fromarray(merged).save(os.path.join(output_dir, f"{name}_mask.png"))

    if save_instances:
        canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
        order = np.argsort(confs)  # low confidence first, high confidence drawn on top
        for draw_idx, det_idx in enumerate(order):
            mask_i = masks_np[det_idx] > 0
            color = INSTANCE_COLORS[draw_idx % len(INSTANCE_COLORS)]
            canvas[mask_i] = color
        Image.fromarray(canvas).save(os.path.join(output_dir, f"{name}_instances.png"))

    if save_overlay:
        orig_img = np.array(Image.open(orig_img_path).convert("RGB").resize((ow, oh)))
        overlay = orig_img.copy()
        order = np.argsort(confs)
        for draw_idx, det_idx in enumerate(order):
            mask_i = masks_np[det_idx] > 0
            color = np.array(INSTANCE_COLORS[draw_idx % len(INSTANCE_COLORS)])
            overlay[mask_i] = (0.5 * orig_img[mask_i] + 0.5 * color).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(output_dir, f"{name}_overlay.png"))


def main():
    parser = argparse.ArgumentParser(description="VALNet inference on raw images")
    parser.add_argument("--images", required=True, help="Directory of images or glob pattern")
    parser.add_argument("--checkpoint", required=True, help="Path to VALNet checkpoint (.pt)")
    parser.add_argument("--output", default="inference_output", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--no-binary", action="store_true", help="Skip binary mask output")
    parser.add_argument("--no-instances", action="store_true", help="Skip instance mask output")
    parser.add_argument("--overlay", action="store_true", help="Save overlay visualization")
    parser.add_argument("--yolo-weights", default="yolov8s-seg.pt", help="Base YOLOv8 weights")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect images
    if os.path.isdir(args.images):
        image_paths = sorted([
            os.path.join(args.images, f)
            for f in os.listdir(args.images)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
    else:
        image_paths = sorted(glob.glob(args.images))

    if not image_paths:
        print(f"No images found in {args.images}")
        return

    print(f"Found {len(image_paths)} images")

    # Build model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = VALNetModel.from_ultralytics(args.yolo_weights, ch=(128, 256, 512))
    model.model = nn.ModuleList([model.backbone_p3, model.backbone_p4, model.backbone_p5, model.neck, model.head])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")

    # Run inference
    for i, img_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img_tensor, orig_size = load_image(img_path, imgsz=args.imgsz)

        masks_np, confs, boxes, n_det = infer_single(
            model, img_tensor, device,
            conf_thres=args.conf,
            iou_thres=args.iou,
        )

        if n_det == 0:
            print(f"[{i+1}/{len(image_paths)}] {name}: no detections")
            # Save blank masks
            oh, ow = orig_size
            if not args.no_binary:
                Image.fromarray(np.zeros((oh, ow), dtype=np.uint8)).save(
                    os.path.join(args.output, f"{name}_mask.png"))
            if not args.no_instances:
                Image.fromarray(np.zeros((oh, ow, 3), dtype=np.uint8)).save(
                    os.path.join(args.output, f"{name}_instances.png"))
            if args.overlay:
                Image.open(img_path).convert("RGB").save(
                    os.path.join(args.output, f"{name}_overlay.png"))
            continue

        save_masks(
            masks_np, confs, orig_size, args.output, name,
            save_binary=not args.no_binary,
            save_instances=not args.no_instances,
            save_overlay=args.overlay,
            orig_img_path=img_path,
        )
        print(f"[{i+1}/{len(image_paths)}] {name}: {n_det} instance(s), max conf {confs.max():.3f}")

    print(f"Done. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
