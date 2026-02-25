"""
VALNet mAP Computation
======================

The Segment head returns DIFFERENT outputs depending on .training:

  Training mode (.training=True):
    dict with keys: boxes [B,64,N], scores [B,80,N], feats, mask_coefficient [B,32,N], proto [B,32,Hp,Wp]
    - boxes are raw DFL logits (NOT decoded bboxes)
    - scores are raw logits (NOT sigmoid)
    - This is what the loss function consumes. NOT suitable for mAP directly.

  Eval mode (.training=False):
    ((y, proto), preds)
    - y: [B, 4+nc+nm, N] decoded boxes + sigmoid scores + mask coefficients
    - proto: [B, 32, Hp, Wp]
    - preds: the raw training dict (for logging)

So the simplest path to mAP is: set model to eval mode, let the head decode everything,
then run NMS + mask assembly + IoU matching.

This module provides two approaches:
  1. compute_map_from_eval()   — recommended, uses eval-mode output
  2. compute_map_from_train()  — decodes training-mode output manually (for debugging)
"""

import torch
import torch.nn.functional as F
import numpy as np
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import mask_iou, box_iou


# ---------------------------------------------------------------------------
#  Approach 1: Eval-mode mAP (recommended)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_map_from_eval(
    model,
    dataloader,
    nc: int = 1,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    iou_range: tuple = (0.5, 0.95, 10),
    device: str = "cuda",
):
    """
    Compute segmentation mAP using eval-mode head output.

    This follows the same pipeline as Ultralytics' SegmentationValidator:
        1. Head outputs decoded boxes + sigmoid scores + mask coefficients
        2. NMS filters detections
        3. Mask prototypes are combined with coefficients -> binary masks
        4. Mask IoU is computed against ground truth
        5. mAP is accumulated across all images

    Args:
        model:      VALNetModel or patched YOLO model (will be set to eval mode)
        dataloader: yields batches with keys:
                        'img' [B,3,H,W] normalized
                        'bboxes' [N_total, 4] in xywh normalized
                        'cls' [N_total, 1]
                        'batch_idx' [N_total]
                        'masks' [B, H, W] or [N_total, H, W]
        nc:         number of classes (1 for RLD runway)
        conf_thres: confidence threshold for NMS
        iou_thres:  IoU threshold for NMS
        max_det:    max detections per image
        iou_range:  (start, stop, num_steps) for mAP IoU thresholds
        device:     device string

    Returns:
        dict with mAP, AP50, AP75, and per-class APs
    """
    model.eval()
    model.to(device)

    # IoU thresholds for mAP computation: [0.50, 0.55, ..., 0.95]
    iou_start, iou_stop, iou_steps = iou_range
    iou_thresholds = torch.linspace(iou_start, iou_stop, iou_steps)
    niou = len(iou_thresholds)

    stats = {
        "tp_m": [],   # mask true positives
        "conf": [],   # confidence scores
        "pred_cls": [],
        "gt_cls": [],
    }

    for batch in dataloader:
        imgs = batch["img"].to(device).float() / 255.0 if batch["img"].max() > 1 else batch["img"].to(device).float()
        gt_bboxes = batch["bboxes"]      # [N_total, 4] xywh normalized
        gt_cls = batch["cls"]            # [N_total, 1]
        batch_idx = batch["batch_idx"]   # [N_total]
        gt_masks = batch["masks"].to(device).float()  # [B, H, W] or similar

        # --- Forward (eval mode) ---
        output = model(imgs)

        # Unpack eval-mode output: ((decoded_preds, proto), raw_dict)
        (decoded, proto), _ = output
        # decoded: [B, 4+nc+nm, N_anchors]
        # proto:   [B, 32, Hp, Wp]

        bs = imgs.shape[0]
        imgsz = imgs.shape[2:]  # (H, W)
        proto_h, proto_w = proto.shape[2:]

        # --- NMS per image ---
        # Permute decoded to [B, N_anchors, 4+nc+nm] for NMS
        nms_input = decoded  # already in correct format for ops.non_max_suppression
        nms_out = non_max_suppression(
            nms_input,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=80,
            # classes=[0],
            multi_label=False,
            max_det=max_det,
        )
        # nms_out: list of B tensors, each [n_det, 6+nm]
        #          columns: x1, y1, x2, y2, conf, cls, mask_coeff_0..mask_coeff_31

        for si in range(bs):
            det = nms_out[si]   # [n_det, 6+nm]
            n_det = det.shape[0]

            # Ground truth for this image
            idx = batch_idx == si
            gt_cls_i = gt_cls[idx].squeeze(-1).to(device)    # [n_gt]
            gt_bbox_i = gt_bboxes[idx].to(device)             # [n_gt, 4] xywh normalized
            n_gt = gt_cls_i.shape[0]

            stats["gt_cls"].append(gt_cls_i.cpu())

            if n_det == 0:
                if n_gt > 0:
                    stats["tp_m"].append(np.zeros((0, niou), dtype=bool))
                    stats["conf"].append(torch.empty(0))
                    stats["pred_cls"].append(torch.empty(0))
                continue

            pred_boxes = det[:, :4]                         # [n_det, 4] x1y1x2y2 pixel
            pred_conf = det[:, 4]                           # [n_det]
            pred_cls = det[:, 5]                            # [n_det]
            mask_coeffs = det[:, 6:]                        # [n_det, nm]

            # --- Assemble masks ---
            pred_masks = ops.process_mask(
                proto[si],          # [32, Hp, Wp]
                mask_coeffs,        # [n_det, 32]
                pred_boxes,         # [n_det, 4]
                shape=imgsz,
                upsample=True,
            )  # [n_det, H, W] uint8

            stats["conf"].append(pred_conf.cpu())
            stats["pred_cls"].append(pred_cls.cpu())

            if n_gt == 0:
                stats["tp_m"].append(np.zeros((n_det, niou), dtype=bool))
                continue

            # --- Ground truth masks for this image ---
            if gt_masks.dim() == 3 and gt_masks.shape[0] == bs:
                # Overlapping mask format: [B, H, W] with instance indices
                gt_mask_i = gt_masks[si]  # [H, W]
                # Expand to per-instance binary masks
                instance_masks = []
                for gi in range(n_gt):
                    instance_masks.append((gt_mask_i == (gi + 1)).float())
                gt_masks_i = torch.stack(instance_masks)  # [n_gt, H, W]
            else:
                gt_masks_i = gt_masks[idx]  # [n_gt, H, W]

            # Resize GT masks to match prediction mask size if needed
            if gt_masks_i.shape[1:] != pred_masks.shape[1:]:
                gt_masks_i = F.interpolate(
                    gt_masks_i.unsqueeze(0).float(),
                    size=pred_masks.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )[0].gt_(0.5).float()

            # --- Mask IoU ---
            iou = mask_iou(pred_masks.flatten(1).float(), gt_masks_i.flatten(1))
            # --- Match predictions to ground truth ---
            tp_m = _match_predictions(pred_cls, gt_cls_i, iou, iou_thresholds)
            stats["tp_m"].append(tp_m)

    # --- Aggregate stats and compute mAP ---
    return _compute_ap(stats, nc, niou, iou_thresholds)


def _match_predictions(pred_cls, gt_cls, iou, iou_thresholds):
    """
    Match predictions to ground truth at multiple IoU thresholds.

    Returns:
        np.ndarray of shape [n_det, n_iou_thresholds] (bool)
    """
    n_det = pred_cls.shape[0]
    n_gt = gt_cls.shape[0]
    niou = len(iou_thresholds)
    tp = np.zeros((n_det, niou), dtype=bool)

    if n_gt == 0 or n_det == 0:
        return tp

    # Only match same class
    correct_cls = pred_cls.unsqueeze(1) == gt_cls.unsqueeze(0)  # [n_det, n_gt]

    for ti, thresh in enumerate(iou_thresholds):
        # For each threshold, greedily match predictions to GT
        matches = (iou >= thresh) & correct_cls.to(iou.device)  # [n_gt, n_det]

        if matches.any():
            # Find best matches (highest IoU)
            # matches is [n_gt, n_det], iou is [n_gt, n_det]
            matched_iou = matches.float() * iou
            # For each detection, find best GT match
            gt_match_vals, gt_match_idx = matched_iou.max(dim=1)  # [n_det]

            # For each GT, only allow one detection (highest IoU)
            used_gt = set()
            # Sort detections by their match IoU descending
            det_order = gt_match_vals.argsort(descending=True)
            for di in det_order:
                di = di.item()
                if gt_match_vals[di] > 0:
                    gi = gt_match_idx[di].item()
                    if gi not in used_gt:
                        tp[di, ti] = True
                        used_gt.add(gi)

    return tp


def _compute_ap(stats, nc, niou, iou_thresholds):
    """
    Compute AP from accumulated statistics.

    Returns:
        dict with mAP, AP50, AP75, per_class_ap
    """
    tp_m = np.concatenate(stats["tp_m"], axis=0)     # [total_dets, niou]
    conf = torch.cat(stats["conf"]).numpy()           # [total_dets]
    pred_cls = torch.cat(stats["pred_cls"]).numpy()   # [total_dets]
    gt_cls = torch.cat(stats["gt_cls"]).numpy()       # [total_gt]

    # Sort by confidence descending
    sort_idx = np.argsort(-conf)
    tp_m = tp_m[sort_idx]
    pred_cls = pred_cls[sort_idx]
    conf = conf[sort_idx]

    # Per-class AP
    ap_per_class = np.zeros((nc, niou))
    n_gt_per_class = np.zeros(nc)

    for ci in range(nc):
        # Detections of this class
        det_mask = pred_cls == ci
        n_det = det_mask.sum()
        n_gt = (gt_cls == ci).sum()
        n_gt_per_class[ci] = n_gt

        if n_gt == 0 or n_det == 0:
            continue

        tp_c = tp_m[det_mask]  # [n_det_class, niou]

        for ti in range(niou):
            tp_cum = np.cumsum(tp_c[:, ti])
            fp_cum = np.cumsum(~tp_c[:, ti])

            recall = tp_cum / n_gt
            precision = tp_cum / (tp_cum + fp_cum)

            # AP = area under PR curve (101-point interpolation like COCO)
            ap_per_class[ci, ti] = _compute_ap_from_pr(recall, precision)

    # Aggregate
    # Only count classes that have GT instances
    valid = n_gt_per_class > 0
    if valid.sum() == 0:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "per_class_ap": ap_per_class}

    mean_ap_per_iou = ap_per_class[valid].mean(axis=0)  # [niou]

    # Standard COCO metrics
    idx_50 = 0                   # IoU=0.50
    idx_75 = 5                   # IoU=0.75
    mAP = mean_ap_per_iou.mean()  # average over all IoU thresholds
    AP50 = mean_ap_per_iou[idx_50]
    AP75 = mean_ap_per_iou[idx_75] if niou > idx_75 else 0.0

    return {
        "mAP": float(mAP),
        "AP50": float(AP50),
        "AP75": float(AP75),
        "per_class_ap": ap_per_class,
        "ap_per_iou": mean_ap_per_iou,
    }


def _compute_ap_from_pr(recall, precision):
    """
    Compute AP using 101-point interpolation (COCO-style).
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_interp = np.linspace(0, 1, 101)
    ap = np.mean(np.interp(recall_interp, mrec, mpre))
    return ap

# ---------------------------------------------------------------------------
#  Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_valnet(model, dataloader, nc=1, device="cuda"):
    """
    One-liner evaluation.

    Args:
        model:      VALNetModel or patched YOLO model
        dataloader: validation dataloader
        nc:         number of classes (1 for RLD)
        device:     device

    Returns:
        dict with mAP, AP50, AP75

    Example:
        metrics = evaluate_valnet(model, val_loader, nc=1)
        print(f"mAP: {metrics['mAP']:.1%}, AP50: {metrics['AP50']:.1%}, AP75: {metrics['AP75']:.1%}")
    """
    return compute_map_from_eval(
        model=model,
        dataloader=dataloader,
        nc=nc,
        device=device,
    )
