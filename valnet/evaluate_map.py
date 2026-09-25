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
from ultralytics.utils.metrics import mask_iou


# ---------------------------------------------------------------------------
#  Approach 1: Eval-mode mAP (recommended)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_map_from_eval(
    model,
    dataloader,
    nc: int = 1,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    iou_range: tuple = (0.5, 0.95, 10),
    device: str = "cuda",
    overlap_mask: bool = True,
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
        overlap_mask: GT mask layout, matching the dataloader's `overlap_mask`
                    (Ultralytics' default, True): one [H, W] map per image
                    whose value k marks the image's k-th instance. False means
                    one binary [H, W] mask per instance.

    Returns:
        dict with mAP, AP50, AP75 (full resolution), the same three suffixed
        _proto (prototype resolution, Ultralytics-comparable), and per-class APs
    """
    model.eval()
    model.to(device)

    # IoU thresholds for mAP computation: [0.50, 0.55, ..., 0.95]
    iou_start, iou_stop, iou_steps = iou_range
    iou_thresholds = torch.linspace(iou_start, iou_stop, iou_steps)
    niou = len(iou_thresholds)

    # Mask IoU is scored at two resolutions, from the same detections:
    #   full  — predicted masks upsampled to the network input (640) and compared
    #           against GT at that size. The strict one: boundary error on thin,
    #           distant runways counts in full. This is the headline metric.
    #   proto — both at the prototype resolution (1/4 of the input), which is
    #           what Ultralytics' SegmentationValidator does. Lenient on thin
    #           objects, but directly comparable to Ultralytics and the paper.
    # Measured on a baseline checkpoint these differ by ~2x in mAP50-95, and
    # shifting the full-res masks by +/-1..3 px only lowers IoU, so the gap is
    # resolution, not misalignment.
    stats = {
        "tp_m": [],        # mask true positives, full resolution
        "tp_m_proto": [],  # mask true positives, prototype resolution
        "conf": [],        # confidence scores
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
        seg_out = model(imgs)
        (decoded, proto), _ = seg_out
        # decoded: [B, 4+nc+nm, N_anchors]
        # proto:   [B, 32, Hp, Wp]

        bs = imgs.shape[0]
        imgsz = imgs.shape[2:]  # (H, W)
        proto_h, proto_w = proto.shape[2:]

        # --- NMS per image ---
        # Permute decoded to [B, N_anchors, 4+nc+nm] for NMS
        nms_input = decoded  # already in correct format for ops.non_max_suppression
        # The head's own class count, derived from the tensor layout:
        # decoded is [B, 4 + nc_head + nm, N] and proto is [B, nm, Hp, Wp].
        # VALNet keeps the pretrained 80-class COCO head, but a model trained
        # by Ultralytics on this dataset has a 1-class head, and passing the
        # wrong nc here silently misreads the score/mask-coefficient split.
        nc_head = decoded.shape[1] - 4 - proto.shape[1]
        # End-to-end heads (YOLO26) skip NMS and already return the final
        # detections as [B, max_det, 6 + nm] = x1 y1 x2 y2 conf cls + coeffs;
        # Ultralytics then only filters by confidence.
        end2end = decoded.shape[-1] == 6 + proto.shape[1]
        nms_out = non_max_suppression(
            nms_input,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=nc_head,
            classes=[0],
            multi_label=False,
            max_det=max_det,
            end2end=end2end,
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
                    stats["tp_m_proto"].append(np.zeros((0, niou), dtype=bool))
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
            pred_masks_proto = ops.process_mask(
                proto[si], mask_coeffs, pred_boxes, shape=imgsz, upsample=False,
            )  # [n_det, Hp, Wp] uint8

            stats["conf"].append(pred_conf.cpu())
            stats["pred_cls"].append(pred_cls.cpu())

            if n_gt == 0:
                stats["tp_m"].append(np.zeros((n_det, niou), dtype=bool))
                stats["tp_m_proto"].append(np.zeros((n_det, niou), dtype=bool))
                continue

            # --- Ground truth masks for this image ---
            if overlap_mask:
                # Unpack the per-image index map. Ultralytics sorts the image's
                # labels so that label k owns mask value k + 1.
                index = torch.arange(1, n_gt + 1, device=device).view(n_gt, 1, 1)
                gt_masks_i = (gt_masks[si] == index).float()  # [n_gt, H, W]
            else:
                gt_masks_i = gt_masks[idx]  # [n_gt, H, W]

            for key, pm in (("tp_m", pred_masks), ("tp_m_proto", pred_masks_proto)):
                # Resize GT masks to match prediction mask size if needed
                gm = gt_masks_i
                if gm.shape[1:] != pm.shape[1:]:
                    gm = F.interpolate(
                        gm.unsqueeze(0).float(),
                        size=pm.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )[0].gt_(0.5).float()

                # --- Mask IoU, then match predictions to ground truth ---
                iou = mask_iou(pm.flatten(1).float(), gm.flatten(1))
                stats[key].append(_match_predictions(pred_cls, gt_cls_i, iou, iou_thresholds))

    # --- Aggregate stats and compute mAP ---
    metrics = _compute_ap(stats, nc, niou, iou_thresholds)
    proto = _compute_ap({**stats, "tp_m": stats["tp_m_proto"]}, nc, niou, iou_thresholds)
    for k in ("mAP", "AP50", "AP75"):
        metrics[f"{k}_proto"] = proto[k]
    return metrics


def _match_predictions(pred_cls, gt_cls, iou, iou_thresholds):
    """
    Match predictions to ground truth at multiple IoU thresholds.

    COCO-style greedy matching: detections are visited in descending confidence
    (NMS already returns them in that order) and each takes the best-IoU ground
    truth still unmatched. This is also what Ultralytics' match_predictions ends
    up doing. Matching each GT to its highest-IoU detection instead marks the
    confident-but-looser mask a false positive at the top of the ranking and
    pushes the true positive down to a low confidence, which cost 0.08 mAP50-95
    on a baseline checkpoint.

    Args:
        pred_cls: [n_det], sorted by descending confidence
        gt_cls:   [n_gt]
        iou:      [n_det, n_gt]

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
    iou = (iou * correct_cls.to(iou.device)).cpu().numpy()

    for ti, thresh in enumerate(iou_thresholds.tolist()):
        used_gt = np.zeros(n_gt, dtype=bool)
        for di in range(n_det):
            cand = np.where(used_gt, -1.0, iou[di])
            gi = int(cand.argmax())
            if cand[gi] >= thresh:
                tp[di, ti] = True
                used_gt[gi] = True

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
