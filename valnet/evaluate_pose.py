"""
Pose evaluation for VALNet.

Reports:
    - Per-axis position error in meters (mean and median)
        - x: along-track (negative before touchdown)
        - y: altitude
        - z: cross-track
    - Rotation error in degrees (geodesic distance on SO(3))
    - Combined Euclidean position error (for reference only)

Run alongside `evaluate_valnet` from evaluate_map.py — they're independent.

Quaternion convention: xyzw, matching the rest of the codebase.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from valnet.pose_loss import denormalize_position


@torch.no_grad()
def evaluate_pose(
    model,
    dataloader,
    yz_scale: float = 100.0,
    device: str = "cuda",
) -> dict:
    """Compute pose metrics over an entire dataloader.

    Args:
        model: VALNetModel (will be set to eval mode).
        dataloader: yields batches with 'img' and 'pose' (normalized).
        yz_scale: must match what the dataset used for normalization.
        device: cuda or cpu.

    Returns:
        dict with keys:
            pos_err_mean_m:   [3] mean |error| in meters per axis (x, y, z)
            pos_err_median_m: [3] median |error| in meters per axis
            pos_err_l2_m:     scalar, mean Euclidean position error
            rot_err_mean_deg: scalar, mean rotation error in degrees
            rot_err_median_deg: scalar, median rotation error
            n_samples:        int
    """
    model.eval()
    model.to(device)

    pos_errs_m = []   # list of [B, 3] tensors in meters (signed error)
    rot_errs_deg = [] # list of [B] tensors in degrees

    for batch in dataloader:
        imgs = batch["img"].to(device).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        pose_gt = batch["pose"].to(device).float()  # [B, 7] normalized

        _, pose_pred = model(imgs)                   # [B, 7] raw

        # --- Position: denormalize both to meters, compute signed error ---
        pos_pred_m = denormalize_position(pose_pred[:, :3], yz_scale=yz_scale)
        pos_gt_m   = denormalize_position(pose_gt[:, :3],   yz_scale=yz_scale)
        pos_errs_m.append((pos_pred_m - pos_gt_m).cpu())

        # --- Rotation: geodesic on SO(3) via |q_pred . q_gt| ---
        q_pred = F.normalize(pose_pred[:, 3:], dim=-1, eps=1e-8)
        q_gt   = pose_gt[:, 3:]  # already unit from dataset

        dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1.0 - 1e-7)
        # Angle between quaternions is 2*acos(|dot|), in radians
        angle_rad = 2.0 * torch.acos(dot)
        rot_errs_deg.append((angle_rad * 180.0 / math.pi).cpu())

    pos_err = torch.cat(pos_errs_m, dim=0)       # [N, 3]
    rot_err = torch.cat(rot_errs_deg, dim=0)     # [N]

    abs_pos = pos_err.abs()
    l2_pos = pos_err.norm(dim=-1)                # [N]

    return {
        "pos_err_mean_m":     abs_pos.mean(dim=0).numpy(),
        "pos_err_median_m":   abs_pos.median(dim=0).values.numpy(),
        "pos_err_l2_m":       float(l2_pos.mean()),
        "rot_err_mean_deg":   float(rot_err.mean()),
        "rot_err_median_deg": float(rot_err.median()),
        "n_samples":          int(pos_err.shape[0]),
    }


def format_pose_metrics(metrics: dict) -> str:
    """Pretty-print for logs."""
    m = metrics
    return (
        f"n={m['n_samples']}  "
        f"pos_mean (x,y,z) = "
        f"({m['pos_err_mean_m'][0]:.2f}, "
        f"{m['pos_err_mean_m'][1]:.2f}, "
        f"{m['pos_err_mean_m'][2]:.2f}) m  "
        f"| pos_median = "
        f"({m['pos_err_median_m'][0]:.2f}, "
        f"{m['pos_err_median_m'][1]:.2f}, "
        f"{m['pos_err_median_m'][2]:.2f}) m  "
        f"| L2 = {m['pos_err_l2_m']:.2f} m  "
        f"| rot = {m['rot_err_mean_deg']:.2f}° (median {m['rot_err_median_deg']:.2f}°)"
    )
