"""
Combined loss for VALNet joint segmentation + pose regression.

Components:
  seg_loss   — Ultralytics v8SegmentationLoss, unchanged
  pos_loss   — per-axis weighted MSE on normalized position
  rot_loss   — 1 - |q_pred · q_gt| on unit-normalized quaternions (xyzw)

All three are summed with configurable weights. Position and rotation GT
are expected to already be in normalized space — the dataloader is
responsible for that conversion via `signed_log1p_position`.

Quaternion convention throughout: scalar-last (qx, qy, qz, qw).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss


# ----------------------------------------------------------------------
# Position normalization helpers
# ----------------------------------------------------------------------

def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """Smooth log compression that preserves sign and is continuous at 0.

    y = sign(x) * log(1 + |x|).  Inverse: `signed_expm1`.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(y: torch.Tensor) -> torch.Tensor:
    """Inverse of `signed_log1p`. Use at eval time to recover meters."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def normalize_position(xyz: torch.Tensor, yz_scale: float = 100.0) -> torch.Tensor:
    """Convert raw runway-relative position (meters) to network-space.

    x (along-track) is signed-log-compressed; y (altitude) and z
    (cross-track) are divided by a nominal scale so all three axes sit
    in roughly the same numeric range.

    Args:
        xyz: [..., 3] tensor of (x, y, z) in meters.
        yz_scale: divisor for y and z. 100m is a reasonable default for
            approach / landing phase; adjust based on your data range.
    """
    x = signed_log1p(xyz[..., 0:1])
    yz = xyz[..., 1:3] / yz_scale
    return torch.cat([x, yz], dim=-1)


def denormalize_position(xyz_norm: torch.Tensor, yz_scale: float = 100.0) -> torch.Tensor:
    """Inverse of `normalize_position`. Recover meters from network output."""
    x = signed_expm1(xyz_norm[..., 0:1])
    yz = xyz_norm[..., 1:3] * yz_scale
    return torch.cat([x, yz], dim=-1)


# ----------------------------------------------------------------------
# Pose loss
# ----------------------------------------------------------------------

class PoseLoss(nn.Module):
    """Weighted position MSE + quaternion dot-product loss.

    Args:
        axis_weights: 3-tuple of loss weights for (x, y, z). x is the
            along-track / depth axis and is usually down-weighted.
        eps: numerical floor for quaternion normalization.
    """

    def __init__(self, axis_weights=(0.3, 1.0, 1.0), eps: float = 1e-8):
        super().__init__()
        self.register_buffer(
            "axis_weights",
            torch.tensor(axis_weights, dtype=torch.float32),
        )
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            pred: [B, 7] raw head output — (pos_norm[3], quat_raw[4])
            gt:   [B, 7] ground truth — (pos_norm[3], quat_unit[4]),
                  already in normalized space with unit quaternion.

        Returns:
            (total, pos_loss, rot_loss) — scalars, for logging the split.
        """
        pos_pred, quat_pred = pred[:, :3], pred[:, 3:]
        pos_gt,   quat_gt   = gt[:, :3],   gt[:, 3:]

        # Per-axis MSE, weighted
        per_axis_sq = (pos_pred - pos_gt) ** 2                # [B, 3]
        pos_loss = (per_axis_sq * self.axis_weights).mean()

        # Normalize predicted quaternion (GT is assumed unit already)
        quat_pred = F.normalize(quat_pred, dim=-1, eps=self.eps)

        # 1 - |dot| handles the q / -q double cover
        dot = (quat_pred * quat_gt).sum(dim=-1)               # [B]
        rot_loss = (1.0 - dot.abs()).mean()

        return pos_loss + rot_loss, pos_loss.detach(), rot_loss.detach()


# ----------------------------------------------------------------------
# Combined loss
# ----------------------------------------------------------------------

class CombinedLoss:
    """Wraps v8SegmentationLoss + PoseLoss.

    Not an nn.Module because v8SegmentationLoss isn't either — matching
    Ultralytics' convention. Call this once per batch.

    Args:
        model: the VALNet model (needed by v8SegmentationLoss for its
            internal config — hyperparameters, nc, etc.).
        lambda_pos: weight on position loss term in the total.
        lambda_rot: weight on rotation loss term in the total.
        axis_weights: forwarded to PoseLoss.
    """

    def __init__(
        self,
        model,
        lambda_pos: float = 1.0,
        lambda_rot: float = 1.0,
        axis_weights=(0.3, 1.0, 1.0),
    ):
        self.seg_loss = v8SegmentationLoss(model)
        self.pose_loss = PoseLoss(axis_weights=axis_weights)
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot

    def to(self, device):
        self.pose_loss = self.pose_loss.to(device)
        return self

    def __call__(self, preds, batch):
        """
        Args:
            preds: tuple (seg_preds, pose_preds) from VALNetModel.forward.
                   seg_preds is whatever the Segment head returns;
                   pose_preds is [B, 7] raw.
            batch: dict with the usual Ultralytics keys plus:
                   'pose': [B, 7] normalized GT (pos_norm, quat_unit).

        Returns:
            total_loss: scalar tensor for backward
            loss_items: dict of detached components for logging
        """
        seg_preds, pose_preds = preds

        # Segmentation
        seg_total, _ = self.seg_loss(seg_preds, batch)
        seg_total = seg_total.sum()

        # Pose — compute position and rotation terms with gradients intact
        pose_gt = batch["pose"].to(pose_preds.device).float()
        pos_pred, pos_gt = pose_preds[:, :3], pose_gt[:, :3]
        quat_pred_raw, quat_gt = pose_preds[:, 3:], pose_gt[:, 3:]

        per_axis_sq = (pos_pred - pos_gt) ** 2
        pos_loss = (per_axis_sq * self.pose_loss.axis_weights).mean()

        quat_pred = F.normalize(quat_pred_raw, dim=-1, eps=self.pose_loss.eps)
        dot = (quat_pred * quat_gt).sum(dim=-1)
        rot_loss = (1.0 - dot.abs()).mean()

        total = seg_total + self.lambda_pos * pos_loss + self.lambda_rot * rot_loss

        loss_items = {
            "seg":   seg_total.detach(),
            "pos":   pos_loss.detach(),
            "rot":   rot_loss.detach(),
            "total": total.detach(),
        }
        return total, loss_items
