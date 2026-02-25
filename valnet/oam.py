"""
OAM (Orientation Adaptation Module) for YOLOv8
Based on VALNet (Wang et al., Remote Sensing 2024) — Figures 6, 11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OAM(nn.Module):
    """
    Orientation Adaptation Module.

    Enhances rotational invariance by fusing features from 90° CW rotation,
    90° CCW rotation, and original orientation via spatial and channel attention.

    Args:
        channels: number of input/output channels (same in, same out)
        reduction: channel reduction ratio for the FC layers in Original-Channel
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # Convolutions for after spatial attention on CL/CR branches
        self.cl_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.cr_spatial= nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # original path: channel attention
        mid = max(channels // reduction, 8)
        self.fc_gap = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_gmp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Linear(mid, channels, bias=False)

        # --- Adaptive fusion weights (Eq. 18) ---
        # α, β, γ — learnable, softmax-normalized at forward time
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    @staticmethod
    def _rotate_cw(x):
        """Rotate feature map 90° clockwise: (B, C, H, W) -> (B, C, W, H)."""
        return x.transpose(-2, -1).flip(-1)

    @staticmethod
    def _rotate_ccw(x):
        """Rotate feature map 90° counter-clockwise: (B, C, H, W) -> (B, C, W, H)."""
        return x.transpose(-2, -1).flip(-2)

    @staticmethod
    def _spatial_pool(x):
        """MaxAvgPool: compute max and mean along channel dim, concat -> (B, 2, H, W)."""
        return torch.cat([
            x.max(dim=1, keepdim=True).values,
            x.mean(dim=1, keepdim=True),
        ], dim=1)

    def _rotation_branch(self, x, rotate_fn, inverse_fn, spatial_attn):
        """
        Shared logic for CR-Channel and CL-Channel.

        1. Rotate input
        2. Compute spatial attention on rotated features
        3. Multiply rotated features by attention
        4. Inverse-rotate back to original orientation

        param[]
        """
        x_rot = rotate_fn(x)

        pool = self._spatial_pool(x_rot)            # (B, 2, H, W)
        attn = spatial_attn(pool)                   # (B, 1, H, W)
        x_rot_attended = x_rot * attn

        y = inverse_fn(x_rot_attended)
        return y

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map from AFPN/neck at one scale
        Returns:
            y: (B, C, H, W) orientation-enhanced feature map
        """
        # clockwise channel
        y_r = self._rotation_branch(x, self._rotate_cw, self._rotate_ccw, self.cr_spatial)

        # counterclockwise channel
        y_l = self._rotation_branch(x, self._rotate_ccw, self._rotate_cw, self.cl_spatial)

        # Original-Channel: channel attention (Eq. 16-17)
        gap = x.mean(dim=[2, 3])                    # (B, C)
        gmp = x.amax(dim=[2, 3])                    # (B, C)
        ch_attn = self.fc_out(self.fc_gap(gap) + self.fc_gmp(gmp))
        ch_attn = ch_attn[..., None, None]          # (B, C, 1, 1)
        y_ur = x * ch_attn

        # Adaptive weighted fusion (Eq. 18)
        w = torch.softmax(self.fusion_weights, dim=0)
        y = w[0] * y_r + w[1] * y_l + w[2] * y_ur

        return y
