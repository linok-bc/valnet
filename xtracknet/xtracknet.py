import torch
import torch.nn as nn
import torch.nn.functional as F


class XTrackNet(nn.Module):
    """
    Two-stream CNN for cross-track displacement regression.

    Inputs:
        rgb:       [B, 3, 720, 1280]  raw RGB image
        mask:      [B, 1, 720, 1280]  binary runway mask
        roll_rad:  [B]                roll angle in radians
        pitch_rad: [B]                pitch angle in radians
        alt_m:     [B]                altitude in meters
    Output:
        xtrack:    [B, 1]             cross-track displacement
    """
    
    def __init__(self, alt_scale=500.0, hidden=128):
        super().__init__()
        self.alt_scale = alt_scale

        # 640 -> 320 -> 160 -> 80 -> 40 (then GAP)
        self.mask_stream = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        feat_dim = 64 * 4 + 128 * 4 + 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def _derotate(self, x, roll_rad, interp):
        """Rotate [B, C, H, W] per-sample by -roll using batched affine_grid."""
        B, C, H, W = x.shape
        cos = torch.cos(-roll_rad)
        sin = torch.sin(-roll_rad)
        zero = torch.zeros_like(cos)
        theta = torch.stack([
            torch.stack([cos, -sin, zero], dim=-1),
            torch.stack([sin,  cos, zero], dim=-1),
        ], dim=-2)  # [B, 2, 3]
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode=interp, padding_mode='zeros', align_corners=False)


    def forward(self, rgb, mask, roll_rad, pitch_rad, alt_m):
        # No resize — input arrives at 640x640 from YOLO's letterbox pipeline
        rgb  = self._derotate(rgb,  roll_rad, interp='bilinear')
        mask = self._derotate(mask, roll_rad, interp='nearest')
    
        f_mask = self.mask_stream(mask)
        f_rgb  = self.rgb_stream(rgb)
    
        alt_norm = alt_m / self.alt_scale
        aux = torch.stack([pitch_rad, alt_norm], dim=-1)
    
        x = torch.cat([f_mask, f_rgb, aux], dim=-1)
        return self.head(x)
