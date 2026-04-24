"""
PoseHead: regress aircraft 6-DOF pose from multi-scale feature maps.

Consumes the three feature maps produced by OAM (post-AFPN, same channel
sizes as the backbone scales) and outputs 7 numbers per image:

    [0:3]  position  = (signed_log(x), y, z)   runway-relative, normalized
    [3:7]  rotation  = quaternion (qx, qy, qz, qw), UNNORMALIZED, xyzw order

Quaternion convention: scalar-last (xyzw), matching scipy / ROS. The
quaternion is output raw (no norm constraint on the layer). The loss
function is responsible for (a) normalizing to unit length before
comparing and (b) handling the q / -q double-cover ambiguity. Doing this
at the loss rather than the head keeps gradients well-behaved.
"""

import torch
import torch.nn as nn


class PoseHead(nn.Module):
    """
    Regress (position, quaternion) from three multi-scale feature maps.

    Args:
        ch: tuple of channel counts for the three input scales, e.g.
            (128, 256, 512) for yolov8s. Must match the channels coming
            out of OAM.
        hidden: width of the MLP hidden layers. Default 256 is a reasonable
            starting point; increase if the model underfits.
    """

    def __init__(self, ch=(128, 256, 512), hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = sum(ch)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 7),
        )

        # Small init on the final layer so the network starts near zero
        # position / near-zero raw quaternion values rather than large
        # random outputs. Stabilizes early training.
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

        # Bias the quaternion toward identity (qx, qy, qz, qw) = (0, 0, 0, 1)
        # at init so early loss values are sane rather than wildly off.
        # In xyzw ordering, w sits at index 6 of the full 7-vector.
        with torch.no_grad():
            self.mlp[-1].bias[6] = 1.0  # qw = 1

    def forward(self, features):
        """
        Args:
            features: iterable of 3 tensors, each [B, C_i, H_i, W_i].
                      Order must match the `ch` tuple passed at init.

        Returns:
            pose: [B, 7] tensor.
                  pose[:, :3] = normalized position (signed-log on x, raw y/z)
                  pose[:, 3:] = raw quaternion (qx, qy, qz, qw), not normalized
        """
        # Pool each scale to [B, C_i], concatenate to [B, sum(C_i)]
        pooled = [self.pool(f).flatten(1) for f in features]
        x = torch.cat(pooled, dim=1)
        return self.mlp(x)
