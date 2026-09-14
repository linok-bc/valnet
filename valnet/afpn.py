"""
AFPN (Asymptotic Feature Pyramid Network) for YOLOv8
Based on VALNet (Wang et al., Remote Sensing 2024) — Figure 10

Architecture:

    Backbone outputs: P3 (high-res, low-level), P4 (mid), P5 (low-res, high-level)

    AFPN is *asymptotic*: it starts from the two adjacent LOW-level features and
    only then incorporates the high-level one — "first fusing features from two
    adjacent layers and gradually incorporating high-level features into the
    fusion process" (p. 16).

    First stage:
        P3, P4, P5 -> ConvBNSiLU -> H1, H2, H3

    Second stage — fuse the adjacent low-level pair (H1, H2), at both scales:
        H1, H2 -> Ha        Ha at H1 scale (P3)
        H1, H2 -> Hb        Hb at H2 scale (P4)

    Third stage — incorporate the high-level feature H3:
        Ha, Hb, H3 -> O1    O1 at Ha scale (P3)
        Ha, Hb, H3 -> O2    O2 at Hb scale (P4)
        Ha, Hb, H3 -> O3    O3 at H3 scale (P5)

    Each output then passes through a final 1x1 convolution (Figure 10).

Fusion is ASFF-style (adaptively spatial feature fusion): the per-input weights
are predicted PER PIXEL from the aligned features and softmaxed across inputs,
so different image regions can prefer different scales.

Resampling: downsampling averages over each target cell so every input pixel
contributes; upsampling is nearest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNSiLU(nn.Module):
    """Standard Conv + BatchNorm + SiLU (matches Ultralytics Conv)."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def resize_to(x, size):
    """Resize x to `size` = (H, W).

    Downsampling averages over each target cell so every input pixel
    contributes. Nearest-neighbour point-sampling would throw most of a
    high-resolution map away — P3 80x80 -> 20x20 keeps only 6.2% of pixels.
    Upsampling stays nearest.
    """
    size = tuple(size)
    if tuple(x.shape[-2:]) == size:
        return x
    if size[0] <= x.shape[-2] and size[1] <= x.shape[-1]:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="nearest")


class ASFFFuseBlock(nn.Module):
    """
    Aligns N inputs to a common channel count and spatial size, then fuses them
    by adaptively spatial feature fusion: one weight map per input, predicted
    per pixel and softmaxed across inputs, followed by a 3x3 conv refinement.

    Args:
        in_chs: channel count of each input, in the order forward() receives them
        out_ch: channel count of the fused output
    """

    def __init__(self, in_chs, out_ch):
        super().__init__()
        self.n = len(in_chs)
        self.aligns = nn.ModuleList(
            ConvBNSiLU(c, out_ch, 1) if c != out_ch else nn.Identity() for c in in_chs
        )

        # ASFF weight branch: compress each aligned feature, concatenate, then
        # predict one weight map per input and softmax across inputs per pixel.
        compress = max(out_ch // 8, 8)
        self.compress = nn.ModuleList(
            nn.Conv2d(out_ch, compress, 1) for _ in range(self.n)
        )
        self.weight = nn.Conv2d(compress * self.n, self.n, 1)

        self.refine = ConvBNSiLU(out_ch, out_ch, 3)

    def forward(self, inputs, target_size=None):
        """
        Fuse N feature maps.

        param[in] inputs: sequence of N feature maps, channels matching `in_chs`
        param[in] target_size: (H, W) to fuse at; defaults to the first input's
        param[out] out: fused feature map [B, out_ch, *target_size]
        """
        feats = [align(x) for align, x in zip(self.aligns, inputs)]
        tgt = target_size if target_size is not None else feats[0].shape[2:]
        feats = [resize_to(f, tgt) for f in feats]

        w = torch.cat([c(f) for c, f in zip(self.compress, feats)], dim=1)
        w = torch.softmax(self.weight(w), dim=1)          # [B, N, H, W]

        out = sum(w[:, i:i + 1] * f for i, f in enumerate(feats))
        return self.refine(out)


class AFPN(nn.Module):
    """
    Asymptotic Feature Pyramid Network (Figure 10 of VALNet).

    See the module docstring for the fusion order and why it starts low-level.
    """

    def __init__(self, ch=(128, 256, 512)):
        """ Make sure that these channels are right """
        super().__init__()
        c3, c4, c5 = ch

        # Stage 1: per-scale projection
        self.layer1_1 = ConvBNSiLU(c3, c3)
        self.layer1_2 = ConvBNSiLU(c4, c4)
        self.layer1_3 = ConvBNSiLU(c5, c5)

        # Stage 2: fuse the adjacent low-level pair (P3, P4)
        self.layer2_1 = ASFFFuseBlock((c3, c4), c3)       # Ha, at P3 scale
        self.layer2_2 = ASFFFuseBlock((c3, c4), c4)       # Hb, at P4 scale

        # Stage 3: incorporate the high-level feature (P5)
        self.layer3_1 = ASFFFuseBlock((c3, c4, c5), c3)   # O1, at P3 scale
        self.layer3_2 = ASFFFuseBlock((c3, c4, c5), c4)   # O2, at P4 scale
        self.layer3_3 = ASFFFuseBlock((c3, c4, c5), c5)   # O3, at P5 scale

        # Final 1x1 convolutions (Figure 10)
        self.out_1 = ConvBNSiLU(c3, c3, 1)
        self.out_2 = ConvBNSiLU(c4, c4, 1)
        self.out_3 = ConvBNSiLU(c5, c5, 1)

    def forward(self, features):
        """
        Fuse feature maps

        param[in] features: tuple containing three feature maps (p3, p4, p5)
        param[out] out: tuple containing three feature maps (o1, o2, o3) with the same shape as features
        """
        p3, p4, p5 = features

        """
        First stage:
        P3, P4, P5 -> ConvBNSiLU -> H1, H2, H3
        """

        h1 = self.layer1_1(p3)
        h2 = self.layer1_2(p4)
        h3 = self.layer1_3(p5)

        """
        Second stage — the adjacent low-level pair fuses first:
        H1, H2 -> Ha        Ha at H1 scale
        H1, H2 -> Hb        Hb at H2 scale
        """

        ha = self.layer2_1((h1, h2), target_size=h1.shape[2:])
        hb = self.layer2_2((h1, h2), target_size=h2.shape[2:])

        """
        Third stage — the high-level feature is incorporated last:
        Ha, Hb, H3 -> O1    O1 at Ha scale
        Ha, Hb, H3 -> O2    O2 at Hb scale
        Ha, Hb, H3 -> O3    O3 at H3 scale
        """

        o1 = self.layer3_1((ha, hb, h3), target_size=ha.shape[2:])
        o2 = self.layer3_2((ha, hb, h3), target_size=hb.shape[2:])
        o3 = self.layer3_3((ha, hb, h3), target_size=h3.shape[2:])

        return (self.out_1(o1), self.out_2(o2), self.out_3(o3))
