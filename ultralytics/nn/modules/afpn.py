"""
AFPN (Asymptotic Feature Pyramid Network) for YOLOv8
Based on VALNet (Wang et al., Remote Sensing 2024) — Figure 10

Architecture (from the diagram):

    Backbone outputs: P3 (large), P4 (mid), P5 (small)

    Progressive bottom-up fusion:
    
    First stage:
        P3, P4, P5 -> ConvBNSiLU -> H1, H2, H3
    
    Second stage:
    H2, H3 -> Ha        Ha at H2 scale
    H3, H2 -> Hb        Hb at H3 scale
    
    Third stage:
    H1, Ha, Hb -> O1    O1 at H1 scale
    Ha, H1, Hb -> O2    O2 at H2 scale
    Hb, Ha, H1 -> O3    O3 at H3 scale

    O1, O2, O3 are almost the outputs; we apply a 1x1 convolution and these are the final outputs
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


class DualAFPNFuseBlock(nn.Module):
    """
    Aligns two inputs to the same channel count and spatial size,
    then fuses via learnable weighted addition + conv refinement.
    """

    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.align1 = ConvBNSiLU(in_ch1, out_ch, 1) if in_ch1 != out_ch else nn.Identity()
        self.align2 = ConvBNSiLU(in_ch2, out_ch, 1) if in_ch2 != out_ch else nn.Identity()
        self.w = nn.Parameter(torch.ones(2) / 2)
        self.refine = ConvBNSiLU(out_ch, out_ch, 3)

    def forward(self, x_main, x_aux, target_size=None):
        """
        Fuses two feature maps; output size is the same size as first input if target_size not specified

        param[in] x_main: primary feature map
        param[in] x_aux:  auxiliary feature map to fuse in
        param[in] target_size: the target size for the final feature map, optional
        param[out] out: 
        """
        f1 = self.align1(x_main)
        f2 = self.align2(x_aux)

        tgt = target_size if target_size is not None else f1.shape[2:]

        if f1.shape[2:] != tgt:
            f1 = F.interpolate(f1, size=tgt, mode="nearest")
        if f2.shape[2:] != tgt:
            f2 = F.interpolate(f2, size=tgt, mode="nearest")

        w = torch.softmax(self.w, dim=0)
        out = self.refine(w[0] * f1 + w[1] * f2)
        return out


class TripleAFPNFuseBlock(nn.Module):
    """
    Aligns three inputs to the same channel count and spatial size,
    then fuses via learnable weighted addition + conv refinement.
    """

    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super().__init__()
        self.align1 = ConvBNSiLU(in_ch1, out_ch, 1) if in_ch1 != out_ch else nn.Identity()
        self.align2 = ConvBNSiLU(in_ch2, out_ch, 1) if in_ch2 != out_ch else nn.Identity()
        self.align3 = ConvBNSiLU(in_ch3, out_ch, 1) if in_ch1 != out_ch else nn.Identity()
        self.w = nn.Parameter(torch.ones(3) / 3)
        self.refine = ConvBNSiLU(out_ch, out_ch, 3)

    def forward(self, x_main, x_aux1, x_aux2, target_size=None):
        """
        Fuses three feature maps; output size is the same size as first input if target_size not specified

        param[in] x_main: primary feature map
        param[in] x_aux1: first auxiliary feature map to fuse in
        param[in] x_aux2: second auxiliary feature map to fuse in
        param[in] target_size: the target size for the final feature map, optional
        param[out] out: 
        """
        f1 = self.align1(x_main)
        f2 = self.align2(x_aux1)
        f3 = self.align3(x_aux2)

        tgt = target_size if target_size is not None else f1.shape[2:]

        if f1.shape[2:] != tgt:
            f1 = F.interpolate(f1, size=tgt, mode="nearest")
        if f2.shape[2:] != tgt:
            f2 = F.interpolate(f2, size=tgt, mode="nearest")
        if f3.shape[2:] != tgt:
            f3 = F.interpolate(f3, size=tgt, mode="nearest")

        w = torch.softmax(self.w, dim=0)
        out = self.refine(w[0] * f1 + w[1] * f2 + w[2] * f3)
        return out


class AFPN(nn.Module):
    """
    Asymptotic Feature Pyramid Network (Figure 10 of VALNet).
    
    See above for explanation of how layers are fused
    """

    def __init__(self, ch=(128, 256, 512)):
        """ Make sure that these channels are right """
        super().__init__()
        c3, c4, c5 = ch

        self.layer1_1 = ConvBNSiLU(c3, c3)
        self.layer1_2 = ConvBNSiLU(c4, c4)
        self.layer1_3 = ConvBNSiLU(c5, c5)

        self.layer2_1 = DualAFPNFuseBlock(c3, c4, c3)
        self.layer2_2 = DualAFPNFuseBlock(c3, c4, c4)

        self.layer3_1 = TripleAFPNFuseBlock(c3, c4, c5, c3)
        self.layer3_2 = TripleAFPNFuseBlock(c3, c4, c5, c4)
        self.layer3_3 = TripleAFPNFuseBlock(c3, c4, c5, c5)

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
        Second stage:
        H2, H3 -> Ha        Ha at H2 scale
        H3, H2 -> Hb        Hb at H3 scale
        """

        ha = self.layer2_1(h1, h2, target_size=h1.size())
        hb = self.layer2_2(h1, h2, target_size=h2.size())
        
        """
        Third stage:
        H1, Ha, Hb -> O1    O1 at H1 scale
        Ha, H1, Hb -> O2    O2 at H2 scale
        Hb, Ha, H1 -> O3    O3 at H3 scale
        """

        o1 = self.layer3_1(h1, ha, hb, target_size=h1.size())
        o2 = self.layer3_2(h1, ha, hb, target_size=ha.size())
        o3 = self.layer3_3(h1, ha, hb, target_size=hb.size())
        out = (o1, o2, o3)

        return out

def register_afpn_in_ultralytics():
    """Call before YOLO('yolov8-afpn.yaml') so Ultralytics can parse AFPN."""
    try:
        import ultralytics.nn.modules as modules
        if not hasattr(modules, "AFPN"):
            modules.AFPN = AFPN
        print("AFPN registered in Ultralytics")
    except ImportError:
        print("Ultralytics not found - use AFPN as standalone nn.Module")
