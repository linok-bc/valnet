"""
VALNet: Integration with Ultralytics YOLOv8
============================================
    
- Load a pretrained YOLOv8-seg model
- Replace the neck (PAFPN) with AFPN
- Insert CEM between backbone and neck
- Wrap each detection/segmentation head scale with OAM
- Retrain

Usage:
    from valnet import build_valnet
    model = build_valnet(base="yolov8s-seg.pt")
    model.train(data="your_rld.yaml", epochs=50, batch=12)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO

# Import your modules (adjust paths as needed)
from valnet.afpn import AFPN, ConvBNSiLU
from valnet.cem import CEM
from valnet.oam import OAM


class VALNetNeck(nn.Module):
    """
    Combined CEM + AFPN neck.
    
    Pipeline per Figure 6:
        Backbone P3, P4, P5
            -> CEM applied to each scale (contextual enhancement)
            -> AFPN (progressive multi-scale fusion)
            -> 3 output feature maps for the head
    """

    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = ch

        # One CEM per backbone scale
        # CEM operates on feature maps, not raw RGB, so in_channels = feature channels
        self.cem_p3 = CEM(in_channels=c3)
        self.cem_p4 = CEM(in_channels=c4)
        self.cem_p5 = CEM(in_channels=c5)

        # AFPN replaces PAFPN
        self.afpn = AFPN(ch=ch)

    def forward(self, features):
        """
        Args:
            features: list of [P3, P4, P5] from backbone
        Returns:
            list of [out_P3, out_P4, out_P5]
        """
        p3, p4, p5 = features

        # CEM contextual enhancement
        p3 = self.cem_p3(p3)
        p4 = self.cem_p4(p4)
        p5 = self.cem_p5(p5)

        # AFPN progressive fusion
        return self.afpn((p3, p4, p5))


class VALNetModel(nn.Module):
    """
    Full VALNet model with clean forward pass.
    
    Use this if you need full control over the architecture
    rather than monkey-patching an Ultralytics model.
    
    Architecture (Figure 6):
        Input -> Backbone -> CEM (per scale) -> AFPN -> OAM (per scale) -> Head -> Output
    """

    def __init__(self, backbone, head, ch=(128, 256, 512)):
        """
        Args:
            backbone: YOLOv8 backbone (extract from pretrained model)
            head: YOLOv8-seg head (extract from pretrained model)  
            ch: channel sizes at P3, P4, P5
        """
        super().__init__()


        # we get the WHOLE backbone from YOLO; split up to get the multi-channel output
        # if not using YOLOv8s, YOU WILL NEED TO CHANGE THESE!!!
        self.backbone = backbone
        self.backbone_p3 = nn.Sequential(*list(self.backbone[:4]))
        self.backbone_p4 = nn.Sequential(*list(self.backbone[4:6]))
        self.backbone_p5 = nn.Sequential(*list(self.backbone[6:8]))
        
        self.neck = VALNetNeck(ch=ch)
        self.oams = nn.ModuleList(OAM(c) for c in ch)
        self.head = head

    def forward(self, x):
        # Backbone: extract multi-scale features
        p3 = self.backbone_p3(x)
        p4 = self.backbone_p4(p3)
        p5 = self.backbone_p5(p4)
        features = (p3, p4, p5)

        # CEM + AFPN neck
        fused = self.neck(features)  # [F3, F4, F5]

        # OAM per scale
        enhanced = [oam(f) for oam, f in zip(self.oams, fused)]

        # Detection/segmentation head
        return self.head(enhanced)

    @classmethod
    def from_ultralytics(cls, model_name="yolov8s-seg.pt", ch=None):
        """
        Extract backbone and head from a pretrained Ultralytics model.

        Usage:
            valnet = VALNetModel.from_ultralytics("yolov8s-seg.pt")
        """
        from ultralytics import YOLO

        base = YOLO(model_name)
        if ch is None:
            ch = get_channel_sizes(base)

        m = base.model

        # Extract backbone layers (indices 0-9 in YOLOv8 typically)
        # and head (last module)
        # This is version-dependent; inspect m.model to verify indices

        print(f"Model structure has {len(m.model)} modules")
        print("Inspect m.model to identify backbone vs neck vs head indices")
        print(f"Using channel sizes: {ch}")

        # For reference, typical YOLOv8-seg structure:
        #   [0-9]:   backbone (Conv, C2f, SPPF)
        #   [10-22]: neck (Upsample, Concat, C2f, Conv)  
        #   [23]:    Segment head
        #
        # You'll need to split at the right index for your version.

        return cls(
            backbone=nn.Sequential(*list(m.model[:10])),
            head=m.model[-1],
            ch=ch,
        )
