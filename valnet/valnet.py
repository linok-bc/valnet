"""
VALNet: Integration with Ultralytics YOLOv8
============================================
    
- Load a pretrained YOLOv8-seg model
- Replace the neck (PAFPN) with AFPN
- Insert CEM between backbone and neck
- Wrap each detection/segmentation head scale with OAM
- Retrain

Usage:
    from valnet.valnet import VALNetModel
    model = VALNetModel.from_ultralytics("yolov8s-seg.pt", ch=(128, 256, 512))

Training lives in scripts/train.py. VALNetModel is a plain nn.Module, so
`model.train()` is nn.Module.train(mode) — it toggles train/eval, it does
not start a training run.
"""

import math

import torch.nn as nn

from valnet.afpn import AFPN
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

    def __init__(self, ch=(128, 256, 512), cem_depthwise=False, afpn_stage1=True):
        super().__init__()
        c3, c4, c5 = ch

        # One CEM per backbone scale
        # CEM operates on feature maps, not raw RGB, so in_channels = feature channels
        self.cem_p3 = CEM(in_channels=c3, patch_size=8, depthwise=cem_depthwise)
        self.cem_p4 = CEM(in_channels=c4, patch_size=8, depthwise=cem_depthwise)
        self.cem_p5 = CEM(in_channels=c5, patch_size=4, depthwise=cem_depthwise)

        # AFPN replaces PAFPN
        self.afpn = AFPN(ch=ch, stage1=afpn_stage1)

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

    def __init__(self, backbone, head, ch=(128, 256, 512), cem_depthwise=False, afpn_stage1=True):
        """
        Args:
            backbone: YOLOv8 backbone (extract from pretrained model)
            head: YOLOv8-seg head (extract from pretrained model)  
            ch: channel sizes at P3, P4, P5
            cem_depthwise, afpn_stage1: architecture options, see CEM and AFPN.
                The defaults build the full-width version (19.3M params);
                cem_depthwise=True, afpn_stage1=False builds the paper-size
                one (12.8M, vs 12.4M in the paper's Table 6).
        """
        super().__init__()


        # we get the WHOLE backbone from YOLO; split up to get the multi-channel output
        # if not using YOLOv8s, YOU WILL NEED TO CHANGE THESE!!!
        # `backbone` is deliberately NOT stored on self: the three slices below
        # already own every one of its layers, so registering it as well would
        # alias each tensor under two state_dict names.
        self.backbone_p3 = nn.Sequential(*list(backbone[:5]))
        self.backbone_p4 = nn.Sequential(*list(backbone[5:7]))
        self.backbone_p5 = nn.Sequential(*list(backbone[7:10]))
        
        self.neck = VALNetNeck(ch=ch, cem_depthwise=cem_depthwise, afpn_stage1=afpn_stage1)
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

        # Segmentation head
        seg_out = self.head(enhanced)
        return seg_out

    @classmethod
    def from_ultralytics(cls, model_name, ch, nc=None, **arch):
        """
        Extract backbone and head from a pretrained Ultralytics model.

        Args:
            model_name: path to (or name of) a YOLOv8-seg checkpoint.
            ch: channel sizes at P3, P4, P5. Required — it depends on the
                YOLOv8 variant, and getting it wrong builds a neck whose
                channels silently disagree with the backbone.
            nc: number of classes. None keeps the checkpoint's head (80 COCO
                classes); otherwise the class branch is resized, see
                `resize_head_classes`.
            **arch: forwarded to __init__ (cem_depthwise, afpn_stage1).

        Usage:
            valnet = VALNetModel.from_ultralytics("yolov8s-seg.pt", ch=(128, 256, 512))
        """
        from ultralytics import YOLO

        base = YOLO(model_name)
        m = base.model

        # Ultralytics weights are frozen by default; we need to unfreeze
        for p in m.parameters():
            p.requires_grad_(True)

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

        head = m.model[-1]
        if nc is not None and nc != head.nc:
            resize_head_classes(head, nc)

        return cls(
            backbone=nn.Sequential(*list(m.model[:10])),
            head=head,
            ch=ch,
            **arch,
        )


def resize_head_classes(head, nc):
    """
    Resize a pretrained Detect/Segment head to `nc` classes, in place.

    Mirrors what Ultralytics does when it trains a COCO checkpoint on a dataset
    with a different class count: every tensor whose shape still matches is
    kept, which for YOLOv8s is everything except the final 1x1 conv of each
    class branch (cv3[i][-1]). The hidden width of cv3 is
    max(ch[0], min(nc, 100)) = 128 for both nc=80 and nc=1, so only that last
    conv changes. It is re-initialized with PyTorch defaults and the same
    prior bias as Detect.bias_init. This makes the head identical in shape
    and initialization to the stock YOLOv8-seg baseline's.
    """
    for i, branch in enumerate(head.cv3):
        old = branch[-1]
        new = nn.Conv2d(old.in_channels, nc, 1).to(old.weight.device)
        new.bias.data[:] = math.log(5 / nc / (640 / float(head.stride[i])) ** 2)
        branch[-1] = new
    head.nc = nc
    head.no = nc + head.reg_max * 4
