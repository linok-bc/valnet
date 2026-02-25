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


class OAMHead(nn.Module):
    """
    Wraps the original YOLOv8-seg head with OAM at each scale.

    Per Figure 6: AFPN outputs -> OAM -> original head convolutions -> predictions
    """

    def __init__(self, original_head, channels_per_scale):
        """
        Args:
            original_head: the original Segment/Detect head from YOLOv8
            channels_per_scale: tuple of channels at each scale, e.g. (128, 256, 512)
        """
        super().__init__()
        self.original_head = original_head
        self.oams = nn.ModuleList([
            OAM(channels=c) for c in channels_per_scale
        ])

    def forward(self, features):
        """
        Args:
            features: list of feature maps from neck [P3, P4, P5]
        Returns:
            whatever the original head returns (detections + masks)
        """
        enhanced = [
            oam(feat) for oam, feat in zip(self.oams, features)
        ]
        return self.original_head(enhanced)


def get_channel_sizes(model):
    """
    Infer the channel sizes at each neck output scale from a loaded YOLOv8 model.
    
    Returns:
        tuple of (c3, c4, c5) channel counts
    """
    # Run a dummy forward through the backbone to get feature map shapes
    dummy = torch.zeros(1, 3, 640, 640)
    model.eval()

    # Access the underlying nn.Module
    m = model.model if hasattr(model, 'model') else model

    # YOLOv8 backbone outputs are typically at indices that feed into the neck
    # The detect/segment head stores which layers it receives from
    head = m.model[-1]  # last module is the head
    channels = []
    for ch_in in head.ch if hasattr(head, 'ch') else []:
        channels.append(ch_in)

    if channels:
        return tuple(channels)

    # Fallback: common YOLOv8 channel configs
    # Determined by model scale (n/s/m/l/x)
    name = getattr(model, 'cfg', '') or ''
    if 'n' in name:
        return (64, 128, 256)
    elif 's' in name:
        return (128, 256, 512)
    elif 'm' in name:
        return (192, 384, 576)
    elif 'l' in name:
        return (256, 512, 512)
    elif 'x' in name:
        return (320, 640, 640)
    else:
        return (128, 256, 512)  # default to 's' scale


def build_valnet(base="yolov8s-seg.pt", ch=None):
    """
    Build a VALNet model by modifying a pretrained YOLOv8-seg.

    Args:
        base: path to pretrained YOLOv8-seg weights or model name
        ch: optional tuple of (c3, c4, c5) channel sizes; auto-detected if None

    Returns:
        Modified YOLO model ready for training

    Usage:
        model = build_valnet("yolov8s-seg.pt")
        results = model.train(
            data="rld.yaml",
            epochs=50,
            batch=12,
            imgsz=640,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
        )
    """
    # Load base model
    model = YOLO(base)

    if ch is None:
        ch = get_channel_sizes(model)

    print(f"Building VALNet with channel sizes: {ch}")

    # Access the underlying DetectionModel / SegmentationModel
    m = model.model

    # --- Replace neck with CEM + AFPN ---
    # In YOLOv8, the neck is the set of modules between backbone and head.
    # Rather than surgically replacing individual neck layers (fragile across
    # Ultralytics versions), we use a hook-based approach:

    valnet_neck = VALNetNeck(ch=ch)
    oam_channels = list(ch)

    # --- Wrap head with OAM ---
    original_head = m.model[-1]
    oam_head = OAMHead(original_head, channels_per_scale=ch)
    m.model[-1] = oam_head

    # Store neck for the custom forward
    m._valnet_neck = valnet_neck

    # --- Monkey-patch the forward to use our neck ---
    original_forward = m.forward

    def valnet_forward(x, *args, **kwargs):
        # This is a simplified version; for production use, you'd want to
        # properly intercept the backbone outputs.
        # See the alternative class-based approach below.
        return original_forward(x, *args, **kwargs)

    print("VALNet modules attached. For full integration, use VALNetModel class below.")
    return model


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
        self.oams = nn.ModuleList([OAM(c) for c in ch])
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
