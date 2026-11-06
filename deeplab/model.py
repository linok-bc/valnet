"""DeepLabV3+ (Chen et al., ECCV 2018) on torchvision's COCO-pretrained DeepLabV3.

torchvision ships DeepLabV3 only (ASPP output upsampled 8x straight to the
input). V3+ adds a decoder that fuses the ASPP output with stride-4 backbone
features before the final upsample, which sharpens boundaries -- the part that
matters for a runway a few pixels wide.

What is pretrained (COCO, VOC label set): the ResNet-50 backbone and ASPP.
What is new: the decoder and a 1-channel runway logit.

Output stride: torchvision's model runs at stride 8 (layer3 and layer4 dilated,
ASPP rates 12/24/36). The V3+ paper trains at stride 16 (only layer4 dilated,
rates 6/12/18), which at 1280x720 is ~2x faster and lighter. Dilation does not
change any weight's shape, so the COCO weights load unchanged; halving the ASPP
rates keeps its receptive field the same in pixels.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


def conv_bn_relu(cin, cout, k):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class DeepLabV3Plus(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, output_stride=16):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        # The COCO weights include the auxiliary head, so aux_loss must match; it is not used.
        base = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=pretrained)
        backbone = base.backbone
        if output_stride == 16:
            backbone = resnet50(replace_stride_with_dilation=[False, False, True])
            missing, unexpected = backbone.load_state_dict(base.backbone.state_dict(), strict=False)
            assert not unexpected and set(missing) == {"fc.weight", "fc.bias"}, (missing, unexpected)
        # Same backbone weights, now also returning layer1 (stride 4, 256 ch).
        self.backbone = IntermediateLayerGetter(backbone, {"layer1": "low", "layer4": "out"})
        self.aspp = base.classifier[0]  # 2048 -> 256, pretrained
        if output_stride == 16:
            for branch in self.aspp.convs[1:4]:
                conv = branch[0]
                conv.dilation = tuple(d // 2 for d in conv.dilation)
                conv.padding = conv.dilation
        self.low_proj = conv_bn_relu(256, 48, 1)  # paper: 48 channels
        self.decoder = nn.Sequential(conv_bn_relu(256 + 48, 256, 3), conv_bn_relu(256, 256, 3))
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.backbone(x)
        y = self.aspp(f["out"])
        low = self.low_proj(f["low"])
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decoder(torch.cat([y, low], 1))
        return F.interpolate(self.head(y), size=(h, w), mode="bilinear", align_corners=False)

    def param_groups(self):
        """Pretrained parts (backbone, ASPP) vs new parts (decoder, head)."""
        pre = list(self.backbone.parameters()) + list(self.aspp.parameters())
        new = list(self.low_proj.parameters()) + list(self.decoder.parameters()) + list(self.head.parameters())
        return pre, new
