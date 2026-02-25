import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

class CEM(nn.Module):
    """
    section 4.2
    Fourier-based contextual enhancement of RGB images
    """

    def __init__(self, in_channels: int = 3, sigma: int = 2, patch_size = 128):
        super().__init__()

        """
        param[in] in_channels: the number of channels for input images
        param[in] sigma: standard deviation for the Gaussian kernel from eq. 2
        param[in] patch_size: patch size for heatmap; reference figure 9
        """

        self.in_channels = in_channels

        self.patch_size = patch_size

        self.lowpass_conv = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.lowpass_pool = nn.AvgPool2d(3)
        self.spatial_attention_kernel = nn.Conv2d(2, 1, 3, padding=1)
        self.highpass_kernel = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)


    def _lowpass(self, x):
        """
        param[in] x: input matrix of shape [B, C, H, W]
        param[out] x3: output matrix of shape [B, C, H, W]
        """
        H, W = x.shape[2:]
        x1 = self.lowpass_conv(x)
        x2 = self.lowpass_pool(x1)
        x3 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
        return x3



    def _generate_heatmap(self, x):
        """
        param[in] x: input matrix of shape [B, C, H, W]
        param[out] out: heatmap of shape [B, H, W]
        """
        out = x.mean(dim=1)
        return out



    def _heatmap_softmax(self, heatmap):
        """
        [in] heatmap: input heatmap of shape [B, H, W]
        [out] patches_weight: softmaxed heatmap corresponding to variable U in equation 6 with shape [B, H, W]
        """

        B, H, W = heatmap.shape
        s = self.patch_size

        # pad heatmap is not directly divisble by s
        pad_y, pad_x = (s - H % s), (s - W % s)
        pad_y_left, pad_y_right = int(np.floor(pad_y / 2)), int(np.ceil(pad_y / 2))
        pad_x_left, pad_x_right = int(np.floor(pad_x / 2)), int(np.ceil(pad_x / 2))
        padded_heatmap = F.pad(heatmap, (pad_x_left, pad_x_right, pad_y_left, pad_y_right))
        Hp, Wp = padded_heatmap.shape[-2:]

        # convert heatmap to relative weight of patches
        patches = padded_heatmap.view(B, Hp//s, s, Wp//s, s)
        patches_logits = torch.sum(torch.sum(patches, dim=-1), dim=-2)      # [B, Hp, Wp]
        patches_weight = F.softmax(patches_logits.view(B, -1), dim=-1).view(B, Hp//s, Wp//s)

        # since we have the padding information in this function, let's expand it back to it's intended shape here
        patches_weight_expanded = torch.repeat_interleave(torch.repeat_interleave(patches_weight, s, -2), s, -1)
        patches_weight_window = patches_weight_expanded[:, pad_y_left:pad_y_left+H, pad_x_left:pad_x_left+W]

        return patches_weight_window


    def _highpass(self, x):
        """
        [in] x: high-pass features corresponding to x_bar from figure 9. has shape [B, C, H, W]
        [out] x_smile: output from the high-path output
        """
        # compute spatial attention
        f_max = torch.max(x, dim=1, keepdim=True).values            # [B, 1, H, W]
        f_mean = torch.mean(x, dim=1, keepdim=True)                 # [B, 1, H, W]
        f_concat = torch.cat([f_max, f_mean], dim=1)                # [B, 2, H, W]
        x_s = F.sigmoid(self.spatial_attention_kernel(f_concat))    # [B, 1, H, W]

        # compute X_smile
        y_smile = (x_s * self.highpass_kernel(x)) + x
        return y_smile


    def forward(self, x):
        # perform heatmap-guided and low-pass filter paths, the combine their results 
        x_tilde = self._lowpass(x)
        u = self._heatmap_softmax(self._generate_heatmap(x))
        x_frown = torch.mul(x_tilde, u[:, None, :, :])
        
        # high-pass filter path
        x_bar = x - x_tilde
        y_smile = self._highpass(x_bar)

        return x_frown + y_smile

def register_cem_in_ultralytics():
    """Call before YOLO('yolov8-cem.yaml') so Ultralytics can parse CEM."""
    try:
        import ultralytics.nn.modules as modules
        if not hasattr(modules, "CEM"):
            modules.CEM = CEM
        print("CEM registered in Ultralytics")
    except ImportError:
        print("Ultralytics not found - use CEM as standalone nn.Module")
