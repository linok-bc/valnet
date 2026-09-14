import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CEM(nn.Module):
    """
    section 4.2
    Fourier-based contextual enhancement of RGB images
    """

    def __init__(self, in_channels: int = 3, sigma: int = 2, patch_size = 8):
        super().__init__()

        """
        param[in] in_channels: the number of channels for input images
        param[in] sigma: standard deviation for the Gaussian kernel from eq. 2
        param[in] patch_size: patch size for heatmap; reference figure 9
        """

        self.in_channels = in_channels

        self.patch_size = patch_size

        self.lowpass_conv = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.lowpass_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
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
        [out] weight: softmaxed heatmap corresponding to variable U in equation 6 with shape [B, H, W]

        Equation 6 softmaxes WITHIN each local s x s region, not across regions:
        "in each local region of size s x s, guided by M, a vector U with the
        highest probability representing the target region is obtained by
        Softmax", with c = s * s and M = {M_1, ..., M_c}. So the competition is
        between the pixels of one region — which is what amplifies the
        high-temperature area S_th that Equation 5 asks for — and each region's
        weights sum to 1 independently of every other region.
        """

        B, H, W = heatmap.shape
        s = self.patch_size

        # pad heatmap is not directly divisble by s
        pad_y, pad_x = (s - H % s) % s, (s - W % s) % s
        pad_y_left, pad_y_right = int(np.floor(pad_y / 2)), int(np.ceil(pad_y / 2))
        pad_x_left, pad_x_right = int(np.floor(pad_x / 2)), int(np.ceil(pad_x / 2))
        # -inf so padded cells get zero softmax weight. Padding is < s on each
        # side, so every region still contains at least one real element.
        padded_heatmap = F.pad(
            heatmap, (pad_x_left, pad_x_right, pad_y_left, pad_y_right),
            value=float("-inf"),
        )
        Hp, Wp = padded_heatmap.shape[-2:]
        nH, nW = Hp // s, Wp // s

        # [B, nH, s, nW, s] -> [B, nH, nW, s, s] so each region is contiguous
        regions = padded_heatmap.view(B, nH, s, nW, s).permute(0, 1, 3, 2, 4)

        # Softmax over the c = s*s elements of each region (equation 6)
        weight = F.softmax(regions.reshape(B, nH, nW, s * s), dim=-1)

        # back to [B, Hp, Wp], then drop the padding we added
        weight = weight.view(B, nH, nW, s, s).permute(0, 1, 3, 2, 4).reshape(B, Hp, Wp)
        return weight[:, pad_y_left:pad_y_left + H, pad_x_left:pad_x_left + W]



    def _highpass(self, x):
        """
        [in] x: high-pass features corresponding to x_bar from figure 9. has shape [B, C, H, W]
        [out] x_smile: output from the high-path output
        """
        # compute spatial attention
        f_max = torch.max(x, dim=1, keepdim=True).values            # [B, 1, H, W]
        f_mean = torch.mean(x, dim=1, keepdim=True)                 # [B, 1, H, W]
        f_concat = torch.cat([f_max, f_mean], dim=1)                # [B, 2, H, W]
        x_s = torch.sigmoid(self.spatial_attention_kernel(f_concat))    # [B, 1, H, W]

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
