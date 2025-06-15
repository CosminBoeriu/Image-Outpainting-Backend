import torch
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

class InpaintGAN2(nn.Module):
    def __init__(self):
        self.name = 'Inpaint_GAN2'
        super().__init__()
        self.relu = nn.ReLU()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gated_256 = GatedSkipConnection(256)
        self.gated_128 = GatedSkipConnection(128)
        self.input_gate_conv = nn.Conv2d(3, 3, kernel_size=1)
        """Initial downsample convolutions, from 256x256 to 64x64"""
        self.pconv1 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.inorm1 = nn.InstanceNorm2d(num_features=64)
        self.pconv2 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.inorm2 = nn.InstanceNorm2d(num_features=128)

        self.downsample_convolutions = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
        )
        self.resnet_block_up = nn.Sequential(
            ResNetBlock(channels=128),
        )

        self.pconv3 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.inorm3 = nn.InstanceNorm2d(num_features=256)
        self.middle_residual_attention_blocks0 = nn.Sequential(
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            SelfAttention(channels=256),
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            ResNetBlock(channels=256),
        )
        self.lowest_convolution = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.lowest_residual_attention_blocks = nn.Sequential(
            AOTBlock(channels=512, rates=[1, 2, 4, 8]),
            SelfAttention(channels=512),
            AOTBlock(channels=512, rates=[1, 2, 4, 8]),
            SelfAttention(channels=512),
            ResNetBlock(channels=512),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        )
        self.middle_residual_attention_blocks1 = nn.Sequential(
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            SelfAttention(channels=256),
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            ResNetBlock(channels=256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=128),
        )

        self.upsample_convolutions = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """Get input and mask from the input tensor."""
        initial_input = x[:, 0:3, :, :]
        original_mask = x[:, 3:4, :, :]
        mask = (original_mask[:, :, :, :] + 1) / 2
        mask = 1 - mask
        x[:, 3:4, :, :] = mask  # Update the mask in the input tensor

        """Downsample with partial convolutions, from 256x256 to 64x64"""
        x, mask = self.pconv1(initial_input, mask)
        x = self.inorm1(x)
        x = self.relu(x)
        x, mask = self.pconv2(x, mask)  # 128 channels
        x = self.inorm2(x)
        input_after_downsample = x
        x = self.relu(x)
        """------------------------------------------------------------"""

        # Up-mid layer
        x = self.resnet_block_up(x)
        residual_up = x  # 128 channels

        """Downsample with partial convolutions, from 64x64 to 32x32"""
        x, mask = self.pconv3(x, mask)
        x = self.inorm3(x)
        x = self.relu(x)
        x = self.middle_residual_attention_blocks0(x)
        residual_middle = x

        # Lower layer
        x = self.lowest_convolution(x)
        x = self.lowest_residual_attention_blocks(x)

        x = self.gated_256(residual_middle, x)
        x = self.relu(x)

        # Middle layer1
        x = self.middle_residual_attention_blocks1(x)
        x = self.gated_128(residual_up, x)
        x = self.relu(x)

        # Up-mid layer
        x = self.resnet_block_up(x)
        x = self.gated_128(input_after_downsample, x)
        x = self.relu(x)
        x = self.upsample_convolutions(x)
        gate = torch.sigmoid(self.input_gate_conv(initial_input))
        x = gate * initial_input + (1 - gate) * x
        x = torch.tanh(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x):
        batch, channels, height, width = x.shape
        # Project input to Q, K, V
        q = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)  # [batch, height*width, channels']
        k = self.key(x).view(batch, -1, height * width)  # [batch, channels', height*width]
        v = self.value(x).view(batch, -1, height * width)  # [batch, channels, height*width]

        # Compute attention scores
        attn = torch.bmm(q, k)  # [batch, height*width, height*width]
        attn = torch.nn.functional.softmax(attn / (k.size(1) ** 0.5), dim=-1)  # Scaled softmax

        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [batch, channels, height*width]
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x  # Residual connection


class ResNetBlock(nn.Module):
    def __init__(self, channels, final_relu=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.final_relu = final_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        if self.final_relu:
            out = self.relu(out)
        return out


class AOTBlock(nn.Module):
    def __init__(self, channels, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(channels, channels // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, padding=0, dilation=1))

    def forward(self, x):
        def my_layer_norm(feat):
            mean = feat.mean((2, 3), keepdim=True)
            std = feat.std((2, 3), keepdim=True) + 1e-9
            feat = 2 * (feat - mean) / std - 1
            feat = 5 * feat
            return feat
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask



class GatedSkipConnection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, skip, decoder):
        gate = torch.sigmoid(self.gate_conv(skip))  # values between 0 and 1
        fused = gate * skip + (1 - gate) * decoder  # dynamic blend
        return fused


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # the real image convolution (e.g. 3→64 channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        # a fixed 1→1 “conv” for counting valid mask pixels
        self.register_buffer('mask_kernel',
                             torch.ones(1, 1, kernel_size, kernel_size))
        self.sum1 = kernel_size * kernel_size

    def forward(self, x, mask):
        # x:    [B, in_ch, H, W]
        # mask: [B,      1, H, W]  with values {0,1}

        # 1) COUNT how many valid (mask==1) pixels under each kernel window
        with torch.no_grad():
            valid_count = torch.nn.functional.conv2d(mask,
                                   self.mask_kernel,
                                   stride=self.conv.stride,
                                   padding=self.conv.padding,
                                   dilation=self.conv.dilation)
            # renormalize factor = (k*k)/(#valid + eps)
            mask_ratio = (self.sum1 / (valid_count + 1e-8)) * (valid_count > 0).float()

        # 2) APPLY conv only on valid pixels
        x_masked = x * mask
        y = self.conv(x_masked)

        # 3) RENORMALIZE and restore bias in empty regions
        if self.conv.bias is not None:
            b = self.conv.bias.view(1, -1, 1, 1)
            y = (y - b) * mask_ratio + b
        else:
            y = y * mask_ratio

        # 4) UPDATE mask: a position is valid if at least one pixel was valid
        new_mask = (valid_count > 0).float()
        return y, new_mask


class CRFillModule(nn.Module):
    def __init__(self, channels, patch_size=3, stride=1, softmax_scale=10.):
        super().__init__()
        self.ps = patch_size
        self.scale = softmax_scale
        self.stride = stride

    def forward(self, feat, mask):
        stride = self.stride
        """
        feat: [B, C, H, W]   bottleneck features  
        mask: [B, 1, H, W]   1=known, 0=hole  
        returns:
          recon:  [B, C, H, W]  reconstructed features in hole
          cr_loss: scalar L2 loss against feat
        """
        B,C,H,W = feat.shape
        # 1) unfold into overlapping patches
        patches = feat.unfold(2, self.ps, stride).unfold(3, self.ps, stride)
        B,C,Ho,Wo,_,_ = patches.shape
        patches = patches.contiguous().view(B, C, Ho*Wo, -1)               # [B,C,Nbg,ps*ps]
        patches_norm = F.normalize(patches, dim=3)                         # unit-normalize

        # 2) query = hole-only patches
        feat_q = feat * (1 - mask)                                         # zero out known
        q_patches = feat_q.unfold(2, self.ps, stride).unfold(3, self.ps, stride)
        q_patches = q_patches.contiguous().view(B, C, Ho*Wo, -1)           # [B,C,Nq,ps*ps]
        q_norm = F.normalize(q_patches, dim=3)

        # 3) similarity & softmax
        # reshape to [B, Nq, C, K] and [B, Nbg, C, K]
        qn = q_norm.permute(0,2,1,3)    # [B,Nq,C,K]
        pn = patches_norm.permute(0,2,1,3) # [B,Nbg,C,K]
        sim = torch.einsum('bnck,bmck->bnm', qn, pn) * self.scale  # [B,Nq,Nbg]
        attn = F.softmax(sim, dim=-1)

        # 4) reconstruct
        recon_p = torch.einsum('bnm,bmck->bnck', attn, pn)  # [B,Nq,C,K]
        recon_p = recon_p.permute(0,2,1,3).view(B,C,Ho,Wo,self.ps,self.ps)
        # fold patches back to [B,C,H,W] (same as unfold-reverse)
        recon = torch.zeros_like(feat)
        # (you can implement a fold or simply use overlap-add with proper strides here)

        # 5) CR loss only in hole
        cr_loss = F.mse_loss(recon * (1-mask), feat * (1-mask))
        return recon, cr_loss