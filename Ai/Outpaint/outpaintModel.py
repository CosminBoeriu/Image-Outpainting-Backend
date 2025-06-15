import torch
import cv2 as cv
import torch.nn as nn

class AOTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'AOTGenerator'
        self.relu = nn.ReLU()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.downsample_convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
        )
        self.resnet_block_up = nn.Sequential(
            ResNetBlock(channels=128),
        )
        self.middle_residual_attention_blocks0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(),
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            SelfAttention(channels=256),
            ResNetBlock(channels=256),
        )
        self.lower_residual_attention_blocks = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.ReLU(),
            AOTBlock(channels=512, rates=[1, 2, 4, 8]),
            SelfAttention(channels=512),
            AOTBlock(channels=512, rates=[1, 2, 4, 8]),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        )
        self.middle_residual_attention_blocks1 = nn.Sequential(
            AOTBlock(channels=256, rates=[1, 2, 4, 8]),
            SelfAttention(channels=256),
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
        # First part
        input = x
        x = self.downsample_convolutions(x)
        input_after_downsample = x
        x = self.relu(x)

        # Up-mid layer
        x = self.resnet_block_up(x)
        residual_up = x

        # Middle layer0
        x = self.middle_residual_attention_blocks0(x)
        residual_middle = x

        # Lower layer
        x = self.lower_residual_attention_blocks(x)
        x = x + residual_middle
        x = self.relu(x)

        # Middle layer1
        x = self.middle_residual_attention_blocks1(x)
        x = x + residual_up
        x = self.relu(x)

        # Up-mid layer
        x = self.resnet_block_up(x)
        x = x + input_after_downsample
        x = self.relu(x)
        x = self.upsample_convolutions(x)
        x = x + self.gamma * input
        x = nn.Tanh()(x)
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


class AOTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = 'full'
        self._number_of_convolution_layers = 5
        self._channels = [3, 64, 128, 256, 512, 1]
        layers = []
        for i in range(self._number_of_convolution_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=self._channels[i], out_channels=self._channels[i + 1], kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2d(self._channels[i + 1]),
                nn.LeakyReLU(),
            ))

        self.model = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x