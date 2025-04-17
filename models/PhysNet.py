""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c),  # Depthwise
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0),  # Pointwise
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TSDM_Module(nn.Module):
    def __init__(self, in_c=3, out_c=3, mode='WTS'):
        """
        mode: 'TSDM' or 'TSDMW'
        """
        super(TSDM_Module, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mode = mode
        self.stem1 = ConvBlock(in_c, out_c)
        self.stem2 = ConvBlock(in_c, out_c)

    def compute_diff(self, x):
        N, C, D, H, W = x.shape
        if self.mode == 'PTS':
            x1 = torch.cat([x[:, :C//2, :1, :, :], x[:, :C//2, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, C//2:, 1:, :, :], x[:, C//2:, D-1:, :, :]], dim=2)
        elif self.mode == 'CTS':
            x1 = torch.cat([x[:, :C//2, D-1:, :, :], x[:, :C//2, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, C//2:, 1:, :, :], x[:, C//2:, :1, :, :]], dim=2)
        elif self.mode == 'WTS':
            zero_pad1 = torch.zeros_like(x[:, :C // 2, :1, :, :])  # 生成全 0 张量
            zero_pad2 = torch.zeros_like(x[:, C // 2:, :1, :, :])  # 确保和 x3 形状匹配
            x1 = torch.cat([zero_pad1, x[:, :C // 2, :D - 1, :, :]], dim=2)  # 用 0 进行填充
            x3 = torch.cat([x[:, C//2:, 1:, :, :], zero_pad2], dim=2)
        return torch.cat([x[:, :C//2, :, :, :] - x1, x3 - x[:, C//2:, :, :, :]], dim=1)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x_diff = self.compute_diff(x)
        x_diff = self.stem1(x_diff)
        x2 = self.stem2(x)
        x = x2 + x_diff
        return x #.view(N, D, self.out_c, H, W).permute(0, 2, 1, 3, 4)


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        self.frames = frames
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            TSDM_Module(3, 3),
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, self.frames)

        return rPPG         #, x_visual, x_visual3232, x_visual1616

if __name__ == '__main__':
    device = 'cuda'
    input = torch.randn((1, 3, 160, 128, 128)).to(device)
    model = PhysNet_padding_Encoder_Decoder_MAX(160).to(device)
    out = model(input)
    print(out.shape)
