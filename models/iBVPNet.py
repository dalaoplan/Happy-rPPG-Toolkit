"""iBVPNet - 3D Convolutional Network.
Proposed along with the iBVP Dataset, see https://doi.org/10.3390/electronics13071334

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        k_t, k_s1, k_s2 = kernel_size
        s_t, s_s1, s_s2 = stride
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (k_t, 1, 1), (s_t, 1, 1), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(in_channel),
            
            nn.Conv3d(in_channel, out_channel, (1, k_s1, k_s2), (1, s_s1, s_s2), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)

# num_filters
nf = [8, 16, 24, 40, 64]

class encoder_block(nn.Module):
    def __init__(self, in_channel, debug=False):
        super(encoder_block, self).__init__()
        # in_channel, out_channel, kernel_size, stride, padding

        self.debug = debug
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Encoder")
            print("x.shape", x.shape)
        st_x = self.spatio_temporal_encoder(x)
        if self.debug:
            print("st_x.shape", st_x.shape)
        t_x = self.temporal_encoder(st_x)
        if self.debug:
            print("t_x.shape", t_x.shape)
        return t_x


class decoder_block(nn.Module):
    def __init__(self, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug
        self.decoder_block = nn.Sequential(
            DeConvBlock3D(nf[4], nf[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeConvBlock3D(nf[3], nf[2], [7, 3, 3], [2, 2, 2], [2, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Decoder")
            print("x.shape", x.shape)
        x = self.decoder_block(x)
        if self.debug:
            print("x.shape", x.shape)
        return x



class iBVPNet(nn.Module):
    def __init__(self, frames, in_channels=3, debug=False):
        super(iBVPNet, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")

        self.ibvpnet = nn.Sequential(
            encoder_block(in_channels, debug),
            decoder_block(debug),
            # spatial adaptive pooling
            nn.AdaptiveMaxPool3d((frames, 1, 1)),
            nn.Conv3d(nf[2], 1, [1, 1, 1], stride=1, padding=0)
        )

        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape

        x = torch.diff(x, dim=2)

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("Diff Normalized shape", x.shape)

        feats = self.ibvpnet(x)
        if self.debug:
            print("feats.shape", feats.shape)
        rPPG = feats.view(-1, length)
        return rPPG
    

if __name__ == "__main__":
    import torch


    batch_size = 1
    frames = 160
    in_channels = 3
    height = 128
    width = 128
    test_data = torch.rand(batch_size, in_channels, frames, height, width)

    net = iBVPNet(in_channels=in_channels, frames=frames, debug=True)
    # print("-"*100)
    # print(net)
    # print("-"*100)
    # pred = net(test_data)
    # print(pred.shape)

    flops, params = profile(net, (test_data,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))