import torch
import torch.nn as nn
import math
from thop import profile
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, expension = 2):
        super(ConvBlock, self).__init__()
        mid_c = in_c * expension
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.dwconv = nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=1, padding=1, groups=mid_c, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))  # 1×1 卷积
        x = F.relu6(self.bn2(self.dwconv(x)))  # 深度可分离卷积
        x = self.bn3(self.conv2(x))  # 1×1 卷积恢复通道数
        return x

class TSDM_Module(nn.Module):
    def __init__(self, in_c=3, out_c=3, shift_method='WTS', use_residual=True):
        """
        shift_method: 'TSDM' or 'TSDMW'
        """
        super(TSDM_Module, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.shift_method = shift_method
        self.use_residual = use_residual
        self.conv = ConvBlock(in_c, out_c)

    def compute_diff(self, x):
        N, C, D, H, W = x.shape
        if self.shift_method == 'WTS':
            x1 = torch.cat([x[:, :C//2, :1, :, :], x[:, :C//2, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, C//2:, 1:, :, :], x[:, C//2:, D-1:, :, :]], dim=2)
        elif self.shift_method == 'CTS':
            x1 = torch.cat([x[:, :C//2, D-1:, :, :], x[:, :C//2, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, C//2:, 1:, :, :], x[:, C//2:, :1, :, :]], dim=2)
        elif self.shift_method == 'PTS':
            zero_pad1 = torch.zeros_like(x[:, :C // 2, :1, :, :])  # 生成全 0 张量
            zero_pad2 = torch.zeros_like(x[:, C // 2:, :1, :, :])  # 确保和 x3 形状匹配
            x1 = torch.cat([zero_pad1, x[:, :C // 2, :D - 1, :, :]], dim=2)  # 用 0 进行填充
            x3 = torch.cat([x[:, C//2:, 1:, :, :], zero_pad2], dim=2)
        return torch.cat([x[:, :C//2, :, :, :] - x1, x3 - x[:, C//2:, :, :, :]], dim=1)

    def forward(self, x):
        N, C, D, H, W = x.shape
        identity = x  # 残差连接

        x = self.compute_diff(x) # 计算时间差分

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, D, C, H, W)
        x = x.view(N * D, C, H, W)  # 合并 N 和 D

        x = self.conv(x)      #
        x = x.view(N, D, C, H, W)  # 还原 N 维度
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, D, H, W)

        if self.use_residual:
            x += identity

        return x


class MT_Block(nn.Module):
    def __init__(self, layer_n, drop = 0.2):
        super().__init__()
        self.layer_n = layer_n
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm3d(64)

        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(64),
                nn.Conv3d(64 , 64 , kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(64 , 64 , (3, 1, 1), stride=1, padding=(1, 0, 0)),
                nn.BatchNorm3d(64),
                nn.ELU(),
                )
            self.upsample_layers.append(upsample_layer)

            block = nn.Sequential(
                # TSDM_Module(64, 64, shift_method='WTS'),
                nn.Conv3d(64, 64, kernel_size=1),
                nn.BatchNorm3d(64),
                nn.GELU(),
                nn.Conv3d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.GELU(),
                nn.Conv3d(64, 64, kernel_size=1),
                nn.BatchNorm3d(64),
            )
            self.blocks.append(block)

    def forward(self, x):
        for i in range(self.layer_n):
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = x + self.drop(self.norm(blk(x)))
        for i in range(self.layer_n):
            x = self.upsample_layers[i](x)

        return x


class PhysNetback1(nn.Module):
    def __init__(self, length = 160, drop = 0.2):
        super().__init__()
        self.layer_n = 3
        # self.TSDM = TSDM_Module(shift_method='WTS')
        self.stages = nn.ModuleList()
        self.drop = nn.Dropout(drop)

        self.Predictor = nn.Conv1d(64, 1, kernel_size=1,stride=1, padding=0)

        self.embedding1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.Dropout(drop),
            nn.ReLU(inplace=True),
            TSDM_Module(16, 16, shift_method='WTS'),
        )

        self.embedding2 = nn.Sequential(
            nn.Conv3d(16, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.Dropout(drop),
            nn.ReLU(inplace=True),
            TSDM_Module(64, 64, shift_method='WTS'),

        )

        self.AvepoolSpa1 = nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4))
        self.AvepoolSpa2 = nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4))

        for i in range(self.layer_n):
            stage = MT_Block(i)
            self.stages.append(stage)


    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.embedding1(x)
        x = self.AvepoolSpa1(x)
        x = self.embedding2(x)
        x = self.AvepoolSpa2(x)

        for i in range(3):
            x = self.stages[i](x)   #[N 64 D 8 8]


        # x = self.poolspa(x)
        features_last = torch.mean(x, 3)  # [N, 64, D, 8]
        features_last = torch.mean(features_last, 3)  # [N, 64, D]

        x = self.Predictor(features_last)

        return x.view(B, T)


if __name__ == '__main__':
    input = torch.randn(1, 3, 160, 128, 128)
    net = PhysNetback1()
    # out = net(input)
    # print(out.shape)
    # out = net(input)
    flops, params = profile(net, (input,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e6 / 1024, params / 1e6))


