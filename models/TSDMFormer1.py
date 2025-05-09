"""
RhythmFormer:Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer
"""
from typing import Optional
import torch
from torch import nn
import math
from typing import Tuple, Union
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import trunc_normal_
from models.base.time_bra import time_BiFormerBlock
from thop import profile

class SEBlock(nn.Module):
    def __init__(self, in_c, reduction=4):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # 输出大小为 [B, C, 1, 1]
            nn.Conv2d(in_c, in_c // reduction, 1), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // reduction, in_c, 1), # 升维
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class ECABlock(nn.Module):
    def __init__(self, in_c, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)        # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1)  # [B, 1, C]
        y = self.conv(y)                    # [B, 1, C]
        y = self.sigmoid(y).permute(0, 2, 1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y


class DWConv(nn.Module):
    def __init__(self, in_c, out_c, dropout=0., attention='se'):
        super(DWConv, self).__init__()

        self.dw_conv = nn.Conv2d(in_c*2, in_c*2, kernel_size=5, stride=2, padding=2, groups=in_c * 2)
        self.pw_conv = nn.Conv2d(in_c*2, out_c, kernel_size=1)

        # 注意力模块选择
        if attention == 'se':
            self.attn = SEBlock(out_c)
        elif attention == 'eca':
            self.attn = ECABlock(out_c)
        else:
            self.attn = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.attn(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x= self.pool(x)
        return x

class TSDM_Module(nn.Module):
    def __init__(self, in_c=3, out_c=3, mode='WTS'):
        """
        mode: 'TSDM' or 'TSDMW'
        """
        super(TSDM_Module, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mode = mode
        self.stem = DWConv(in_c, out_c, 0.1)

    def compute_diff(self, x):
        N, C, D, H, W = x.shape
        if self.mode == 'CTS':
            x1 = torch.cat([x[:, :, :1, :, :], x[:, :, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, :, 1:, :, :], x[:, :, D-1:, :, :]], dim=2)
        elif self.mode == 'WTS':
            x1 = torch.cat([x[:, :, D-1:, :, :], x[:, :, :D-1, :, :]], dim=2)
            x3 = torch.cat([x[:, :, 1:, :, :], x[:, :, :1, :, :]], dim=2)
        elif self.mode == 'PTS':
            zero_pad1 = torch.zeros_like(x[:, :, :1, :, :])  # 生成全 0 张量
            zero_pad2 = torch.zeros_like(x[:, :, :1, :, :])  # 确保和 x3 形状匹配
            x1 = torch.cat([zero_pad1, x[:, :, :D-1, :, :]], dim=2)  # 用 0 进行填充
            x3 = torch.cat([x[:, :, 1:, :, :], zero_pad2], dim=2)
        return torch.cat([x - x1, x3 - x], dim=1)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x_diff = self.compute_diff(x)
        x = self.stem(x_diff.transpose(1, 2).contiguous().view(N*D, 6, H, W))
        return x

class TPT_Block(nn.Module):
    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=3):
        super().__init__()
        self.dim = dim
        self.depth = depth
        ############ downsample layers & upsample layers #####################
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch, 2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(dim),
                nn.ELU(),
            )
            self.upsample_layers.append(upsample_layer)
        ######################################################################
        self.blocks = nn.ModuleList([
            time_BiFormerBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                num_heads=num_heads,
                t_patch=t_patch,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_dwconv=side_dwconv,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """Definition of TPT_Block.
        Args:
          x [N,D,C,H,W]
        Returns:
          x [N,D,C,H,W]
        """

        for i in range(self.layer_n):
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = blk(x)
        for i in range(self.layer_n):
            x = self.upsample_layers[i](x)

        return x


class TSDMFormer1(nn.Module):

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            dim: int = 96, frame: int = 160,
            image_size: Optional[int] = (160, 128, 128),
            in_chans=24, head_dim=16,
            stage_n=3,
            embed_dim=[96, 96, 96], mlp_ratios=[2, 2, 2],
            depth=[2, 2, 2],
            t_patchs: Union[int, Tuple[int]] = (2, 4, 8), 
            topks: Union[int, Tuple[int]] = (20, 20, 20),
            side_dwconv: int = 3,
            drop_path_rate=0.,
            use_checkpoint_stages=[],
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.stage_n = stage_n

        # self.Fusion_Stem = Fusion_Stem()
        self.TSDM = TSDM_Module(3,  24, 'WTS')
        self.patch_embedding = nn.Sequential(
                    nn.Conv3d(in_chans, embed_dim[0] // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
                    nn.BatchNorm3d(embed_dim[0] // 2),
                    nn.ReLU(),
                    nn.AvgPool3d((1, 4, 4)),
                    nn.Conv3d(embed_dim[0] // 2, embed_dim[0], kernel_size=(1, 3, 3), stride=(1, 1, 1)),
                    nn.BatchNorm3d(embed_dim[0]),
                    nn.ReLU(),
                    nn.AvgPool3d((1, 4, 4))
        )
        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1, stride=1, padding=0)

        ##########################################################################
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in embed_dim]  # [4, 4, 4]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(stage_n):
            stage = TPT_Block(dim=embed_dim[i],
                              depth=depth[i],
                              num_heads=nheads[i],
                              mlp_ratio=mlp_ratios[i],
                              drop_path=dp_rates[sum(depth[:i]):sum(depth[:i + 1])],
                              t_patch=t_patchs[i], topk=topks[i], side_dwconv=side_dwconv
                              )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
        ##########################################################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, C, D, H, W = x.shape  # [N D 3 128 128]
        # x = x.transpose(1, 2)
        x = self.TSDM(x)
        x = x.view(N, D, 24, H // 4, W // 4).permute(0, 2, 1, 3, 4)  # [N 64 D 32 32]
        x = self.patch_embedding(x)  # [N 64 D 8 8]
        
        x = torch.mean(x, 3)  # [N, 64, D, 8]
        x = torch.mean(x, 3)  # [N, 64, D]

        for i in range(3):
            x = self.stages[i](x)  


        rPPG = self.ConvBlockLast(x)  # [N, 1, D]
        rPPG = rPPG.squeeze(1)
        return rPPG


if __name__ == '__main__':
    input = torch.randn(1, 3, 160, 128, 128) # (1, 3, 160, 128, 128))
    net = TSDMFormer1()
    flops, params = profile(net, (input,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    # out = net(input)
    # print(out.shape)