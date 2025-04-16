"""This file is the official PhysFormer implementation, but set the input as diffnormalized data
   https://github.com/ZitongYu/PhysFormer

   model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from models.base.physformer_layer import Transformer_ST_TDC_gra_sharp


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.2,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            # positional_embedding: str = '1d',
            in_channels: int = 3,
            frame: int = 160,
            theta: float = 0.2,
            image_size: Optional[int] = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t // ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x, gra_sharp=2.0):

        N, C, D, H, W = x.shape
        x = self.Stem0(x)  # [N, 24, 160, 64, 64]
        x = self.Stem1(x)  # [N, 48, 160, 32, 32]
        x = self.Stem2(x)  # [N, 96, 160, 16, 16]
        x = self.patch_embedding(x)  # [N, 96, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [N, 40*4*4, C]: [N, 40*4*4, 96]

        Trans_features, Score1 = self.transformer1(x, gra_sharp)  # [N, 4*4*40, C]
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp)  # [N, 4*4*40, C]
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp)  # [N, 4*4*40, C]

        # upsampling heads
        features_last = Trans_features3.transpose(1, 2).view(N, self.dim, D // 4, 4, 4)  # [N, C, 40, 4, 4]

        features_last = self.upsample(features_last)  # x [N, C, 7*7, 80]
        features_last = self.upsample2(features_last)  # x [N, C, 7*7, 160]

        features_last = torch.mean(features_last, 3)  # x [N, C, 160, 4]
        features_last = torch.mean(features_last, 3)  # x [N, C, 160]
        rPPG = self.ConvBlockLast(features_last)  # x [N, 1, 160]

        rPPG = rPPG.squeeze(1)

        return rPPG  # , Score1, Score2, Score3


if __name__ == '__main__':
    input = torch.randn(1, 3, 300, 128, 128)
    net = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(300, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
                                           num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    # flops, params = ViT_ST_ST_Compact3_TDC_gra_sharp(net, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
    out = net(input)
    print(out.shape)