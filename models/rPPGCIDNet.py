import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[32, 32, 64, 128],
                 heads=[1, 2, 4, 8],
                 norm=False
                 ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()

        self.I_start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= 16, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )

        self.HV_start = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 16, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )

        self.HV_spapool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.I_spapool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.HV_spapool2 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.I_spapool2 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.HV_spapool3 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.I_spapool3 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.dense1 = nn.Sequential(nn.Linear(1024, 128),
                                    nn.Tanh())
        self.dense2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [640, 3, 32, 32]
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)  # [640, 3, 32, 32]
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)  # [640, 1, 32, 32]

        i = self.I_start(i)
        hvi = self.HV_start(hvi)
        # low
        i_enc0 = self.IE_block0(i)  # [640, 1, 32, 32]
        i_enc1 = self.IE_block1(i_enc0)  # [640, 32, 16, 16]
        hv_0 = self.HVE_block0(hvi)  # [640, 2, 32, 32]
        hv_1 = self.HVE_block1(hv_0)  # [640, 32, 16, 16]
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)  # [640, 32, 16, 16]
        hv_2 = self.HV_LCA1(hv_1, i_enc1)  # [640, 32, 16, 16]
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)  # [640, 32, 16, 16]
        hv_2 = self.HVE_block2(hv_2)  # [640, 32, 16, 16]

        i_enc3 = self.I_LCA2(i_enc2, hv_2)  # [640, 72, 8, 8]
        hv_3 = self.HV_LCA2(hv_2, i_enc2)  # [640, 72, 8, 8]
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)  # [640, 144, 4, 4]
        hv_3 = self.HVE_block3(hv_2)  # [640, 144, 4, 4]

        i_enc4 = self.I_LCA3(i_enc3, hv_3)  # [640, 144, 4, 4]
        hv_4 = self.HV_LCA3(hv_3, i_enc3)  # [640, 144, 4, 4]

        i_dec4 = self.I_LCA4(i_enc4, hv_4)  # [640, 144, 4, 4]
        hv_4 = self.HV_LCA4(hv_4, i_enc4)  # [640, 144, 4, 4]

        hv_3 = self.HVD_block3(hv_4, self.HV_spapool1(hv_jump2))  # [640, 72, 8, 8]
        i_dec3 = self.ID_block3(i_dec4, self.I_spapool1(v_jump2))  # [640, 72, 8, 8]
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, self.HV_spapool2(hv_jump1))
        i_dec2 = self.ID_block2(i_dec3, self.I_spapool2(v_jump1))

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, self.I_spapool3(i_jump0))
        # i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, self.HV_spapool3(hv_jump0))
        # hv_0 = self.HVD_block0(hv_1)
        output_hvi = torch.cat([hv_1, i_dec1], dim=1)
        out = output_hvi.view(output_hvi.size(0), -1)
        out = self.dense1(out)
        out = self.drop(out)
        out = self.dense2(out)
        # output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        # output_rgb = self.trans.PHVIT(output_hvi)

        return out.view(B, T)

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi


if __name__ == '__main__':
    input = torch.rand(4, 160, 3, 128, 128).to('cuda')
    model = CIDNet().to('cuda')
    out = model(input)

    print(out.shape)