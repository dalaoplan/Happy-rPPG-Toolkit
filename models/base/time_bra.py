"""
Adapted from here: https://github.com/rayleizhu/BiFormer
"""
from typing import List, Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
import math
from timm.layers import DropPath

from models.base.temporal import temporal_regional_attention

class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, C, N = x.shape
        x = self.fc1(x).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        # x = x.to(torch.float32)
        x = x.real

        x = self.fc2(x.transpose(1, 2))
        return x

class Time_bra(nn.Module):

    def __init__(self, dim, num_heads=8, t_patch=2, qk_scale=None, topk=40,  side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 
        self.topk = topk
        self.t_patch = t_patch  # frame of patch
        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv1d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        ##########################################
        self.output_linear = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.proj_q = nn.Sequential(
            nn.Conv1d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm1d(dim),
        )
        self.proj_k = nn.Sequential(
            nn.Conv1d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm1d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv1d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        if attn_backend == 'torch':
            self.attn_fn = temporal_regional_attention
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x:Tensor):

        N, C, T = x.size()
        region_size = max(4 // self.t_patch , 1)

        # STEP 1: linear projection
        # q,k,v shape is [b, c, t]
        q , k , v = self.proj_q(x) , self.proj_k(x) ,self.proj_v(x)

        # STEP 2: pre attention
        # q_r,k_r shape is [b, c, t//2]
        q_r = F.avg_pool1d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool1d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) 
        
        # a_r shape is [b, t//2, t//2]
        a_r = q_r.transpose(-1, -2) @ k_r 
        # idx_r shape is [b, t//2, k]
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) 
        # dix_r -> [b, n_heads, t//2, k]
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        # STEP 3: refined attention
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        routing_graph=idx_r, region_size=region_size)
        
        output = output + self.lepe(v) # nct
        output = self.output_linear(output) # nct

        return output

class time_BiFormerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., num_heads=4, t_patch=2, qk_scale=None, topk=40, mlp_ratio=2, side_dwconv=3):
        super().__init__()
        self.t_patch = t_patch
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = Time_bra(dim=dim, num_heads=num_heads, t_patch=t_patch,qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        self.norm2 = nn.BatchNorm1d(dim)
        self.mlp = Frequencydomain_FFN(dim=dim, mlp_ratio=mlp_ratio)

        # self.mlp = nn.Sequential(nn.Conv1d(dim, int(mlp_ratio*dim), kernel_size=1),
        #                          nn.BatchNorm1d(int(mlp_ratio*dim)),
        #                          nn.GELU(),
        #                          nn.Conv1d(int(mlp_ratio*dim),  int(mlp_ratio*dim), 3, stride=1, padding=1),  
        #                          nn.BatchNorm1d(int(mlp_ratio*dim)),
        #                          nn.GELU(),
        #                          nn.Conv1d(int(mlp_ratio*dim), dim, kernel_size=1),
        #                          nn.BatchNorm1d(dim),
        #                         )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

if __name__ == '__main__':
    model = video_BiFormerBlock(dim=64)
    x = torch.randn(2, 64, 160)  # B, C, T, H, W
    out = model(x)
    print("Output shape:", out.shape)


