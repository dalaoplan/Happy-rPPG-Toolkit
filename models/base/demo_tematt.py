import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTopKAttention(nn.Module):
    def __init__(self, dim, num_heads=4, region_size=4, topk=3):
        super().__init__()
        self.num_heads = num_heads
        self.region_size = region_size
        self.topk = topk
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.size()
        R = self.region_size
        num_regions = T // R

        # Padding if needed
        if T % R != 0:
            pad_len = R - (T % R)
            x = F.pad(x, (0, 0, 0, pad_len))
            T = x.size(1)
            num_regions = T // R

        # Project Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        # Group into regions
        q = q.view(B, self.num_heads, num_regions, R, self.head_dim)  # (B, H, Nr, R, D)
        k = k.view(B, self.num_heads, num_regions, R, self.head_dim)
        v = v.view(B, self.num_heads, num_regions, R, self.head_dim)

        # Compute region-level features by averaging
        k_avg = k.mean(dim=3)  # (B, H, Nr, D)
        q_avg = q.mean(dim=3)  # (B, H, Nr, D)

        # Compute attention score between query region and key regions
        attn_score = torch.einsum("bhqd,bhkd->bhqk", q_avg, k_avg) / (self.head_dim ** 0.5)  # (B, H, Nr, Nr)

        # Select top-k for each query region
        topk_val, topk_idx = torch.topk(attn_score, self.topk, dim=-1)  # (B, H, Nr, topk)

        # Gather relevant key and value regions
        k_selected = self._gather_regions(k, topk_idx)  # (B, H, Nr, topk, R, D)
        v_selected = self._gather_regions(v, topk_idx)

        # Expand q to match shape
        q = q.unsqueeze(3)  # (B, H, Nr, 1, R, D)

        # Compute attention within region pairs
        attn = torch.einsum("bhnqrd,bhnkrd->bhnqrk", q, k_selected) / (self.head_dim ** 0.5)  # (B, H, Nr, R, topk*R)
        attn = attn.view(B, self.num_heads, num_regions, R, -1)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to value
        v_selected = v_selected.view(B, self.num_heads, num_regions, self.topk * R, self.head_dim)
        out = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_selected)  # (B, H, Nr, R, D)

        # Merge heads and regions
        out = out.reshape(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(out)
        return out

    def _gather_regions(self, x, idx):
        # x: (B, H, Nr, R, D), idx: (B, H, Nr, topk)
        B, H, Nr, R, D = x.size()
        topk = idx.size(-1)

        # Expand idx to (B, H, Nr, topk, R, D)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, R, D)

        # Expand x to (B, H, 1, Nr, R, D)
        x_exp = x.unsqueeze(2).expand(-1, -1, Nr, -1, -1, -1)

        selected = torch.gather(x_exp, dim=3, index=idx_exp)  # (B, H, Nr, topk, R, D)
        return selected


# Test script
if __name__ == '__main__':
    B, T, C = 2, 32, 64
    x = torch.randn(B, T, C)

    attn = TemporalTopKAttention(dim=C, num_heads=4, region_size=4, topk=3)
    y = attn(x)
    print("Input:", x.shape)
    print("Output:", y.shape)  # Should be (B, T, C)
