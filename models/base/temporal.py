"""
Adapted from here: https://github.com/rayleizhu/BiFormer
"""
import torch
from torch import Tensor, LongTensor , nn
import torch.nn.functional as F
from typing import Optional, Tuple
            
def _grid2seq_time(x: Tensor, region_size: int, num_heads: int):
    """
    x: (B, C, T)
    return: (B, num_heads, n_region, region_size, head_dim)
    """
    B, C, T = x.shape
    assert T % region_size == 0, 'T must be divisible by region_size.'
    n_region = T // region_size
    x = x.view(B, num_heads, C // num_heads, n_region, region_size)
    x = x.permute(0, 1, 3, 4, 2)  # (bs, num_heads, n_region, region_size, head_dim)
    return x, n_region

    
def _seq2grid_time(x: Tensor, region_size: int):
    """
    x: (B, num_heads, n_region, region_size, head_dim)
    return: (B, C, T)
    """
    B, num_heads, n_region, region_size, head_dim = x.shape
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    x = x.view(B, num_heads * head_dim, n_region * region_size)
    return x

def temporal_regional_attention(
        query: Tensor, key: Tensor, value: Tensor, scale: float, 
        routing_graph:LongTensor, region_size: int, kv_region_size:Optional[Tuple[int]]=None,):
    
    
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = routing_graph.shape

    """
    query, key, value: (B, C, T)
    return: (B, C, T)
    """

    # q,k,v [b,  nhead, n_region, region_size, head_dim]
    q, n_region = _grid2seq_time(query, region_size, nhead)
    k, _ = _grid2seq_time(key, region_size, nhead)
    v, _ = _grid2seq_time(value, region_size, nhead)

    # Step 2: routing_graph 是 [B, nhead, n_region, topk]，gather 的维度是 region_index（dim=2）
    # 先扩展为 [B, nhead, n_region, topk, region_size, head_dim]
    # key/v shape: [B, nhead, kv_nregion, region_size, head_dim]
    B, nhead, kv_nregion, kv_region_size, head_dim = k.shape
    _, _, q_nregion, topk = routing_graph.shape

    broadcasted_routing_graph = routing_graph.view(B, nhead, q_nregion, topk, 1, 1).expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_gather = torch.gather(
        k.unsqueeze(2).expand(-1, -1, q_nregion, -1, -1, -1),  # [B, nhead, q_nregion, kv_nregion, kv_region_size, d]
        dim=3,
        index=broadcasted_routing_graph
    )  # → [B, nhead, q_nregion, topk, kv_region_size, d]

    value_gather = torch.gather(
        v.unsqueeze(2).expand(-1, -1, q_nregion, -1, -1, -1),
        dim=3,
        index=broadcasted_routing_graph
    )  # [B, nhead, q_nregion, topk, kv_region_size, d]

    # Attention计算
    # [b, nhead, n_region, resion_size, head_dim] * [b, nhead, n_region, head_dim, topk*resion_size]
    # [b, nhead, n_region, region_size, topk*resion_size]
    attn = (q * scale) @ key_gather.flatten(-3, -2).transpose(-1, -2)  
    attn = torch.softmax(attn, dim=-1)

    # [b, nhead, n_region, region_size, topk*resion_size] * [b, nhead, n_region, topk*region_size, head_dim]
    # out -> # [b, nhead, n_region, region_size, head_dim]
    out = attn @ value_gather.flatten(-3, -2)  

    # out -> [b, c, t]
    out = _seq2grid_time(out, region_size)
    return out, attn
    



