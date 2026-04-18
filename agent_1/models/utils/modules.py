from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_dim: tuple, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_dim = patch_dim # (H, W)/(T, H, W)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        if len(patch_dim) == 3:
            self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_dim,
                stride=patch_dim,
            )
        elif len(patch_dim) == 2:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_dim,
                stride=patch_dim,
            )
        else:
            raise ValueError(f"Invalid patch dimension: {patch_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            return rearrange(self.conv(x), "b c t h w -> b (t h w) c")
        elif len(x.shape) == 4:
            return rearrange(self.conv(x), "b c h w -> b (h w) c")
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)

def build_rope_cache(
    dim: int,
    positions: torch.Tensor,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()

class RoPE(nn.Module):
    """Axial RoPE for 1D sequences and 2D/3D grids.

    dim_partitions specifies the subspace dims to partition head_dim into.
    e.g. (16, 24, 24) for a 3D grid with head_dim=64.
    """

    def __init__(
        self,
        head_dim: int,
        grid_size: tuple[int, ...],
        dim_partitions: tuple[int, ...],
        theta: float = 10000.0,
    ):
        super().__init__()
        assert len(dim_partitions) == len(grid_size), "need one partition per axis"
        assert sum(dim_partitions) == head_dim, f"partitions must sum to head_dim ({head_dim})"
        assert all(d % 2 == 0 for d in dim_partitions), "each partition must be even"

        coords = torch.meshgrid(
            *[torch.arange(s, dtype=torch.float32) for s in grid_size],
            indexing="ij",
        )

        cos_parts, sin_parts = [], []
        for i, p in enumerate(dim_partitions):
            cos_i, sin_i = build_rope_cache(p, coords[i].flatten(), theta)
            cos_parts.append(cos_i)
            sin_parts.append(sin_i)

        self.register_buffer("cos", torch.cat(cos_parts, dim=-1))
        self.register_buffer("sin", torch.cat(sin_parts, dim=-1))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos[: q.shape[-2]], self.sin[: q.shape[-2]]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rope: RoPE = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope = rope
        embed_dim = num_heads * head_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # todo: add positions to handle masked values from original grid
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> b three h n d", three=3, h=self.num_heads
        ).unbind(dim=1)
        
        q, k = norm(q), norm(k)
        
        if self.rope is not None:
            q, k = self.rope(q, k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.proj(rearrange(x, "b h n d -> b n (h d)"))
        return x


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ReLU2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # TODO: implement ReLU^2 MLP
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: pass


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        rope: RoPE = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.attn = Attention(num_heads, head_dim)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        patch_dim: tuple,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        grid_size: tuple[int, ...],
        dim_partitions: tuple[int, ...],
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_dim=patch_dim,
            in_channels=hidden_size,
            embed_dim=hidden_size,
        )
        self.rope = RoPE(
            head_dim=head_dim,
            grid_size=grid_size,
            dim_partitions=dim_partitions,
            theta=rope_theta,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                head_dim=head_dim,
                rope=self.rope
            ) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patch_embed(x)
        if mask is not None:
            x = x * mask
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x