from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat


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
        return rearrange(self.conv(x), "b c ... -> b (...) c")


def apply_masks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather tokens by per-sample index. x: [B, N, D], indices: [B, K] -> [B, K, D]."""
    return torch.gather(x, 1, repeat(indices, "b k -> b k d", d=x.size(-1)))


def grid_positions(
    indices: torch.Tensor,
    grid_size: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """Decompose flat indices in [0, prod(grid_size)) into per-axis coordinates."""
    coords = []
    for i, axis_len in enumerate(grid_size):
        stride = 1
        for axis in grid_size[i + 1:]:
            stride *= axis
        coords.append((indices // stride) % axis_len)
    return tuple(coords)


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
    """Axial RoPE for 1D/2D/3D inputs.

    dim_partitions specifies the subspace dims to partition head_dim into.
    e.g. (16, 24, 24) for a 3D grid with head_dim=64.

    Forward accepts per-axis position tensors so masked / reordered sequences
    rotate by their original-grid coordinates rather than sequential 0..N-1.
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

        self.head_dim = head_dim
        self.grid_size = grid_size
        self.dim_partitions = dim_partitions

        for i, (axis_len, p) in enumerate(zip(grid_size, dim_partitions)):
            cos_i, sin_i = build_rope_cache(p, torch.arange(axis_len, dtype=torch.float32), theta)
            self.register_buffer(f"cos_{i}", cos_i, persistent=False)
            self.register_buffer(f"sin_{i}", sin_i, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            flat = torch.arange(q.shape[-2], device=q.device)
            positions = grid_positions(flat, self.grid_size)

        cos_parts, sin_parts = [], []
        for i in range(len(self.grid_size)):
            cos_parts.append(getattr(self, f"cos_{i}")[positions[i]])
            sin_parts.append(getattr(self, f"sin_{i}")[positions[i]])
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)

        # broadcast over the heads axis between batch and seq
        if cos.ndim == 2:
            cos = rearrange(cos, "n d -> 1 1 n d")
            sin = rearrange(sin, "n d -> 1 1 n d")
        elif cos.ndim == 3:
            cos = rearrange(cos, "b n d -> b 1 n d")
            sin = rearrange(sin, "b n d -> b 1 n d")

        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope = rope
        embed_dim = num_heads * head_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> b three h n d", three=3, h=self.num_heads
        ).unbind(dim=1)

        q, k = norm(q), norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k, positions)
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.attn = Attention(num_heads, head_dim, rope=rope)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        patch_dim: tuple,
        in_channels: int,
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
        self.grid_size = grid_size
        self.patch_embed = PatchEmbed(
            patch_dim=patch_dim,
            in_channels=in_channels,
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

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        if indices is not None:
            x = apply_masks(x, indices)
            positions = grid_positions(indices, self.grid_size)
        else:
            positions = None
        for block in self.blocks:
            x = block(x, positions)
        return self.norm(x)
