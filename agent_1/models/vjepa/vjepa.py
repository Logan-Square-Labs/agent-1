import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from agent_1.models.utils.modules import (
    RoPE,
    TransformerBlock,
    ViT,
    apply_masks,
    grid_positions,
)


class Predictor(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        grid_size: tuple[int, ...],
        dim_partitions: tuple[int, ...],
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.grid_size = grid_size

        self.embed = nn.Linear(encoder_dim, predictor_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.rope = RoPE(
            head_dim=head_dim,
            grid_size=grid_size,
            dim_partitions=dim_partitions,
            theta=rope_theta,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=predictor_dim,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                head_dim=head_dim,
                rope=self.rope,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(predictor_dim)
        self.proj = nn.Linear(predictor_dim, encoder_dim, bias=False)

    def forward(
        self,
        z: torch.Tensor,
        masks_enc: torch.Tensor,
        masks_pred: torch.Tensor,
    ) -> torch.Tensor:
        B, K_pred = masks_pred.shape

        z = self.embed(z)
        targets = repeat(self.mask_token, "1 1 d -> b k d", b=B, k=K_pred)
        seq = torch.cat([z, targets], dim=1)
        positions = grid_positions(torch.cat([masks_enc, masks_pred], dim=1), self.grid_size)

        for block in self.blocks:
            seq = block(seq, positions)

        return self.proj(self.norm(seq[:, -K_pred:, :]))


class VJEPA(nn.Module):
    def __init__(
        self,
        patch_dim: tuple,
        in_channels: int,
        grid_size: tuple[int, ...],
        encoder_dim: int,
        encoder_intermediate_size: int,
        encoder_num_heads: int,
        encoder_head_dim: int,
        encoder_num_layers: int,
        encoder_dim_partitions: tuple[int, ...],
        predictor_dim: int,
        predictor_intermediate_size: int,
        predictor_num_heads: int,
        predictor_head_dim: int,
        predictor_num_layers: int,
        predictor_dim_partitions: tuple[int, ...],
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.encoder = ViT(
            patch_dim=patch_dim,
            in_channels=in_channels,
            hidden_size=encoder_dim,
            intermediate_size=encoder_intermediate_size,
            num_heads=encoder_num_heads,
            head_dim=encoder_head_dim,
            num_layers=encoder_num_layers,
            grid_size=grid_size,
            dim_partitions=encoder_dim_partitions,
            rope_theta=rope_theta,
        )
        self.predictor = Predictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            intermediate_size=predictor_intermediate_size,
            num_heads=predictor_num_heads,
            head_dim=predictor_head_dim,
            num_layers=predictor_num_layers,
            grid_size=grid_size,
            dim_partitions=predictor_dim_partitions,
            rope_theta=rope_theta,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.no_grad():
            h_full = self.target_encoder(x)
            h_full = F.rms_norm(h_full, (h_full.size(-1),))

        zs, hs = [], []
        for m_enc, m_pred in zip(masks_enc, masks_pred):
            z = self.encoder(x, m_enc)
            z = self.predictor(z, m_enc, m_pred)
            h = apply_masks(h_full, m_pred)
            zs.append(z)
            hs.append(h)
        return zs, hs

    @torch.no_grad()
    def update_target(self, m: float) -> None:
        params_q = list(self.encoder.parameters())
        params_k = list(self.target_encoder.parameters())
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1.0 - m)


def vjepa_loss(
    zs: list[torch.Tensor],
    hs: list[torch.Tensor],
    p: float = 1.0,
) -> torch.Tensor:
    return sum((z - h).abs().pow(p).mean() / p for z, h in zip(zs, hs)) / len(zs)
