# inspired by https://github.com/facebookresearch/jepa/blob/main/src/masks/multiblock3d.py
import math

import torch
from torch.utils.data import default_collate


class MaskCollator:
    def __init__(
        self,
        mask_configs: list[dict],
    ):
        self.mask_configs = mask_configs
        self.mask_generators = [
            MaskGenerator(
                grid_size=config["grid_size"],
                mask_area_ratio=config["mask_area_ratio"],
                mask_ar_range=config["mask_ar_range"],
                num_sub_masks=config["num_sub_masks"],
            ) for config in mask_configs
        ]

    def __call__(self, batch: list[dict]) -> dict:
        collated = default_collate(batch)
        batch_size = collated["video"].shape[0]

        masks_enc, masks_pred = [], []
        for generator in self.mask_generators:
            m_enc, m_pred = generator(batch_size)
            masks_enc.append(m_enc)
            masks_pred.append(m_pred)

        return {**collated, "masks_enc": masks_enc, "masks_pred": masks_pred}


class MaskGenerator:
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        mask_area_ratio: float,
        mask_ar_range: tuple[float, float],
        num_sub_masks: int,
    ):
        self.grid_size = grid_size
        self.mask_area_ratio = mask_area_ratio
        self.mask_ar_range = mask_ar_range
        self.num_sub_masks = num_sub_masks

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        t_grid, h_grid, w_grid = self.grid_size
        num_patches = t_grid * h_grid * w_grid
        block_size = _sample_block_size(self.grid_size, self.mask_area_ratio, self.mask_ar_range)

        enc_indices, pred_indices = [], []
        min_keep_enc, min_keep_pred = num_patches, num_patches
        for _ in range(batch_size):
            while True:
                mask = torch.ones(self.grid_size, dtype=torch.int32)
                for _ in range(self.num_sub_masks):
                    mask *= _sample_block_mask(self.grid_size, block_size)
                mask = mask.flatten()
                m_enc = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                m_pred = torch.nonzero(mask == 0, as_tuple=False).squeeze(-1)
                if len(m_enc) > 0 and len(m_pred) > 0:
                    break

            min_keep_enc = min(min_keep_enc, len(m_enc))
            min_keep_pred = min(min_keep_pred, len(m_pred))
            enc_indices.append(m_enc)
            pred_indices.append(m_pred)

        # truncate to min kept indices for uniform sequence len across batch for collation
        enc_indices = [m[:min_keep_enc] for m in enc_indices]
        pred_indices = [m[:min_keep_pred] for m in pred_indices]
        return default_collate(enc_indices), default_collate(pred_indices)


def _sample_block_size(
    grid_size: tuple[int, int, int],
    area_ratio: float,
    ar_range: tuple[float, float],
) -> tuple[int, int, int]:
    t_grid, h_grid, w_grid = grid_size
    min_ar, max_ar = ar_range
    aspect_ratio = min_ar + torch.rand(1).item() * (max_ar - min_ar)

    spatial_keep = area_ratio * h_grid * w_grid
    h = min(h_grid, max(1, int(round(math.sqrt(spatial_keep * aspect_ratio)))))
    w = min(w_grid, max(1, int(round(math.sqrt(spatial_keep / aspect_ratio)))))
    return (t_grid, h, w)


def _sample_block_mask(
    grid_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
) -> torch.Tensor:
    t_grid, h_grid, w_grid = grid_size
    t, h, w = block_size
    top = torch.randint(0, h_grid - h + 1, (1,)).item()
    left = torch.randint(0, w_grid - w + 1, (1,)).item()
    start = torch.randint(0, t_grid - t + 1, (1,)).item()

    mask = torch.ones(grid_size, dtype=torch.int32)
    mask[start : start + t, top : top + h, left : left + w] = 0
    return mask
