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
                allow_overlap=config.get("allow_overlap", True),
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
        allow_overlap: bool = True,
    ):
        self.grid_size = grid_size
        self.mask_area_ratio = mask_area_ratio
        self.mask_ar_range = mask_ar_range
        self.num_sub_masks = num_sub_masks
        self.allow_overlap = allow_overlap

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = _sample_block_size(self.grid_size, self.mask_area_ratio, self.mask_ar_range)

        # sample a shared mask layout for the batch to ensure uniform patch counts
        while True:
            shared_mask = torch.ones(self.grid_size, dtype=torch.int32)
            occupied = torch.zeros(self.grid_size, dtype=torch.bool) if not self.allow_overlap else None
            for _ in range(self.num_sub_masks):
                block_mask, occupied = _sample_block_mask(self.grid_size, block_size, occupied)
                if block_mask is not None:
                    shared_mask *= block_mask
            shared_mask = shared_mask.flatten()
            m_enc = torch.nonzero(shared_mask, as_tuple=False).squeeze(-1)
            m_pred = torch.nonzero(shared_mask == 0, as_tuple=False).squeeze(-1)
            if len(m_enc) > 0 and len(m_pred) > 0:
                break

        enc_indices = m_enc.unsqueeze(0).expand(batch_size, -1)
        pred_indices = m_pred.unsqueeze(0).expand(batch_size, -1)
        return enc_indices.contiguous(), pred_indices.contiguous()


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
    occupied: torch.Tensor,
    max_attempts: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_grid, h_grid, w_grid = grid_size
    t, h, w = block_size

    # if checking for overlap, try max_attempts times to find a non-overlapping block
    for _ in range(max_attempts):
        top = torch.randint(0, h_grid - h + 1, (1,)).item()
        left = torch.randint(0, w_grid - w + 1, (1,)).item()
        start = torch.randint(0, t_grid - t + 1, (1,)).item()
        if occupied is None or not occupied[start : start + t, top : top + h, left : left + w].any():
            mask = torch.ones(grid_size, dtype=torch.int32)
            mask[start : start + t, top : top + h, left : left + w] = 0
            if occupied is not None:
                occupied = occupied.clone()
                occupied[start : start + t, top : top + h, left : left + w] = True
            return mask, occupied

    # if we can't find a non-overlapping block after max_attempts, return None
    return None, occupied
