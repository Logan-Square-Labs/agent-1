"""V-JEPA pretraining entrypoint.

Example:
    uv run python scripts/training/train_video_enc.py \\
        --config agent_1/configs/video_encoder/vjepa.yaml \\
        --shards "data/shards/{000000..000099}.tar" \\
        --val-shards "data/shards/{000100..000109}.tar" \\
        --wandb-run-name vjepa-tiny-run-01
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import yaml
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from agent_1.data.dataset import make_dataset, s3_worker_init
from agent_1.models.vjepa.mask import MaskCollator
from agent_1.models.vjepa.vjepa import VJEPA
from agent_1.trainers.vjepa_trainer import LitVJEPA

from dotenv import load_dotenv

load_dotenv()

def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_ns(v) for v in d]
    return d


def build_model(cfg) -> VJEPA:
    return VJEPA(
        patch_dim=tuple(cfg.patch.patch_dim),
        in_channels=cfg.data.channels,
        grid_size=tuple(cfg.patch.grid_size),
        encoder_dim=cfg.encoder.dim,
        encoder_intermediate_size=cfg.encoder.intermediate_size,
        encoder_num_heads=cfg.encoder.num_heads,
        encoder_head_dim=cfg.encoder.head_dim,
        encoder_num_layers=cfg.encoder.num_layers,
        encoder_dim_partitions=tuple(cfg.encoder.dim_partitions),
        predictor_dim=cfg.predictor.dim,
        predictor_intermediate_size=cfg.predictor.intermediate_size,
        predictor_num_heads=cfg.predictor.num_heads,
        predictor_head_dim=cfg.predictor.head_dim,
        predictor_num_layers=cfg.predictor.num_layers,
        predictor_dim_partitions=tuple(cfg.predictor.dim_partitions),
    )


def build_lit_config(cfg) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_lr=cfg.optim.encoder_lr,
        predictor_lr=cfg.optim.predictor_lr,
        use_lr_schedule=cfg.optim.use_lr_schedule,
        training_steps=cfg.train.max_steps,
        compile=cfg.train.compile,
        loss_exp=cfg.optim.loss_exp,
        ema_start=cfg.ema.start,
        ema_end=cfg.ema.end,
    )


def _prep_video(sample):
    v = sample["video"].float() / 255.0
    return {**sample, "video": rearrange(v, "t c h w -> c t h w")}


def build_dataloader(cfg, shards: str, *, train: bool) -> DataLoader:
    dataset = make_dataset(
        shards,
        shuffle_buffer=5000 if train else 0,
        shardshuffle=train,
    ).map(_prep_video)
    mask_configs = [
        {
            "grid_size": tuple(cfg.patch.grid_size),
            "mask_area_ratio": m.mask_area_ratio,
            "mask_ar_range": tuple(m.mask_ar_range),
            "num_sub_masks": m.num_sub_masks,
        }
        for m in cfg.mask
    ]
    uses_s3 = shards.startswith("s3://")
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=MaskCollator(mask_configs),
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=s3_worker_init if uses_s3 else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--shards", required=True, type=str, help="WebDataset shard URLs (brace expansion supported)")
    p.add_argument("--val-shards", default=None, type=str)
    p.add_argument("--ckpt-dir", default=Path("checkpoints"), type=Path)
    p.add_argument("--wandb-project", default="agent-1")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--devices", default=1, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)

    raw_cfg = yaml.safe_load(args.config.read_text())
    cfg = to_ns(raw_cfg)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    lit_model = LitVJEPA(model, build_lit_config(cfg))

    train_loader = build_dataloader(cfg, args.shards, train=True)
    val_loader = (
        build_dataloader(cfg, args.val_shards, train=False)
        if args.val_shards else None
    )

    logger = (
        None if args.no_wandb else
        WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(args.ckpt_dir),
        )
    )
    if logger is not None:
        logger.log_hyperparams(raw_cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            save_last=True,
            every_n_train_steps=cfg.train.ckpt_every_n_steps,
            save_top_k=-1,
        ),
    ]

    trainer = L.Trainer(
        max_steps=cfg.train.max_steps,
        precision=cfg.train.precision,
        accelerator="auto",
        devices=args.devices,
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval if args.val_shards else None,
        callbacks=callbacks,
        default_root_dir=str(args.ckpt_dir),
    )
    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
