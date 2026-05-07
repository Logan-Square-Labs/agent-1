"""V-JEPA hyperparameter sweep entrypoint (W&B sweep agent).

Example:
    # Create sweep and launch a local agent:
    uv run python scripts/training/sweep_video_enc.py \\
        --config agent_1/configs/video_encoder/vjepa.yaml \\
        --sweep-config agent_1/configs/video_encoder/vjepa_sweep.yaml \\
        --shards "data/shards/{000000..000099}.tar" \\
        --val-shards "data/shards/{000100..000109}.tar" \\
        --count 20

    # Resume an existing sweep:
    uv run python scripts/training/sweep_video_enc.py \\
        --sweep-id <sweep-id> \\
        --config agent_1/configs/video_encoder/vjepa.yaml \\
        --shards "data/shards/{000000..000099}.tar" \\
        --count 10

Sweep config parameters use dot-path notation to address nested config keys.
List indices are expressed as integers in the path (e.g. mask.0.mask_area_ratio).
"""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import wandb
import yaml

from dotenv import load_dotenv

from scripts.training.train_video_enc import train

load_dotenv()


def _deep_set(d, path: str, value) -> None:
    """Set a nested value in d using dot-path notation, e.g. 'mask.0.mask_area_ratio'."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d[int(key)] if isinstance(d, list) else d[key]
    final = keys[-1]
    if isinstance(d, list):
        d[int(final)] = value
    else:
        d[final] = value


def _make_agent_fn(base_cfg: dict, run_args):
    def _run():
        with wandb.init() as run:
            cfg = copy.deepcopy(base_cfg)
            for path, value in dict(wandb.config).items():
                _deep_set(cfg, path, value)
            args = copy.copy(run_args)
            args.wandb_run_name = run.name
            train(cfg, args, wandb_run=run)
    return _run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--sweep-config", default=None, type=Path, help="W&B sweep spec YAML; required unless --sweep-id is given")
    p.add_argument("--sweep-id", default=None, type=str, help="Resume an existing sweep instead of creating one")
    p.add_argument("--shards", required=True, type=str)
    p.add_argument("--val-shards", default=None, type=str)
    p.add_argument("--ckpt-dir", default=Path("checkpoints"), type=Path)
    p.add_argument("--wandb-project", default="agent-1")
    p.add_argument("--devices", default=1, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--count", default=None, type=int, help="Max trials for this agent (None = run until sweep ends)")
    args = p.parse_args()

    if args.sweep_id is None and args.sweep_config is None:
        p.error("one of --sweep-config or --sweep-id is required")

    base_cfg = yaml.safe_load(args.config.read_text())

    run_args = SimpleNamespace(
        shards=args.shards,
        val_shards=args.val_shards,
        ckpt_dir=args.ckpt_dir,
        devices=args.devices,
        seed=args.seed,
        no_wandb=False,
        wandb_project=args.wandb_project,
        wandb_run_name=None,  # overwritten per trial by agent fn
    )

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_cfg = yaml.safe_load(args.sweep_config.read_text())
        sweep_id = wandb.sweep(sweep_cfg, project=args.wandb_project)

    wandb.agent(sweep_id, function=_make_agent_fn(base_cfg, run_args), count=args.count)


if __name__ == "__main__":
    main()
