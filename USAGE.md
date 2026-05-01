# Usage

## Video Encoder (V-JEPA)

The video encoder is pretrained with a [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) objective: the encoder sees unmasked patches and a target encoder (updated via EMA) produces latents for the masked regions; a predictor learns to reconstruct those latents from the visible context.

### Prerequisites

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Data format

Training consumes [WebDataset](https://github.com/webdataset/webdataset) `.tar` shards. Each sample in a shard must contain a `video` key holding a short MP4 clip (16 frames, 160×144 grayscale). Shards can live locally, on S3, or any URL scheme WebDataset supports (`pipe:`, `s3://`, etc.).

The `--shards` and `--val-shards` arguments accept brace-expansion globs:

```
data/shards/{000000..000099}.tar          # local
s3://my-bucket/shards/{000000..000697}.tar
pipe:gsutil cat gs://bucket/shard-{0..99}.tar
```

See `data/README.md` for how this project's shards are organised in the R2 bucket.

### Configuration

All model, masking, optimiser, and training hyperparameters live in `agent_1/configs/video_encoder/vjepa.yaml`. The sections are:

| Section | Key fields | Notes |
|---|---|---|
| `data` | `resolution`, `channels`, `num_frames`, `batch_size`, `num_workers` | Must match the clip dimensions in your shards |
| `patch` | `patch_dim`, `grid_size` | `grid_size` = `[T/patchT, H/patchH, W/patchW]`; must divide evenly |
| `encoder` | `dim`, `num_layers`, `num_heads`, `head_dim`, `intermediate_size`, `dim_partitions` | `dim_partitions` are per-axis RoPE dims for (T, H, W); must sum to `head_dim` |
| `predictor` | same fields as encoder | Lighter than the encoder by convention |
| `mask` | list of `{mask_area_ratio, mask_ar_range, num_sub_masks}` | Two masks are used: large contiguous blocks (hard) and small scattered blocks (easy) |
| `ema` | `start`, `end` | Target-encoder momentum interpolates from `start` → `end` over `train.max_steps` |
| `optim` | `encoder_lr`, `predictor_lr`, `loss_exp`, `use_lr_schedule` | `loss_exp=1.0` → L1, `loss_exp=2.0` → L2; schedule is cosine-to-zero |
| `train` | `max_steps`, `precision`, `compile`, `log_every_n_steps`, `val_check_interval`, `ckpt_every_n_steps` | Progress is in steps, not epochs, because data streams from shards |

### Running a training job

#### Minimal local run (no W&B)

```bash
uv run python scripts/training/train_video_enc.py \
    --config agent_1/configs/video_encoder/vjepa.yaml \
    --shards "data/shards/{000000..000099}.tar" \
    --no-wandb
```

#### Full run with validation and W&B logging

```bash
uv run python scripts/training/train_video_enc.py \
    --config agent_1/configs/video_encoder/vjepa.yaml \
    --shards "data/shards/{000000..0000646}.tar" \
    --val-shards "data/shards/{000647..000697}.tar" \
    --wandb-run-name vjepa-base-run-01 \
    --wandb-project agent-1 \
    --ckpt-dir checkpoints/vjepa-base-run-01
```

#### Multi-GPU (single node)

```bash
uv run python scripts/training/train_video_enc.py \
    --config agent_1/configs/video_encoder/vjepa.yaml \
    --shards "data/shards/{000000..000099}.tar" \
    --val-shards "data/shards/{000100..000109}.tar" \
    --wandb-run-name vjepa-base-run-01 \
    --devices 4
```

Lightning's `accelerator="auto"` picks CUDA/MPS/CPU automatically. `--devices` controls how many GPUs to use on one node.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--config` | _(required)_ | Path to YAML config file |
| `--shards` | _(required)_ | WebDataset shard URL pattern for training data |
| `--val-shards` | `None` | WebDataset shard URL pattern for validation data (optional) |
| `--ckpt-dir` | `checkpoints/` | Directory for saved checkpoints |
| `--wandb-project` | `agent-1` | W&B project name |
| `--wandb-run-name` | `None` | W&B run name (auto-generated if omitted) |
| `--devices` | `1` | Number of GPUs |
| `--seed` | `42` | Global random seed |
| `--no-wandb` | `False` | Disable W&B logging entirely |

### Checkpoints

Checkpoints are saved by Lightning's `ModelCheckpoint` callback to `--ckpt-dir`. Two files are written:

- `last.ckpt` — updated every `train.ckpt_every_n_steps` steps; safe to use for resuming
- step-numbered snapshots — one per `ckpt_every_n_steps` interval, kept indefinitely (`save_top_k=-1`)

To resume from a checkpoint, pass `--ckpt-path last.ckpt` (Lightning standard flag, not shown in the script's argparser — pass it after all other flags and Lightning will pick it up).

### Logged metrics

| Metric | When |
|---|---|
| `train/loss` | Every `log_every_n_steps` steps |
| `train/ema` | Every step (target-encoder momentum value) |
| `val/loss` | Every `val_check_interval` steps (only if `--val-shards` is set) |

---

## Inverse Dynamics Model (IDM)

_Not yet implemented. The IDM will be a Masked Diffusion Language Model (MDLM) that unmasks action tokens interleaved between video embeddings produced by the trained encoder._

---

## Forward Dynamics Model (FDM)

_Not yet implemented. The FDM will be an autoregressive transformer that takes a sequence of video embeddings interleaved with past action tokens and predicts the next action from a discrete action space (`Up`, `Down`, `Left`, `Right`, `A`, `B`, `Start`, `Select`, `NoOp`)._
