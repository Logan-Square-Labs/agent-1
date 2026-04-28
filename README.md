# agent-1
A GameBoy-using agent inspired by [FDM-1](https://si.inc/posts/fdm1/) and [VPT](https://arxiv.org/abs/2206.11795).

## Architecture
### Video Encoder
Video Encoder based on [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) which we can augment to use Token Merging ([ToMe](https://arxiv.org/abs/2210.09461), [Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604)) to produce compressed, variable-length embedding sequences.

#### Pretraining
Edit `agent_1/configs/video_encoder/vjepa.yaml` to set model/optim/mask hyperparameters, then run:

```
uv run python scripts/training/train_video_enc.py \
    --config agent_1/configs/video_encoder/vjepa.yaml \
    --shards "data/shards/{000000..000099}.tar" \
    --val-shards "data/shards/{000100..000109}.tar" \
    --wandb-run-name vjepa-tiny-01
```

`--shards` accepts any WebDataset URL (local path, brace-expansion glob, `s3://`, `pipe:`, etc.). Pass `--no-wandb` for local runs without logging, `--devices N` to scale across GPUs, and `--ckpt-dir <path>` to override the checkpoint location. The trainer streams from WebDataset shards, so progress is tracked in `train.max_steps` (set in the YAML) rather than epochs.

### Inverse Dynamics Model
A [Masked Diffusion Language Model (MDLM)](https://s-sahoo.com/mdlm/) model to unmask action tokens interleaved between observations.

### Forward Dynamics Model
An autoregressive transformer model that takes a sequence of video embeddings (variable length, delimited by a special `<end_of_obs>` token) interleaved with previous action tokens and predicts the next action token (delimited with `<action_start>` and `<action_end>` tokens) from a discrete action space (`Up`, `Down`, `Left`, `Right`, `A`, `B`, `Start`, `Select`, `NoOp`).

Example input sequence:
```
[video_embeddings]<end_of_obs><action_start>[action(s)]<action_end>[video_embeddings]<end_of_obs><action_start>[action(s)]<action_end>...
```
TODOs:
- [X] write pipeline script to:
  1. download longplay videos
  2. extract frames
  3. convert to grayscale
  4. resize
  5. chunk
  6. package into tar files
  7. upload to R2 bucket

- [X] Implement V-JEPA video encoder
- [ ] Implement Inverse Dynamics Model
- [ ] Implement Forward Dynamics Model