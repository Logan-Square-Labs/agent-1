# agent-1
A GameBoy-using agent inspired by [FDM-1](https://si.inc/posts/fdm1/) and [VPT](https://arxiv.org/abs/2206.11795).

## Architecture
### Video Encoder
Video Encoder based on [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) which we can augment to use Token Merging ([ToMe](https://arxiv.org/abs/2210.09461), [Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604)) to produce compressed, variable-length embedding sequences.

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

- [ ] Implement V-JEPA video encoder
- [ ] Implement Inverse Dynamics Model
- [ ] Implement Forward Dynamics Model