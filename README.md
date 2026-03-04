# agent-1
A replication of [FDM-1](https://si.inc/posts/fdm1/) to train an agent to play GameBoy games.

## Architecture
### Video Encoder
A family of video encoders based on [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/).
This currently produces fixed-size embedding sequences, but will be updated to produce 
variable-length embedding sequences with methods like [ToMe](https://arxiv.org/abs/2210.09461).

### Inverse Dynamics Model
A [Masked Diffusion Language Model (MDLM)](https://s-sahoo.com/mdlm/) model to unmask action tokens taken between video embeddings.

### Forward Dynamics Model
An autoregressive transformer model that takes a sequence of interleaved video embeddings and previous action tokens and predicts
the next action token.

TODOs:
- [ ] write pipeline script to:
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