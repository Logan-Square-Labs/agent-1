import io
import json
from typing import Literal

import av
import numpy as np
import torch
import webdataset as wds
from einops import rearrange

PixelFormat = Literal["gray", "rgb"]

# Maps our public format names to PyAV format strings and channel counts.
_FORMAT_INFO: dict[PixelFormat, tuple[str, int]] = {
    "gray": ("gray", 1),
    "rgb": ("rgb24", 3),
}


def _decode_video(mp4_bytes: bytes, pixel_format: PixelFormat = "gray") -> torch.Tensor:
    """Decode mp4 bytes into a uint8 (T, C, H, W) tensor.

    PyAV converts from the source pixel format, so this works regardless
    of how the video was encoded. Callers handle normalization.
    """
    av_fmt, channels = _FORMAT_INFO[pixel_format]
    with av.open(io.BytesIO(mp4_bytes)) as container:
        frames = [
            frame.to_ndarray(format=av_fmt)
            for frame in container.decode(video=0)
        ]
    video = torch.from_numpy(np.stack(frames))
    if channels == 1:
        return rearrange(video, "t h w -> t 1 h w")
    return rearrange(video, "t h w c -> t c h w")


def make_dataset(
    urls: str | list[str],
    *,
    shuffle_buffer: int = 5000,
    shardshuffle: bool = True,
) -> wds.WebDataset:
    """Create an IterableDataset that streams video clips from WebDataset shards.

    Args:
        urls: WebDataset-compatible shard URL(s). Supports local paths,
              brace expansion ("shards/{000000..000099}.tar"), and
              http/s3/pipe URLs for remote storage.
        shuffle_buffer: Number of samples to buffer for shuffling.
            Set to 0 to disable sample-level shuffling.
        shardshuffle: Whether to shuffle shard order each epoch.
            Should be True for training, False for deterministic eval.

    Returns:
        A torch IterableDataset yielding dicts with:
            "video": float32 tensor of shape (T, 1, H, W)
            "source_video": str
            "clip_number": int
            "start_frame": int
            "end_frame": int
    """

    def process_sample(sample):
        video = _decode_video(sample["mp4"])
        meta = json.loads(sample["json"])
        return {"video": video, **meta}

    dataset = wds.WebDataset(urls, shardshuffle=shardshuffle)

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(process_sample)

    return dataset
