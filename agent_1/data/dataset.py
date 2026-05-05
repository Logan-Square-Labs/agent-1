import io
import json
import os
from typing import Literal
from urllib.parse import urlparse

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


def _process_sample(sample):
    video = _decode_video(sample["mp4"])
    meta = json.loads(sample["json"])
    return {"video": video, **meta}


# Module-level boto3 client, created lazily once per process.
# Using a module-level (not closure) variable makes _gopen_s3 picklable,
# which is required when DataLoader spawns workers rather than forking.
_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
    return _s3_client


def _gopen_s3(url, mode="rb", bufsize=8192, **kw):
    import botocore.exceptions
    parsed = urlparse(url)
    try:
        return _get_s3_client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise FileNotFoundError(f"{url}: key not found") from None
        raise OSError(f"{url}: S3 error {code}") from None


def _ensure_s3_handler() -> None:
    if "s3" not in wds.gopen_schemes:
        wds.gopen_schemes["s3"] = _gopen_s3


def s3_worker_init(worker_id: int) -> None:
    """DataLoader worker_init_fn that re-registers the S3 gopen handler.

    Spawned workers start with a fresh module state and lose any handler
    registered in the main process, so each worker must re-register.
    """
    _ensure_s3_handler()


def _maybe_register_s3_handler(urls: str | list[str]) -> None:
    urls_list = [urls] if isinstance(urls, str) else urls
    if any(u.startswith("s3://") for u in urls_list):
        _ensure_s3_handler()


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
    _maybe_register_s3_handler(urls)
    dataset = wds.WebDataset(urls, shardshuffle=shardshuffle)

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(_process_sample)

    return dataset
