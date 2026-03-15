"""Package preprocessed mp4 longplays into globally-shuffled WebDataset shards.

Each shard sample contains a 16-frame mp4 clip and a metadata JSON file.
Completed shards are uploaded to an R2/S3 bucket and removed locally.

Usage:
    uv run python scripts/data/package_videos.py /path/to/video/dir /path/to/shards
"""

import io
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import boto3
import click
import webdataset as wds
from tqdm import tqdm


def _encode_clip(
    frames: list[av.VideoFrame], width: int, height: int, fps: int
) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.pix_fmt = "gray"
        stream.width = width
        stream.height = height
        for i, frame in enumerate(frames):
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def _extract_clips(
    video_path: Path, clip_frames: int, clip_dir: str
) -> list[dict]:
    """Extract clips from a video, write each to disk, return metadata entries."""
    entries: list[dict] = []
    frames: list[av.VideoFrame] = []
    clip_num = 0
    video_name = video_path.stem

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = int(stream.average_rate)
        w, h = stream.width, stream.height

        for frame in container.decode(stream):
            frames.append(frame)
            if len(frames) == clip_frames:
                clip_path = os.path.join(
                    clip_dir, f"{video_name}_{clip_num:06d}.mp4"
                )
                mp4_bytes = _encode_clip(frames, w, h, fps)
                with open(clip_path, "wb") as f:
                    f.write(mp4_bytes)
                entries.append({
                    "path": clip_path,
                    "video_name": video_name,
                    "clip_number": clip_num,
                })
                frames = []
                clip_num += 1

    return entries


def _make_upload_callback(
    s3_client: boto3.client, bucket: str, prefix: str
):
    def _upload_and_cleanup(shard_path: str) -> None:
        key = prefix + Path(shard_path).name
        s3_client.upload_file(shard_path, bucket, key)
        Path(shard_path).unlink()
        click.echo(f"[uploaded] {key}")

    return _upload_and_cleanup


@click.command()
@click.argument("video_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("shard_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--clip-frames", default=16, show_default=True, help="Frames per clip.")
@click.option("--samples-per-shard", default=1000, show_default=True, help="Samples per shard.")
@click.option("--max-workers", default=4, show_default=True, help="Parallel video extraction workers.")
@click.option("--seed", default=42, show_default=True, help="Random seed for shuffle.")
@click.option("--r2-bucket", default="datasets", show_default=True, help="R2 bucket name.")
@click.option("--r2-prefix", default="vision_encoder/DMG/longplays/", show_default=True, help="Key prefix in the bucket.")
@click.option("--r2-endpoint-url", envvar="R2_ENDPOINT_URL", required=True, help="R2 endpoint URL (or R2_ENDPOINT_URL env var).")
@click.option("--r2-access-key-id", envvar="R2_ACCESS_KEY_ID", required=True, help="R2 access key ID (or R2_ACCESS_KEY_ID env var).")
@click.option("--r2-secret-access-key", envvar="R2_SECRET_ACCESS_KEY", required=True, help="R2 secret access key (or R2_SECRET_ACCESS_KEY env var).")
def main(
    video_dir: Path,
    shard_dir: Path,
    clip_frames: int,
    samples_per_shard: int,
    max_workers: int,
    seed: int,
    r2_bucket: str,
    r2_prefix: str,
    r2_endpoint_url: str,
    r2_access_key_id: str,
    r2_secret_access_key: str,
) -> None:
    """Package preprocessed mp4s into globally-shuffled WebDataset shards."""
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise click.ClickException(f"No mp4 files found in {video_dir}")

    shard_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="clips_") as clip_dir:
        # -- Phase 1: extract clips to disk ------------------------------------
        clip_index: list[dict] = []

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_extract_clips, vp, clip_frames, clip_dir): vp
                for vp in videos
            }
            with tqdm(as_completed(futures), total=len(futures),
                      desc="Extracting clips", unit="video") as pbar:
                for future in pbar:
                    entries = future.result()
                    clip_index.extend(entries)
                    if entries:
                        pbar.set_postfix(
                            last=entries[0]["video_name"], clips=len(clip_index)
                        )

        # -- Phase 2: shuffle index, stream clips into shards ------------------
        random.seed(seed)
        random.shuffle(clip_index)

        s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
        )
        post_cb = _make_upload_callback(s3, r2_bucket, r2_prefix)

        pattern = str(shard_dir / "%06d.tar")
        with wds.ShardWriter(pattern, maxcount=samples_per_shard, post=post_cb) as sink:
            for idx, entry in tqdm(
                enumerate(clip_index), total=len(clip_index),
                desc="Writing shards", unit="clip",
            ):
                with open(entry["path"], "rb") as f:
                    mp4_bytes = f.read()
                meta = {
                    "source_video": entry["video_name"],
                    "clip_number": entry["clip_number"],
                    "start_frame": entry["clip_number"] * clip_frames,
                    "end_frame": (entry["clip_number"] + 1) * clip_frames,
                }
                sink.write({
                    "__key__": f"{idx:08d}",
                    "mp4": mp4_bytes,
                    "json": json.dumps(meta).encode(),
                })


if __name__ == "__main__":
    main()
