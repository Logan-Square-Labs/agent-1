"""Converts mp4 files in a directory to grayscale at 30 fps with frame averaging.

Uses the tmix filter to blend neighbouring frames before dropping to 30 fps,
so the result is temporally anti-aliased rather than just frame-dropped.

Processed files are written to a `processed/` subdirectory inside the target
directory. Originals are never modified. Writes to a .part temp file so an
interrupted conversion never leaves a corrupt output.

Usage:
    uv run python scripts/data/preprocess_videos.py /path/to/video/dir
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

MAX_CONCURRENT = 2


async def _process(src: Path, out_dir: Path, sem: asyncio.Semaphore) -> None:
    async with sem:
        dest = out_dir / src.name
        if dest.exists():
            print(f"[skip] {src.name} already processed")
            return

        partial = dest.with_suffix(".mp4.part")
        print(f"[processing] {src.name}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(src),
                "-vf", "format=gray,scale=160:144:flags=neighbor,tmix=frames=2:weights=1 1,fps=30",
                "-c:v", "libx264",
                "-pix_fmt", "gray",
                "-an",
                "-movflags", "+faststart",
                "-f", "mp4",
                "-y", str(partial),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        except Exception as exc:
            print(f"[error] {src.name}: {exc}")
            partial.unlink(missing_ok=True)
            return

        if proc.returncode != 0:
            err_tail = stderr.decode(errors="replace").strip().splitlines()[-5:]
            detail = "\n  ".join(err_tail)
            print(f"[error] {src.name}: ffmpeg exited with code {proc.returncode}\n  {detail}")
            partial.unlink(missing_ok=True)
            return

        if not partial.exists() or partial.stat().st_size == 0:
            print(f"[error] {src.name}: output file missing or empty")
            partial.unlink(missing_ok=True)
            return

        partial.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"[done] {dest.name} ({size_mb:.1f} MB)")


async def main():
    parser = argparse.ArgumentParser(
        description="Convert mp4 videos to grayscale at 30 fps with frame averaging"
    )
    parser.add_argument("target_dir", type=Path, help="directory containing mp4 files")
    args = parser.parse_args()

    if not args.target_dir.is_dir():
        sys.exit(f"Not a directory: {args.target_dir}")

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found – make sure it is installed and on PATH")

    sources = sorted(
        p for p in args.target_dir.iterdir()
        if p.suffix.lower() == ".mp4"
    )

    if not sources:
        print("No mp4 files found.")
        return

    out_dir = args.target_dir.parent / "processed"
    out_dir.mkdir(exist_ok=True)

    print(f"Processing {len(sources)} file(s) from {args.target_dir} -> {out_dir}\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    await asyncio.gather(*[_process(src, out_dir, sem) for src in sources])

    print("\nAll processing complete.")


if __name__ == "__main__":
    asyncio.run(main())
