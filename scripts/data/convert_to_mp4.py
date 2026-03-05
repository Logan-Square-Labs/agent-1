"""Converts non-mp4 video files in a directory to mp4 using ffmpeg, then removes the originals.

Conversion writes to a temporary file first and only replaces/removes the
original after ffmpeg exits successfully, so a failed or interrupted conversion
never causes data loss.

Usage:
    uv run python scripts/data/convert_to_mp4.py /path/to/video/dir
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mkv", ".webm", ".flv", ".mov", ".wmv", ".mpg", ".mpeg", ".ts"}
MAX_CONCURRENT = 3


async def _convert(src: Path, sem: asyncio.Semaphore) -> None:
    async with sem:
        dest = src.with_suffix(".mp4")
        if dest.exists():
            print(f"[skip] {src.name} -> {dest.name} already exists")
            return

        partial = dest.with_suffix(".mp4.part")
        print(f"[converting] {src.name}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(src),
                "-c:v", "libx264", "-c:a", "aac",
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
        src.unlink()
        size_mb = dest.stat().st_size / 1e6
        print(f"[done] {dest.name} ({size_mb:.1f} MB)")


async def main():
    parser = argparse.ArgumentParser(
        description="Convert non-mp4 videos to mp4 with ffmpeg"
    )
    parser.add_argument("target_dir", type=Path, help="directory containing videos")
    args = parser.parse_args()

    if not args.target_dir.is_dir():
        sys.exit(f"Not a directory: {args.target_dir}")

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found – make sure it is installed and on PATH")

    sources = sorted(
        p for p in args.target_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not sources:
        print("No non-mp4 video files found.")
        return

    print(f"Converting {len(sources)} file(s) in {args.target_dir}\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    await asyncio.gather(*[_convert(src, sem) for src in sources])

    print("\nAll conversions complete.")


if __name__ == "__main__":
    asyncio.run(main())
