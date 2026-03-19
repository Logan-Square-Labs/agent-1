"""Downloads Game Boy longplay videos from archive.org URLs in data/longplays.json.

Usage:
    uv run python scripts/data/download_longplays.py /path/to/output/dir
"""

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

LONGPLAYS_JSON = (
    Path(__file__).resolve().parents[2] / "data" / "longplays.json"
)
MAX_CONCURRENT = 3


def _filename_from_url(url: str) -> str:
    return unquote(urlparse(url).path.rsplit("/", 1)[-1])


async def _download(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        if dest.exists():
            print(f"[skip] {dest.name}")
            return

        print(f"[downloading] {dest.name}")
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            async with client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(partial, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                        f.write(chunk)

            size_mb = partial.stat().st_size / 1e6
            partial.rename(dest)
            print(f"[done] {dest.name} ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"[error] {dest.name}: {exc}")
            partial.unlink(missing_ok=True)


async def main():
    parser = argparse.ArgumentParser(
        description="Download Game Boy longplay videos from archive.org"
    )
    parser.add_argument("target_dir", type=Path, help="directory to save downloads to")
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    with open(LONGPLAYS_JSON) as f:
        longplays = json.load(f)

    urls = [url for entry in longplays for url in entry.get("downloads", [])]
    if not urls:
        print("No download URLs found in longplays.json")
        return

    print(f"Downloading {len(urls)} file(s) to {args.target_dir}\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _download(client, url, args.target_dir / _filename_from_url(url), sem)
            for url in urls
        ]
        await asyncio.gather(*tasks)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    asyncio.run(main())
