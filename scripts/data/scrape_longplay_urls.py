"""Scrapes longplays.org for Game Boy longplay archive.org download links
and updates data/longplays.json with them.

Usage:
    uv run python scripts/data/longplays_pipeline.py
"""

import asyncio
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

GAMEBOY_CATEGORY_URL = (
    "https://longplays.org/infusions/longplays/longplays.php?cat_id=30"
)
LONGPLAYS_JSON = Path(__file__).resolve().parents[2] / "data" / "longplays.json"

MAX_CONCURRENT_SCRAPES = 3
MAX_CONCURRENT_RESOLVES = 3


def _normalize(name: str) -> str:
    """Strip a game name down to lowercase ASCII alphanumerics for fuzzy matching."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^\w\s]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _match_games(
    wanted: list[dict], catalog: list[dict]
) -> list[tuple[int, dict] | tuple[int, None]]:
    """Match each entry in `wanted` to a catalog entry scraped from the site.

    Returns a list parallel to `wanted`: each element is (index, catalog_entry)
    or (index, None) if no match was found.
    """
    by_norm = {_normalize(entry["name"]): entry for entry in catalog}

    results: list[tuple[int, dict] | tuple[int, None]] = []
    for i, game in enumerate(wanted):
        norm = _normalize(game["game"])
        if norm in by_norm:
            results.append((i, by_norm[norm]))
            continue

        # Fallback: substring containment (handles minor naming differences)
        match = next(
            (entry for key, entry in by_norm.items() if norm in key or key in norm),
            None,
        )
        results.append((i, match))

    return results


async def _scrape_catalog(page) -> list[dict]:
    """Scrape the Game Boy longplays listing page for all entries."""
    await page.goto(GAMEBOY_CATEGORY_URL)
    await page.wait_for_selector("tbody")
    html = await page.inner_html("tbody")

    soup = BeautifulSoup(html, "html.parser")
    return [
        {"name": a.text.strip(), "url": urljoin(GAMEBOY_CATEGORY_URL, a["href"])}
        for a in soup.find_all("a", href=lambda h: h and "longplay_id=" in h)
    ]


async def _scrape_file_ids(longplay_url: str, browser) -> list[str]:
    """Scrape a longplay detail page for its file_id download links."""
    page = await browser.new_page()
    try:
        await page.goto(longplay_url, timeout=60_000)
        await page.wait_for_selector("table.tblDetail", timeout=30_000)
        html = await page.content()
    finally:
        await page.close()

    soup = BeautifulSoup(html, "html.parser")
    return [
        urljoin(longplay_url, a["href"])
        for a in soup.find_all("a", href=lambda h: h and "file_id=" in h)
    ]


async def _resolve_archive_url(file_url: str, browser) -> str | None:
    """Follow a file_id URL and intercept the redirect to capture the archive.org link."""
    ctx = await browser.new_context()
    page = await ctx.new_page()
    resolved: dict[str, str | None] = {"url": None}

    async def on_download(download):
        resolved["url"] = download.url
        await download.cancel()

    page.on("download", on_download)
    try:
        await page.goto(file_url, timeout=60_000)
    except Exception:
        pass
    finally:
        try:
            await ctx.close()
        except Exception:
            pass

    url = resolved["url"]
    if url and url.startswith("http"):
        return url
    return None


def _save(longplays: list[dict]) -> None:
    with open(LONGPLAYS_JSON, "w", encoding="utf-8") as f:
        json.dump(longplays, f, indent=4, ensure_ascii=False)


async def scrape_downloads(longplays: list[dict]) -> list[dict]:
    """Populate each entry in `longplays` with a ``downloads`` list of
    archive.org URLs.  Entries that already have downloads are skipped.
    Progress is saved after every game so partial runs are resumable.

    Returns the mutated list.
    """
    async with async_playwright() as pw:
        browser = await pw.firefox.launch(headless=True)

        print("Scraping longplays.org catalog...")
        listing_page = await browser.new_page()
        catalog = await _scrape_catalog(listing_page)
        await listing_page.close()
        print(f"  Found {len(catalog)} entries in catalog")

        matches = _match_games(longplays, catalog)
        matched = [(i, entry) for i, entry in matches if entry is not None]
        unmatched = [longplays[i]["game"] for i, entry in matches if entry is None]

        if unmatched:
            print(f"  Could not match: {unmatched}")
        print(f"  Matched {len(matched)}/{len(longplays)} games\n")

        resolve_sem = asyncio.Semaphore(MAX_CONCURRENT_RESOLVES)

        for idx, catalog_entry in matched:
            game_name = longplays[idx]["game"]
            if longplays[idx].get("downloads"):
                print(f"[skip] {game_name} — already has downloads")
                continue

            try:
                await asyncio.sleep(random.uniform(1, 3))
                print(f"[scrape] {game_name}")
                file_urls = await _scrape_file_ids(catalog_entry["url"], browser)

                if not file_urls:
                    print(f"  ↳ no download links found")
                    longplays[idx]["downloads"] = []
                else:
                    async def resolve_one(url: str) -> str | None:
                        async with resolve_sem:
                            await asyncio.sleep(random.uniform(0.5, 2))
                            return await _resolve_archive_url(url, browser)

                    archive_urls = await asyncio.gather(
                        *[resolve_one(u) for u in file_urls]
                    )
                    downloads = [u for u in archive_urls if u]
                    longplays[idx]["downloads"] = downloads
                    print(
                        f"  ↳ resolved {len(downloads)}/{len(file_urls)} download link(s)"
                    )

                _save(longplays)
            except Exception as exc:
                print(f"  ↳ ERROR processing {game_name}: {exc}")

        await browser.close()

    return longplays


async def main():
    with open(LONGPLAYS_JSON) as f:
        longplays = json.load(f)
    print(f"Loaded {len(longplays)} games from {LONGPLAYS_JSON.name}\n")

    await scrape_downloads(longplays)
    print(f"\nDone. Results saved to {LONGPLAYS_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
