"""Download Premier League match results from football-data.co.uk.

One CSV per season into ``data/raw/matches/``. Files are cached: an existing file is
left alone unless ``--force`` is passed, so re-running is cheap and offline-friendly.

Usage:
    python -m src.data.fetch_matches
    python -m src.data.fetch_matches --season 2025/26 --force
"""

from __future__ import annotations

import argparse
import sys

import requests

from src.config import (
    MATCHES_PER_SEASON,
    RAW_MATCHES_DIR,
    SEASONS,
    SEASONS_BY_LABEL,
    UPCOMING_SEASON,
    Season,
)

TIMEOUT_SECONDS = 30


def raw_path(season: Season):
    """Where this season's CSV lives on disk."""
    return RAW_MATCHES_DIR / f"E0_{season.slug}.csv"


def download_season(season: Season, force: bool = False) -> tuple[bool, str]:
    """Download one season's CSV.

    Returns (downloaded, message). ``downloaded`` is False when the cached file was
    kept or the download failed.
    """
    destination = raw_path(season)

    # A finished season's file never changes, so the cache is the whole point. The season
    # being *played* changes every week, and caching it is actively harmful: before the
    # season starts football-data serves a placeholder at this address holding another
    # division's matches, and a cached copy of that would keep clean_matches skipping the
    # season for months after the real fixtures appeared.
    if destination.exists() and not force and season != UPCOMING_SEASON:
        return False, f"{season.label}  cached ({destination.name})"

    try:
        response = requests.get(season.matches_url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, f"{season.label}  FAILED: {exc}"

    # football-data serves latin-1 with a BOM; decoding here keeps the file usable by
    # anything downstream that assumes utf-8.
    text = response.content.decode("utf-8-sig", errors="replace")

    RAW_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="")

    rows = _count_data_rows(text)
    # A season still being played is *meant* to be short, so the count is reported
    # rather than flagged. Whether the file holds the right division is checked in
    # clean_matches, which is the stage that can refuse to write.
    if season == UPCOMING_SEASON:
        return True, f"{season.label}  downloaded, {rows} matches so far (in progress)"
    warning = "" if rows == MATCHES_PER_SEASON else f"  <-- expected {MATCHES_PER_SEASON}"
    return True, f"{season.label}  downloaded, {rows} matches{warning}"


def _count_data_rows(text: str) -> int:
    """Data rows in a CSV, ignoring the header and any trailing blank lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        action="append",
        metavar="LABEL",
        help='season to fetch, e.g. "2025/26" (repeatable; default: all)',
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args(argv)

    # The season being played is fetchable by name and included by default, even though
    # it is not in SEASONS. football-data updates its file within a day or two of each
    # round, and those results are what let the report score a stored prediction.
    known_seasons = {**SEASONS_BY_LABEL, UPCOMING_SEASON.label: UPCOMING_SEASON}

    if args.season:
        try:
            seasons = [known_seasons[label] for label in args.season]
        except KeyError as exc:
            known = ", ".join(known_seasons)
            parser.error(f"unknown season {exc}. Known seasons: {known}")
    else:
        seasons = [*SEASONS, UPCOMING_SEASON]

    failures = 0
    for season in seasons:
        _, message = download_season(season, force=args.force)
        if "FAILED" in message:
            failures += 1
        print(message)

    if failures:
        print(f"\n{failures} season(s) failed to download.", file=sys.stderr)
        return 1

    print(f"\nMatch data in {RAW_MATCHES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
