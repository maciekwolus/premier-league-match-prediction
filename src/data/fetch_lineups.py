"""Download lineups and per-player match data from Understat.

Two stages, both cached to disk so an interrupted run resumes where it stopped:

1. ``matches``  one request per season -> ``raw/lineups/matches_{season}.json``
   The season fixture list, which carries Understat's own match ids.
2. ``rosters``  one request per match  -> ``raw/lineups/rosters/{match}.json``
   Who played, their position, minutes, xG, xA and cards.

Stage 2 is ~2,660 requests, so it is slow by design: a delay between requests keeps
this polite towards a small free site. Run it in the background.

Usage:
    python -m src.data.fetch_lineups --stage matches
    python -m src.data.fetch_lineups --stage rosters
    python -m src.data.fetch_lineups --season "2025/26" --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING

from src.config import MATCHES_PER_SEASON, RAW_LINEUPS_DIR, SEASONS, SEASONS_BY_LABEL, Season

if TYPE_CHECKING:  # the annotations below need the name, not the package at runtime
    from understatapi import UnderstatClient

ROSTER_DIR = RAW_LINEUPS_DIR / "rosters"

DEFAULT_DELAY_SECONDS = 0.5


def season_matches_path(season: Season):
    return RAW_LINEUPS_DIR / f"matches_{season.slug}.json"


def roster_path(understat_match_id: str):
    return ROSTER_DIR / f"{understat_match_id}.json"


def load_season_matches(season: Season) -> list[dict]:
    """Read a cached season fixture list."""
    path = season_matches_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.fetch_lineups --stage matches"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_season_matches(
    client: UnderstatClient, season: Season, force: bool = False
) -> tuple[int, str]:
    """Download one season's fixture list. Returns (match count, message)."""
    path = season_matches_path(season)

    if path.exists() and not force:
        matches = json.loads(path.read_text(encoding="utf-8"))
        return len(matches), f"{season.label}  cached, {len(matches)} matches"

    matches = client.league(league="EPL").get_match_data(season=season.understat)

    RAW_LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matches, indent=1), encoding="utf-8")

    warning = "" if len(matches) == MATCHES_PER_SEASON else f"  <-- expected {MATCHES_PER_SEASON}"
    return len(matches), f"{season.label}  downloaded, {len(matches)} matches{warning}"


def fetch_rosters(
    client: UnderstatClient,
    season: Season,
    force: bool = False,
    delay: float = DEFAULT_DELAY_SECONDS,
) -> tuple[int, int, list[str]]:
    """Download every roster for one season.

    Returns (downloaded, cached, failed match ids). Individual failures are collected
    rather than raised, so one bad match does not discard the whole season's work.
    """
    matches = load_season_matches(season)
    ROSTER_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = cached = 0
    failures: list[str] = []

    for index, match in enumerate(matches, start=1):
        match_id = match["id"]
        path = roster_path(match_id)

        if path.exists() and not force:
            cached += 1
            continue

        try:
            roster = client.match(match=match_id).get_roster_data()
        except Exception as exc:  # noqa: BLE001 - one bad match must not stop the run
            failures.append(f"{match_id}: {type(exc).__name__} {exc}")
            continue

        path.write_text(json.dumps(roster, indent=1), encoding="utf-8")
        downloaded += 1

        if downloaded % 25 == 0:
            print(f"  {season.label}  {index}/{len(matches)}", flush=True)

        time.sleep(delay)

    return downloaded, cached, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("matches", "rosters", "all"),
        default="all",
        help="which stage to run (default: all)",
    )
    parser.add_argument(
        "--season",
        action="append",
        metavar="LABEL",
        help='season to fetch, e.g. "2025/26" (repeatable; default: all)',
    )
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"seconds between roster requests (default: {DEFAULT_DELAY_SECONDS})",
    )
    args = parser.parse_args(argv)

    if args.season:
        try:
            seasons = [SEASONS_BY_LABEL[label] for label in args.season]
        except KeyError as exc:
            known = ", ".join(s.label for s in SEASONS)
            parser.error(f"unknown season {exc}. Known seasons: {known}")
    else:
        seasons = list(SEASONS)

    all_failures: list[str] = []

    # Imported here rather than at module scope. understatapi pins its transitive
    # dependencies exactly - urllib3==1.26.5, idna==2.10 - and clean_lineups imports this
    # module for two helpers, so a module-level import dragged those pins into anything
    # that touched a lineup, including the report and the test suite. Only this function
    # actually talks to Understat.
    from understatapi import UnderstatClient

    with UnderstatClient() as client:
        if args.stage in ("matches", "all"):
            print("Season fixture lists")
            for season in seasons:
                _, message = fetch_season_matches(client, season, force=args.force)
                print(f"  {message}", flush=True)

        if args.stage in ("rosters", "all"):
            print("\nRosters")
            for season in seasons:
                downloaded, cached, failures = fetch_rosters(
                    client, season, force=args.force, delay=args.delay
                )
                all_failures.extend(failures)
                note = f"  {len(failures)} FAILED" if failures else ""
                print(
                    f"  {season.label}  {downloaded} downloaded, {cached} cached{note}",
                    flush=True,
                )

    if all_failures:
        print(f"\n{len(all_failures)} roster(s) failed:", file=sys.stderr)
        for failure in all_failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        print("Re-run to retry - successful downloads are cached.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
