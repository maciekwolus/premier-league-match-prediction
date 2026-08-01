"""Download the Fantasy Premier League feed.

The Premier League's own Fantasy service publishes the JSON its website runs on. It is
free, needs no authentication, covers all twenty clubs and updates before every gameweek,
which makes it the only live source in this project that is not a file someone places by
hand.

**It is undocumented, so treat a schema change as expected rather than exceptional.**
Nothing here trusts a field to exist: the cleaners validate before writing, in the same
shape as every other source.

Two endpoints matter:

``bootstrap-static/``
    Clubs, players, and per-player availability - ``status``,
    ``chance_of_playing_next_round`` and a ``news`` string.

``fixtures/``
    All 380 fixtures with kickoff times and *official* gameweek numbers, which is better
    than the derived gameweek in ``data/gameweeks.py`` for a season FPL covers. The
    derivation still earns its place for the seven historical seasons, which FPL does not
    reach.

Usage:
    python -m src.data.fetch_fpl
    python -m src.data.fetch_fpl --force
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

from src.config import RAW_FPL_DIR

BASE_URL = "https://fantasy.premierleague.com/api"

# The endpoints we cache, and the filename each lands in.
ENDPOINTS = {
    "bootstrap-static/": "bootstrap.json",
    "fixtures/": "fixtures.json",
}

TIMEOUT_SECONDS = 30

# The API rejects a request with no user agent.
HEADERS = {"User-Agent": "premier-league-match-prediction (github.com/maciekwolus)"}


def fetch(endpoint: str, force: bool = False) -> dict | list:
    """One endpoint, cached on disk. Re-downloads only when asked."""
    path = RAW_FPL_DIR / ENDPOINTS[endpoint]
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    response = requests.get(f"{BASE_URL}/{endpoint}", timeout=TIMEOUT_SECONDS, headers=HEADERS)
    response.raise_for_status()
    payload = response.json()

    RAW_FPL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    args = parser.parse_args(argv)

    for endpoint in ENDPOINTS:
        try:
            payload = fetch(endpoint, force=args.force)
        except requests.RequestException as error:
            print(f"failed to fetch {endpoint}: {error}", file=sys.stderr)
            return 1

        size = len(payload) if isinstance(payload, list) else len(payload.keys())
        print(f"{endpoint:20} -> {RAW_FPL_DIR / ENDPOINTS[endpoint]}  ({size} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
