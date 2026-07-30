"""Turn the FPL feed into the clubs and fixtures of the season about to start.

This is the answer to a gap the pipeline has had since it was written: ``season_clubs()``
reads participants from ``matches.parquet``, which is empty for a season nobody has played
yet. Until now the twenty clubs of an upcoming season simply could not be known.

Follows the same contract as the other cleaners - **validate before writing, and raise
rather than emit a suspect table**. The guards are the ones the project uses everywhere:
exactly 20 clubs, exactly 380 fixtures, and 19 home and 19 away for each club. A promoted
club we failed to map shows up as 19 clubs rather than as a silent gap.

Usage:
    python -m src.data.clean_fpl
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import PROCESSED_DIR, UPCOMING_SEASON
from src.data.fetch_fpl import fetch
from src.matching.team_names import fpl_to_football_data

FPL_FIXTURES_PARQUET = PROCESSED_DIR / "fpl_fixtures.parquet"

CLUBS_PER_SEASON = 20
MATCHES_PER_SEASON = 380
MATCHES_PER_CLUB_AT_HOME = 19


def club_names(bootstrap: dict | None = None) -> dict[int, str]:
    """FPL club id -> football-data club name.

    Raises on an unmapped club, because the FPL club list *is* the division: a name we
    cannot place is a promoted side nobody has added, not a foreign club to ignore.
    """
    bootstrap = bootstrap if bootstrap is not None else fetch("bootstrap-static/")
    return {team["id"]: fpl_to_football_data(team["name"]) for team in bootstrap["teams"]}


def upcoming_fixtures(bootstrap: dict | None = None, fixtures: list | None = None) -> pd.DataFrame:
    """All fixtures of the upcoming season, named as football-data names them.

    Carries FPL's ``gameweek`` because it is authoritative for this season - unlike the
    derived one, it survives a postponement without inferring anything.
    """
    bootstrap = bootstrap if bootstrap is not None else fetch("bootstrap-static/")
    fixtures = fixtures if fixtures is not None else fetch("fixtures/")
    clubs = club_names(bootstrap)

    rows = [
        {
            "season": UPCOMING_SEASON.label,
            "gameweek": fixture["event"],
            "date": pd.to_datetime(fixture["kickoff_time"], errors="coerce", utc=True),
            "home_team": clubs[fixture["team_h"]],
            "away_team": clubs[fixture["team_a"]],
        }
        for fixture in fixtures
        if fixture.get("team_h") in clubs and fixture.get("team_a") in clubs
    ]

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Kickoff times are published in UTC; the rest of the pipeline works in naive dates.
    frame["date"] = frame["date"].dt.tz_localize(None)
    return frame.sort_values(["gameweek", "date"]).reset_index(drop=True)


def validate(fixtures: pd.DataFrame, clubs: dict[int, str]) -> list[str]:
    """Problems with the schedule. An empty list means it is clean.

    Returns rather than raises so a caller can report every problem at once - the same
    shape as ``clean_matches.validate_season``.
    """
    problems = []

    if len(clubs) != CLUBS_PER_SEASON:
        problems.append(f"expected {CLUBS_PER_SEASON} clubs, got {len(clubs)}")

    distinct = set(clubs.values())
    if len(distinct) != len(clubs):
        problems.append("two FPL clubs mapped to the same football-data name")

    if len(fixtures) != MATCHES_PER_SEASON:
        problems.append(f"expected {MATCHES_PER_SEASON} fixtures, got {len(fixtures)}")

    undated = int(fixtures["date"].isna().sum())
    if undated:
        # Normal early in a season for televised rounds not yet scheduled; reported so it
        # is a known quantity rather than a surprise when a fixture has no date.
        problems.append(f"{undated} fixture(s) have no kickoff time yet")

    for side, expected in (
        ("home_team", MATCHES_PER_CLUB_AT_HOME),
        ("away_team", MATCHES_PER_CLUB_AT_HOME),
    ):
        counts = fixtures[side].value_counts()
        wrong = counts[counts != expected]
        if not wrong.empty:
            problems.append(f"{side}: {wrong.to_dict()} (expected {expected} each)")

    return problems


def build(strict: bool = True) -> pd.DataFrame:
    """Fixtures for the upcoming season, validated and written to disk."""
    bootstrap = fetch("bootstrap-static/")
    raw_fixtures = fetch("fixtures/")

    clubs = club_names(bootstrap)
    fixtures = upcoming_fixtures(bootstrap, raw_fixtures)

    problems = validate(fixtures, clubs)
    if problems and strict:
        raise ValueError("FPL fixtures failed validation:\n  " + "\n  ".join(problems))
    for problem in problems:
        print(f"  warning: {problem}", file=sys.stderr)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fixtures.to_parquet(FPL_FIXTURES_PARQUET, index=False)
    return fixtures


def main(argv: list[str] | None = None) -> int:
    fixtures = build()
    clubs = sorted(set(fixtures["home_team"]))

    print(f"{UPCOMING_SEASON.label}: {len(fixtures)} fixtures, {len(clubs)} clubs")
    print(f"  gameweeks {fixtures['gameweek'].min()}-{fixtures['gameweek'].max()}")
    print(f"  first kickoff {fixtures['date'].min()}")
    print(f"  clubs: {', '.join(clubs)}")
    print(f"\nwritten to {FPL_FIXTURES_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
