"""Join Understat lineups onto the football-data match table.

Produces two tables:

``data/processed/understat_matches.parquet``
    One row per match: our ``match_id``, Understat's own id, and match-level xG.

``data/processed/lineups.parquet``
    One row per player per match, with position, minutes, xG, xA and cards.

**The join is the whole point of this module.** Understat and football-data are separate
scrapes of the same fixtures, and a fixture that fails to match would silently vanish
from every downstream feature. Matching is therefore on
``(season, home_team, away_team)`` - unique within a season, and immune to the timezone
and date-format differences that make date joins fragile - with kickoff dates and final
scores cross-checked afterwards as independent evidence the join is correct.

Usage:
    python -m src.data.clean_lineups
"""

from __future__ import annotations

import html
import json
import sys

import pandas as pd

from src.config import MATCHES_PER_SEASON, PROCESSED_DIR, SEASONS, Season
from src.data.clean_matches import MATCHES_PARQUET
from src.data.fetch_lineups import load_season_matches, roster_path
from src.matching.team_names import understat_to_football_data

UNDERSTAT_MATCHES_PARQUET = PROCESSED_DIR / "understat_matches.parquet"
LINEUPS_PARQUET = PROCESSED_DIR / "lineups.parquet"

STARTERS_PER_TEAM = 11

# Understat roster field -> our name
PLAYER_FIELDS = {
    "player": "player",
    "player_id": "understat_player_id",
    "position": "position",
    "positionOrder": "position_order",
    "time": "minutes",
    "goals": "goals",
    "own_goals": "own_goals",
    "assists": "assists",
    "shots": "shots",
    "key_passes": "key_passes",
    "xG": "xg",
    "xA": "xa",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
    "yellow_card": "yellow_cards",
    "red_card": "red_cards",
}

NUMERIC_FIELDS = [
    "minutes",
    "goals",
    "own_goals",
    "assists",
    "shots",
    "key_passes",
    "xg",
    "xa",
    "xg_chain",
    "xg_buildup",
    "yellow_cards",
    "red_cards",
]


def load_matches() -> pd.DataFrame:
    """The football-data match table - the canonical fixture list."""
    if not MATCHES_PARQUET.exists():
        raise FileNotFoundError(
            f"{MATCHES_PARQUET} not found. Run: python -m src.data.clean_matches"
        )
    return pd.read_parquet(MATCHES_PARQUET)


def understat_season_frame(season: Season, known_teams: set[str]) -> pd.DataFrame:
    """One season of Understat fixtures, with team names translated."""
    records = []

    for match in load_season_matches(season):
        records.append(
            {
                "understat_match_id": match["id"],
                "season_slug": season.slug,
                "home_team": understat_to_football_data(match["h"]["title"], known_teams),
                "away_team": understat_to_football_data(match["a"]["title"], known_teams),
                "understat_home_goals": int(match["goals"]["h"]),
                "understat_away_goals": int(match["goals"]["a"]),
                "xg_home": float(match["xG"]["h"]),
                "xg_away": float(match["xG"]["a"]),
                "understat_datetime": pd.to_datetime(match["datetime"]),
            }
        )

    return pd.DataFrame(records)


def link_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach Understat ids and xG to our matches. Returns (linked, problems)."""
    known_teams = set(matches["home_team"]) | set(matches["away_team"])
    problems: list[str] = []

    understat = pd.concat(
        [understat_season_frame(season, known_teams) for season in SEASONS],
        ignore_index=True,
    )

    key = ["season_slug", "home_team", "away_team"]

    duplicates = understat.duplicated(key).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate fixtures in Understat data")

    linked = matches.merge(understat, on=key, how="left", validate="one_to_one")

    unmatched = linked["understat_match_id"].isna()
    if unmatched.any():
        examples = (
            linked.loc[unmatched, ["season", "date", "home_team", "away_team"]]
            .head(10)
            .to_string(index=False)
        )
        problems.append(f"{unmatched.sum()} fixtures had no Understat match:\n{examples}")
        return linked, problems

    # Independent evidence the join is right: both sources should agree on the score,
    # and on the kickoff date to within a day (Understat timestamps are UK local time,
    # so a late kickoff can land either side of midnight).
    score_mismatch = (linked["home_goals"] != linked["understat_home_goals"]) | (
        linked["away_goals"] != linked["understat_away_goals"]
    )
    if score_mismatch.any():
        examples = (
            linked.loc[
                score_mismatch,
                [
                    "match_id",
                    "home_goals",
                    "away_goals",
                    "understat_home_goals",
                    "understat_away_goals",
                ],
            ]
            .head(10)
            .to_string(index=False)
        )
        problems.append(
            f"{score_mismatch.sum()} fixtures where the two sources disagree on the "
            f"score - the join is wrong:\n{examples}"
        )

    date_gap = (linked["understat_datetime"].dt.normalize() - linked["date"]).dt.days.abs()
    far_apart = date_gap > 1
    if far_apart.any():
        problems.append(f"{far_apart.sum()} fixtures whose dates differ by more than a day")

    for season in SEASONS:
        count = (linked["season_slug"] == season.slug).sum()
        if count != MATCHES_PER_SEASON:
            problems.append(
                f"{season.label}: {count} linked matches, expected {MATCHES_PER_SEASON}"
            )

    return linked, problems


def roster_rows(understat_match_id: str, match_id: str, home: str, away: str) -> list[dict]:
    """Flatten one match's roster JSON into player rows."""
    path = roster_path(understat_match_id)
    if not path.exists():
        raise FileNotFoundError(path)

    roster = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for side, team in (("h", home), ("a", away)):
        for player in roster[side].values():
            row = {source: player[source] for source in PLAYER_FIELDS}
            row = {PLAYER_FIELDS[k]: v for k, v in row.items()}
            # Understat serves names HTML-escaped, so apostrophes arrive as "&#039;"
            # ("Dara O&#039;Shea"). Phase 4 matches on these strings.
            row["player"] = html.unescape(row["player"])
            row.update(
                {
                    "match_id": match_id,
                    "understat_match_id": understat_match_id,
                    "team": team,
                    "side": "home" if side == "h" else "away",
                    "is_starter": player["position"] != "Sub",
                }
            )
            rows.append(row)

    return rows


def build_lineups(linked: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Flatten every roster into one table. Returns (lineups, problems)."""
    rows: list[dict] = []
    missing: list[str] = []
    problems: list[str] = []

    for match in linked.itertuples():
        try:
            rows.extend(
                roster_rows(
                    match.understat_match_id, match.match_id, match.home_team, match.away_team
                )
            )
        except FileNotFoundError:
            missing.append(match.match_id)

    if missing:
        problems.append(
            f"{len(missing)} matches have no roster file, e.g. {missing[:3]}. "
            f"Run: python -m src.data.fetch_lineups --stage rosters"
        )
        return pd.DataFrame(rows), problems

    lineups = pd.DataFrame(rows)

    for column in NUMERIC_FIELDS:
        lineups[column] = pd.to_numeric(lineups[column], errors="coerce")
    lineups["position_order"] = pd.to_numeric(lineups["position_order"], errors="coerce").astype(
        "Int64"
    )

    starters = lineups[lineups["is_starter"]].groupby(["match_id", "side"]).size()
    wrong = starters[starters != STARTERS_PER_TEAM]
    if not wrong.empty:
        problems.append(
            f"{len(wrong)} team-matches without exactly {STARTERS_PER_TEAM} starters, "
            f"e.g.\n{wrong.head(5).to_string()}"
        )

    ordered = [
        "match_id",
        "understat_match_id",
        "team",
        "side",
        "player",
        "understat_player_id",
        "position",
        "position_order",
        "is_starter",
        *NUMERIC_FIELDS,
    ]
    return lineups[ordered], problems


def build(strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_matches()
    print(f"{len(matches)} matches from football-data")

    linked, problems = link_matches(matches)
    if not problems:
        print(f"{len(linked)} linked to Understat  (scores and dates agree)")

    lineups = pd.DataFrame()
    if not problems:
        lineups, problems = build_lineups(linked)
        if not problems:
            starters = lineups["is_starter"].sum()
            print(
                f"{len(lineups)} player rows, {starters} starters, "
                f"{lineups['player'].nunique()} distinct players"
            )

    if problems:
        message = "Validation failed:\n" + "\n".join(f"  {p}" for p in problems)
        if strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    understat_columns = [
        "match_id",
        "understat_match_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "xg_home",
        "xg_away",
    ]
    return linked[understat_columns], lineups


def main() -> int:
    understat_matches, lineups = build()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    understat_matches.to_parquet(UNDERSTAT_MATCHES_PARQUET, index=False)
    lineups.to_parquet(LINEUPS_PARQUET, index=False)

    print(f"\nwritten to {UNDERSTAT_MATCHES_PARQUET}")
    print(f"written to {LINEUPS_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
