"""Clean the raw football-data.co.uk CSVs into a single tidy match table.

Reads ``data/raw/matches/E0_*.csv`` and writes ``data/processed/matches.parquet``:
one row per match, snake_case columns, a stable ``match_id``, and every season stacked
together.

Raw files carry 106-132 columns depending on the season (football-data keeps adding
bookmakers). We take only the stable subset, which is present in all seasons.

Usage:
    python -m src.data.clean_matches
"""

from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from src.config import (
    MATCHES_PER_SEASON,
    PROCESSED_DIR,
    RAW_MATCHES_DIR,
    SEASONS,
    Season,
)

MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"

DATE_FORMAT = "%d/%m/%Y"

# Raw column -> our name. Anything not listed is dropped.
COLUMN_MAP = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",
    # full time - the prediction targets
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    # half time
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    # match statistics - POST-match, only ever usable as rolling form over
    # *previous* matches. See the leakage rule in CLAUDE.md.
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Bet365 opening odds
    "B365H": "odds_open_home",
    "B365D": "odds_open_draw",
    "B365A": "odds_open_away",
    # Bet365 closing odds - sharper, they absorb team news
    "B365CH": "odds_close_home",
    "B365CD": "odds_close_draw",
    "B365CA": "odds_close_away",
    # market average closing odds - the benchmark to beat
    "AvgCH": "odds_close_avg_home",
    "AvgCD": "odds_close_avg_draw",
    "AvgCA": "odds_close_avg_away",
}

COUNT_COLUMNS = [
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
]

ODDS_COLUMNS = [name for name in COLUMN_MAP.values() if name.startswith("odds_")]


def slugify_team(name: str) -> str:
    """Team name to a filename- and id-safe token.

    "Nott'm Forest" -> "nottm_forest",  "Man United" -> "man_united"
    """
    normalised = unicodedata.normalize("NFKD", name)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    # Drop apostrophes rather than turning them into separators, so "Nott'm Forest"
    # becomes "nottm_forest" and not "nott_m_forest".
    without_apostrophes = re.sub(r"['’]", "", ascii_only.lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", without_apostrophes)
    return cleaned.strip("_")


def build_match_id(season: Season, date: pd.Timestamp, home: str, away: str) -> str:
    """Stable identifier that Phase 2 can join Understat lineups onto."""
    return f"{season.slug}_{date:%Y%m%d}_{slugify_team(home)}_{slugify_team(away)}"


def load_season(season: Season) -> pd.DataFrame:
    """Read and tidy one season's raw CSV."""
    path = RAW_MATCHES_DIR / f"E0_{season.slug}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.fetch_matches")

    raw = pd.read_csv(path)

    missing = [column for column in COLUMN_MAP if column not in raw.columns]
    if missing:
        raise ValueError(f"{season.label}: raw CSV is missing columns {missing}")

    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    df["date"] = pd.to_datetime(raw["Date"], format=DATE_FORMAT)
    # Time is occasionally blank in older files; midnight is a harmless placeholder
    # because nothing downstream needs kickoff time to the minute.
    df["kickoff"] = pd.to_datetime(
        raw["Date"] + " " + raw["Time"].fillna("00:00"),
        format=f"{DATE_FORMAT} %H:%M",
        errors="coerce",
    )

    df["season"] = season.label
    df["season_slug"] = season.slug

    df["match_id"] = [
        build_match_id(season, date, home, away)
        for date, home, away in zip(df["date"], df["home_team"], df["away_team"], strict=True)
    ]

    for column in COUNT_COLUMNS:
        df[column] = df[column].astype("Int64")

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_difference"] = df["home_goals"] - df["away_goals"]

    ordered = (
        ["match_id", "season", "season_slug", "date", "kickoff", "home_team", "away_team"]
        + ["home_goals", "away_goals", "result", "total_goals", "goal_difference"]
        + [
            c
            for c in df.columns
            if c
            not in (
                "match_id",
                "season",
                "season_slug",
                "date",
                "kickoff",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
                "total_goals",
                "goal_difference",
            )
        ]
    )
    return df[ordered].sort_values("date").reset_index(drop=True)


def validate_season(df: pd.DataFrame, season: Season) -> list[str]:
    """Return a list of problems found in one season. Empty list means clean."""
    problems: list[str] = []

    if len(df) != MATCHES_PER_SEASON:
        problems.append(f"{len(df)} matches, expected {MATCHES_PER_SEASON}")

    teams = set(df["home_team"]) | set(df["away_team"])
    if len(teams) != 20:
        problems.append(f"{len(teams)} distinct teams, expected 20")

    home_counts = df["home_team"].value_counts()
    away_counts = df["away_team"].value_counts()
    for team in sorted(teams):
        home, away = home_counts.get(team, 0), away_counts.get(team, 0)
        if home != 19 or away != 19:
            problems.append(f"{team}: {home} home / {away} away, expected 19/19")

    if df["match_id"].duplicated().any():
        duplicates = df.loc[df["match_id"].duplicated(), "match_id"].tolist()
        problems.append(f"duplicate match_id: {duplicates}")

    for column in ("home_goals", "away_goals", "result", "date"):
        nulls = df[column].isna().sum()
        if nulls:
            problems.append(f"{nulls} null values in {column}")

    # A result letter that disagrees with the goals means the source file is corrupt.
    # Rows with a missing score are skipped - they are already reported as nulls above,
    # and comparing against NA would raise rather than record a problem.
    scored = df[df["home_goals"].notna() & df["away_goals"].notna()]
    if not scored.empty:
        home = scored["home_goals"].astype(int)
        away = scored["away_goals"].astype(int)
        expected = np.where(home > away, "H", np.where(home < away, "A", "D"))
        mismatches = int((expected != scored["result"]).sum())
        if mismatches:
            problems.append(f"{mismatches} rows where result disagrees with the score")

    return problems


def build(strict: bool = True) -> pd.DataFrame:
    """Load, validate and concatenate every season."""
    frames = []
    all_problems: dict[str, list[str]] = {}

    for season in SEASONS:
        df = load_season(season)
        problems = validate_season(df, season)
        if problems:
            all_problems[season.label] = problems
        odds_coverage = 100 * (1 - df[ODDS_COLUMNS].isna().mean().mean())
        print(
            f"{season.label}  {len(df)} matches  "
            f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}  "
            f"odds {odds_coverage:.0f}% complete" + ("  PROBLEMS" if problems else "")
        )
        frames.append(df)

    if all_problems:
        lines = [
            f"  {season}: {problem}"
            for season, problems in all_problems.items()
            for problem in problems
        ]
        message = "Validation failed:\n" + "\n".join(lines)
        if strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    combined = pd.concat(frames, ignore_index=True).sort_values("date")
    return combined.reset_index(drop=True)


def main() -> int:
    matches = build()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(MATCHES_PARQUET, index=False)

    print(f"\n{len(matches)} matches, {len(matches.columns)} columns")
    print(f"written to {MATCHES_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
