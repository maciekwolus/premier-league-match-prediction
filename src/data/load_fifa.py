"""Load FIFA / EA FC player ratings and reduce them to Premier League squads.

Reads one CSV per edition from ``data/raw/fifa/`` and writes
``data/processed/fifa_players.parquet``: one row per player per season, with club
names already translated to football-data form so Phase 4 can match players.

**These files are not downloaded automatically.** Kaggle requires an account, so the
CSVs are placed by hand - see the README. Name each file after its edition:

    data/raw/fifa/fifa20.csv   fifa21.csv   fifa22.csv   fifa23.csv
                  fc24.csv     fc25.csv     fc26.csv

**Column names vary between editions**, because the community datasets covering
FIFA 20-23 and EA FC 24-26 come from different authors. Rather than assume one layout,
every known spelling of each field is listed in ``COLUMN_ALIASES`` and matched
case-insensitively. A missing *required* field raises; a missing *optional* one is
reported and left null, so one sparse edition does not block the rest.

Usage:
    python -m src.data.load_fifa
"""

from __future__ import annotations

import sys
from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, RAW_FIFA_DIR, SEASONS, Season
from src.matching.team_names import PREMIER_LEAGUE_CLUBS_PER_SEASON, fifa_to_football_data

FIFA_PLAYERS_PARQUET = PROCESSED_DIR / "fifa_players.parquet"

# Our name -> every source spelling we have seen, lowercase.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "player_name": ("short_name", "name", "player", "player_name", "known_as"),
    "long_name": ("long_name", "full_name", "fullname"),
    "club": ("club_name", "club", "team", "team_name", "current_club"),
    "age": ("age",),
    "overall": ("overall", "overall_rating", "ovr", "rating"),
    "potential": ("potential", "pot", "potential_rating"),
    "positions": ("player_positions", "positions", "position", "best_position"),
    "value_eur": ("value_eur", "value", "market_value", "value_euro"),
    "pace": ("pace", "speed"),
    "shooting": ("shooting",),
    "passing": ("passing",),
    "dribbling": ("dribbling",),
    "defending": ("defending",),
    "physic": ("physic", "physicality", "physical"),
}

REQUIRED = ("player_name", "club", "overall")
NUMERIC = (
    "age",
    "overall",
    "potential",
    "value_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
)

# A Premier League squad is 25 registered players plus under-21s; anything far outside
# this range means the club filter or the source file is wrong.
MIN_PLAYERS_PER_CLUB = 15
MAX_PLAYERS_PER_CLUB = 60


def raw_path(season: Season):
    """Where this season's ratings CSV is expected."""
    return RAW_FIFA_DIR / f"{season.fifa_slug}.csv"


# Some Kaggle datasets ship every edition in one file, distinguished by a version
# column - the FC 24 dataset's male_players.csv covers FIFA 15 through FC 24. Accepting
# that shape directly saves splitting it by hand.
COMBINED_FILENAMES = ("male_players.csv", "players.csv", "all_players.csv")
VERSION_ALIASES = ("fifa_version", "version", "edition", "game_version", "fifa_edition")


def combined_path():
    """A multi-edition file in ``data/raw/fifa/``, if one was placed there."""
    for name in COMBINED_FILENAMES:
        candidate = RAW_FIFA_DIR / name
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def read_combined(path_str: str) -> pd.DataFrame:
    """Read the multi-edition file once, since it can run to hundreds of thousands of rows."""
    return pd.read_csv(path_str, low_memory=False)


def version_column(columns) -> str | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    for alias in VERSION_ALIASES:
        if alias in lookup:
            return lookup[alias]
    return None


def edition_rows(season: Season) -> pd.DataFrame | None:
    """This edition's rows from a combined file, or ``None`` if unavailable."""
    path = combined_path()
    if path is None:
        return None

    combined = read_combined(str(path))
    column = version_column(combined.columns)
    if column is None:
        return None

    # Versions appear as 24, "24", "FIFA 24" or "EA FC 24" depending on the author.
    versions = pd.to_numeric(
        combined[column].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    rows = combined[versions == season.fifa_version]
    return rows.reset_index(drop=True) if not rows.empty else None


def missing_editions() -> list[Season]:
    """Seasons with neither their own CSV nor rows in a combined file."""
    return [
        season
        for season in SEASONS
        if not raw_path(season).exists() and edition_rows(season) is None
    ]


def resolve_columns(columns) -> dict[str, str]:
    """Map our field names onto whatever this file happens to call them."""
    lookup = {str(column).strip().lower(): column for column in columns}
    resolved = {}

    for ours, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[ours] = lookup[alias]
                break

    return resolved


def load_edition(season: Season) -> tuple[pd.DataFrame, list[str]]:
    """Read one edition and cut it down to Premier League players.

    A per-edition file wins if present; otherwise this edition's rows are taken from a
    combined multi-edition file.
    """
    path = raw_path(season)

    if path.exists():
        raw = pd.read_csv(path, low_memory=False)
    else:
        raw = edition_rows(season)
        if raw is None:
            raise FileNotFoundError(
                f"{path} not found. Download {season.fifa_edition} from Kaggle and save "
                f"it as {path.name}, or place a multi-edition file "
                f"({' / '.join(COMBINED_FILENAMES)}) covering it - see the README."
            )

    resolved = resolve_columns(raw.columns)

    absent_required = [field for field in REQUIRED if field not in resolved]
    if absent_required:
        raise ValueError(
            f"{season.fifa_edition} ({path.name}) is missing required column(s) "
            f"{absent_required}. Columns present: {sorted(raw.columns)[:25]}... "
            f"Add the spelling used here to COLUMN_ALIASES."
        )

    df = pd.DataFrame({ours: raw[source] for ours, source in resolved.items()})

    notes = []
    for field in COLUMN_ALIASES:
        if field not in resolved:
            df[field] = pd.NA
            notes.append(field)

    df["club_fd"] = df["club"].map(fifa_to_football_data)
    premier_league = df[df["club_fd"].notna()].copy()

    premier_league["season"] = season.label
    premier_league["season_slug"] = season.slug
    premier_league["fifa_edition"] = season.fifa_edition

    for field in NUMERIC:
        premier_league[field] = pd.to_numeric(premier_league[field], errors="coerce")

    problems = []
    if notes:
        problems.append(f"{season.fifa_edition}: no column found for {sorted(notes)}")

    return premier_league, problems


def validate_edition(df: pd.DataFrame, season: Season) -> list[str]:
    """Return a list of problems with one edition. Empty list means clean."""
    problems: list[str] = []

    clubs = sorted(df["club_fd"].unique())
    if len(clubs) != PREMIER_LEAGUE_CLUBS_PER_SEASON:
        problems.append(
            f"{season.fifa_edition}: matched {len(clubs)} Premier League clubs, expected "
            f"{PREMIER_LEAGUE_CLUBS_PER_SEASON}. Matched: {clubs}. A club missing here "
            f"means its FIFA spelling is absent from FIFA_TO_FOOTBALL_DATA."
        )

    squad_sizes = df.groupby("club_fd").size()
    odd = squad_sizes[(squad_sizes < MIN_PLAYERS_PER_CLUB) | (squad_sizes > MAX_PLAYERS_PER_CLUB)]
    if not odd.empty:
        problems.append(f"{season.fifa_edition}: implausible squad sizes\n{odd.to_string()}")

    if df["player_name"].isna().any():
        problems.append(
            f"{season.fifa_edition}: {df['player_name'].isna().sum()} null player names"
        )

    ratings = df["overall"].dropna()
    if not ratings.empty and not (40 <= ratings.min() and ratings.max() <= 99):
        problems.append(
            f"{season.fifa_edition}: overall ratings outside 40-99 "
            f"(min {ratings.min()}, max {ratings.max()})"
        )

    return problems


def build(strict: bool = True) -> pd.DataFrame:
    """Load, validate and stack every edition."""
    absent = missing_editions()
    if absent:
        wanted = "\n".join(f"  {s.fifa_edition:10} -> {raw_path(s)}" for s in absent)
        combined = combined_path()
        found = (
            f"\nA combined file was found at {combined.name}, but it has no rows for "
            f"{', '.join(s.fifa_edition for s in absent)}."
            if combined is not None
            else f"\nAlternatively place one multi-edition file "
            f"({' / '.join(COMBINED_FILENAMES)}) in {RAW_FIFA_DIR}."
        )
        raise FileNotFoundError(
            f"{len(absent)} edition(s) missing. Download from Kaggle and save as:\n"
            f"{wanted}{found}\nSee the README for the download steps."
        )

    frames = []
    all_problems: list[str] = []

    for season in SEASONS:
        df, problems = load_edition(season)
        problems += validate_edition(df, season)
        all_problems += problems

        print(
            f"{season.label}  {season.fifa_edition:9} {len(df):4} players, "
            f"{df['club_fd'].nunique()} clubs, mean overall {df['overall'].mean():.1f}"
            + ("  PROBLEMS" if problems else "")
        )
        frames.append(df)

    if all_problems:
        message = "Validation failed:\n" + "\n".join(f"  {p}" for p in all_problems)
        if strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    combined = pd.concat(frames, ignore_index=True)

    ordered = [
        "season",
        "season_slug",
        "fifa_edition",
        "player_name",
        "long_name",
        "club",
        "club_fd",
        "positions",
        *NUMERIC,
    ]
    return combined[ordered]


def main() -> int:
    try:
        players = build()
    except FileNotFoundError as exc:
        # Absent ratings files are the expected first-run state, not a crash: the CSVs
        # have to be downloaded by hand. Say so plainly instead of via a traceback.
        print(exc, file=sys.stderr)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    players.to_parquet(FIFA_PLAYERS_PARQUET, index=False)

    print(f"\n{len(players)} player-seasons, {players['player_name'].nunique()} distinct names")
    print(f"written to {FIFA_PLAYERS_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
