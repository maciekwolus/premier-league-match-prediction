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

import argparse
import re
import sys
from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, RAW_FIFA_DIR, SEASONS, Season
from src.data.clean_matches import MATCHES_PARQUET
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
    "dob": ("dob", "birth_date", "date_of_birth", "birthday"),
    # Short codes come first deliberately. Datasets that use them (EA FC 26) also carry
    # detailed skills under the long names - "Dribbling" there is ball control, not the
    # summary stat - so matching "dri" first keeps the six face stats consistent across
    # editions. Datasets using long names (male_players.csv) have no short codes to hit.
    "pace": ("pac", "pace", "speed"),
    "shooting": ("sho", "shooting"),
    "passing": ("pas", "passing"),
    "dribbling": ("dri", "dribbling"),
    "defending": ("def", "defending"),
    "physic": ("phy", "physic", "physicality", "physical"),
}

REQUIRED = ("player_name", "club", "overall")

# FIFA's six summary "face" stats. They are published as a set, so we take them only
# when all six resolve. Some SoFIFA exports carry the *detailed* skills instead - a
# column literally named "dribbling" that means ball control, alongside acceleration
# and sprint_speed rather than pace. Accepting that one column would put a different
# quantity in the same field for one season, which is worse than leaving it null.
FACE_STATS = ("pace", "shooting", "passing", "dribbling", "defending", "physic")
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

# FIFA lists the full club roster - senior squad plus youth and players out on loan -
# so a legitimate club runs from about 22 to the low 60s. The range is only here to
# catch a broken filter, not to police squad registration rules.
MIN_PLAYERS_PER_CLUB = 15
MAX_PLAYERS_PER_CLUB = 80


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

    if not all(stat in resolved for stat in FACE_STATS):
        for stat in FACE_STATS:
            resolved.pop(stat, None)

    return resolved


def parse_money(value):
    """Turn "€115.5M" into 115500000. Values that are already numeric pass through."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, int | float):
        return value

    match = re.search(r"([\d.]+)\s*([KMB])?", str(value), re.IGNORECASE)
    if match is None:
        return pd.NA

    amount = float(match.group(1))
    scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get((match.group(2) or "").lower(), 1)
    return amount * scale


def age_at_season_start(dob, season: Season):
    """Age on 1 August of the season's opening year, when only a birth date is given."""
    birth = pd.to_datetime(dob, errors="coerce")
    start = pd.Timestamp(year=int(season.label[:4]), month=8, day=1)
    return ((start - birth).dt.days / 365.25).round(0)


@lru_cache(maxsize=1)
def _matches_clubs() -> dict[str, frozenset[str]]:
    """Season label -> the clubs that actually played it, from the match table."""
    if not MATCHES_PARQUET.exists():
        raise FileNotFoundError(
            f"{MATCHES_PARQUET} not found. Run: python -m src.data.clean_matches"
        )
    matches = pd.read_parquet(MATCHES_PARQUET, columns=["season", "home_team"])
    return {label: frozenset(group["home_team"]) for label, group in matches.groupby("season")}


def season_clubs(season: Season) -> frozenset[str]:
    """The 20 clubs that played this season.

    Ratings files cover every division, so filtering on "is this ever a Premier League
    club" lets Championship sides through - Leeds and Sunderland are in the file every
    year. Only the clubs that actually played the season belong here.
    """
    return _matches_clubs()[season.label]


def load_edition(
    season: Season, clubs: frozenset[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Read one edition and cut it down to the clubs that played that season.

    A per-edition file wins if present; otherwise this edition's rows are taken from a
    combined multi-edition file. ``clubs`` defaults to the season's real participants.
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
    if clubs is None:
        clubs = season_clubs(season)
    premier_league = df[df["club_fd"].isin(clubs)].copy()

    premier_league["season"] = season.label
    premier_league["season_slug"] = season.slug
    premier_league["fifa_edition"] = season.fifa_edition

    # Some exports append a dangling separator to the name ("Rodri -"). Phase 4 matches
    # on these strings, so tidy them at the source rather than in every consumer.
    for field in ("player_name", "long_name"):
        premier_league[field] = (
            premier_league[field].astype("string").str.strip().str.strip("-").str.strip()
        )

    premier_league["value_eur"] = premier_league["value_eur"].map(parse_money)

    for field in NUMERIC:
        premier_league[field] = pd.to_numeric(premier_league[field], errors="coerce")

    # Where the source gives a birth date but no age, derive it.
    if premier_league["age"].isna().all() and premier_league["dob"].notna().any():
        premier_league["age"] = age_at_season_start(premier_league["dob"], season)

    # `dob` is an input for deriving age, not an output field, and age is only genuinely
    # absent if the derivation could not fill it either.
    reportable = set(notes) - {"dob"}
    if premier_league["age"].notna().any():
        reportable.discard("age")

    problems = []
    if reportable:
        problems.append(f"{season.fifa_edition}: no column found for {sorted(reportable)}")

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


def build(strict: bool = True, allow_missing: bool = False) -> pd.DataFrame:
    """Load, validate and stack every edition.

    ``allow_missing`` builds from whatever editions are present. Useful while chasing
    down a source for one edition, but the seasons it skips will have no squad-quality
    features at all, so it is not the default.
    """
    absent = missing_editions()
    if absent and allow_missing:
        print(
            f"Skipping {len(absent)} edition(s) with no data: "
            f"{', '.join(s.fifa_edition for s in absent)}",
            file=sys.stderr,
        )
    elif absent:
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
    available = [s for s in SEASONS if s not in absent]

    for season in available:
        # Notes are informational - an edition lacking an optional column is still
        # usable. Only validation problems are fatal.
        df, notes = load_edition(season)
        problems = validate_edition(df, season)
        all_problems += problems

        print(
            f"{season.label}  {season.fifa_edition:9} {len(df):4} players, "
            f"{df['club_fd'].nunique()} clubs, mean overall {df['overall'].mean():.1f}"
            + ("  PROBLEMS" if problems else "")
        )
        for note in notes:
            print(f"           note: {note}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="build from the editions present instead of requiring all seven",
    )
    args = parser.parse_args(argv)

    try:
        players = build(allow_missing=args.allow_missing)
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
