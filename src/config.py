"""Central configuration: filesystem paths, season definitions and data source URLs.

Everything that varies by season lives in `SEASONS`. Adding next season should mean
adding one row here and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_MATCHES_DIR = DATA_DIR / "raw" / "matches"
RAW_LINEUPS_DIR = DATA_DIR / "raw" / "lineups"
RAW_FIFA_DIR = DATA_DIR / "raw" / "fifa"
RAW_FPL_DIR = DATA_DIR / "raw" / "fpl"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MANUAL_DIR = DATA_DIR / "manual"  # hand-written override files, committed to git

# Each stage creates the directory it writes to, so there is no setup step to forget.

# --------------------------------------------------------------------------- sources

# football-data.co.uk publishes one CSV per competition per season.
# E0 is the Premier League.
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

MATCHES_PER_SEASON = 380  # 20 teams, home and away


@dataclass(frozen=True)
class Season:
    """One Premier League season and its identifier in each upstream data source."""

    label: str  # human readable, e.g. "2019/20"
    code: str  # football-data.co.uk URL segment, e.g. "1920"
    understat: str  # Understat season key (the starting year), e.g. "2019"
    fifa_edition: str  # game whose ratings apply to this season

    @property
    def slug(self) -> str:
        """Filename-safe form, e.g. "2019_20"."""
        return self.label.replace("/", "_")

    @property
    def matches_url(self) -> str:
        return FOOTBALL_DATA_URL.format(code=self.code)

    @property
    def fifa_slug(self) -> str:
        """Filename form of the ratings edition: "FIFA 20" -> "fifa20", "EA FC 24" -> "fc24".

        This is the name the ratings CSV must be saved under in ``data/raw/fifa/``.
        """
        return self.fifa_edition.lower().replace("ea ", "").replace(" ", "")

    @property
    def fifa_version(self) -> int:
        """Edition number: "FIFA 20" -> 20, "EA FC 24" -> 24.

        Datasets that bundle several editions into one file identify them this way.
        """
        return int("".join(c for c in self.fifa_edition if c.isdigit()))


# EA renamed the series after FIFA 23, hence the inconsistent-looking labels.
SEASONS: tuple[Season, ...] = (
    Season("2019/20", "1920", "2019", "FIFA 20"),
    Season("2020/21", "2021", "2020", "FIFA 21"),
    Season("2021/22", "2122", "2021", "FIFA 22"),
    Season("2022/23", "2223", "2022", "FIFA 23"),
    Season("2023/24", "2324", "2023", "EA FC 24"),
    Season("2024/25", "2425", "2024", "EA FC 25"),
    Season("2025/26", "2526", "2025", "EA FC 26"),
)

SEASONS_BY_CODE: dict[str, Season] = {s.code: s for s in SEASONS}
SEASONS_BY_LABEL: dict[str, Season] = {s.label: s for s in SEASONS}

# The season being predicted, kept out of SEASONS because every ingestion stage treats
# that tuple as "seasons with complete data" and would demand a ratings edition that does
# not exist yet. Its *results* are still ingested, as a partial season - see
# `clean_matches.load_in_progress` - because scoring a stored prediction against what
# happened is the only reason to keep the archive at all.
#
# `fifa_edition` deliberately points at the *previous* game: a season starts in August
# and its own edition is not published until late September, so the newest ratings that
# exist are last year's. Change this line when the new edition lands. Until then the
# transfer window makes squad membership wrong rather than the ratings, which is what
# `predict.transfers` corrects from the FPL squad lists.
UPCOMING_SEASON = Season("2026/27", "2627", "2026", "EA FC 26")


# Timestamp resolution for anything written to parquet.
#
# **A committed parquet must not depend on which machine wrote it.** pandas 3 defaults
# datetimes to microseconds where pandas 2 used nanoseconds, so the scheduled job (Linux,
# pandas 3) and a developer laptop (pandas 2) wrote byte-different files holding
# identical values. The job noticed a change every single day, committed "Update results"
# with 0 insertions and 0 deletions, and redeployed the site for nothing - and the next
# local rebuild flipped it straight back. Microseconds because that is parquet's native
# resolution and what the newer default produces, so this converts on the way out rather
# than on the way in.
PARQUET_TIME_UNIT = "us"


def normalise_for_parquet(frame):
    """Copy of a frame with every datetime column at ``PARQUET_TIME_UNIT``."""
    import pandas as pd

    stable = frame.copy()
    for column in stable.columns:
        if isinstance(stable[column].dtype, pd.DatetimeTZDtype):
            stable[column] = stable[column].dt.as_unit(PARQUET_TIME_UNIT)
        elif str(stable[column].dtype).startswith("datetime64"):
            stable[column] = stable[column].astype(f"datetime64[{PARQUET_TIME_UNIT}]")
    return stable


def write_parquet(frame, path) -> bool:
    """Write a dataframe to parquet, but **leave the file alone if the data is the same**.

    Returns whether anything was written.

    Use this for anything under version control; ``to_parquet`` directly is fine for
    throwaway output.

    Normalising the timestamp resolution is not enough on its own. pyarrow stamps its own
    version into every file it writes, so a laptop on pyarrow 20 and a runner on pyarrow
    24 produce different bytes from identical data no matter what. The scheduled job
    rebuilds this table daily, and byte-comparison is what git does - so it committed
    "Update results" with 0 insertions and 0 deletions every day, redeployed the site for
    nothing, and buried the real results updates in the noise.

    Comparing the *data* instead is the only version-proof answer, and it is the honest
    one: a rebuild that produces the same table has not changed anything.
    """
    import pandas as pd

    stable = normalise_for_parquet(frame)

    if path.exists():
        try:
            if pd.read_parquet(path).equals(stable):
                return False
        except Exception:  # noqa: BLE001 - unreadable or written by a future format
            pass  # fall through and rewrite it

    path.parent.mkdir(parents=True, exist_ok=True)
    stable.to_parquet(path, index=False)
    return True
