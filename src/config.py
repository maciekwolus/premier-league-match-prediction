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
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MANUAL_DIR = DATA_DIR / "manual"  # hand-written override files, committed to git

ALL_DATA_DIRS = (
    RAW_MATCHES_DIR,
    RAW_LINEUPS_DIR,
    RAW_FIFA_DIR,
    PROCESSED_DIR,
    FINAL_DIR,
    MANUAL_DIR,
)

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


def ensure_data_dirs() -> None:
    """Create every data directory if it does not already exist."""
    for directory in ALL_DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
