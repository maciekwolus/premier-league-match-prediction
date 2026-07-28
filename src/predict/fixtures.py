"""Where the next round of fixtures comes from.

football-data publishes a rolling ``fixtures.csv`` covering the next few days across
every competition it tracks. That is enough during a season and empty outside one, so a
hand-written list in ``data/manual/upcoming_fixtures.csv`` takes precedence - it is also
how you predict a specific round rather than whatever happens to be next.
"""

from __future__ import annotations

import pandas as pd
import requests

from src.config import MANUAL_DIR, UPCOMING_SEASON

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
MANUAL_FIXTURES_CSV = MANUAL_DIR / "upcoming_fixtures.csv"

PREMIER_LEAGUE = "E0"
TIMEOUT_SECONDS = 30


def manual_fixtures() -> pd.DataFrame:
    """Fixtures typed by hand. Empty frame when the file has only its header."""
    if not MANUAL_FIXTURES_CSV.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    df = pd.read_csv(MANUAL_FIXTURES_CSV)
    required = {"date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{MANUAL_FIXTURES_CSV} is missing column(s) {sorted(missing)}")

    if df.empty:
        return df.assign(date=pd.to_datetime(df["date"], errors="coerce"))

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


def download_fixtures() -> pd.DataFrame:
    """The next few days of Premier League fixtures, as football-data has them."""
    response = requests.get(FIXTURES_URL, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    from io import StringIO

    raw = pd.read_csv(StringIO(response.content.decode("utf-8-sig", errors="replace")))
    if "Div" not in raw.columns:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    english = raw[raw["Div"] == PREMIER_LEAGUE]
    if english.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(english["Date"], dayfirst=True, errors="coerce"),
            "home_team": english["HomeTeam"],
            "away_team": english["AwayTeam"],
        }
    )

    # The feed carries a market average when one exists. It is the *opening* line rather
    # than the closing one - the fixture has not kicked off - but showing the model
    # against the market is the point of the report, so take it when offered.
    for source, ours in (
        ("AvgH", "odds_close_avg_home"),
        ("AvgD", "odds_close_avg_draw"),
        ("AvgA", "odds_close_avg_away"),
    ):
        if source in english.columns:
            fixtures[ours] = pd.to_numeric(english[source], errors="coerce").to_numpy()

    return fixtures.dropna(subset=["date"]).reset_index(drop=True)


def upcoming_fixtures(allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """The fixtures to predict, and where they came from.

    Returns (fixtures, source). The hand-written file wins when it has rows, because
    naming the round you want is more useful than accepting whatever is next.
    """
    manual = manual_fixtures()
    if not manual.empty:
        return manual, str(MANUAL_FIXTURES_CSV)

    if not allow_download:
        return manual, "manual (empty)"

    downloaded = download_fixtures()
    return downloaded, FIXTURES_URL


def as_matches(fixtures: pd.DataFrame, season: str = UPCOMING_SEASON.label) -> pd.DataFrame:
    """Shape fixtures like rows of the match table, with the results left blank.

    The feature builder works on a team-match table that expects match-shaped rows, so
    upcoming fixtures are appended to history in exactly that form and simply carry no
    goals. Everything downstream then treats them uniformly.
    """
    from src.data.clean_matches import build_match_id

    if fixtures.empty:
        return pd.DataFrame()

    season_object = UPCOMING_SEASON if season == UPCOMING_SEASON.label else None
    slug = season.replace("/", "_")

    rows = fixtures.copy()
    rows["season"] = season
    rows["match_id"] = [
        build_match_id(season_object, date, home, away)
        if season_object is not None
        else f"{slug}_{date:%Y%m%d}_{home}_{away}"
        for date, home, away in zip(rows["date"], rows["home_team"], rows["away_team"], strict=True)
    ]

    for column in (
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_target",
        "away_shots_target",
        "home_corners",
        "away_corners",
    ):
        rows[column] = pd.NA

    return rows
