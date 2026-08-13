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
from src.data.clean_fpl import FPL_FIXTURES_PARQUET

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


def next_fpl_gameweek(fixtures: pd.DataFrame, played_before: pd.Timestamp | None = None):
    """The earliest gameweek in ``fixtures`` that has not kicked off yet.

    A round is "next" when its *first* fixture is still ahead. Using the last fixture
    instead would keep a round current while most of it had already been played.
    """
    if fixtures.empty:
        return None

    now = played_before or pd.Timestamp.now()
    upcoming = fixtures[fixtures["date"] >= now]
    return None if upcoming.empty else int(upcoming["gameweek"].min())


def fpl_schedule() -> pd.DataFrame:
    """The whole stored FPL season, or an empty frame when it has not been built."""
    if not FPL_FIXTURES_PARQUET.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team", "gameweek"])
    return pd.read_parquet(FPL_FIXTURES_PARQUET)


def fpl_fixtures(gameweek: int | None = None) -> pd.DataFrame:
    """One round from the stored FPL schedule, defaulting to the next one.

    Returns an empty frame when the schedule has not been built, so the caller falls
    through to its other sources rather than failing.
    """
    fixtures = fpl_schedule()
    if fixtures.empty:
        return fixtures

    gameweek = gameweek if gameweek is not None else next_fpl_gameweek(fixtures)
    if gameweek is None:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    return fixtures[fixtures["gameweek"] == gameweek].reset_index(drop=True)


# How close to kickoff an unpredicted round has to be before the scheduled job predicts
# it. Later is better - team news, suspensions and transfers all keep arriving - but the
# job only gets one attempt per day, so this is also how many attempts a round gets before
# it kicks off. Three days is two spare attempts if a run fails or the FPL feed is down.
DEFAULT_LEAD_DAYS = 3


def due_round(
    fixtures: pd.DataFrame, now: pd.Timestamp | None = None, within_days: int = DEFAULT_LEAD_DAYS
) -> int | None:
    """The gameweek an unattended run should predict, or None if none is due.

    Due means the next round's *first* kickoff is ahead of us and no further away than
    ``within_days``. Both halves matter:

    **A round that has already started is never due**, whatever went wrong. If the job was
    down for the whole window, the honest outcome is a missing round - a "prediction" made
    after kickoff is not one, and the archive exists precisely to be trustworthy about
    when it was written.

    **A round further out than the window is not due yet**, so the job waits rather than
    committing to a team sheet a fortnight early. A stored round is never rewritten, so
    predicting early is a decision that cannot be taken back.
    """
    if fixtures.empty or "gameweek" not in fixtures.columns:
        return None

    now = now if now is not None else pd.Timestamp.now()
    gameweek = next_fpl_gameweek(fixtures, now)
    if gameweek is None:
        return None

    kickoffs = fixtures.loc[fixtures["gameweek"] == gameweek, "date"].dropna()
    if kickoffs.empty:
        return None

    first = kickoffs.min()
    if first <= now or first - now > pd.Timedelta(days=within_days):
        return None
    return int(gameweek)


def upcoming_fixtures(
    allow_download: bool = True, gameweek: int | None = None
) -> tuple[pd.DataFrame, str]:
    """The fixtures to predict, and where they came from.

    Returns (fixtures, source), in order of preference:

    1. The hand-written file, when it has rows - naming the round you want beats
       accepting whatever is next.
    2. The stored Fantasy Premier League schedule, which covers a whole season and so
       works in the summer, when football-data's rolling feed is empty. This is what
       makes predicting the opening round possible at all.
    3. football-data's rolling feed, which only ever covers the next few days.

    ``gameweek`` names a round instead of taking the next one. It skips the hand-written
    file, which describes one specific round and would otherwise silently answer for a
    different one.
    """
    if gameweek is not None:
        stored = fpl_fixtures(gameweek)
        return stored, f"FPL schedule, gameweek {gameweek}"

    manual = manual_fixtures()
    if not manual.empty:
        return manual, str(MANUAL_FIXTURES_CSV)

    stored = fpl_fixtures()
    if not stored.empty:
        gameweek = int(stored["gameweek"].iloc[0])
        return stored, f"FPL schedule, gameweek {gameweek}"

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
