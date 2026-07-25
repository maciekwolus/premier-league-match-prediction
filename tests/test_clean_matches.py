"""Tests for match cleaning and validation.

These build synthetic seasons rather than reading the downloaded files, so they run
offline and stay meaningful even if football-data.co.uk changes.
"""

import pandas as pd
import pytest

from src.config import SEASONS
from src.data.clean_matches import build_match_id, slugify_team, validate_season

SEASON = SEASONS[0]


def make_valid_season() -> pd.DataFrame:
    """A structurally perfect season: 20 teams, 380 matches, 19 home / 19 away each."""
    teams = [f"Team {i:02d}" for i in range(20)]
    rows = []
    date = pd.Timestamp("2019-08-09")

    for home in teams:
        for away in teams:
            if home == away:
                continue
            rows.append(
                {
                    "match_id": build_match_id(SEASON, date, home, away),
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 1,
                    "away_goals": 0,
                    "result": "H",
                }
            )
            date += pd.Timedelta(days=1)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- slugify


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Man United", "man_united"),
        ("Nott'm Forest", "nottm_forest"),
        ("Sheffield United", "sheffield_united"),
        ("Arsenal", "arsenal"),
        ("Brighton", "brighton"),
    ],
)
def test_slugify_team(name, expected):
    assert slugify_team(name) == expected


def test_slugify_strips_accents():
    assert slugify_team("Atlético Madrid") == "atletico_madrid"


# -------------------------------------------------------------------- match_id


def test_match_id_is_stable_and_readable():
    match_id = build_match_id(SEASON, pd.Timestamp("2019-08-09"), "Man United", "Chelsea")
    assert match_id == "2019_20_20190809_man_united_chelsea"


def test_match_id_distinguishes_home_and_away_fixtures():
    date = pd.Timestamp("2019-08-09")
    assert build_match_id(SEASON, date, "Arsenal", "Chelsea") != build_match_id(
        SEASON, date, "Chelsea", "Arsenal"
    )


# ------------------------------------------------------------------ validation


def test_valid_season_has_no_problems():
    assert validate_season(make_valid_season(), SEASON) == []


def test_detects_wrong_match_count():
    df = make_valid_season().iloc[:-1]
    problems = validate_season(df, SEASON)
    assert any("379 matches" in p for p in problems)


def test_detects_duplicate_match_id():
    df = make_valid_season()
    df.loc[df.index[1], "match_id"] = df.loc[df.index[0], "match_id"]
    problems = validate_season(df, SEASON)
    assert any("duplicate match_id" in p for p in problems)


def test_detects_result_disagreeing_with_score():
    df = make_valid_season()
    df.loc[df.index[0], "result"] = "A"  # says away win, score says 1-0 home
    problems = validate_season(df, SEASON)
    assert any("result disagrees" in p for p in problems)


def test_detects_null_goals():
    df = make_valid_season()
    df["home_goals"] = df["home_goals"].astype("Int64")
    df.loc[df.index[0], "home_goals"] = pd.NA
    problems = validate_season(df, SEASON)
    assert any("null values in home_goals" in p for p in problems)


def test_detects_wrong_team_count():
    """A 21st team appearing means the source file mixed in another competition."""
    df = make_valid_season()
    df.loc[df.index[0], "home_team"] = "Team 99"
    problems = validate_season(df, SEASON)
    assert any("distinct teams" in p for p in problems)


def test_detects_unbalanced_home_and_away():
    df = make_valid_season()
    mask = df["home_team"] == "Team 00"
    df.loc[mask, "home_team"] = "Team 01"
    problems = validate_season(df, SEASON)
    assert any("home /" in p for p in problems)
