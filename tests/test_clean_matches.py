"""Tests for match cleaning and validation.

These build synthetic seasons rather than reading the downloaded files, so they run
offline and stay meaningful even if football-data.co.uk changes.
"""

import pandas as pd
import pytest

from src.config import SEASONS
from src.data.clean_matches import (
    build_match_id,
    divisions_in,
    slugify_team,
    validate_season,
)

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


# ------------------------------------------------------- a season still being played

# The count checks are what protect a completed season: "exactly 380 matches" catches
# almost any corruption by accident. A partial season has no such backstop, so these
# pin what replaces it.


def partial_season(rounds: int = 3) -> pd.DataFrame:
    """The opening few rounds of a season: 20 teams, nobody played twice at home."""
    full = make_valid_season()
    return full.head(10 * rounds).reset_index(drop=True)


def test_a_partial_season_is_valid_while_a_complete_one_would_not_be():
    df = partial_season()

    assert validate_season(df, SEASON, partial=True) == []
    assert validate_season(df, SEASON) != []


def test_an_empty_partial_season_is_valid():
    """Between the fixture list appearing and the first match being played."""
    empty = partial_season().iloc[0:0]
    assert validate_season(empty, SEASON, partial=True) == []


def test_a_partial_season_still_rejects_more_than_a_division_of_teams():
    """The check that replaces the count. A 21st club means the file is mixing
    divisions - which is exactly what football-data served at the Premier League
    address before the 2026/27 season started."""
    df = partial_season()
    df.loc[df.index[0], "home_team"] = "Team 99"

    assert any("more than a division's 20" in p for p in validate_season(df, SEASON, partial=True))


def test_a_partial_season_still_rejects_a_club_playing_home_too_often():
    df = make_valid_season()
    df.loc[df["home_team"] == "Team 01", "home_team"] = "Team 00"

    assert any("more than 19 each" in p for p in validate_season(df, SEASON, partial=True))


def test_a_partial_season_never_exceeds_a_full_one():
    df = pd.concat([make_valid_season(), partial_season(1)], ignore_index=True)

    assert any("more than a season's" in p for p in validate_season(df, SEASON, partial=True))


def test_a_partial_season_still_rejects_a_missing_score():
    """Not relaxed, deliberately: football-data lists a match only once it has been
    played, so a row with no score is corrupt rather than a fixture still to come."""
    df = partial_season()
    df["home_goals"] = df["home_goals"].astype("Int64")
    df.loc[df.index[0], "home_goals"] = pd.NA

    assert any("null values in home_goals" in p for p in validate_season(df, SEASON, partial=True))


def test_a_partial_season_still_rejects_duplicates():
    df = pd.concat([partial_season(1), partial_season(1)], ignore_index=True)

    assert any("duplicate match_id" in p for p in validate_season(df, SEASON, partial=True))


# --------------------------------------------------------------- which division is it

# football-data serves a file at the Premier League address before the season starts, and
# in August 2026 that file held twelve National League matches. Team names are well-formed
# and this source needs no mapping table, so nothing else here would have objected.


def test_the_division_column_is_found_despite_the_byte_order_mark():
    """football-data writes a UTF-8 BOM, so the header reads "\ufeffDiv" rather than "Div" -
    which is precisely how a column this important goes unread."""
    raw = pd.DataFrame({"\ufeffDiv": ["E0", "E0"], "HomeTeam": ["A", "B"]})

    assert divisions_in(raw) == {"E0"}


def test_the_division_column_is_found_without_one():
    raw = pd.DataFrame({"Div": ["E0"], "HomeTeam": ["A"]})
    assert divisions_in(raw) == {"E0"}


def test_a_file_with_no_division_column_reports_nothing_rather_than_guessing():
    assert divisions_in(pd.DataFrame({"HomeTeam": ["A"]})) == set()


def test_another_division_is_detected():
    """EC is the National League. Ingesting it would put those results in the Premier
    League table, and every downstream stage would believe them."""
    raw = pd.DataFrame({"\ufeffDiv": ["EC"] * 12, "HomeTeam": ["Altrincham"] * 12})

    assert divisions_in(raw) == {"EC"}
