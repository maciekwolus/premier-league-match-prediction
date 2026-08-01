"""Tests for the Fantasy Premier League source and the promoted-club fallback.

Two risks here, both quiet. A club we fail to map disappears from a division of twenty
without anything raising, and a promoted club with no history gets no squad quality at
all - which downstream becomes the training median, describing a newly promoted side to
the model as an average Premier League squad. That is the exact shape of the Phase 9 bug.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean_fpl import club_names, upcoming_fixtures, validate
from src.matching.team_names import UnknownTeamError, fpl_to_football_data
from src.predict.squads import FALLBACK_SHAPE, ratings_eleven

FPL_CLUBS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds",
    "Liverpool",
    "Man City",
    "Man Utd",
    "Newcastle",
    "Nott'm Forest",
    "Spurs",
    "Sunderland",
]


def bootstrap(clubs=None) -> dict:
    clubs = clubs or FPL_CLUBS
    return {"teams": [{"id": index + 1, "name": name} for index, name in enumerate(clubs)]}


def full_schedule(clubs=None) -> list[dict]:
    """A complete double round-robin, one gameweek a week."""
    ids = list(range(1, len(clubs or FPL_CLUBS) + 1))
    rotating = list(ids)
    rounds = []
    for _ in range(len(ids) - 1):
        rounds.append([(rotating[i], rotating[-1 - i]) for i in range(len(ids) // 2)])
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    rounds += [[(a, h) for h, a in rnd] for rnd in rounds]

    fixtures = []
    for number, pairs in enumerate(rounds, start=1):
        for home, away in pairs:
            fixtures.append(
                {
                    "event": number,
                    "kickoff_time": (
                        pd.Timestamp("2026-08-21") + pd.Timedelta(days=7 * (number - 1))
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "team_h": home,
                    "team_a": away,
                }
            )
    return fixtures


# --------------------------------------------------------------------- club names


def test_the_five_differing_names_are_translated():
    assert fpl_to_football_data("Man Utd") == "Man United"
    assert fpl_to_football_data("Spurs") == "Tottenham"
    assert fpl_to_football_data("Coventry City") == "Coventry"
    assert fpl_to_football_data("Hull City") == "Hull"
    assert fpl_to_football_data("Ipswich Town") == "Ipswich"


def test_names_that_already_agree_pass_through():
    for name in ("Arsenal", "Liverpool", "Nott'm Forest", "Crystal Palace"):
        assert fpl_to_football_data(name) == name


def test_an_unmapped_club_raises_rather_than_entering_the_pipeline():
    """The FPL club list *is* the division, so an unknown name is a promoted club nobody
    added - not a foreign club to ignore."""
    with pytest.raises(UnknownTeamError, match="FPL_TO_FOOTBALL_DATA"):
        fpl_to_football_data("Wrexham")


def test_every_club_of_the_new_season_maps():
    mapped = club_names(bootstrap())

    assert len(mapped) == 20
    assert len(set(mapped.values())) == 20


# --------------------------------------------------------------------- the schedule


def test_a_full_season_is_read_and_named_in_football_data_terms():
    fixtures = upcoming_fixtures(bootstrap(), full_schedule())

    assert len(fixtures) == 380
    assert "Man United" in set(fixtures["home_team"])
    assert "Man Utd" not in set(fixtures["home_team"])
    assert fixtures["gameweek"].min() == 1
    assert fixtures["gameweek"].max() == 38


def test_kickoff_times_lose_their_timezone():
    """The rest of the pipeline works in naive dates; a tz-aware column breaks comparisons."""
    fixtures = upcoming_fixtures(bootstrap(), full_schedule())
    assert fixtures["date"].dt.tz is None


def test_a_clean_schedule_has_no_problems():
    fixtures = upcoming_fixtures(bootstrap(), full_schedule())
    assert validate(fixtures, club_names(bootstrap())) == []


def test_a_missing_fixture_is_caught():
    schedule = full_schedule()[:-1]
    fixtures = upcoming_fixtures(bootstrap(), schedule)

    problems = validate(fixtures, club_names(bootstrap()))

    assert any("379" in problem for problem in problems)


def test_a_club_playing_the_wrong_number_of_home_games_is_caught():
    """The guard that catches a schedule that is the right size and the wrong shape."""
    schedule = full_schedule()
    schedule[0]["team_h"], schedule[0]["team_a"] = schedule[0]["team_a"], schedule[0]["team_h"]

    problems = validate(upcoming_fixtures(bootstrap(), schedule), club_names(bootstrap()))

    assert any("home_team" in problem for problem in problems)


def test_a_fixture_without_a_kickoff_time_is_reported_not_dropped():
    """Normal for televised rounds not yet scheduled - but it must be visible."""
    schedule = full_schedule()
    schedule[0]["kickoff_time"] = None

    fixtures = upcoming_fixtures(bootstrap(), schedule)
    problems = validate(fixtures, club_names(bootstrap()))

    assert len(fixtures) == 380
    assert any("kickoff" in problem for problem in problems)


# ------------------------------------------------------- the promoted-club fallback XI


def make_ratings(club="Coventry", season="2025/26", count=25) -> pd.DataFrame:
    """A squad deep enough to pick a shape from, best players first."""
    positions = ["GK", "GK", "CB", "CB", "CB", "LB", "RB", "CDM", "CM", "CM", "CAM", "LM", "RM"]
    positions += ["ST", "ST", "LW", "RW"] + ["CM"] * 8
    return pd.DataFrame(
        [
            {
                "season": season,
                "club_fd": club,
                "player_name": f"Player {index:02d}",
                "positions": positions[index % len(positions)],
                "overall": 80 - index,
            }
            for index in range(count)
        ]
    )


def test_a_ratings_eleven_is_eleven_players():
    eleven = ratings_eleven(make_ratings(), "Coventry", "2025/26")

    assert len(eleven) == sum(FALLBACK_SHAPE.values()) == 11


def test_a_ratings_eleven_has_one_goalkeeper():
    eleven = ratings_eleven(make_ratings(), "Coventry", "2025/26")
    assert (eleven["line"] == "gk").sum() == 1


def test_a_ratings_eleven_takes_the_best_of_each_line():
    """It answers "who are this club's best players", which is the only question the
    ratings can answer for a club nobody has seen play in this division."""
    ratings = make_ratings()
    eleven = ratings_eleven(ratings, "Coventry", "2025/26")

    keeper = eleven[eleven["line"] == "gk"]["player"].iloc[0]
    keepers = ratings[ratings["positions"] == "GK"].sort_values("overall", ascending=False)
    assert keeper == keepers["player_name"].iloc[0]


def test_starts_are_zero_because_the_club_has_started_nobody():
    eleven = ratings_eleven(make_ratings(), "Coventry", "2025/26")
    assert set(eleven["starts"]) == {0}


def test_a_club_with_no_ratings_yields_nothing_rather_than_raising():
    assert ratings_eleven(make_ratings(), "Wrexham", "2025/26").empty


def test_the_wrong_season_yields_nothing():
    assert ratings_eleven(make_ratings(), "Coventry", "2019/20").empty


def test_an_empty_ratings_table_yields_nothing():
    assert ratings_eleven(pd.DataFrame(), "Coventry", "2025/26").empty
