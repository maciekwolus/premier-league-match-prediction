"""Tests for deriving a gameweek from the schedule.

No upstream source publishes a round number, so this is inferred - which means a wrong
inference looks exactly like a right one. These pin the two properties that matter: a
clean season splits into 38 rounds of 10, and a rescheduled match does not shift every
fixture after it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.gameweeks import assign_gameweeks, gameweek_sizes


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """A double round-robin as a list of rounds, using the standard circle method."""
    rotating = list(teams)
    first_half = []
    for _ in range(len(teams) - 1):
        pairs = [(rotating[i], rotating[-1 - i]) for i in range(len(teams) // 2)]
        first_half.append(pairs)
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    # Second half is the same fixtures with home and away swapped.
    return first_half + [[(away, home) for home, away in rnd] for rnd in first_half]


def season_frame(team_count: int = 20, season: str = "2025/26") -> pd.DataFrame:
    """A complete synthetic season, one round every seven days."""
    teams = [f"Team {chr(ord('A') + i)}" for i in range(team_count)]
    start = pd.Timestamp("2025-08-15")

    rows = []
    for number, fixtures in enumerate(round_robin(teams)):
        for home, away in fixtures:
            rows.append(
                {
                    "season": season,
                    "date": start + pd.Timedelta(days=7 * number),
                    "home_team": home,
                    "away_team": away,
                }
            )
    return pd.DataFrame(rows)


def test_a_clean_season_splits_into_38_rounds_of_10():
    matches = season_frame()
    gameweeks = assign_gameweeks(matches)

    assert gameweeks.min() == 1
    assert gameweeks.max() == 38
    assert set(gameweek_sizes(matches, gameweeks)) == {10}


def test_every_club_appears_once_per_gameweek():
    """The property the whole derivation rests on - a club plays each round exactly once."""
    matches = season_frame().assign(gameweek=lambda df: assign_gameweeks(df))

    for _, rows in matches.groupby("gameweek"):
        clubs = [*rows["home_team"], *rows["away_team"]]
        assert len(clubs) == len(set(clubs))


def test_a_postponed_match_does_not_shift_the_rest_of_the_season():
    """The reason for counting matches rather than clustering dates.

    Moving one fixture to the end of the season must leave every other fixture's round
    untouched - only the two clubs involved are affected.
    """
    matches = season_frame().sort_values("date").reset_index(drop=True)
    before = assign_gameweeks(matches)

    postponed = matches.copy()
    postponed.loc[0, "date"] = postponed["date"].max() + pd.Timedelta(days=7)
    after = assign_gameweeks(postponed)

    moved = set(matches.loc[0, ["home_team", "away_team"]])
    untouched = matches[~matches["home_team"].isin(moved) & ~matches["away_team"].isin(moved)].index

    pd.testing.assert_series_equal(before[untouched], after[untouched])


def test_a_rescheduled_match_lands_in_a_later_round_not_its_original():
    """Honest about what the rule does. A match played out of order counts where it was
    played, so its round can hold 11 fixtures and another 9 - which is why callers are
    told not to assume 10."""
    matches = season_frame().sort_values("date").reset_index(drop=True)
    postponed = matches.copy()
    postponed.loc[0, "date"] = postponed["date"].max() + pd.Timedelta(days=7)

    sizes = gameweek_sizes(postponed)

    assert set(sizes) != {10}
    assert sizes.sum() == len(matches)


def test_seasons_are_counted_separately():
    """Gameweek 1 exists in every season; the count must reset."""
    first = season_frame(team_count=4, season="2024/25")
    second = season_frame(team_count=4, season="2025/26")
    both = pd.concat([first, second], ignore_index=True)

    gameweeks = assign_gameweeks(both)

    assert gameweeks[both["season"] == "2024/25"].max() == 6
    assert gameweeks[both["season"] == "2025/26"].max() == 6
    assert (gameweeks == 1).sum() == 4  # two matches per round, two seasons


def test_gameweeks_align_to_the_original_row_order():
    """Assignment goes back to rows, not to positions - the frame is not date-sorted."""
    matches = season_frame(team_count=4)
    shuffled = matches.sample(frac=1, random_state=0)

    gameweeks = assign_gameweeks(shuffled)

    assert list(gameweeks.index) == list(shuffled.index)
    for index in shuffled.index:
        # The same fixture must get the same round however the frame is ordered.
        assert gameweeks.at[index] == assign_gameweeks(matches).at[index]


def test_missing_columns_raise_rather_than_guess():
    with pytest.raises(ValueError, match="home_team"):
        assign_gameweeks(pd.DataFrame({"season": ["2025/26"], "date": [pd.Timestamp.now()]}))


def test_two_clubs_playing_a_game_in_hand_do_not_break_the_count():
    """A midweek fixture brings two clubs a round ahead. They must not be pulled back."""
    teams = ["A", "B", "C", "D"]
    rows = [
        # Round 1 for everyone.
        ("2025-08-15", "A", "B"),
        ("2025-08-15", "C", "D"),
        # A and C play their second match early, before anyone else's round 2.
        ("2025-08-18", "A", "C"),
        # Round 2 for B and D.
        ("2025-08-22", "B", "D"),
    ]
    matches = pd.DataFrame(
        [
            {"season": "2025/26", "date": pd.Timestamp(d), "home_team": h, "away_team": a}
            for d, h, a in rows
        ]
    )

    gameweeks = list(assign_gameweeks(matches))

    assert gameweeks[:2] == [1, 1]
    assert gameweeks[2] == 2  # A and C's second match
    assert gameweeks[3] == 2  # B and D's second, despite being played later
    assert set(teams) == {"A", "B", "C", "D"}
