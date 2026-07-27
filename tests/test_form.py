"""Tests for rolling form, Elo and rest days.

The important test here is `test_a_match_cannot_influence_its_own_features`. Every other
check is a detail; that one is the rule the whole phase exists to obey, and it is tested
by *changing a result* rather than by inspecting column names - the only way to catch a
window that quietly includes the match it describes.
"""

import pandas as pd
import pytest

from src.features.form import (
    ELO_START,
    add_elo,
    add_rest_days,
    add_rolling,
    build_team_matches,
    team_match_table,
)


def make_matches(results) -> pd.DataFrame:
    """A fixture list from (date, home, away, home_goals, away_goals) tuples."""
    rows = []
    for date, home, away, home_goals, away_goals in results:
        rows.append(
            {
                "match_id": f"{date}_{home}_{away}".replace("-", ""),
                "season": "2019/20",
                "date": pd.Timestamp(date),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_shots": 10,
                "away_shots": 8,
                "home_shots_target": 5,
                "away_shots_target": 3,
                "home_corners": 6,
                "away_corners": 4,
            }
        )
    return pd.DataFrame(rows)


def make_understat(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "xg_home": 1.5,
            "xg_away": 1.1,
        }
    )


SEQUENCE = [
    ("2019-08-01", "Arsenal", "Chelsea", 2, 0),
    ("2019-08-08", "Chelsea", "Arsenal", 1, 1),
    ("2019-08-15", "Arsenal", "Chelsea", 0, 3),
    ("2019-08-22", "Chelsea", "Arsenal", 2, 2),
]


# ------------------------------------------------------------------ table shape


def test_each_fixture_becomes_two_rows():
    matches = make_matches(SEQUENCE)
    table = team_match_table(matches, make_understat(matches))

    assert len(table) == 2 * len(matches)
    assert table.groupby("match_id").size().eq(2).all()
    assert table.groupby("match_id")["is_home"].sum().eq(1).all()


def test_goals_are_oriented_per_team():
    matches = make_matches(SEQUENCE[:1])
    table = team_match_table(matches, make_understat(matches))

    arsenal = table[table["team"] == "Arsenal"].iloc[0]
    chelsea = table[table["team"] == "Chelsea"].iloc[0]

    assert (arsenal["goals_for"], arsenal["goals_against"]) == (2, 0)
    assert (chelsea["goals_for"], chelsea["goals_against"]) == (0, 2)


def test_points_follow_the_result():
    matches = make_matches(SEQUENCE)
    table = team_match_table(matches, make_understat(matches))

    first = table[table["match_id"] == table["match_id"].iloc[0]]
    assert set(first["points"]) == {3, 0}

    draw = table[table["goals_for"] == table["goals_against"]]
    assert (draw["points"] == 1).all()


# --------------------------------------------------------------- the leakage rule


def test_first_match_has_no_rolling_history():
    matches = make_matches(SEQUENCE)
    table = add_rolling(team_match_table(matches, make_understat(matches)))

    first = table.sort_values("date").iloc[0]
    assert pd.isna(first["goals_for_last5"])
    assert first["matches_played"] == 0


def test_rolling_average_excludes_the_current_match():
    """Arsenal score 2 then 0. Before match two the average must be 2, not 1."""
    matches = make_matches(SEQUENCE[:3])
    table = add_rolling(team_match_table(matches, make_understat(matches)))

    arsenal = table[table["team"] == "Arsenal"].sort_values("date")
    assert arsenal.iloc[1]["goals_for_last5"] == pytest.approx(2.0)
    assert arsenal.iloc[2]["goals_for_last5"] == pytest.approx(1.5)  # (2 + 1) / 2


def test_a_match_cannot_influence_its_own_features():
    """Change one match's score; its own features must not move.

    This is the check that catches a missing `.shift(1)`. A column-name audit cannot -
    the offending column has a perfectly innocent name.
    """
    matches = make_matches(SEQUENCE)
    understat = make_understat(matches)
    original = build_team_matches(matches, understat)

    altered_matches = matches.copy()
    altered_matches.loc[1, "home_goals"] = 9  # rewrite the second fixture
    altered = build_team_matches(altered_matches, understat)

    changed_id = matches.loc[1, "match_id"]
    feature_columns = [
        column
        for column in original.columns
        if column.endswith("_last5") or column in ("elo_before", "rest_days")
    ]

    before = original[original["match_id"] == changed_id].set_index("team")[feature_columns]
    after = altered[altered["match_id"] == changed_id].set_index("team")[feature_columns]

    pd.testing.assert_frame_equal(before, after)


def test_a_changed_result_does_move_later_features():
    """The mirror of the leakage test: history must actually propagate forwards.

    Without this, a feature that is always NaN would pass the leakage test trivially.
    """
    matches = make_matches(SEQUENCE)
    understat = make_understat(matches)
    original = build_team_matches(matches, understat)

    altered_matches = matches.copy()
    altered_matches.loc[1, "home_goals"] = 9
    altered = build_team_matches(altered_matches, understat)

    later_id = matches.loc[3, "match_id"]
    before = original[original["match_id"] == later_id].set_index("team")["goals_against_last5"]
    after = altered[altered["match_id"] == later_id].set_index("team")["goals_against_last5"]

    assert not before.equals(after)


# -------------------------------------------------------------------------- elo


def test_elo_starts_level():
    matches = make_matches(SEQUENCE[:1])
    table = add_elo(team_match_table(matches, make_understat(matches)))

    assert (table["elo_before"] == ELO_START).all()


def test_elo_rewards_the_winner_next_time():
    matches = make_matches(SEQUENCE[:2])
    table = add_elo(team_match_table(matches, make_understat(matches))).sort_values("date")

    second = table[table["match_id"] == matches.loc[1, "match_id"]].set_index("team")
    assert second.loc["Arsenal", "elo_before"] > ELO_START  # won the opener
    assert second.loc["Chelsea", "elo_before"] < ELO_START


def test_elo_is_zero_sum_within_a_match():
    matches = make_matches(SEQUENCE)
    table = add_elo(team_match_table(matches, make_understat(matches)))

    total = table.groupby("match_id")["elo_before"].sum()
    assert total.round(6).nunique() == 1  # two teams, ratings only ever swap points


def test_every_row_gets_an_elo():
    matches = make_matches(SEQUENCE)
    table = add_elo(team_match_table(matches, make_understat(matches)))

    assert table["elo_before"].notna().all()


# ------------------------------------------------------------------- rest days


def test_rest_days_measure_the_gap_since_the_previous_match():
    matches = make_matches(SEQUENCE)
    table = add_rest_days(team_match_table(matches, make_understat(matches)))

    arsenal = table[table["team"] == "Arsenal"].sort_values("date")
    assert pd.isna(arsenal.iloc[0]["rest_days"])
    assert arsenal.iloc[1]["rest_days"] == 7


def test_rolling_window_crosses_season_boundaries():
    """August round one should use the previous May, not start blind."""
    matches = make_matches(
        [("2020-05-01", "Arsenal", "Chelsea", 3, 0), ("2020-08-01", "Arsenal", "Chelsea", 0, 0)]
    )
    matches.loc[1, "season"] = "2020/21"
    table = add_rolling(team_match_table(matches, make_understat(matches)))

    arsenal = table[table["team"] == "Arsenal"].sort_values("date")
    assert arsenal.iloc[1]["goals_for_last5"] == pytest.approx(3.0)
    assert arsenal.iloc[1]["season_matches_played"] == 0  # but the season counter resets
