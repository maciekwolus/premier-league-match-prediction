"""Tests for feature-table assembly and its leakage guard."""

import pandas as pd

from src.features.build import FORBIDDEN, NO_DIFFERENCE, TEAM_FEATURES, validate


def make_features(rows=2660, **overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(rows)],
            "season": "2019/20",
            "home_elo_before": 1500.0,
            "away_elo_before": 1500.0,
            "home_squad_overall_mean": 75.0,
        }
    )
    for column, value in overrides.items():
        df[column] = value
    return df


def test_a_clean_table_has_no_problems():
    assert validate(make_features()) == []


def test_detects_a_post_match_column():
    """The named-column guard: cheap, and catches a careless passthrough."""
    problems = validate(make_features(home_shots=12))

    assert any("post-match columns" in p for p in problems)
    assert any("home_shots" in p for p in problems)


def test_detects_half_time_score_leaking_in():
    problems = validate(make_features(ht_home_goals=1))
    assert any("ht_home_goals" in p for p in problems)


def test_detects_match_xg_leaking_in():
    """Match xG is a post-match measurement; only rolling xG over past matches is fair."""
    problems = validate(make_features(xg_home=1.8))
    assert any("xg_home" in p for p in problems)


def test_detects_wrong_row_count():
    problems = validate(make_features(rows=100))
    assert any("100 rows" in p for p in problems)


def test_detects_duplicate_matches():
    df = make_features()
    df.loc[df.index[1], "match_id"] = df.loc[df.index[0], "match_id"]
    assert any("duplicate match_id" in p for p in validate(df))


def test_detects_missing_elo():
    df = make_features()
    df.loc[df.index[0], "home_elo_before"] = None
    assert any("null values in home_elo_before" in p for p in validate(df))


def test_detects_mostly_missing_squad_ratings():
    df = make_features()
    df.loc[df.index[: int(0.5 * len(df))], "home_squad_overall_mean"] = None
    assert any("squad ratings" in p for p in validate(df))


def test_forbidden_list_covers_every_post_match_statistic():
    """A new statistic added to matches.parquet must be classified, not forgotten."""
    post_match = {
        "home_shots",
        "away_shots",
        "home_shots_target",
        "away_shots_target",
        "home_corners",
        "away_corners",
        "home_yellows",
        "away_yellows",
        "home_reds",
        "away_reds",
        "home_fouls",
        "away_fouls",
        "ht_home_goals",
        "ht_away_goals",
        "ht_result",
        "xg_home",
        "xg_away",
    }
    assert post_match <= set(FORBIDDEN)


def test_context_features_have_no_difference_column():
    """ "Matches played" is context; a home-minus-away difference of it means nothing."""
    assert NO_DIFFERENCE <= set(TEAM_FEATURES)
