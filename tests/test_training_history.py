"""Tests for the training table the prediction path fits on.

The season being played has results but no engineered features, so it cannot go in
``features.parquet``. Withholding it from training is what these are about: a club's
strength in October is not May's, and a promoted club has no parameter at all until its
own results supply one.

**The cutoff is the test that matters.** Appending every played match is safe only for a
round that has not been played, and two of the three ways to invoke the predictor
(``--replay``, ``--gameweek N``) predict rounds that have. Without the cutoff a replayed
round would be trained on its own result, the output would look entirely normal, and the
archive would be worthless.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from src.features.build import MATCH_COLUMNS, training_history


@pytest.fixture
def tables(tmp_path):
    """A complete season on disk as features, and a season in progress as matches.

    The feature table carries engineered columns; the match table does not, which is the
    whole shape of the problem.
    """
    complete = pd.DataFrame(
        {
            "match_id": ["2025_26_20250810_arsenal_chelsea", "2025_26_20250817_chelsea_arsenal"],
            "season": ["2025/26", "2025/26"],
            "date": pd.to_datetime(["2025-08-10", "2025-08-17"]),
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [2.0, 1.0],
            "away_goals": [1.0, 1.0],
            "home_squad_overall_mean": [80.5, 79.0],
            "away_squad_overall_mean": [79.0, 80.5],
            "home_elo_before": [1600.0, 1550.0],
            "away_elo_before": [1550.0, 1600.0],
            "is_promoted_home": [False, False],
            "is_promoted_away": [False, False],
        }
    )

    # Three rounds of the season in progress: two played, one still to come.
    in_progress = pd.DataFrame(
        {
            "match_id": [
                "2026_27_20260815_coventry_hull",
                "2026_27_20260822_hull_coventry",
                "2026_27_20260829_coventry_arsenal",
            ],
            "season": ["2026/27"] * 3,
            "date": pd.to_datetime(["2026-08-15", "2026-08-22", "2026-08-29"]),
            "home_team": ["Coventry", "Hull", "Coventry"],
            "away_team": ["Hull", "Coventry", "Arsenal"],
            "home_goals": [3.0, 0.0, None],
            "away_goals": [1.0, 2.0, None],
        }
    )

    features_path = tmp_path / "features.parquet"
    matches_path = tmp_path / "matches.parquet"
    complete.to_parquet(features_path, index=False)
    pd.concat([complete[MATCH_COLUMNS[:7]], in_progress], ignore_index=True).to_parquet(
        matches_path, index=False
    )
    return features_path, matches_path


def test_the_complete_seasons_are_always_there(tables):
    features_path, matches_path = tables
    history = training_history(features_path=features_path, matches_path=matches_path)

    assert "2025_26_20250810_arsenal_chelsea" in set(history["match_id"])
    assert set(history.columns) == set(pd.read_parquet(features_path).columns)


def test_a_played_match_the_feature_table_lacks_is_appended(tables):
    """The point of the whole thing: Coventry and Hull have no history at all, so without
    this they inherit the league average for the entire season."""
    features_path, matches_path = tables
    history = training_history(features_path=features_path, matches_path=matches_path)

    appended = history[history["season"] == "2026/27"]
    assert len(appended) == 2
    assert set(appended["home_team"]) == {"Coventry", "Hull"}


def test_an_unplayed_fixture_is_never_appended(tables):
    """A fixture with no score is not a training example, it is next week."""
    features_path, matches_path = tables
    history = training_history(features_path=features_path, matches_path=matches_path)

    assert "2026_27_20260829_coventry_arsenal" not in set(history["match_id"])
    assert history["home_goals"].notna().all()


def test_an_appended_row_carries_its_result_and_no_engineered_features(tables):
    """Results yes, squad quality and Elo no - there are no lineups for a season being
    played. The models treat those nulls as "no information" rather than as zero."""
    features_path, matches_path = tables
    history = training_history(features_path=features_path, matches_path=matches_path)

    row = history[history["match_id"] == "2026_27_20260815_coventry_hull"].iloc[0]

    assert (row["home_goals"], row["away_goals"]) == (3.0, 1.0)
    assert pd.isna(row["home_squad_overall_mean"])
    assert pd.isna(row["home_elo_before"])


def test_the_cutoff_withholds_the_round_being_predicted(tables):
    """The leakage guard, and the reason ``before`` is not optional in practice.

    ``--replay`` predicts the most recent *played* round, which now comes from the season
    in progress. Train on everything played and the model learns the result of the very
    round it is about to forecast - and nothing in the output would look wrong.
    """
    features_path, matches_path = tables

    history = training_history(
        before="2026-08-22", features_path=features_path, matches_path=matches_path
    )
    stored = set(history["match_id"])

    assert "2026_27_20260815_coventry_hull" in stored
    assert "2026_27_20260822_hull_coventry" not in stored


def test_the_cutoff_excludes_a_match_kicking_off_at_that_moment(tables):
    """Strictly before, so a fixture in the round being predicted cannot slip in on an
    equal timestamp."""
    features_path, matches_path = tables

    history = training_history(
        before="2026-08-15", features_path=features_path, matches_path=matches_path
    )

    assert not (history["season"] == "2026/27").any()


def test_nothing_to_append_leaves_the_table_alone(tables):
    """Once the season ends and the features are rebuilt, this quietly becomes a no-op -
    which is why it keys on what the feature table holds rather than on a season name."""
    features_path, _ = tables

    history = training_history(features_path=features_path, matches_path=features_path)

    pd.testing.assert_frame_equal(history, pd.read_parquet(features_path))


def test_a_missing_match_table_is_not_fatal(tables, tmp_path):
    """A fresh clone has features and nothing else; that should still train."""
    features_path, _ = tables

    history = training_history(
        features_path=features_path, matches_path=tmp_path / "absent.parquet"
    )

    assert len(history) == 2


def test_the_result_stays_in_date_order(tables):
    features_path, matches_path = tables
    history = training_history(features_path=features_path, matches_path=matches_path)

    assert history["date"].is_monotonic_increasing


def test_a_boolean_feature_is_not_widened_into_being_dropped():
    """The regression that would have cost two features without a word.

    ``models.base.feature_columns`` admits a column only if it is numeric or boolean.
    Concatenating nulls onto a ``bool`` column turns it into ``object``, which is neither
    - so ``is_promoted_home`` and ``is_promoted_away`` would have vanished from the GLM
    and the GBM, with no error and no warning, leaving them two columns short of the set
    they were measured on.
    """
    import pandas as pd

    from src.models.base import feature_columns

    # built inline rather than from the fixture so the assertion names what it relies on
    complete = pd.DataFrame(
        {
            "match_id": ["a"],
            "season": ["2025/26"],
            "date": pd.to_datetime(["2025-08-10"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [2.0],
            "away_goals": [1.0],
            "is_promoted_home": [False],
        }
    )
    played = pd.DataFrame(
        {
            "match_id": ["b"],
            "season": ["2026/27"],
            "date": pd.to_datetime(["2026-08-15"]),
            "home_team": ["Coventry"],
            "away_team": ["Hull"],
            "home_goals": [3.0],
            "away_goals": [1.0],
        }
    )

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        complete.to_parquet(root / "f.parquet", index=False)
        pd.concat([complete, played], ignore_index=True).to_parquet(root / "m.parquet", index=False)

        history = training_history(
            features_path=root / "f.parquet", matches_path=root / "m.parquet"
        )

    assert pd.api.types.is_bool_dtype(history["is_promoted_home"])
    assert "is_promoted_home" in feature_columns(history)
    assert feature_columns(history) == feature_columns(complete)
