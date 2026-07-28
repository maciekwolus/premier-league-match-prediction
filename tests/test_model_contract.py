"""Contract tests every model must pass, whatever is inside it.

These run against all model families at once. A model that fails here is unsafe to
compare, however good its score looks - and the scores are precisely what a leaking
model gets wrong in a flattering direction.

AutoGluon is excluded: it needs a real training run per fit, which is far too slow for
a unit test. It is exercised through the full backtest instead.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import EloModel, LeagueAverageModel, TeamAverageModel
from src.models.dixon_coles import DixonColesModel
from src.models.poisson_glm import PoissonRegressionModel

MODELS = [
    pytest.param(LeagueAverageModel, id="league-average"),
    pytest.param(TeamAverageModel, id="team-average"),
    pytest.param(EloModel, id="elo"),
    pytest.param(DixonColesModel, id="dixon-coles"),
    pytest.param(PoissonRegressionModel, id="poisson-glm"),
]

TEAMS = [f"Team {index:02d}" for index in range(6)]


def make_features(seed: int = 0, seasons=("2019/20", "2020/21")) -> pd.DataFrame:
    """A miniature feature table with the columns every model relies on."""
    rng = np.random.default_rng(seed)
    rows = []
    date = pd.Timestamp("2019-08-01")

    for season in seasons:
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                date += pd.Timedelta(days=1)
                rows.append(
                    {
                        "match_id": f"{season}_{date:%Y%m%d}_{home}_{away}".replace(" ", ""),
                        "season": season,
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": int(rng.poisson(1.6)),
                        "away_goals": int(rng.poisson(1.2)),
                        "home_elo_before": 1500 + rng.normal(0, 60),
                        "away_elo_before": 1500 + rng.normal(0, 60),
                        "diff_squad_overall_mean": rng.normal(0, 4),
                        "home_points_last5": rng.uniform(0, 3),
                        "away_points_last5": rng.uniform(0, 3),
                        "home_xg_for_last5": rng.uniform(0.5, 2.5),
                        "away_xg_for_last5": rng.uniform(0.5, 2.5),
                        "odds_close_avg_home": 2.5,
                        "odds_close_avg_draw": 3.4,
                        "odds_close_avg_away": 2.9,
                    }
                )

    df = pd.DataFrame(rows)
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"],
        "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A"),
    )
    return df


def split(features: pd.DataFrame):
    seasons = sorted(features["season"].unique())
    return (
        features[features["season"] == seasons[0]],
        features[features["season"] == seasons[1]],
    )


@pytest.mark.parametrize("model_class", MODELS)
def test_predict_returns_one_lambda_pair_per_row(model_class):
    train, test = split(make_features())
    model = model_class()
    model.fit(train)
    home, away = model.predict(test)

    assert len(home) == len(test)
    assert len(away) == len(test)


@pytest.mark.parametrize("model_class", MODELS)
def test_lambdas_are_positive_and_finite(model_class):
    """A non-positive or infinite lambda makes the Poisson conversion meaningless."""
    train, test = split(make_features())
    model = model_class()
    model.fit(train)
    home, away = model.predict(test)

    assert np.isfinite(home).all() and np.isfinite(away).all()
    assert (home > 0).all() and (away > 0).all()


@pytest.mark.parametrize("model_class", MODELS)
def test_lambdas_are_plausible_football_scores(model_class):
    """Expected goals outside roughly 0-6 means the fit has gone wrong, not that a
    team is very good."""
    train, test = split(make_features())
    model = model_class()
    model.fit(train)
    home, away = model.predict(test)

    assert home.mean() < 6 and away.mean() < 6
    assert home.mean() > 0.3 and away.mean() > 0.3


@pytest.mark.parametrize("model_class", MODELS)
def test_predictions_ignore_the_test_results(model_class):
    """**The leakage test.** Rewrite every test result; predictions must not move.

    A model that consults the outcome it is predicting scores brilliantly here and
    uselessly on Saturday. Nothing else in the suite catches it, because a leaking model
    produces perfectly well-formed output.
    """
    features = make_features()
    train, test = split(features)

    model = model_class()
    model.fit(train)
    home, away = model.predict(test)

    corrupted = test.copy()
    corrupted["home_goals"] = 9
    corrupted["away_goals"] = 0
    corrupted["result"] = "H"

    corrupted_home, corrupted_away = model.predict(corrupted)

    np.testing.assert_allclose(home, corrupted_home)
    np.testing.assert_allclose(away, corrupted_away)


@pytest.mark.parametrize("model_class", MODELS)
def test_promoted_teams_do_not_crash(model_class):
    """Three clubs each season have never been seen before. They must not raise."""
    features = make_features()
    train, test = split(features)

    newcomers = test.copy()
    newcomers["home_team"] = newcomers["home_team"].replace(TEAMS[0], "Newly Promoted")

    model = model_class()
    model.fit(train)
    home, away = model.predict(newcomers)

    assert np.isfinite(home).all() and np.isfinite(away).all()


@pytest.mark.parametrize("model_class", MODELS)
def test_models_expose_a_name(model_class):
    assert isinstance(model_class().name, str)
    assert model_class().name


@pytest.mark.parametrize("model_class", MODELS)
def test_refitting_is_deterministic(model_class):
    """The walk-forward loop refits six times; a model that wanders between identical
    fits would make the comparison noise rather than signal."""
    train, test = split(make_features())

    first = model_class()
    first.fit(train)
    second = model_class()
    second.fit(train)

    np.testing.assert_allclose(first.predict(test)[0], second.predict(test)[0], rtol=1e-6)
