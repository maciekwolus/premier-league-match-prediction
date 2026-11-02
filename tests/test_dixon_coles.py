"""The prior that holds a barely-seen club to the league mean.

The failure this guards against is quiet: a club fitted on one match produces well-formed
numbers, so nothing raises and the contract tests pass, and a 10-0 goes into an archive
that can never be corrected. Every check that matters here therefore runs both ways, with
the prior off as well as on, because a test that cannot fail without the fix proves nothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import (
    MIN_LAMBDA,
    PRIOR_FADES_BY,
    PRIOR_STRENGTH,
    DixonColesModel,
)
from src.models.dixon_coles_squad import SquadDixonColesModel
from src.models.score_matrix import MAX_GOALS, score_matrix, top_scorelines
from src.predict.gameweek import DEFAULT_MODEL, load_model

MODELS = [
    pytest.param(DixonColesModel, id="dixon-coles"),
    pytest.param(SquadDixonColesModel, id="dixon-coles-squad"),
]

CLUBS = [f"Club {index:02d}" for index in range(8)]
LAST_DAY = pd.Timestamp("2021-05-23")


def established_league(seed: int = 0, rounds: int = 3) -> pd.DataFrame:
    """Eight clubs, each with far more than PRIOR_FADES_BY recent matches."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "home_team": home,
            "away_team": away,
            "home_goals": int(rng.poisson(1.5)),
            "away_goals": int(rng.poisson(1.2)),
        }
        for _ in range(rounds)
        for home in CLUBS
        for away in CLUBS
        if home != away
    ]
    league = pd.DataFrame(rows)
    league["date"] = LAST_DAY - pd.to_timedelta(np.arange(len(league))[::-1], unit="D")
    return league


def debut(league: pd.DataFrame, *matches: tuple) -> pd.DataFrame:
    """The league plus each ``(home, away, home_goals, away_goals)``, played last."""
    rows = pd.DataFrame(matches, columns=["home_team", "away_team", "home_goals", "away_goals"])
    rows["date"] = LAST_DAY
    return pd.concat([league, rows], ignore_index=True)


def fixture(home: str, away: str) -> pd.DataFrame:
    return pd.DataFrame({"home_team": [home], "away_team": [away]})


def expected_goals(model, home: str, away: str) -> tuple[float, float]:
    lambda_home, lambda_away = model.predict(fixture(home, away))
    return float(lambda_home[0]), float(lambda_away[0])


# Leeds lost 4-3 at Liverpool and Fulham 3-0 at home to Arsenal, then met each other.
LEEDS_AND_FULHAM = (
    (CLUBS[0], "Leeds", 4, 3),
    ("Fulham", CLUBS[1], 0, 3),
)


@pytest.mark.parametrize("model_class", MODELS)
def test_two_debutants_meeting_is_not_a_ten_nil(model_class):
    """The reproduction: one match each, then each other."""
    train = debut(established_league(), *LEEDS_AND_FULHAM)

    unprotected = model_class(prior_strength=0)
    unprotected.fit(train)
    home, away = expected_goals(unprotected, "Leeds", "Fulham")
    assert home > 6
    assert away == pytest.approx(MIN_LAMBDA)

    model = model_class()
    model.fit(train)
    home, away = expected_goals(model, "Leeds", "Fulham")
    assert 1 < home < 4
    assert 0.3 < away < 3

    top_home, top_away, _ = top_scorelines(score_matrix(home, away), 1)[0]
    assert max(top_home, top_away) < MAX_GOALS / 2


@pytest.mark.parametrize("model_class", MODELS)
def test_one_clean_sheet_does_not_floor_the_next_opponent(model_class):
    """The commoner shape: it takes one debutant, not two.

    Seven backtested fixtures sat on MIN_LAMBDA, and only Leeds v Fulham was between two
    new clubs. A debutant's clean sheet alone drives its defence towards minus infinity,
    so an established side is told it will not score.
    """
    train = debut(established_league(), (CLUBS[0], "Sunderland", 0, 3))

    unprotected = model_class(prior_strength=0)
    unprotected.fit(train)
    assert expected_goals(unprotected, CLUBS[2], "Sunderland")[0] == pytest.approx(MIN_LAMBDA)

    model = model_class()
    model.fit(train)
    assert expected_goals(model, CLUBS[2], "Sunderland")[0] > 0.3


@pytest.mark.parametrize("model_class", MODELS)
def test_a_debutant_that_fails_to_score_keeps_a_finite_attack(model_class):
    """Without the prior the best-fit attack here is minus infinity."""
    train = debut(established_league(), ("Fulham", CLUBS[1], 0, 3))

    model = model_class()
    model.fit(train)

    assert abs(model.attack["Fulham"] - model.mean_attack) < 1


@pytest.mark.parametrize("model_class", MODELS)
def test_established_clubs_feel_nothing(model_class):
    """**What stops this being a hedge.** A prior on everyone scores better by dragging
    the whole league towards the mean, and 1-1 climbs with it. Past the fading point the
    objective has to be the plain likelihood, so the strengths must be identical."""
    train = established_league()

    plain = model_class(prior_strength=0)
    plain.fit(train)
    model = model_class()
    model.fit(train)

    assert model.attack == pytest.approx(plain.attack, abs=1e-12)
    assert model.defence == pytest.approx(plain.defence, abs=1e-12)


@pytest.mark.parametrize("model_class", MODELS)
def test_fixtures_between_established_clubs_barely_move_when_a_debutant_is_fitted(model_class):
    train = debut(established_league(), *LEEDS_AND_FULHAM)

    plain = model_class(prior_strength=0)
    plain.fit(train)
    model = model_class()
    model.fit(train)

    for home, away in [(CLUBS[2], CLUBS[3]), (CLUBS[4], CLUBS[5]), (CLUBS[6], CLUBS[7])]:
        np.testing.assert_allclose(
            expected_goals(model, home, away), expected_goals(plain, home, away), rtol=0.01
        )


@pytest.mark.parametrize("model_class", MODELS)
def test_a_debutants_results_still_count(model_class):
    """Shrinking towards the mean must not become assigning the mean: a club that won its
    first match has to come out above one that lost it, or the season in progress has
    stopped teaching the model anything about promoted sides."""
    train = debut(
        established_league(),
        ("Winners", CLUBS[0], 4, 0),
        ("Losers", CLUBS[1], 0, 4),
    )

    model = model_class()
    model.fit(train)

    assert model.attack["Winners"] > model.mean_attack > model.attack["Losers"]
    assert model.defence["Winners"] < model.mean_defence < model.defence["Losers"]


@pytest.mark.parametrize("model_class", MODELS)
def test_the_hold_loosens_as_a_club_plays(model_class):
    """A newcomer outscoring everyone should be believed more each match it keeps doing it."""
    league = established_league()
    gaps = []
    for played in (1, 5, 15):
        matches = [(("Newcomer", CLUBS[index % len(CLUBS)], 3, 0)) for index in range(played)]
        model = model_class()
        model.fit(debut(league, *matches))
        gaps.append(model.attack["Newcomer"] - model.mean_attack)

    assert gaps[0] < gaps[1] < gaps[2]


def test_precision_is_full_for_nobody_seen_and_gone_by_the_fading_point():
    model = DixonColesModel()
    home = np.array([0, 0, 1, 1])
    away = np.array([1, 2, 2, 2])
    # Club 0 appears twice, club 1 three times, club 2 three times.
    weights = np.array([0.0, 0.0, PRIOR_FADES_BY, PRIOR_FADES_BY / 2])

    precision = model._prior_precision(home, away, weights, 4)

    assert precision[0] == pytest.approx(PRIOR_STRENGTH)
    assert precision[1] == 0
    assert precision[2] == 0
    assert precision[3] == pytest.approx(PRIOR_STRENGTH)

    halfway = model._prior_precision(np.array([0]), np.array([1]), np.array([9.5]), 2)
    np.testing.assert_allclose(halfway, [PRIOR_STRENGTH / 2, PRIOR_STRENGTH / 2])


def numerical_gradient(objective, parameters: np.ndarray, step: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(parameters)
    for position in range(len(parameters)):
        shift = np.zeros_like(parameters)
        shift[position] = step
        gradient[position] = (objective(parameters + shift)[0] - objective(parameters - shift)[0]) / (
            2 * step
        )
    return gradient


def gradient_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    count, matches = 6, 40
    home = rng.integers(0, count, matches)
    away = (home + rng.integers(1, count, matches)) % count
    return {
        "home": home,
        "away": away,
        "home_goals": rng.poisson(1.5, matches).astype(float),
        "away_goals": rng.poisson(1.2, matches).astype(float),
        "weights": rng.uniform(0.2, 1, matches),
        "count": count,
        # Deliberately uneven: the gradient of the moving mean only matters when the clubs
        # are held with different strengths, so equal precisions would hide a missing term.
        "precision": rng.uniform(0, PRIOR_STRENGTH, count),
    }


def test_parent_gradient_matches_the_objective():
    """A wrong gradient does not raise. L-BFGS-B just stops somewhere that is not a minimum."""
    problem = gradient_problem()
    parameters = np.random.default_rng(1).normal(0, 0.3, 2 * problem["count"] + 1)
    parameters[: problem["count"]] -= parameters[: problem["count"]].mean()

    def objective(values):
        return DixonColesModel._negative_log_likelihood(values, **problem)

    np.testing.assert_allclose(
        objective(parameters)[1], numerical_gradient(objective, parameters), atol=1e-4
    )


def test_squad_gradient_matches_the_objective():
    """The squad model has its own likelihood, so it needs its own check: a prior added
    only to the parent would pass everything else here for the parent and never run."""
    problem = gradient_problem()
    rng = np.random.default_rng(2)
    parameters = rng.normal(0, 0.3, 2 * problem["count"] + 3)
    parameters[: problem["count"]] -= parameters[: problem["count"]].mean()
    matches = len(problem["home"])
    deltas = {"delta_home": rng.normal(0, 1, matches), "delta_away": rng.normal(0, 1, matches)}

    def objective(values):
        return SquadDixonColesModel._negative_log_likelihood(values, **problem, **deltas)

    np.testing.assert_allclose(
        objective(parameters)[1], numerical_gradient(objective, parameters), atol=1e-4
    )


def test_the_deployed_model_carries_the_prior():
    """A tested function is not a connected one: go through what the prediction path calls."""
    model = load_model(DEFAULT_MODEL)
    assert model.prior_strength == PRIOR_STRENGTH

    model.fit(debut(established_league(), *LEEDS_AND_FULHAM))
    assert expected_goals(model, "Leeds", "Fulham")[0] < 4
