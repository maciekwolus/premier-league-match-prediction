"""Baseline goals models - the floor every real model must beat.

Each one knows less than the next: league-wide scoring rates only, then each team's own
attack and defence, then the Elo rating already computed in the feature table. None of
them look at squad quality, form, or anything else Phase 5 built. If a model with real
features cannot beat these, its complexity is not earning its keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# score_matrix.py feeds these lambdas straight into a Poisson pmf, which is undefined at
# zero. A team that never scored in training, or a fixture with an extreme Elo gap, would
# otherwise ask for a probability-zero rate.
GOAL_FLOOR = 0.05

# Matches features/form.py's ELO_START - a fixture with no rating history (or a promoted
# club) is assumed exactly average, not advantaged either way.
ELO_START = 1500.0


class LeagueAverageModel:
    """Predicts the training set's mean home and away goals for every fixture.

    The "knows nothing except that home teams score more" floor - no team identity, no
    form, just two numbers repeated for every match. Every smarter model must beat this.
    """

    name = "baseline-league-average"

    def __init__(self) -> None:
        self.lambda_home = GOAL_FLOOR
        self.lambda_away = GOAL_FLOOR

    def fit(self, train: pd.DataFrame) -> None:
        self.lambda_home = max(float(train["home_goals"].mean()), GOAL_FLOOR)
        self.lambda_away = max(float(train["away_goals"].mean()), GOAL_FLOOR)

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matches = len(test)
        return (
            np.full(matches, self.lambda_home, dtype=float),
            np.full(matches, self.lambda_away, dtype=float),
        )


class TeamAverageModel:
    """Combines each team's own attack and defence rate with the league average.

    ``lambda_home = league_home_average * home_attack_ratio * away_defence_ratio``, and
    symmetrically for the away side. This is the standard multiplicative shape behind
    most Poisson goal models (Maher 1982, and Dixon-Coles after it): a team's attack
    ratio is how much more it scores than the league average, its defence ratio how much
    more it concedes, and the two combine multiplicatively because who a team plays and
    how strong that team's attack is are treated as independent effects.

    A club unseen in training - a promotion - has no ratio to look up, so it falls back
    to 1.0: exactly the league average, for both attack and defence.
    """

    name = "baseline-team-average"

    def __init__(self) -> None:
        self.league_home_average = GOAL_FLOOR
        self.league_away_average = GOAL_FLOOR
        self.home_attack: dict[str, float] = {}
        self.home_defence: dict[str, float] = {}
        self.away_attack: dict[str, float] = {}
        self.away_defence: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> None:
        self.league_home_average = max(float(train["home_goals"].mean()), GOAL_FLOOR)
        self.league_away_average = max(float(train["away_goals"].mean()), GOAL_FLOOR)

        # A team's home record gives its home attack (goals scored) and home defence
        # (goals conceded, i.e. the away side's goals); its away record gives the mirror.
        home = train.groupby("home_team").agg(
            scored=("home_goals", "mean"), conceded=("away_goals", "mean")
        )
        away = train.groupby("away_team").agg(
            scored=("away_goals", "mean"), conceded=("home_goals", "mean")
        )

        self.home_attack = (home["scored"] / self.league_home_average).to_dict()
        self.home_defence = (home["conceded"] / self.league_away_average).to_dict()
        self.away_attack = (away["scored"] / self.league_away_average).to_dict()
        self.away_defence = (away["conceded"] / self.league_home_average).to_dict()

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home_attack = test["home_team"].map(self.home_attack).fillna(1.0).to_numpy(dtype=float)
        home_defence = test["home_team"].map(self.home_defence).fillna(1.0).to_numpy(dtype=float)
        away_attack = test["away_team"].map(self.away_attack).fillna(1.0).to_numpy(dtype=float)
        away_defence = test["away_team"].map(self.away_defence).fillna(1.0).to_numpy(dtype=float)

        lambda_home = self.league_home_average * home_attack * away_defence
        lambda_away = self.league_away_average * away_attack * home_defence

        return np.clip(lambda_home, GOAL_FLOOR, None), np.clip(lambda_away, GOAL_FLOOR, None)


def _elo_difference(df: pd.DataFrame) -> np.ndarray:
    """Home minus away Elo, defaulting missing ratings to average so the gap is 0."""
    if "home_elo_before" in df.columns:
        home = df["home_elo_before"].fillna(ELO_START)
    else:
        home = pd.Series(ELO_START, index=df.index)

    if "away_elo_before" in df.columns:
        away = df["away_elo_before"].fillna(ELO_START)
    else:
        away = pd.Series(ELO_START, index=df.index)

    return (home - away).to_numpy(dtype=float)


def _fit_poisson_glm(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``lambda = exp(intercept + slope * x)`` by maximum likelihood.

    A single-predictor Poisson regression has no closed form, but it is a small enough
    problem that L-BFGS on the negative log-likelihood converges reliably from a sane
    starting point - the log of the mean rate, with zero slope (i.e. Elo tells you
    nothing yet). ``log(y!)`` is dropped from the likelihood since it does not depend on
    the parameters being optimised.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_rate = max(float(y.mean()), GOAL_FLOOR)

    def negative_log_likelihood(params: np.ndarray) -> float:
        intercept, slope = params
        # Clipped in log-space: an optimiser step that sends the linear predictor far
        # from the data would otherwise overflow exp() into inf/nan and stall the fit.
        linear = np.clip(intercept + slope * x, -20.0, 20.0)
        rate = np.exp(linear)
        return float(np.sum(rate - y * linear))

    result = minimize(
        negative_log_likelihood,
        x0=np.array([np.log(mean_rate), 0.0]),
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])


class EloModel:
    """Maps the Elo rating gap already in the feature table onto expected goals.

    Elo is built to predict outcomes, not goal counts, so turning a rating gap into a
    scoring rate needs its own fit: two one-parameter Poisson regressions of goals on
    ``home_elo_before - away_elo_before``, one for the home side's goals and one for the
    away side's. A bigger home Elo advantage should raise home goals and lower away
    goals, so the two slopes are expected to land with opposite signs.
    """

    name = "baseline-elo"

    def __init__(self) -> None:
        self.home_params = (np.log(GOAL_FLOOR), 0.0)
        self.away_params = (np.log(GOAL_FLOOR), 0.0)

    def fit(self, train: pd.DataFrame) -> None:
        elo_diff = _elo_difference(train)
        self.home_params = _fit_poisson_glm(elo_diff, train["home_goals"].to_numpy(dtype=float))
        self.away_params = _fit_poisson_glm(elo_diff, train["away_goals"].to_numpy(dtype=float))

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        elo_diff = _elo_difference(test)

        home_intercept, home_slope = self.home_params
        away_intercept, away_slope = self.away_params

        lambda_home = np.exp(home_intercept + home_slope * elo_diff)
        lambda_away = np.exp(away_intercept + away_slope * elo_diff)

        return np.clip(lambda_home, GOAL_FLOOR, None), np.clip(lambda_away, GOAL_FLOOR, None)
