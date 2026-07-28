"""The Dixon-Coles / Maher goals model, fitted by maximum likelihood.

This is the baseline every richer model has to beat, and it is deliberately blind: it
sees results and nothing else. No form, no squad ratings, no odds. Each team carries an
attack strength and a defence strength, one global home advantage covers the rest, and
the goals each side scores are Poisson counts around

    log(lambda_home) = attack[home] + defence[away] + home_advantage
    log(lambda_away) = attack[away] + defence[home]

Its value is as a floor. A model fed 90 engineered features that cannot beat 57
parameters learned from scorelines alone is not adding information, it is adding noise -
and that is far easier to see against this than against a coin flip.

The low-score correction Dixon and Coles are best known for lives in
``score_matrix.py``, not here, because it acts on the scoreline distribution rather than
on the expected goals. This module fits the strengths; that one shapes the matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Exponential time decay, in units of 1/day. 0.0018 puts the half-life at just under a
# year, so a match from two seasons ago counts about a quarter of one from last month.
# Squads turn over every summer, which is what the decay is really modelling - but decay
# too fast and the fit is starved, since a single season gives a team only 38 matches.
DEFAULT_XI = 0.0018

# Attack and defence are only identified up to a constant: add c to every attack and
# subtract it from every defence and the likelihood is unchanged. Pinning the attack
# strengths to sum to zero fixes that. A quadratic penalty is used rather than
# eliminating a parameter because it keeps the problem smooth and every team symmetric -
# no arbitrary reference club. The weight is large enough that the constraint binds to
# numerical precision and small enough not to distort the curvature L-BFGS-B relies on.
SUM_TO_ZERO_PENALTY = 1e4

# A lambda of zero is a likelihood of zero for any goal scored, and the matrix builder
# would rather not be handed one. Nothing real gets near this floor.
MIN_LAMBDA = 0.05


class DixonColesModel:
    """Team attack and defence strengths from results alone.

    ``xi`` is the time-decay rate in 1/days; pass 0 to weight every training match
    equally, which is worse but occasionally useful for comparison.
    """

    name = "dixon-coles"

    def __init__(self, xi: float = DEFAULT_XI, max_iterations: int = 500) -> None:
        self.xi = xi
        self.max_iterations = max_iterations
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage: float = 0.0
        # What an unseen team inherits. Held as attributes rather than recomputed at
        # predict time so the fallback is inspectable after a fit.
        self.mean_attack: float = 0.0
        self.mean_defence: float = 0.0

    def fit(self, train: pd.DataFrame) -> None:
        """Maximise the weighted Poisson likelihood over the training matches."""
        matches = train.loc[
            :, ["home_team", "away_team", "home_goals", "away_goals", "date"]
        ].dropna()

        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {team: position for position, team in enumerate(self.teams)}
        count = len(self.teams)

        home = matches["home_team"].map(index).to_numpy()
        away = matches["away_team"].map(index).to_numpy()
        home_goals = matches["home_goals"].to_numpy(dtype=float)
        away_goals = matches["away_goals"].to_numpy(dtype=float)
        weights = self._weights(matches["date"])

        # Attacks start at zero and defences at the log of the mean goals per team-match,
        # so the very first evaluation already sits at roughly the right overall scoring
        # rate and the optimiser only has to find the differences between teams.
        mean_goals = max(float(np.concatenate([home_goals, away_goals]).mean()), 1e-3)
        start = np.concatenate([np.zeros(count), np.full(count, np.log(mean_goals)), [0.0]])

        result = minimize(
            self._negative_log_likelihood,
            start,
            args=(home, away, home_goals, away_goals, weights, count),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations},
        )

        attack = result.x[:count]
        defence = result.x[count : 2 * count]

        self.attack = dict(zip(self.teams, attack, strict=True))
        self.defence = dict(zip(self.teams, defence, strict=True))
        self.home_advantage = float(result.x[-1])
        self.mean_attack = float(attack.mean())
        self.mean_defence = float(defence.mean())

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for each fixture, one entry per row of ``test``."""
        attack_home = self._lookup(test["home_team"], self.attack, self.mean_attack)
        attack_away = self._lookup(test["away_team"], self.attack, self.mean_attack)
        defence_home = self._lookup(test["home_team"], self.defence, self.mean_defence)
        defence_away = self._lookup(test["away_team"], self.defence, self.mean_defence)

        lambda_home = np.exp(attack_home + defence_away + self.home_advantage)
        lambda_away = np.exp(attack_away + defence_home)

        return np.maximum(lambda_home, MIN_LAMBDA), np.maximum(lambda_away, MIN_LAMBDA)

    def _weights(self, dates: pd.Series) -> np.ndarray:
        """Exponential decay measured back from the most recent training match.

        Anchoring on the end of the training window rather than on the date being
        predicted means the weights do not change from fixture to fixture, so one fit
        serves a whole test season. Within a season that costs very little - the anchor
        moves by months, the half-life is a year - and it avoids refitting 380 times.
        """
        days = (pd.to_datetime(dates).max() - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
        return np.exp(-self.xi * days)

    @staticmethod
    def _negative_log_likelihood(
        parameters: np.ndarray,
        home: np.ndarray,
        away: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        count: int,
    ) -> tuple[float, np.ndarray]:
        """Weighted negative Poisson log likelihood and its gradient.

        The analytic gradient is worth the extra lines: without it L-BFGS-B has to
        finite-difference 57 parameters, which is 57 extra likelihood evaluations per
        step, and the whole backtest refits six times.
        """
        attack = parameters[:count]
        defence = parameters[count : 2 * count]
        home_advantage = parameters[-1]

        lambda_home = np.exp(attack[home] + defence[away] + home_advantage)
        lambda_away = np.exp(attack[away] + defence[home])

        # Constant terms in log(y!) are dropped - they do not depend on the parameters.
        log_likelihood = weights * (
            home_goals * np.log(lambda_home)
            - lambda_home
            + away_goals * np.log(lambda_away)
            - lambda_away
        )

        total = attack.sum()
        penalty = SUM_TO_ZERO_PENALTY * total**2

        # d/dtheta of (lambda - y) for a log link is (lambda - y) times d(log lambda)/dtheta,
        # and every derivative below is 1, so these residuals just need summing per team.
        residual_home = weights * (lambda_home - home_goals)
        residual_away = weights * (lambda_away - away_goals)

        gradient_attack = np.bincount(home, residual_home, count) + np.bincount(
            away, residual_away, count
        )
        gradient_defence = np.bincount(away, residual_home, count) + np.bincount(
            home, residual_away, count
        )
        gradient_attack = gradient_attack + 2 * SUM_TO_ZERO_PENALTY * total

        gradient = np.concatenate([gradient_attack, gradient_defence, [residual_home.sum()]])

        return float(-log_likelihood.sum() + penalty), gradient

    @staticmethod
    def _lookup(teams: pd.Series, strengths: dict[str, float], fallback: float) -> np.ndarray:
        """Fitted strengths for a column of team names, average for anyone unseen.

        Promoted clubs are the whole reason this exists: three arrive every August with
        no Premier League history, and a model that raised KeyError on them would be
        useless for exactly the fixtures people care about. Average is a generous guess -
        promoted sides are usually worse than average - but inventing a promotion penalty
        here would be a judgement the results have not earned.
        """
        return teams.map(strengths).fillna(fallback).to_numpy(dtype=float)
