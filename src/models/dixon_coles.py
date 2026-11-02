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
#
# Re-swept after the season in progress joined the training table, since that adds matches
# newer than anything the rate was first chosen against, and again once PRIOR_STRENGTH
# arrived. It is still the optimum and the curve around it is smooth, so leave it alone:
# 0.0009 gives 0.20302, 0.0014 0.20284, 0.0018 0.20281, 0.0024 0.20291, 0.0030 0.20311, and
# no decay at all 0.20414, walk-forward over the usual 2,280 matches. A single minimum on a
# smooth curve is a finding; chasing the fourth decimal across a jagged one would just be
# fitting the backtest.
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

# How hard a club with almost no matches is held to the league mean, and how many
# time-weighted matches it takes for that hold to fade to nothing.
#
# Maximum likelihood has no answer for a club seen once. One that fails to score has a
# best-fit attack of minus infinity, and one that keeps a clean sheet the same for its
# defence, so its next opponent lands on MIN_LAMBDA. Leeds v Fulham in round 2 of 2020/21
# came out 10.27 goals to 0.05 and a top scoreline of 10-0 this way, and seven fixtures of
# the 2,280 backtested sat on the floor, one or more in five seasons of six. That needs only
# one debutant blanking in round 1, so it is an annual event rather than a freak.
#
# **The prior fades out, and that is what stops it being a hedge.** Applied to every club, a
# ridge scores better - 0.2024 at strength 8 - by pulling established sides together too,
# and 1-1 between them climbs from 53.7% to 60.6%. Here a club past 19 weighted matches
# feels nothing and the objective is the plain likelihood. Measured walk-forward, RPS moves
# 0.2033 to 0.2028 with both tails gone (peak 10.27 to 4.64, floor 0.05 to 0.35), 1-1
# between established clubs stays at 53.8% with their spread no narrower, and the gain is
# where it should be: 0.2583 to 0.1990 where a club has been seen under two weighted
# matches, 0.2029 to 0.2028 where both are past 19. Nineteen is half a season, the window
# RECENT_MATCHES settled on for the same reason.
#
# **The strength is not chosen by RPS, which is flat from 5 to 12.** Past that plateau's
# start the score stops improving while 1-1 on the thinnest fixtures keeps rising, 43.8% to
# 52.1%, so anything above 5 is a hedge that pays nothing. At 5 that share sits beside the
# market's own 41.7%. Do not lower it on the strength of the peak either: 2 clears both
# tails and still gives Leeds v Fulham 88% where the closing line had 57%, against 66% here.
PRIOR_STRENGTH = 5.0
PRIOR_FADES_BY = 19.0


def strength_prior(strengths: np.ndarray, precision: np.ndarray) -> tuple[float, np.ndarray]:
    """A Gaussian penalty around the league mean for one block of strengths, with gradient.

    The mean moves with the strengths, so every club carries a share of the gradient of
    every other's deviation, which is the ``- pull.mean()``. It vanishes only when every
    club is held equally hard, and here they are not.
    """
    deviation = strengths - strengths.mean()
    pull = precision * deviation
    return 0.5 * float((pull * deviation).sum()), pull - pull.mean()


class DixonColesModel:
    """Team attack and defence strengths from results alone.

    ``xi`` is the time-decay rate in 1/days; pass 0 to weight every training match
    equally, which is worse but occasionally useful for comparison. ``prior_strength``
    likewise: pass 0 for the plain likelihood, which is what produced the 10-0.
    """

    name = "dixon-coles"

    def __init__(
        self,
        xi: float = DEFAULT_XI,
        max_iterations: int = 500,
        prior_strength: float = PRIOR_STRENGTH,
    ) -> None:
        self.xi = xi
        self.max_iterations = max_iterations
        self.prior_strength = prior_strength
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
        precision = self._prior_precision(home, away, weights, count)

        # Attacks start at zero and defences at the log of the mean goals per team-match,
        # so the very first evaluation already sits at roughly the right overall scoring
        # rate and the optimiser only has to find the differences between teams.
        mean_goals = max(float(np.concatenate([home_goals, away_goals]).mean()), 1e-3)
        start = np.concatenate([np.zeros(count), np.full(count, np.log(mean_goals)), [0.0]])

        result = minimize(
            self._negative_log_likelihood,
            start,
            args=(home, away, home_goals, away_goals, weights, count, precision),
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

    def _prior_precision(
        self, home: np.ndarray, away: np.ndarray, weights: np.ndarray, count: int
    ) -> np.ndarray:
        """How hard each club is held to the league mean, given how much it has been seen.

        Counted in time-weighted matches rather than raw ones, because the question is how
        much the fit actually leans on them: a club back after two seasons away has plenty
        of rows and very little evidence.
        """
        seen = np.bincount(home, weights, count) + np.bincount(away, weights, count)
        return self.prior_strength * np.clip(1 - seen / PRIOR_FADES_BY, 0, None)

    @staticmethod
    def _negative_log_likelihood(
        parameters: np.ndarray,
        home: np.ndarray,
        away: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        count: int,
        precision: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Weighted negative Poisson log likelihood and its gradient, plus the prior.

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
        prior_attack, pull_attack = strength_prior(attack, precision)
        prior_defence, pull_defence = strength_prior(defence, precision)
        penalty = SUM_TO_ZERO_PENALTY * total**2 + prior_attack + prior_defence

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
        gradient_attack = gradient_attack + 2 * SUM_TO_ZERO_PENALTY * total + pull_attack
        gradient_defence = gradient_defence + pull_defence

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
