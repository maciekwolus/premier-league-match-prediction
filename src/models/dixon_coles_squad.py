"""Dixon-Coles with squad quality folded into the likelihood.

Plain Dixon-Coles reads results and nothing else, which makes it sharp but slow to react:
a club that has just lost its best striker looks exactly as strong as it did last month,
until enough matches accumulate to move its attack parameter.

The obvious fix - adding squad rating as a covariate - does not work, because a club's
attack strength and its average squad rating say almost the same thing, and the two
compete to explain the same variation. What FIFA ratings know that results do not is
narrower and more useful: **how this XI compares with the one that club usually fields.**

So the term added here is a deviation, not a level:

    delta = (this XI's mean overall - the club's usual mean) / scale

    log(lambda_home) = attack[h] + defence[a] + home_adv + b_att*delta_h - b_def*delta_a
    log(lambda_away) = attack[a] + defence[h]            + b_att*delta_a - b_def*delta_h

``b_att`` is how much a stronger-than-usual XI lifts a side's own scoring; ``b_def`` how
much a stronger-than-usual opponent suppresses it. Two parameters on top of the original
57, both fitted jointly rather than bolted on afterwards.

A club with no history gets its deviation measured against the league instead of against
itself, so a weak promoted side starts below average rather than at it - which is the
known weakness of the parent model, and the one place absolute rating is the right
quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.dixon_coles import (
    MIN_LAMBDA,
    SUM_TO_ZERO_PENALTY,
    DixonColesModel,
)

SQUAD_COLUMNS = ("home_squad_overall_mean", "away_squad_overall_mean")

# Deviations are divided by this many rating points before fitting, so the coefficients
# come out around order one and the optimiser sees a well-scaled problem. Roughly the
# spread of squad ratings within a club across a season.
SQUAD_SCALE = 2.0


class SquadDixonColesModel(DixonColesModel):
    """Team strengths from results, adjusted by how strong today's XI is."""

    name = "dixon-coles-squad"

    def __init__(self, xi: float | None = None, max_iterations: int = 500) -> None:
        super().__init__(**({} if xi is None else {"xi": xi}), max_iterations=max_iterations)
        self.club_baseline: dict[str, float] = {}
        self.league_baseline: float = 0.0
        self.squad_attack: float = 0.0
        self.squad_defence: float = 0.0

    # ------------------------------------------------------------------ deviations

    def _learn_baselines(self, train: pd.DataFrame) -> None:
        """Each club's usual squad rating, and the league's."""
        if not all(column in train.columns for column in SQUAD_COLUMNS):
            self.club_baseline, self.league_baseline = {}, 0.0
            return

        ratings = pd.concat(
            [
                train[["home_team", "home_squad_overall_mean"]].rename(
                    columns={"home_team": "team", "home_squad_overall_mean": "rating"}
                ),
                train[["away_team", "away_squad_overall_mean"]].rename(
                    columns={"away_team": "team", "away_squad_overall_mean": "rating"}
                ),
            ]
        ).dropna()

        if ratings.empty:
            self.club_baseline, self.league_baseline = {}, 0.0
            return

        self.club_baseline = ratings.groupby("team")["rating"].mean().to_dict()
        self.league_baseline = float(ratings["rating"].mean())

    def _deviations(self, frame: pd.DataFrame, side: str) -> np.ndarray:
        """How far this side's XI sits from what that club normally fields.

        Zero means "no information": either the ratings are missing for this match, or
        no baseline exists to compare against. Zero is the honest default because it
        leaves the parent model's behaviour untouched.
        """
        column = f"{side}_squad_overall_mean"
        if column not in frame.columns or not self.club_baseline:
            return np.zeros(len(frame))

        ratings = pd.to_numeric(frame[column], errors="coerce")
        teams = frame[f"{side}_team"]

        # A club we have never seen has no "usual" XI, so its deviation is measured
        # against the league. That is the one case where the absolute rating is what
        # matters, and it stops promoted sides defaulting to league-average strength.
        baselines = teams.map(self.club_baseline).fillna(self.league_baseline)

        deviation = (ratings - baselines) / SQUAD_SCALE
        return deviation.fillna(0.0).to_numpy(dtype=float)

    # ------------------------------------------------------------------ fitting

    def fit(self, train: pd.DataFrame) -> None:
        self._learn_baselines(train)

        columns = ["home_team", "away_team", "home_goals", "away_goals", "date"]
        matches = train.loc[:, columns + [c for c in SQUAD_COLUMNS if c in train.columns]].dropna(
            subset=columns
        )

        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {team: position for position, team in enumerate(self.teams)}
        count = len(self.teams)

        home = matches["home_team"].map(index).to_numpy()
        away = matches["away_team"].map(index).to_numpy()
        home_goals = matches["home_goals"].to_numpy(dtype=float)
        away_goals = matches["away_goals"].to_numpy(dtype=float)
        weights = self._weights(matches["date"])

        delta_home = self._deviations(matches, "home")
        delta_away = self._deviations(matches, "away")

        mean_goals = max(float(np.concatenate([home_goals, away_goals]).mean()), 1e-3)
        start = np.concatenate(
            [np.zeros(count), np.full(count, np.log(mean_goals)), [0.0, 0.0, 0.0]]
        )

        result = minimize(
            self._negative_log_likelihood,
            start,
            args=(home, away, home_goals, away_goals, weights, count, delta_home, delta_away),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations},
        )

        attack = result.x[:count]
        defence = result.x[count : 2 * count]

        self.attack = dict(zip(self.teams, attack, strict=True))
        self.defence = dict(zip(self.teams, defence, strict=True))
        self.home_advantage = float(result.x[-3])
        self.squad_attack = float(result.x[-2])
        self.squad_defence = float(result.x[-1])
        self.mean_attack = float(attack.mean())
        self.mean_defence = float(defence.mean())

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        attack_home = self._lookup(test["home_team"], self.attack, self.mean_attack)
        attack_away = self._lookup(test["away_team"], self.attack, self.mean_attack)
        defence_home = self._lookup(test["home_team"], self.defence, self.mean_defence)
        defence_away = self._lookup(test["away_team"], self.defence, self.mean_defence)

        delta_home = self._deviations(test, "home")
        delta_away = self._deviations(test, "away")

        lambda_home = np.exp(
            attack_home
            + defence_away
            + self.home_advantage
            + self.squad_attack * delta_home
            - self.squad_defence * delta_away
        )
        lambda_away = np.exp(
            attack_away
            + defence_home
            + self.squad_attack * delta_away
            - self.squad_defence * delta_home
        )

        return np.maximum(lambda_home, MIN_LAMBDA), np.maximum(lambda_away, MIN_LAMBDA)

    @staticmethod
    def _negative_log_likelihood(
        parameters: np.ndarray,
        home: np.ndarray,
        away: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        count: int,
        delta_home: np.ndarray,
        delta_away: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        attack = parameters[:count]
        defence = parameters[count : 2 * count]
        home_advantage, squad_attack, squad_defence = parameters[-3:]

        lambda_home = np.exp(
            attack[home]
            + defence[away]
            + home_advantage
            + squad_attack * delta_home
            - squad_defence * delta_away
        )
        lambda_away = np.exp(
            attack[away] + defence[home] + squad_attack * delta_away - squad_defence * delta_home
        )

        log_likelihood = weights * (
            home_goals * np.log(lambda_home)
            - lambda_home
            + away_goals * np.log(lambda_away)
            - lambda_away
        )

        total = attack.sum()
        penalty = SUM_TO_ZERO_PENALTY * total**2

        residual_home = weights * (lambda_home - home_goals)
        residual_away = weights * (lambda_away - away_goals)

        gradient_attack = np.bincount(home, residual_home, count) + np.bincount(
            away, residual_away, count
        )
        gradient_defence = np.bincount(away, residual_home, count) + np.bincount(
            home, residual_away, count
        )
        gradient_attack = gradient_attack + 2 * SUM_TO_ZERO_PENALTY * total

        # Each squad coefficient appears in both lambdas, with the sign flipping between
        # a side's own deviation and its opponent's.
        gradient_squad_attack = float(
            (residual_home * delta_home).sum() + (residual_away * delta_away).sum()
        )
        gradient_squad_defence = float(
            -(residual_home * delta_away).sum() - (residual_away * delta_home).sum()
        )

        gradient = np.concatenate(
            [
                gradient_attack,
                gradient_defence,
                [residual_home.sum(), gradient_squad_attack, gradient_squad_defence],
            ]
        )

        return float(-log_likelihood.sum() + penalty), gradient
