"""Gradient-boosted goals models, via AutoGluon.

Two regressions - home goals and away goals - over the whole feature table. AutoGluon
trains several tree ensembles and stacks them, which is the point: it explores model
families and hyperparameters that would otherwise be a phase of work on their own.

This is also the variant that can be handed the bookmaker's line. Set ``use_odds=True``
and the closing odds join the feature set, which measures how much the market knows that
our own features do not. It is deliberately a *separate* model, because a model trained
on odds cannot then be honestly compared against them.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from src.models.base import feature_columns

# AutoGluon is conversational by default and would bury the backtest output.
logging.getLogger("autogluon").setLevel(logging.ERROR)

# Squeezing more out of 380-2280 rows is not where the remaining gains are, and a long
# preset would make the six-fit walk-forward loop painful to iterate on.
DEFAULT_PRESET = "medium_quality"
DEFAULT_TIME_LIMIT = 60

GOAL_FLOOR = 0.05
GOAL_CEILING = 8.0


class GradientBoostingModel:
    """Two AutoGluon regressors, one per side's goals."""

    def __init__(
        self,
        use_odds: bool = False,
        preset: str = DEFAULT_PRESET,
        time_limit: int = DEFAULT_TIME_LIMIT,
        verbosity: int = 0,
    ) -> None:
        self.use_odds = use_odds
        self.preset = preset
        self.time_limit = time_limit
        self.verbosity = verbosity
        self.name = "gbm-with-odds" if use_odds else "gbm"

        self.columns: list[str] = []
        self._home_predictor = None
        self._away_predictor = None

    def _train_one(self, train: pd.DataFrame, target: str):
        from autogluon.tabular import TabularPredictor

        frame = train[[*self.columns, target]].copy()

        # A fresh directory per fit: the walk-forward loop refits six times per model,
        # and AutoGluon would otherwise reuse or clash over a previous run's artefacts.
        path = Path(mkdtemp(prefix="ag_"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = TabularPredictor(
                label=target,
                problem_type="regression",
                eval_metric="root_mean_squared_error",
                path=str(path),
                verbosity=self.verbosity,
            ).fit(frame, presets=self.preset, time_limit=self.time_limit)

        return predictor

    def fit(self, train: pd.DataFrame) -> None:
        self.columns = feature_columns(train, use_odds=self.use_odds)
        self._home_predictor = self._train_one(train, "home_goals")
        self._away_predictor = self._train_one(train, "away_goals")

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._home_predictor is None or self._away_predictor is None:
            raise RuntimeError("fit must be called before predict")

        frame = test[self.columns]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home = np.asarray(self._home_predictor.predict(frame), dtype=float)
            away = np.asarray(self._away_predictor.predict(frame), dtype=float)

        # A regressor has no notion that goals cannot be negative, so the floor is not
        # cosmetic - a negative lambda would make the Poisson conversion meaningless.
        return (
            np.clip(home, GOAL_FLOOR, GOAL_CEILING),
            np.clip(away, GOAL_FLOOR, GOAL_CEILING),
        )
