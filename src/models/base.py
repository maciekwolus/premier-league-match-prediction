"""The contract every model implements.

Models differ enormously inside - one is a hand-rolled likelihood, another an AutoML
ensemble - but they all answer the same question: given matches whose results we know,
how many goals will each side score in these later ones?

Keeping that interface narrow is what makes the comparison fair. Nothing downstream
knows which model produced a prediction, so scoring, the scoreline matrix and the report
are identical for all of them.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class GoalsModel(Protocol):
    """Predicts expected goals for each side of a fixture."""

    name: str

    def fit(self, train: pd.DataFrame) -> None:
        """Learn from matches whose results are known.

        ``train`` is a slice of the feature table and always precedes the test set in
        time - the harness guarantees it, so a model never has to think about leakage
        across the split.
        """

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (expected home goals, expected away goals), one entry per row."""


# Columns a model must never train on. The targets themselves, the identifiers, and the
# bookmaker line - which is the benchmark, and is only fed to the deliberately separate
# odds variant.
NON_FEATURES = (
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)

ODDS_COLUMNS = (
    "odds_close_home",
    "odds_close_draw",
    "odds_close_away",
    "odds_close_avg_home",
    "odds_close_avg_draw",
    "odds_close_avg_away",
)


def feature_columns(features: pd.DataFrame, use_odds: bool = False) -> list[str]:
    """The numeric columns a model may train on.

    Odds are excluded by default. They are the benchmark: a model that trains on them
    largely learns to restate the bookmaker's opinion, so measuring yourself against
    them afterwards would be circular.
    """
    excluded = set(NON_FEATURES)
    if not use_odds:
        excluded |= set(ODDS_COLUMNS)

    def usable(column: str) -> bool:
        if column in excluded:
            return False
        values = features[column]
        return pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)

    return [column for column in features.columns if usable(column)]
