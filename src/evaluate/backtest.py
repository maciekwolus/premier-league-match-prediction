"""Walk-forward backtesting.

Train on seasons 1..n, test on n+1, roll forward. **Never a random split** - a random
split lets a model learn from matches that had not happened yet, which inflates every
score and tells you nothing about Saturday.

The first season can only ever be training data. That costs 380 matches of evaluation
and is the honest price of the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.metrics import implied_probabilities, score_all
from src.models.base import GoalsModel
from src.models.score_matrix import outcome_frame


def season_splits(features: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding train / single-season test pairs, in chronological order."""
    seasons = sorted(features["season"].unique())
    splits = []

    for index in range(1, len(seasons)):
        train = features[features["season"].isin(seasons[:index])]
        test = features[features["season"] == seasons[index]]
        splits.append((train, test))

    return splits


def backtest(model: GoalsModel, features: pd.DataFrame) -> pd.DataFrame:
    """Run one model across every walk-forward split.

    Returns one row per test match with the predicted goals and outcome probabilities,
    so scoring and the per-season breakdown are both derivable from a single object.
    """
    predictions = []

    for train, test in season_splits(features):
        model.fit(train)
        lambda_home, lambda_away = model.predict(test)

        probabilities = outcome_frame(lambda_home, lambda_away)
        predictions.append(
            pd.DataFrame(
                {
                    "match_id": test["match_id"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "result": test["result"].to_numpy(),
                    "lambda_home": lambda_home,
                    "lambda_away": lambda_away,
                    "prob_home": probabilities[:, 0],
                    "prob_draw": probabilities[:, 1],
                    "prob_away": probabilities[:, 2],
                }
            )
        )

    return pd.concat(predictions, ignore_index=True)


def score_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    probabilities = predictions[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    return score_all(probabilities, predictions["result"])


def score_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in predictions.groupby("season"):
        scores = score_predictions(group)
        rows.append({"season": season, "matches": len(group), **scores})
    return pd.DataFrame(rows).set_index("season")


def bookmaker_benchmark(features: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    """Score the closing line over exactly the matches a model was tested on.

    Comparing against the market on a different set of matches would be meaningless, so
    this deliberately restricts itself to the backtest's own test rows.
    """
    tested = features[features["match_id"].isin(predictions["match_id"])]
    tested = tested.set_index("match_id").loc[predictions["match_id"]].reset_index()

    probabilities = implied_probabilities(
        tested["odds_close_avg_home"],
        tested["odds_close_avg_draw"],
        tested["odds_close_avg_away"],
    )
    return score_all(probabilities, tested["result"])


def compare(results: dict[str, pd.DataFrame], features: pd.DataFrame) -> pd.DataFrame:
    """One table of every model's scores, with the bookmaker line as the last row."""
    rows = []

    for name, predictions in results.items():
        rows.append({"model": name, "matches": len(predictions), **score_predictions(predictions)})

    any_predictions = next(iter(results.values()))
    rows.append(
        {
            "model": "bookmaker (closing)",
            "matches": len(any_predictions),
            **bookmaker_benchmark(features, any_predictions),
        }
    )

    return pd.DataFrame(rows).set_index("model").sort_values("rps")


def calibration(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed home-win rate, bucketed.

    A model can score well on RPS while being systematically overconfident; this is
    where that shows up.
    """
    df = predictions.copy()
    df["bucket"] = pd.cut(df["prob_home"], np.linspace(0, 1, bins + 1))
    df["home_won"] = df["result"] == "H"

    return df.groupby("bucket", observed=True).agg(
        matches=("home_won", "size"),
        predicted=("prob_home", "mean"),
        observed=("home_won", "mean"),
    )
