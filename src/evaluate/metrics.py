"""Scoring rules for probabilistic match predictions.

**Ranked Probability Score is the headline metric**, not accuracy. Football outcomes are
ordered - home win, draw, away win - and RPS is the only common metric that knows it.
Predicting a draw when the home side wins is a smaller error than predicting an away win,
and accuracy cannot express that. Lower is better; 0 is perfect.

Accuracy is reported alongside because it is intuitive, but it is a poor guide here: a
model that always predicts the favourite scores well on accuracy while being useless for
anything that depends on the probabilities being right.
"""

from __future__ import annotations

import numpy as np

# Outcome order matters for RPS: these are ranked, not categorical.
OUTCOMES = ("H", "D", "A")


def to_onehot(results) -> np.ndarray:
    """Actual outcomes as an (n, 3) indicator array in H/D/A order."""
    lookup = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    onehot = np.zeros((len(results), len(OUTCOMES)))
    for row, result in enumerate(results):
        onehot[row, lookup[result]] = 1.0
    return onehot


def ranked_probability_score(probabilities: np.ndarray, results) -> float:
    """Mean RPS over matches. Lower is better.

    For ordered outcomes this compares *cumulative* distributions, which is what makes
    a near-miss cost less than a wrong end of the scale.
    """
    predicted = np.asarray(probabilities, dtype=float)
    actual = to_onehot(results)

    cumulative_predicted = np.cumsum(predicted, axis=1)
    cumulative_actual = np.cumsum(actual, axis=1)

    # Only the first r-1 terms are free; the last is 1 by construction.
    squared = (cumulative_predicted[:, :-1] - cumulative_actual[:, :-1]) ** 2
    return float(squared.sum(axis=1).mean() / (len(OUTCOMES) - 1))


def log_loss(probabilities: np.ndarray, results, epsilon: float = 1e-15) -> float:
    """Mean negative log likelihood of the actual outcomes. Lower is better.

    Punishes confident mistakes far more harshly than RPS, which makes it the better
    check on whether a model is overstating its certainty.
    """
    predicted = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    actual = to_onehot(results)
    return float(-np.log((predicted * actual).sum(axis=1)).mean())


def accuracy(probabilities: np.ndarray, results) -> float:
    """Share of matches whose most likely outcome was the one that happened."""
    predicted = np.asarray(probabilities, dtype=float)
    chosen = [OUTCOMES[index] for index in predicted.argmax(axis=1)]
    return float(np.mean([a == b for a, b in zip(chosen, results, strict=True)]))


def implied_probabilities(home_odds, draw_odds, away_odds) -> np.ndarray:
    """Bookmaker decimal odds converted to probabilities, with the margin removed.

    Raw reciprocals sum to more than 1 - the overround is the bookmaker's margin.
    Normalising to 1 is the standard, if slightly crude, way to recover a fair line;
    it assumes the margin is spread proportionally across outcomes.
    """
    raw = np.column_stack(
        [
            1 / np.asarray(home_odds, dtype=float),
            1 / np.asarray(draw_odds, dtype=float),
            1 / np.asarray(away_odds, dtype=float),
        ]
    )
    return raw / raw.sum(axis=1, keepdims=True)


def score_all(probabilities: np.ndarray, results) -> dict[str, float]:
    """Every metric at once, for one set of predictions."""
    return {
        "rps": ranked_probability_score(probabilities, results),
        "log_loss": log_loss(probabilities, results),
        "accuracy": accuracy(probabilities, results),
    }
