"""Tests for the scoring rules.

RPS decides which model wins, so an error here would silently pick the wrong one.
"""

import numpy as np
import pytest

from src.evaluate.metrics import (
    accuracy,
    implied_probabilities,
    log_loss,
    ranked_probability_score,
)

CERTAIN_HOME = np.array([[1.0, 0.0, 0.0]])
CERTAIN_AWAY = np.array([[0.0, 0.0, 1.0]])
CERTAIN_DRAW = np.array([[0.0, 1.0, 0.0]])
IGNORANT = np.array([[1 / 3, 1 / 3, 1 / 3]])


def test_perfect_prediction_scores_zero():
    assert ranked_probability_score(CERTAIN_HOME, ["H"]) == pytest.approx(0.0)


def test_completely_wrong_prediction_scores_one():
    """Predicting a home win when the away side wins is the worst possible RPS."""
    assert ranked_probability_score(CERTAIN_HOME, ["A"]) == pytest.approx(1.0)


def test_rps_knows_the_outcomes_are_ordered():
    """The point of RPS: predicting a draw when the home side wins is a near miss,
    predicting an away win is not. Accuracy cannot tell those apart."""
    near_miss = ranked_probability_score(CERTAIN_DRAW, ["H"])
    far_miss = ranked_probability_score(CERTAIN_AWAY, ["H"])

    assert near_miss < far_miss
    assert near_miss == pytest.approx(0.5)


def test_uninformed_prediction_sits_between():
    uninformed = ranked_probability_score(IGNORANT, ["H"])
    assert 0 < uninformed < ranked_probability_score(CERTAIN_AWAY, ["H"])


def test_rps_averages_over_matches():
    probabilities = np.vstack([CERTAIN_HOME, CERTAIN_HOME])
    assert ranked_probability_score(probabilities, ["H", "A"]) == pytest.approx(0.5)


def test_hedging_beats_being_confidently_wrong():
    """The property that makes RPS worth optimising."""
    confident_wrong = ranked_probability_score(np.array([[0.9, 0.05, 0.05]]), ["A"])
    hedged = ranked_probability_score(np.array([[0.4, 0.3, 0.3]]), ["A"])

    assert hedged < confident_wrong


# ------------------------------------------------------------------- log loss


def test_log_loss_rewards_the_truth():
    assert log_loss(np.array([[0.9, 0.05, 0.05]]), ["H"]) < log_loss(IGNORANT, ["H"])


def test_log_loss_punishes_confident_mistakes_harder_than_rps():
    """Log loss is the check on overconfidence; RPS is comparatively forgiving.

    Comparing the two metrics' *ratios* is the meaningful test. Both agree the confident
    mistake is worse; the claim being made is that log loss says so far more emphatically,
    which is why both are reported rather than just the headline one.
    """
    mild, severe = np.array([[0.5, 0.3, 0.2]]), np.array([[0.98, 0.01, 0.01]])

    log_loss_ratio = log_loss(severe, ["A"]) / log_loss(mild, ["A"])
    rps_ratio = ranked_probability_score(severe, ["A"]) / ranked_probability_score(mild, ["A"])

    assert log_loss_ratio > rps_ratio


def test_log_loss_does_not_return_infinity():
    """A probability of exactly zero on the true outcome must not break the metric."""
    assert np.isfinite(log_loss(CERTAIN_HOME, ["A"]))


# ------------------------------------------------------------------- accuracy


def test_accuracy_counts_the_most_likely_outcome():
    probabilities = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert accuracy(probabilities, ["H", "A"]) == pytest.approx(1.0)
    assert accuracy(probabilities, ["H", "H"]) == pytest.approx(0.5)


# ------------------------------------------------------------- bookmaker odds


def test_implied_probabilities_sum_to_one():
    probabilities = implied_probabilities([2.5], [3.4], [2.9])
    assert probabilities.sum() == pytest.approx(1.0)


def test_the_overround_is_removed():
    """Raw reciprocals sum above 1 - that excess is the bookmaker's margin."""
    raw = 1 / 2.5 + 1 / 3.4 + 1 / 2.9
    assert raw > 1.0
    assert implied_probabilities([2.5], [3.4], [2.9]).sum() == pytest.approx(1.0)


def test_shorter_odds_mean_higher_probability():
    probabilities = implied_probabilities([1.5], [4.0], [7.0])[0]
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_evens_across_the_board_is_a_three_way_split():
    probabilities = implied_probabilities([3.0], [3.0], [3.0])[0]
    assert probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])
