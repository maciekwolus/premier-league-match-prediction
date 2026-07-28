"""Tests for the expected-goals to scoreline conversion.

Every model funnels through here, so an error in this file would corrupt the comparison
between them without making any single model look wrong.
"""

import numpy as np
import pytest

from src.models.score_matrix import (
    outcome_probabilities,
    score_matrix,
    top_scorelines,
)


def test_matrix_is_a_distribution():
    matrix = score_matrix(1.5, 1.2)

    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_matrix_covers_zero_to_max_goals():
    matrix = score_matrix(1.5, 1.2, max_goals=6)
    assert matrix.shape == (7, 7)


def test_stronger_home_side_shifts_mass_to_home_wins():
    home, draw, away = outcome_probabilities(score_matrix(2.5, 0.8))
    assert home > away
    assert home > draw


def test_symmetric_lambdas_give_symmetric_outcomes():
    home, _, away = outcome_probabilities(score_matrix(1.4, 1.4))
    assert home == pytest.approx(away, abs=1e-9)


def test_outcome_probabilities_sum_to_one():
    probabilities = outcome_probabilities(score_matrix(1.7, 1.1))
    assert sum(probabilities) == pytest.approx(1.0)


def test_equal_lambdas_maximise_the_draw():
    """Draw probability should peak when the sides are evenly matched."""
    _, level_draw, _ = outcome_probabilities(score_matrix(1.4, 1.4))
    _, lopsided_draw, _ = outcome_probabilities(score_matrix(2.6, 0.6))
    assert level_draw > lopsided_draw


def test_reproduces_the_real_scoreline_distribution():
    """League-average goals must yield roughly the league's actual scorelines.

    Across the 2,660 matches gathered, 1-1 occurs 11.2%, 2-1 8.4% and 1-0 8.3%. If this
    conversion is right, feeding it the league's average goals should land near those.
    """
    matrix = score_matrix(1.55, 1.31)
    top = {(home, away): probability for home, away, probability in top_scorelines(matrix, 3)}

    assert (1, 1) in top
    assert top[(1, 1)] == pytest.approx(0.112, abs=0.02)


def test_reproduces_the_real_outcome_split():
    """Actual split across seven seasons: 43.4% home, 23.6% draw, 32.9% away."""
    home, draw, away = outcome_probabilities(score_matrix(1.55, 1.31))

    assert home == pytest.approx(0.434, abs=0.03)
    assert draw == pytest.approx(0.236, abs=0.03)
    assert away == pytest.approx(0.329, abs=0.03)


def test_dixon_coles_correction_lifts_the_low_draws():
    """The correction exists because independent Poisson under-predicts 0-0 and 1-1."""
    corrected = score_matrix(1.4, 1.2, rho=-0.03)
    independent = score_matrix(1.4, 1.2, rho=0.0)

    assert corrected[0, 0] > independent[0, 0]
    assert corrected[1, 1] > independent[1, 1]


def test_top_scorelines_are_ordered():
    scorelines = top_scorelines(score_matrix(1.6, 1.2), n=5)
    probabilities = [probability for _, _, probability in scorelines]

    assert probabilities == sorted(probabilities, reverse=True)
    assert len(scorelines) == 5


def test_no_scoreline_is_ever_near_certain():
    """The project's central claim: exact scores top out around 12%.

    If this ever fails, either the conversion is broken or the lambdas fed to it are.
    """
    for lambda_home in (0.5, 1.0, 1.5, 2.0, 3.0):
        for lambda_away in (0.5, 1.0, 1.5, 2.0, 3.0):
            best = top_scorelines(score_matrix(lambda_home, lambda_away), 1)[0]
            assert best[2] < 0.40


def test_zero_lambdas_do_not_break():
    matrix = score_matrix(0.0, 0.0)
    assert matrix.sum() == pytest.approx(1.0)
    assert matrix[0, 0] > 0.99


def test_matrix_is_indexed_home_then_away():
    """[i, j] must mean i home goals and j away goals, not the reverse."""
    matrix = score_matrix(3.0, 0.3)
    assert matrix[3, 0] > matrix[0, 3]


def test_outcome_regions_do_not_overlap():
    """Home wins, draws and away wins must partition the matrix exactly once."""
    matrix = score_matrix(1.5, 1.5)
    home, draw, away = outcome_probabilities(matrix)
    assert home + draw + away == pytest.approx(matrix.sum())
    assert np.trace(matrix) == pytest.approx(draw)
