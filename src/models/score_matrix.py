"""Turn expected goals into a scoreline probability matrix.

Every model in this project predicts the same two numbers - the goals each side is
expected to score - and this module converts that pair into the actual product: a
probability for every plausible scoreline, and the home/draw/away split implied by it.

Doing it here rather than in each model means the models are directly comparable, and
that improving the conversion improves all of them at once.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# 0-0 through 9-9. Beyond this the probabilities are negligible, and truncating too
# early would quietly discard mass that belongs somewhere.
MAX_GOALS = 10

# Dixon-Coles low-score correction. Independent Poisson under-predicts 0-0 and 1-1 and
# over-predicts 1-0 and 0-1, because the two scores are not independent at low values.
# Negative rho shifts mass towards the draws. -0.03 is the usual empirical value.
DEFAULT_RHO = -0.03


def dixon_coles_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> np.ndarray:
    """The low-score adjustment, applied to the four scorelines where it is defined."""
    tau = np.ones_like(home_goals, dtype=float)

    tau = np.where((home_goals == 0) & (away_goals == 0), 1 - lambda_home * lambda_away * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1 + lambda_home * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1 + lambda_away * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, tau)

    return tau


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Probability of every scoreline, indexed ``[home_goals, away_goals]``.

    Rows and columns each run 0..max_goals, and the matrix sums to 1.
    """
    lambda_home = max(float(lambda_home), 1e-6)
    lambda_away = max(float(lambda_away), 1e-6)

    goals = np.arange(max_goals + 1)
    home_probability = poisson.pmf(goals, lambda_home)
    away_probability = poisson.pmf(goals, lambda_away)

    matrix = np.outer(home_probability, away_probability)

    grid_home, grid_away = np.meshgrid(goals, goals, indexing="ij")
    matrix = matrix * dixon_coles_tau(grid_home, grid_away, lambda_home, lambda_away, rho)

    # Truncation and the correction both cost a little mass; renormalising keeps the
    # matrix a valid distribution.
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Collapse a scoreline matrix into (home win, draw, away win)."""
    home = float(np.tril(matrix, -1).sum())  # home goals > away goals
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


def top_scorelines(matrix: np.ndarray, n: int = 3) -> list[tuple[int, int, float]]:
    """The ``n`` most likely scorelines as (home_goals, away_goals, probability)."""
    flat = np.argsort(matrix, axis=None)[::-1][:n]
    rows, columns = np.unravel_index(flat, matrix.shape)
    return [(int(h), int(a), float(matrix[h, a])) for h, a in zip(rows, columns, strict=True)]


def outcome_frame(lambdas_home, lambdas_away, rho: float = DEFAULT_RHO) -> np.ndarray:
    """Outcome probabilities for many matches at once, as an (n, 3) array of H/D/A."""
    return np.array(
        [
            outcome_probabilities(score_matrix(home, away, rho))
            for home, away in zip(lambdas_home, lambdas_away, strict=True)
        ]
    )
