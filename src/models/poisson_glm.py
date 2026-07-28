"""Poisson regression on the feature table.

Goals are counts, so a Poisson likelihood with a log link is the natural first model:
``log(lambda) = intercept + X @ beta`` guarantees a positive rate and makes every
coefficient a multiplicative effect on expected goals, which is how football effects
actually behave - a strong attack scores *proportionally* more, not a flat goal more.

Two independent regressions are fitted, one per side, rather than a single symmetric
model. The home and away columns are already mirror images of each other in the feature
table, so a shared model would buy nothing, and separate fits let home advantage live in
the two intercepts where it is easy to read off.

The pair of lambdas is all this module produces; ``score_matrix`` turns them into
scorelines, including the Dixon-Coles correction for the dependence between the two
scores that independent Poissons cannot express.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base import feature_columns

# Around 85 features against as few as 380 training rows, and most of them near-duplicates
# - squad overall, squad potential, form and the six FIFA face stats all measure much the
# same underlying strength, and home/away/diff triples are linearly dependent by
# construction. Unpenalised, the likelihood spends that collinearity on large offsetting
# coefficients that fit the training seasons and generalise badly: ridge 0 scores RPS
# 0.2218 across the walk-forward backtest against 0.2040 here, so the penalty is worth more
# than any feature in the table.
#
# The default came from sweeping the backtest across five decades of magnitude. The curve
# is a broad basin between roughly 2.5 and 8 - 0.2040 to 0.2043, which is noise on 2,280
# matches - and climbs away either side. 4.0 sits near the middle of that basin on a log
# scale, which matters more than the exact minimum: a value picked off the floor of a flat
# region is far likelier to hold up on a season the sweep never saw.
DEFAULT_RIDGE = 4.0

# exp() of a linear predictor is easy to overflow while the optimiser is exploring, and a
# football side has never been a plausible 3,000 goals. Clamping the predictor keeps the
# objective finite without distorting anything in the region the fit actually lives in.
MAX_LINEAR_PREDICTOR = 5.0

# Lambdas handed downstream. The floor keeps the Poisson pmf defined; the ceiling is far
# above any real fixture and only ever catches a pathological extrapolation.
MIN_LAMBDA = 0.05
MAX_LAMBDA = 8.0


def _negative_log_likelihood(
    parameters: np.ndarray,
    design: np.ndarray,
    goals: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    """Mean Poisson deviance-equivalent plus an L2 penalty, with its gradient.

    The ``log(y!)`` term of the true log likelihood is dropped: it does not involve the
    parameters, so it shifts the objective without moving the optimum.

    The likelihood is averaged over rows while the penalty is not, which makes ``ridge``
    mean the same thing whether the training set is one season or six. Under a summed
    likelihood the same number would regularise the first walk-forward split six times as
    hard as the last, and the model would silently change character across the backtest.

    The intercept is deliberately left out of the penalty. Shrinking it would pull the
    baseline scoring rate towards one goal per game rather than towards the truth, and
    unlike the coefficients it has no collinearity problem to fix.
    """
    intercept = parameters[0]
    coefficients = parameters[1:]

    linear = intercept + design @ coefficients
    linear = np.clip(linear, -MAX_LINEAR_PREDICTOR, MAX_LINEAR_PREDICTOR)
    rate = np.exp(linear)

    rows = len(goals)
    loss = float((rate - goals * linear).sum() / rows + 0.5 * ridge * coefficients @ coefficients)

    residual = (rate - goals) / rows
    gradient = np.empty_like(parameters)
    gradient[0] = residual.sum()
    gradient[1:] = design.T @ residual + ridge * coefficients

    return loss, gradient


class PoissonRegressionModel:
    """Ridge-penalised Poisson regression for each side's expected goals."""

    name = "poisson-glm"

    def __init__(
        self,
        use_odds: bool = False,
        ridge: float = DEFAULT_RIDGE,
        max_iterations: int = 500,
    ) -> None:
        self.use_odds = use_odds
        self.ridge = ridge
        self.max_iterations = max_iterations

        self.columns: list[str] = []
        self.medians: np.ndarray = np.empty(0)
        self.means: np.ndarray = np.empty(0)
        self.deviations: np.ndarray = np.empty(0)
        self.home_parameters: np.ndarray = np.empty(0)
        self.away_parameters: np.ndarray = np.empty(0)

    @property
    def variant(self) -> str:
        return f"{self.name}-odds" if self.use_odds else self.name

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Feature values as floats, with booleans and nullable integers flattened.

        ``astype`` is what turns pandas' ``NA`` into ``NaN``; the imputation below only
        recognises the latter.
        """
        return frame.reindex(columns=self.columns).astype("float64").to_numpy()

    def _prepare(self, frame: pd.DataFrame) -> np.ndarray:
        """Impute and standardise using statistics learned in ``fit``.

        Test rows are transformed with the *training* median, mean and deviation. Using
        the test set's own statistics would leak the future into the present - a promoted
        side's missing form would be filled from a distribution that includes matches not
        yet played - and would also make a single prediction depend on which other
        fixtures happened to be in the batch.
        """
        design = self._matrix(frame)
        missing = np.isnan(design)
        if missing.any():
            design = np.where(missing, self.medians, design)
        return (design - self.means) / self.deviations

    def fit(self, train: pd.DataFrame) -> None:
        candidates = feature_columns(train, use_odds=self.use_odds)
        values = train.reindex(columns=candidates).astype("float64").to_numpy()

        # An all-null column makes ``nanmedian`` warn and return NaN, which is exactly the
        # signal wanted a few lines below, so the warning is suppressed rather than avoided.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(values, axis=0)

        imputed = np.where(np.isnan(values), medians, values)
        deviations = imputed.std(axis=0)

        # A column all-null in training has no median to impute with and would poison every
        # coefficient with NaN; a constant column carries no information and cannot be
        # standardised. Both are real here - the six FIFA face stats are absent for whole
        # editions - so they are dropped rather than patched, which keeps the surviving
        # coefficients interpretable. Whether a column is usable is a property of the
        # training data alone, so this decision never sees the test set.
        keep = np.isfinite(medians) & (deviations > 1e-8)

        self.columns = [column for column, usable in zip(candidates, keep, strict=True) if usable]
        self.medians = medians[keep]
        self.means = imputed[:, keep].mean(axis=0)
        self.deviations = deviations[keep]

        design = (imputed[:, keep] - self.means) / self.deviations

        self.home_parameters = self._fit_one(design, train["home_goals"].to_numpy(dtype=float))
        self.away_parameters = self._fit_one(design, train["away_goals"].to_numpy(dtype=float))

    def _fit_one(self, design: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Maximise the penalised likelihood for one side.

        Started from the intercept-only solution - ``log`` of the mean goals scored, with
        every coefficient at zero. That is already the best model that ignores the
        features, so the optimiser only ever has to find what the features add, and a run
        that converges badly degrades towards the base rate instead of somewhere absurd.
        """
        start = np.zeros(design.shape[1] + 1)
        start[0] = np.log(max(goals.mean(), MIN_LAMBDA))

        result = minimize(
            _negative_log_likelihood,
            start,
            args=(design, goals, self.ridge),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations},
        )
        return result.x

    def _rate(self, design: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        linear = parameters[0] + design @ parameters[1:]
        rate = np.exp(np.clip(linear, -MAX_LINEAR_PREDICTOR, MAX_LINEAR_PREDICTOR))
        return np.clip(rate, MIN_LAMBDA, MAX_LAMBDA)

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        design = self._prepare(test)
        return self._rate(design, self.home_parameters), self._rate(design, self.away_parameters)
