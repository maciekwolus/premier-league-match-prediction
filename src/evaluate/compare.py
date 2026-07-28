"""Run every model through the same walk-forward backtest and rank them.

The bookmaker's closing line appears as a row in the table, scored over exactly the
matches the models were tested on. That is the honest comparison: not "is this model
good" but "is it better than the number you could have read off a screen for free".

Usage:
    python -m src.evaluate.compare
    python -m src.evaluate.compare --fast      # skip AutoGluon, which dominates runtime
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from src.evaluate.backtest import backtest, calibration, compare, score_by_season
from src.features.build import FEATURES_PARQUET
from src.models.baselines import EloModel, LeagueAverageModel, TeamAverageModel
from src.models.dixon_coles import DixonColesModel
from src.models.dixon_coles_squad import SquadDixonColesModel
from src.models.poisson_glm import PoissonRegressionModel

PREDICTIONS_DIR = FEATURES_PARQUET.parent / "predictions"


def build_models(fast: bool = False) -> list:
    """Every model family, cheapest first so a --fast run still covers the range."""
    models = [
        LeagueAverageModel(),
        TeamAverageModel(),
        EloModel(),
        DixonColesModel(),
        SquadDixonColesModel(),
        PoissonRegressionModel(),
        PoissonRegressionModel(use_odds=True),
    ]
    # The odds variant needs a distinct name, since it is a different question.
    models[-1].name = "poisson-glm-with-odds"

    if not fast:
        from src.models.gbm import GradientBoostingModel

        models.append(GradientBoostingModel())
        models.append(GradientBoostingModel(use_odds=True))

    return models


def run(fast: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"{FEATURES_PARQUET} not found. Run: python -m src.features.build")

    features = pd.read_parquet(FEATURES_PARQUET)
    results: dict[str, pd.DataFrame] = {}

    for model in build_models(fast):
        print(f"  {model.name} ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[model.name] = backtest(model, features)

    return compare(results, features), results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true", help="skip AutoGluon, which dominates runtime"
    )
    parser.add_argument("--save", action="store_true", help="write per-match predictions")
    args = parser.parse_args(argv)

    print("Backtesting (train on seasons 1..n, test on n+1):")
    table, results = run(fast=args.fast)

    print("\n" + "=" * 66)
    print("RANKED BY RPS - lower is better, bookmaker is the line to beat")
    print("=" * 66)
    print(table.round(4).to_string())

    best = table.index[0]
    if best != "bookmaker (closing)":
        print(f"\n{best} beats the closing line. Verify this before believing it.")
    else:
        gap = table["rps"].iloc[1] - table["rps"].iloc[0]
        print(f"\nThe market wins by {gap:.4f} RPS. Nothing beats it, which is normal.")

    strongest = next(name for name in table.index if name != "bookmaker (closing)")
    print(f"\nPer season, {strongest}:")
    print(score_by_season(results[strongest]).round(4).to_string())

    print(f"\nCalibration, {strongest} (predicted vs observed home win rate):")
    print(calibration(results[strongest]).round(3).to_string())

    if args.save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        for name, predictions in results.items():
            predictions.to_parquet(PREDICTIONS_DIR / f"{name}.parquet", index=False)
        print(f"\npredictions written to {PREDICTIONS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
