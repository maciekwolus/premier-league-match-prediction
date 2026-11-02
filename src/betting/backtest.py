"""What the deployed model would have picked for every match it could have been tested on.

The live archive starts in August 2026, so 2020/21 to 2025/26 have to be reconstructed.
That is done the way the model actually runs rather than the way ``evaluate.compare`` runs
it: **refitted before every round on everything played so far**, the season in progress
included. Fitting once per season would simulate a model nobody deployed, and measured
over these matches that gap is worth 0.0058 RPS. It costs 228 fits instead of 6.

One difference from live cannot be removed: a historical squad rating comes from the XI
that really started, where a live round has to guess it. That flatters these seasons
slightly, which is one more reason the page keeps them apart from the real record.

Usage:
    python -m src.betting.backtest
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.betting.prices import closing_prices
from src.betting.simulate import BACKTEST_PARQUET, TOP_SCORES, priced
from src.config import write_parquet
from src.data.gameweeks import assign_gameweeks
from src.evaluate.metrics import OUTCOMES, ranked_probability_score
from src.features.build import FEATURES_PARQUET, MATCH_COLUMNS
from src.models.dixon_coles_squad import SquadDixonColesModel
from src.models.score_matrix import outcome_probabilities, score_matrix, top_scorelines
from src.report.results import outcome_of


def walk_forward(features: pd.DataFrame, make_model=SquadDixonColesModel) -> pd.DataFrame:
    """One row per testable match: what would have been picked, knowing only the past.

    The first season has nothing before it to learn from. The cutoff is the round's first
    day, so a round never sees its own results or a later round's.
    """
    frame = features.sort_values("date", kind="stable", ignore_index=True)
    frame = frame.assign(gameweek=assign_gameweeks(frame))
    seasons = sorted(frame["season"].unique())
    carried = [column for column in MATCH_COLUMNS if column in frame.columns]

    rows = []
    for season in seasons[1:]:
        before = frame[frame["season"] < season]
        current = frame[frame["season"] == season]
        for gameweek, fixtures in current.groupby("gameweek", sort=True):
            played = current[current["date"] < fixtures["date"].min()]
            # Results only, as ``training_history`` supplies them live: a season in
            # progress has no lineups or ratings yet, and training on these rows with
            # the real squad columns scored 0.2025, a model nobody can run.
            played = played[carried].reindex(columns=frame.columns)

            model = make_model()
            model.fit(pd.concat([before, played], ignore_index=True))
            lambda_home, lambda_away = model.predict(fixtures)

            for match, home, away in zip(
                fixtures.itertuples(), lambda_home, lambda_away, strict=True
            ):
                matrix = score_matrix(home, away)
                probabilities = outcome_probabilities(matrix)
                top = [f"{home}-{away}" for home, away, _ in top_scorelines(matrix, TOP_SCORES)]
                rows.append(
                    {
                        "season_slug": season.replace("/", "_"),
                        "gameweek": int(gameweek),
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "prob_home": probabilities[0],
                        "prob_draw": probabilities[1],
                        "prob_away": probabilities[2],
                        "pick_outcome": OUTCOMES[int(np.argmax(probabilities))],
                        "pick_score": top[0],
                        "pick_scores": ",".join(top),
                        "home_goals": int(match.home_goals),
                        "away_goals": int(match.away_goals),
                    }
                )
        print(f"  {season}: walked forward", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    picks = walk_forward(pd.read_parquet(FEATURES_PARQUET))
    table = priced(picks, closing_prices())

    if len(table) != len(picks):
        print(f"{len(picks) - len(table)} matches had no price and carry no bet.")

    # Reproducing this number is how the backtest proves it is the deployed model.
    probabilities = table[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    actual = [
        outcome_of(home, away)
        for home, away in zip(table["home_goals"], table["away_goals"], strict=True)
    ]
    print(f"{len(table)} matches, RPS {ranked_probability_score(probabilities, actual):.4f}")

    write_parquet(table, BACKTEST_PARQUET)
    print(f"Wrote {BACKTEST_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
