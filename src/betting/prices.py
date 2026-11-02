"""What a simulated bet would have been paid at.

**These are not Polish bookmaker prices, and everything downstream has to say so.** The
tracker was meant to use STS. OddsPortal does hold STS's closing prices, but its robots.txt
disallows every archived season for every crawler and its terms forbid copying without
written permission, so that archive was not collected. No API found offers Polish prices
for the correct-score market. The two markets therefore rest on different evidence:

- **Home/draw/away is a real price**: Bet365's close, from football-data, which any bettor
  could have taken. A Polish book's margin differs a little, but the number is not invented.
- **Exact score is an estimate.** No source available here records a correct-score price
  for these matches at all, so one is derived from the same closing line (below) and
  labelled ``estimated`` wherever it reaches a page.

To use real prices later, this is the only module that has to change.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.data.clean_matches import MATCHES_PARQUET
from src.evaluate.metrics import implied_probabilities
from src.models.score_matrix import outcome_probabilities, score_matrix
from src.report.results import fixture_key

BOOKMAKER = "Bet365"

PRICE_COLUMNS = ("odds_close_home", "odds_close_draw", "odds_close_away")

# How much a correct-score book adds up to, which is far worse than the 1.05 of home/draw/
# away. Read off two whole correct-score tables on OddsPortal: 1.25 for one book on Arsenal
# v Brighton in May 2021, and 1.15 for the *best* of four books on Southampton v
# Arsenal in May 2025, which is flattering because nobody offers the best of four. Two
# matches is thin: across the backtest, 15% gives -7,418 zl and 35% gives -9,697 zl, so the
# verdict survives the guess even though the exact figure does not.
CORRECT_SCORE_OVERROUND = 1.25

# Nothing is priced shorter than this, however likely the score.
MINIMUM_PRICE = 1.01


@lru_cache(maxsize=8192)
def market_matrix(home_odds: float, draw_odds: float, away_odds: float) -> np.ndarray:
    """The scoreline distribution the closing line implies.

    Home/draw/away pins down two free numbers and a goals model has two, so the expected
    goals of each side can be solved for directly. This is deliberately the *market's*
    view and never our model's: pricing a bet from the model that chose it would mean
    agreeing with ourselves about what it was worth.

    Cached because a report rerun asks for the same few hundred matches every time.
    """
    target = implied_probabilities([home_odds], [draw_odds], [away_odds])[0]

    def gap(logs: np.ndarray) -> np.ndarray:
        matrix = score_matrix(np.exp(logs[0]), np.exp(logs[1]))
        return np.array(outcome_probabilities(matrix)) - target

    solved = least_squares(gap, x0=np.log([1.5, 1.2]), bounds=(np.log(0.05), np.log(8.0)))
    return score_matrix(*np.exp(solved.x))


def estimated_score_price(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    home_goals: int,
    away_goals: int,
    overround: float = CORRECT_SCORE_OVERROUND,
) -> float:
    """An estimated correct-score price: the market's fair price, less a book's margin."""
    matrix = market_matrix(float(home_odds), float(draw_odds), float(away_odds))
    probability = float(matrix[home_goals, away_goals])
    return max(round(1 / (probability * overround), 2), MINIMUM_PRICE)


def outcome_price(prices: tuple[float, float, float], outcome: str) -> float:
    """The closing price of H, D or A."""
    return prices["HDA".index(outcome)]


def closing_prices(path=None) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Bet365's closing home/draw/away, keyed by fixture.

    Keyed the way scoring is keyed and for the same reason: ``match_id`` carries the
    scheduled date, so a postponed match would silently lose its price. A fixture without
    a complete price is absent, and absent means no bet was placed.
    """
    matches = pd.read_parquet(
        path or MATCHES_PARQUET, columns=["season", "home_team", "away_team", *PRICE_COLUMNS]
    ).dropna()

    return {
        fixture_key(row.season.replace("/", "_"), row.home_team, row.away_team): (
            float(row.odds_close_home),
            float(row.odds_close_draw),
            float(row.odds_close_away),
        )
        for row in matches.itertuples()
    }
