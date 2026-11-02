"""Settling simulated bets: ten zloty on what the model said, every match, no selection.

Three markets per fixture: the outcome the model thinks most likely, the scoreline it puts
first, and the three scorelines it puts first. The first two are exactly the calls the
scorecard already grades, so the money and the hit counts can never describe different
picks.

**The top three is three separate bets, 30 zl a match, and at most one can win.** It is the
same stake per bet as the other markets, so the three can be compared per zloty staked, and
it is not a safer version of the exact score: it wins more often at the same bad prices.

**A Polish coupon does not stake what it says.** Legal bookmakers there carry a 12% tax on
stakes and usually pass it on, so 10 zl buys 8.80 zl of bet and a winner pays 8.80 times
the price. That is why the headline balance has the tax in it, with the untaxed figure
beside it for the promotions where the bookmaker absorbs it. The 10% tax on a win above
2,280 zl is left out: at this stake it needs a price of about 260, and the longest winner
in 2,280 backtested matches paid 15.59.

Nothing here is selective, and that is the point. A bet is placed on every fixture that
has a price, so the balance cannot flatter itself by skipping the hard ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.betting.prices import estimated_score_price, outcome_price
from src.config import FINAL_DIR
from src.report.results import fixture_key, outcome_of

BACKTEST_PARQUET = FINAL_DIR / "betting_backtest.parquet"

STAKE = 10.0
STAKE_TAX = 0.12

MARKETS = ("outcome", "score", "top3")

# The report's card lists three scorelines, so this bets on what the page shows.
TOP_SCORES = 3
BETS_PER_MATCH = {"outcome": 1, "score": 1, "top3": TOP_SCORES}

PICK_COLUMNS = [
    "season_slug",
    "gameweek",
    "home_team",
    "away_team",
    "pick_outcome",
    "pick_score",
    "pick_scores",
    "home_goals",
    "away_goals",
]
BET_COLUMNS = [
    *PICK_COLUMNS,
    "price_outcome",
    "price_score",
    "price_top3",
    "won_outcome",
    "won_score",
    "won_top3",
]


def priced(picks: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """The picks with what each would have paid, dropping any match nobody priced."""
    rows = []
    for row in picks.to_dict("records"):
        closing = prices.get(fixture_key(row["season_slug"], row["home_team"], row["away_team"]))
        if closing is None:
            continue

        pick_home, pick_away = (int(goals) for goals in row["pick_score"].split("-"))
        result = outcome_of(row["home_goals"], row["away_goals"])
        final_score = f"{row['home_goals']}-{row['away_goals']}"
        top_three_hit = final_score in row["pick_scores"].split(",")
        rows.append(
            {
                **row,
                "price_outcome": outcome_price(closing, row["pick_outcome"]),
                "price_score": estimated_score_price(*closing, pick_home, pick_away),
                # What the top three paid, so only the score that came in needs a price.
                "price_top3": (
                    estimated_score_price(*closing, row["home_goals"], row["away_goals"])
                    if top_three_hit
                    else float("nan")
                ),
                "won_outcome": row["pick_outcome"] == result,
                "won_score": row["pick_score"] == final_score,
                "won_top3": top_three_hit,
            }
        )

    extra = [column for column in picks.columns if column not in BET_COLUMNS]
    return pd.DataFrame(rows, columns=[*BET_COLUMNS, *extra])


def settle(bets: pd.DataFrame, market: str, tax: float = STAKE_TAX) -> dict:
    """What one market staked and returned over these bets."""
    won = bets[f"won_{market}"].astype(bool)
    prices = bets[f"price_{market}"]

    count = len(bets) * BETS_PER_MATCH[market]
    staked = count * STAKE
    payouts = prices.where(won, 0.0).to_numpy(dtype=float) * STAKE * (1 - tax)
    returned = float(payouts.sum())
    return {
        "bets": count,
        "won": int(won.sum()),
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "balance": round(returned - staked, 2),
        "roi": (returned - staked) / staked if staked else None,
        # How far one standard deviation of luck moves this balance over this many bets.
        # A balance inside it says nothing about the picks, in either direction.
        "luck": round(float(payouts.std() * np.sqrt(len(bets))), 2) if len(bets) else 0.0,
    }


def both_markets(bets: pd.DataFrame) -> dict:
    """Every market over these bets, with the balance had no tax been deducted."""
    summary: dict = {"matches": len(bets)}
    for market in MARKETS:
        summary[market] = settle(bets, market)
        summary[market]["untaxed_balance"] = settle(bets, market, tax=0.0)["balance"]
    return summary


def by_round(bets: pd.DataFrame) -> list[dict]:
    """Each gameweek in order, with the balance so far alongside the round's own."""
    rounds = []
    for gameweek, fixtures in bets.groupby("gameweek", sort=True):
        row = {"gameweek": int(gameweek), **both_markets(fixtures)}
        # Settled afresh rather than added up: summing rounded balances drifts by a grosz,
        # and the last running figure then disagrees with the total printed beneath it.
        so_far = bets[bets["gameweek"] <= gameweek]
        for market in MARKETS:
            row[market]["running"] = settle(so_far, market)["balance"]
        rounds.append(row)
    return rounds
