"""The betting record: stored rounds as they settle, and the backtest behind them.

Computed on read from the archive and ``matches.parquet``, the same way the scorecard is.
So the scheduled job needs no bookkeeping step and there is no ledger to fall out of step
with the archive: the morning's results refresh is what settles the bets.

Usage:
    python -m src.betting.tracker           # markdown, for the scheduled job's run summary
"""

from __future__ import annotations

import sys

import pandas as pd

from src.betting.prices import (
    BOOKMAKER,
    CORRECT_SCORE_OVERROUND,
    OVERROUND_RANGE,
    closing_prices,
)
from src.betting.simulate import (
    BACKTEST_PARQUET,
    MARKETS,
    PICK_COLUMNS,
    STAKE,
    STAKE_TAX,
    both_markets,
    by_round,
    priced,
    settle,
)
from src.config import UPCOMING_SEASON
from src.predict.archive import available_rounds, load_round
from src.report.results import (
    attach_results,
    once_per_fixture,
    predicted_outcome,
    predicted_score,
)

LABELS = {"outcome": "Home/draw/away", "score": "Exact score (estimated prices)"}


def stored_bets(
    season_slug: str,
    root=None,
    results: dict | None = None,
    prices: dict | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Every bet the archive implies for a season: (settled, awaiting a result, unpriced).

    Replayed rounds are left out. A replay is predicted after the fact from only what was
    known beforehand, which is a fair test and a fair scorecard entry, but nobody could
    have placed that bet, so it does not belong in a record of money.

    A postponed fixture is one bet, on the first forecast: ``once_per_fixture`` is what
    stops it being staked twice and settled twice against a single result.
    """
    matches: list[dict] = []
    for stored_season, gameweek in available_rounds(root):
        if stored_season == season_slug:
            matches.extend(load_round(stored_season, gameweek, root))

    live = [match for match in matches if match.get("mode") != "replay"]
    attached = once_per_fixture(attach_results(live, results))
    played = [match for match in attached if match.get("actual")]

    picks = pd.DataFrame(
        [
            {
                "season_slug": match["season_slug"],
                "gameweek": match["gameweek"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "pick_outcome": predicted_outcome(match),
                "pick_score": predicted_score(match),
                "home_goals": match["actual"]["home_goals"],
                "away_goals": match["actual"]["away_goals"],
            }
            for match in played
        ],
        columns=PICK_COLUMNS,
    )

    bets = priced(picks, closing_prices() if prices is None else prices)
    return bets, len(attached) - len(played), len(picks) - len(bets)


def tracker(season_slug: str, root=None, results=None, prices=None) -> dict:
    """One season round by round, and in total."""
    bets, pending, unpriced = stored_bets(season_slug, root, results, prices)
    return {
        "season": season_slug.replace("_", "/"),
        "total": both_markets(bets),
        "rounds": by_round(bets),
        "pending": pending,
        "unpriced": unpriced,
    }


def history(path=None) -> dict | None:
    """The six-season backtest, or None on a checkout that does not have it."""
    source = path or BACKTEST_PARQUET
    if not source.exists():
        return None

    bets = pd.read_parquet(source)
    return {
        "seasons": [
            {"season": slug.replace("_", "/"), **both_markets(rows)}
            for slug, rows in bets.groupby("season_slug", sort=True)
        ],
        "total": both_markets(bets),
        # The one number here that rests on a judgement, so show what it is worth.
        "overrounds": [
            {
                "overround": overround,
                "balance": settle(
                    bets.assign(
                        price_score=bets["price_score"] * CORRECT_SCORE_OVERROUND / overround
                    ),
                    "score",
                )["balance"],
            }
            for overround in OVERROUND_RANGE
        ],
    }


def signed(amount: float) -> str:
    return f"{amount:+,.2f} zł"


def format_summary(record: dict, past: dict | None) -> str:
    """The run summary: where each balance stands, as markdown."""
    lines = [
        f"## Betting tracker, {record['season']}",
        "",
        f"{STAKE:.0f} zł on every match, {STAKE_TAX:.0%} stake tax deducted. Prices are "
        f"{BOOKMAKER} closing odds, not a Polish bookmaker's, and exact-score prices are "
        "estimated.",
        "",
        "| | Bets | Won | Staked | Returned | Balance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for market in MARKETS:
        row = record["total"][market]
        lines.append(
            f"| {LABELS[market]} | {row['bets']} | {row['won']} | {row['staked']:,.0f} zł "
            f"| {row['returned']:,.2f} zł | {signed(row['balance'])} |"
        )

    if record["rounds"]:
        lines += [
            "",
            "| Round | Home/draw/away | Running | Exact score | Running |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in record["rounds"]:
            outcome, score = row["outcome"], row["score"]
            lines.append(
                f"| GW {row['gameweek']} | {signed(outcome['balance'])} "
                f"| {signed(outcome['running'])} | {signed(score['balance'])} "
                f"| {signed(score['running'])} |"
            )

    if record["pending"]:
        lines += ["", f"Waiting on a result: {record['pending']}."]
    if record["unpriced"]:
        lines += ["", f"Played but unpriced, so no bet: {record['unpriced']}."]
    if past:
        total = past["total"]
        lines += [
            "",
            f"Backtest over {total['matches']:,} matches: home/draw/away "
            f"{signed(total['outcome']['balance'])}, exact score "
            f"{signed(total['score']['balance'])}.",
        ]
    return "\n".join(lines)


def main() -> int:
    # Written straight into GitHub's run summary, and a Windows console would choke on zł.
    sys.stdout.reconfigure(encoding="utf-8")
    print(format_summary(tracker(UPCOMING_SEASON.slug), history()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
