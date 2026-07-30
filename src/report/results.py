"""Scoring stored predictions against what actually happened.

The archive says what the model expected; ``matches.parquet`` says what occurred. Joining
them is the only way the report can make a claim about itself that is checkable.

**The claim worth making is not "we called it".** Any model calls some matches right, and
a page that showed only its hits would be advertising. The honest comparison is our RPS
beside the bookmaker's over the same fixtures - the same benchmark the project has used
throughout, now applied to predictions that were stored before kickoff rather than to a
backtest.

Small samples are stated as such. Ten matches is a round, not evidence; the difference
between two models over ten fixtures is mostly noise, and the page says so rather than
letting a good week read as skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean_matches import MATCHES_PARQUET
from src.evaluate.metrics import OUTCOMES, ranked_probability_score

# Below this a round's scoring is reported but flagged as too small to read anything into.
# A Premier League round is 10 matches, so this is deliberately above one round: the point
# is that a single good week is not a result.
MEANINGFUL_SAMPLE = 30


def actual_results(path=None) -> dict[str, dict]:
    """Final scores keyed by ``match_id``, for matches that have been played.

    Unplayed fixtures are absent rather than present with nulls, so a caller checking
    membership gets the right answer without also testing for NaN.
    """
    matches = pd.read_parquet(
        path or MATCHES_PARQUET, columns=["match_id", "home_goals", "away_goals"]
    )
    played = matches.dropna(subset=["home_goals", "away_goals"])
    return {
        row.match_id: {"home_goals": int(row.home_goals), "away_goals": int(row.away_goals)}
        for row in played.itertuples()
    }


def outcome_of(home_goals: int, away_goals: int) -> str:
    """H, D or A - the same labels the metrics use."""
    if home_goals > away_goals:
        return "H"
    return "D" if home_goals == away_goals else "A"


def attach_results(predictions: list[dict], results: dict[str, dict] | None = None) -> list[dict]:
    """Copy of the predictions with an ``actual`` block where the match has been played.

    Returns copies rather than mutating: the archive is a record, and a display concern
    has no business writing into it even in memory.
    """
    results = actual_results() if results is None else results

    attached = []
    for match in predictions:
        match = dict(match)
        actual = results.get(match["match_id"])
        if actual:
            home, away = actual["home_goals"], actual["away_goals"]
            match["actual"] = {
                "home_goals": home,
                "away_goals": away,
                "score": f"{home}-{away}",
                "outcome": outcome_of(home, away),
            }
        attached.append(match)
    return attached


def verdict(match: dict) -> dict | None:
    """How one prediction fared, or None if the match has not been played.

    ``exact`` is whether our leading scoreline was the score. ``outcome`` is the weaker
    and more meaningful claim: whether the most likely of home/draw/away happened.
    """
    actual = match.get("actual")
    if not actual:
        return None

    scorelines = match.get("scorelines") or []
    top_score = scorelines[0]["score"] if scorelines else None

    probabilities = match.get("outcome") or {}
    ordered = [probabilities.get(key, 0.0) for key in ("home", "draw", "away")]
    predicted_outcome = OUTCOMES[int(np.argmax(ordered))] if any(ordered) else None

    return {
        "score": actual["score"],
        "exact": top_score == actual["score"],
        "outcome": predicted_outcome == actual["outcome"],
        "predicted_score": top_score,
    }


def _probability_rows(matches: list[dict], key: str) -> np.ndarray:
    if key == "outcome":
        return np.array(
            [[m["outcome"]["home"], m["outcome"]["draw"], m["outcome"]["away"]] for m in matches]
        )
    return np.array(
        [[m["bookmaker"]["home"], m["bookmaker"]["draw"], m["bookmaker"]["away"]] for m in matches]
    )


def scorecard(predictions: list[dict]) -> dict:
    """How the stored predictions did, against the market where it priced the same games.

    ``rps`` and ``market_rps`` are computed over *the same* fixtures - only those both
    played and priced - because comparing scores taken over different match sets is the
    easiest way to produce a flattering and meaningless number.
    """
    played = [match for match in predictions if match.get("actual")]
    empty = {
        "played": 0,
        "exact": 0,
        "outcome": 0,
        "rps": None,
        "market_rps": None,
        "compared": 0,
        "small_sample": True,
    }
    if not played:
        return empty

    verdicts = [verdict(match) for match in played]
    card = {
        "played": len(played),
        "exact": sum(1 for v in verdicts if v["exact"]),
        "outcome": sum(1 for v in verdicts if v["outcome"]),
        "rps": None,
        "market_rps": None,
        "compared": 0,
        "small_sample": len(played) < MEANINGFUL_SAMPLE,
    }

    actual = [match["actual"]["outcome"] for match in played]
    card["rps"] = round(ranked_probability_score(_probability_rows(played, "outcome"), actual), 4)

    priced = [match for match in played if match.get("bookmaker")]
    if priced:
        priced_actual = [match["actual"]["outcome"] for match in priced]
        card["compared"] = len(priced)
        card["market_rps"] = round(
            ranked_probability_score(_probability_rows(priced, "bookmaker"), priced_actual), 4
        )
        # Recompute ours over the priced subset only, so the two numbers describe the
        # same fixtures and the difference between them means something.
        card["rps"] = round(
            ranked_probability_score(_probability_rows(priced, "outcome"), priced_actual), 4
        )

    return card
