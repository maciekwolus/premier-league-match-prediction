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
import pyarrow.parquet as pq

from src.data.clean_matches import MATCHES_PARQUET
from src.evaluate.metrics import OUTCOMES, implied_probabilities, ranked_probability_score

# Below this a round's scoring is reported but flagged as too small to read anything into.
# A Premier League round is 10 matches, so this is deliberately above one round: the point
# is that a single good week is not a result.
MEANINGFUL_SAMPLE = 30


def fixture_key(season_slug: str, home_team: str, away_team: str) -> tuple[str, str, str]:
    """The identity of a fixture, and deliberately not its date.

    A home/away pairing occurs exactly once per season, which is the join key every
    cross-source stage in this project uses. **``match_id`` cannot be used here** even
    though both sides carry one: it encodes the *scheduled* date, and football-data
    records a postponed match under the date it was actually played. The two ids then
    never meet, the fixture reads as never played, and it leaves the scorecard silently -
    which on a real stored round turned 10 scored matches into 9 and moved the reported
    RPS from 0.2407 to 0.2513 with nothing on the page to say anything had gone missing.
    """
    return (str(season_slug), str(home_team), str(away_team))


def _key_of(match: dict) -> tuple[str, str, str]:
    """The fixture key of one stored prediction."""
    return fixture_key(
        match.get("season_slug", ""), match.get("home_team", ""), match.get("away_team", "")
    )


ODDS_COLUMNS = ("odds_close_avg_home", "odds_close_avg_draw", "odds_close_avg_away")


def actual_results(path=None) -> dict[tuple[str, str, str], dict]:
    """Final scores keyed by fixture, for matches that have been played.

    Unplayed fixtures are absent rather than present with nulls, so a caller checking
    membership gets the right answer without also testing for NaN.

    Each entry also carries the **closing** market probabilities where the source has
    them. A round predicted weeks ahead has no odds in it - the market has not formed
    that far out - so all four stored 2026/27 rounds were priced 0 of 10, and without
    this the season's scorecard would show our score beside a blank forever. Closing
    odds are the project's benchmark everywhere else, and they only exist once a match
    is close, so this is where they have to come from.
    """
    source = path or MATCHES_PARQUET
    required = ["season", "home_team", "away_team", "home_goals", "away_goals"]

    # Odds are read only if the source has them. A table built before this column set
    # existed is still perfectly good for saying who won, and should not fail to load
    # over a column that only affects the comparison.
    available = set(pq.ParquetFile(source).schema.names)
    odds = [column for column in ODDS_COLUMNS if column in available]

    matches = pd.read_parquet(source, columns=[*required, *odds])
    played = matches.dropna(subset=["home_goals", "away_goals"])

    priced = played.dropna(subset=odds) if len(odds) == len(ODDS_COLUMNS) else played.iloc[0:0]
    market = {}
    if not priced.empty:
        probabilities = implied_probabilities(*(priced[column] for column in ODDS_COLUMNS))
        market = {
            fixture_key(season.replace("/", "_"), home, away): {
                "home": round(float(row[0]), 3),
                "draw": round(float(row[1]), 3),
                "away": round(float(row[2]), 3),
            }
            for season, home, away, row in zip(
                priced["season"],
                priced["home_team"],
                priced["away_team"],
                probabilities,
                strict=True,
            )
        }

    results = {}
    for row in played.itertuples():
        # matches.parquet spells a season "2025/26"; the archive uses the slug form.
        key = fixture_key(row.season.replace("/", "_"), row.home_team, row.away_team)
        results[key] = {
            "home_goals": int(row.home_goals),
            "away_goals": int(row.away_goals),
            "bookmaker": market.get(key),
        }
    return results


def outcome_of(home_goals: int, away_goals: int) -> str:
    """H, D or A - the same labels the metrics use."""
    if home_goals > away_goals:
        return "H"
    return "D" if home_goals == away_goals else "A"


def attach_results(predictions: list[dict], results: dict | None = None) -> list[dict]:
    """Copy of the predictions with an ``actual`` block where the match has been played.

    Returns copies rather than mutating: the archive is a record, and a display concern
    has no business writing into it even in memory.
    """
    results = actual_results() if results is None else results

    attached = []
    for match in predictions:
        match = dict(match)
        actual = results.get(_key_of(match))
        if actual:
            home, away = actual["home_goals"], actual["away_goals"]
            match["actual"] = {
                "home_goals": home,
                "away_goals": away,
                "score": f"{home}-{away}",
                "outcome": outcome_of(home, away),
            }
            # A prediction made before the market formed carries no odds. Fill them in
            # from the closing line now that the match has been played - never overwrite
            # odds the prediction already recorded, which are what we actually saw.
            if not match.get("bookmaker") and actual.get("bookmaker"):
                match["bookmaker"] = actual["bookmaker"]
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


def once_per_fixture(predictions: list[dict]) -> list[dict]:
    """The predictions with repeat forecasts of the same fixture removed.

    A postponed match gets predicted twice: once for the round it was scheduled in, and
    again for the round it is moved to. Both then join to the same result, so counting
    both would score one fixture twice.

    **The earliest is the one kept**, by ``predicted_at``. It is the less informed of the
    two - made furthest from a kickoff that had not yet been rearranged - so this cannot
    flatter the record. Keeping the later one would mean scoring ourselves on the forecast
    made with more information, which is the direction of error worth refusing.
    """
    ordered = sorted(
        enumerate(predictions), key=lambda pair: (pair[1].get("predicted_at") or "", pair[0])
    )
    first: dict[tuple[str, str, str], dict] = {}
    for _, match in ordered:
        first.setdefault(_key_of(match), match)
    return list(first.values())


def scorecard(predictions: list[dict]) -> dict:
    """How the stored predictions did, against the market where it priced the same games.

    ``rps`` and ``market_rps`` are computed over *the same* fixtures - only those both
    played and priced - because comparing scores taken over different match sets is the
    easiest way to produce a flattering and meaningless number.
    """
    played = [match for match in once_per_fixture(predictions) if match.get("actual")]
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
