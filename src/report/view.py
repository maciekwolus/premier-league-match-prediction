"""Shaping predictions for display.

Kept out of the Streamlit file so it can be tested without a browser: the formatting
decisions here are the ones that could quietly mislead - a rounded percentage that no
longer sums to 100, or a disagreement with the market shown without its sign.
"""

from __future__ import annotations

from pathlib import Path

from src.predict.archive import available_rounds, load_round
from src.report.results import attach_results, scorecard

OUTCOMES = ("home", "draw", "away")

# Below this the model and the market are saying the same thing, given that neither is
# precise to a percentage point. Above it, the disagreement is worth pointing at.
NOTABLE_DISAGREEMENT = 0.10


def round_options(root: Path | None = None) -> list[dict]:
    """Every stored round, newest first, as the selector needs it.

    Newest first because the round you want is almost always the most recent one, and a
    selector that opens on gameweek 1 in April makes you scroll past the whole season.
    """
    options = []
    for season_slug, gameweek in reversed(available_rounds(root)):
        options.append(
            {
                "season_slug": season_slug,
                "gameweek": gameweek,
                "label": f"{season_slug.replace('_', '/')}  ·  GW {gameweek}",
            }
        )
    return options


def load_round_predictions(
    season_slug: str,
    gameweek: int,
    root: Path | None = None,
    results: dict[str, dict] | None = None,
) -> list[dict]:
    """One stored round, with actual results attached where the matches were played.

    ``results`` is injectable so this can be exercised without reading ``data/`` - the
    tests build synthetic seasons and must stay meaningful when the real table changes.
    """
    return attach_results(load_round(season_slug, gameweek, root), results)


def season_scorecard(
    season_slug: str, root: Path | None = None, results: dict[str, dict] | None = None
) -> dict:
    """How every stored round of one season has fared, taken together.

    Season-to-date rather than per-round, because a round is ten matches and the honest
    reading of ten matches is that it says almost nothing.
    """
    matches: list[dict] = []
    for stored_season, gameweek in available_rounds(root):
        if stored_season == season_slug:
            matches.extend(load_round(stored_season, gameweek, root))
    return scorecard(attach_results(matches, results))


def as_percent(probability: float) -> str:
    return f"{round(probability * 100)}%"


def scoreline_rows(match: dict) -> list[dict]:
    """The most likely scorelines, with a bar width relative to the best of them.

    Scaling to the leader rather than to 100 is deliberate: the top scoreline is around
    12%, so bars drawn against a full scale would all be slivers and convey nothing.
    """
    scorelines = match.get("scorelines", [])
    if not scorelines:
        return []

    best = max(entry["probability"] for entry in scorelines) or 1.0
    return [
        {
            "score": entry["score"],
            "probability": entry["probability"],
            "label": as_percent(entry["probability"]),
            "width": entry["probability"] / best,
        }
        for entry in scorelines
    ]


def outcome_rows(match: dict) -> list[dict]:
    """Home / draw / away, with the bookmaker's view alongside where there is one."""
    outcome = match.get("outcome", {})
    bookmaker = match.get("bookmaker")

    rows = []
    for name in OUTCOMES:
        model_probability = outcome.get(name, 0.0)
        market_probability = bookmaker.get(name) if bookmaker else None
        rows.append(
            {
                "outcome": name,
                "model": model_probability,
                "model_label": as_percent(model_probability),
                "bookmaker": market_probability,
                "bookmaker_label": as_percent(market_probability) if bookmaker else "—",
                "edge": (model_probability - market_probability) if bookmaker else None,
            }
        )
    return rows


def most_likely_outcome(match: dict) -> str:
    outcome = match.get("outcome", {})
    if not outcome:
        return "—"
    best = max(OUTCOMES, key=lambda name: outcome.get(name, 0.0))
    return {"home": "Home win", "draw": "Draw", "away": "Away win"}[best]


def disagreement(match: dict) -> tuple[str, float] | None:
    """The outcome where the model differs most from the market, if it differs much.

    This is the only genuinely interesting thing a model can say once the market exists:
    not "who wins" - the odds already answer that - but "where do I disagree".
    """
    bookmaker = match.get("bookmaker")
    if not bookmaker:
        return None

    outcome = match.get("outcome", {})
    gaps = {name: outcome.get(name, 0.0) - bookmaker.get(name, 0.0) for name in OUTCOMES}

    # Both sets of probabilities sum to one, so whenever the draw agrees the home and
    # away gaps are exact mirrors and "largest" is a tie. Ties resolve to the first in
    # H/D/A order, which is arbitrary but stable - and the two readings say the same
    # thing anyway, since rating the home side higher *is* rating the away side lower.
    name = max(gaps, key=lambda key: abs(gaps[key]))

    if abs(gaps[name]) < NOTABLE_DISAGREEMENT:
        return None
    return name, gaps[name]


def summarise(predictions: list[dict]) -> dict:
    """Headline numbers for the top of the page."""
    if not predictions:
        return {
            "fixtures": 0,
            "model": "—",
            "date_range": "—",
            "with_odds": 0,
            "mode": "upcoming",
        }

    dates = sorted({match["date"] for match in predictions})
    date_range = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"

    return {
        "fixtures": len(predictions),
        "model": predictions[0].get("model", "—"),
        "date_range": date_range,
        "with_odds": sum(1 for match in predictions if match.get("bookmaker")),
        "mode": predictions[0].get("mode", "upcoming"),
    }


def modal_scoreline_share(predictions: list[dict]) -> float:
    """How often the same scoreline tops the list.

    Worth surfacing, because 1-1 leading nearly every card looks like a broken report
    when it is largely a property of the sport: with both sides expected to score around
    1.3 goals, 1-1 stays the single most likely result until one team is expected to
    score roughly 2.4. It only stops being true for genuine mismatches.
    """
    if not predictions:
        return 0.0

    leaders = [match["scorelines"][0]["score"] for match in predictions if match.get("scorelines")]
    if not leaders:
        return 0.0
    return max(leaders.count(score) for score in set(leaders)) / len(leaders)
