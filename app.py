"""Streamlit report: this round's fixtures with predicted scorelines.

Run with:
    streamlit run app.py

Reads ``data/final/predictions.json``, so produce that first:
    python -m src.predict.gameweek --replay
"""

from __future__ import annotations

import streamlit as st

from src.report.view import (
    disagreement,
    load_predictions,
    modal_scoreline_share,
    most_likely_outcome,
    outcome_rows,
    scoreline_rows,
    summarise,
)

OUTCOME_LABELS = {"home": "Home", "draw": "Draw", "away": "Away"}

# Anything smaller than this rounds to nothing and is not a disagreement worth showing.
MEANINGFUL_EDGE = 0.02

st.set_page_config(page_title="Premier League predictions", page_icon="⚽", layout="centered")


def render_scorelines(match: dict) -> None:
    st.caption("Most likely scorelines")
    for row in scoreline_rows(match):
        bar, label = st.columns([5, 1])
        with bar:
            st.progress(row["width"], text=row["score"])
        with label:
            st.markdown(f"**{row['label']}**")


def render_outcomes(match: dict) -> None:
    rows = outcome_rows(match)
    has_market = rows[0]["bookmaker"] is not None

    columns = st.columns(3)
    for column, row in zip(columns, rows, strict=True):
        with column:
            # Only label a gap worth noticing. A rounded "+0 pts" is noise, and reads
            # as a finding when it is agreement.
            edge = row["edge"]
            delta = (
                f"{edge * 100:+.0f} pts vs market"
                if edge is not None and abs(edge) >= MEANINGFUL_EDGE
                else None
            )
            st.metric(
                OUTCOME_LABELS[row["outcome"]], row["model_label"], delta=delta, delta_color="off"
            )

    if has_market:
        market = " · ".join(
            f"{OUTCOME_LABELS[row['outcome']]} {row['bookmaker_label']}" for row in rows
        )
        st.caption(f"Bookmaker: {market}")
    else:
        st.caption("No bookmaker line for this fixture yet.")


def render_match(match: dict) -> None:
    with st.container(border=True):
        heading, verdict = st.columns([3, 1])
        with heading:
            st.subheader(f"{match['home_team']} vs {match['away_team']}")
            st.caption(match["date"])
        with verdict:
            st.caption("Most likely")
            st.markdown(f"**{most_likely_outcome(match)}**")

        render_scorelines(match)
        st.divider()
        render_outcomes(match)

        gap = disagreement(match)
        if gap:
            outcome, edge = gap
            direction = "higher" if edge > 0 else "lower"
            st.info(
                f"Biggest disagreement: the model rates **{OUTCOME_LABELS[outcome].lower()}** "
                f"{abs(edge) * 100:.0f} points {direction} than the market."
            )


def main() -> None:
    st.title("Premier League predictions")

    predictions = load_predictions()

    if not predictions:
        st.warning("No predictions yet.")
        st.markdown(
            "Generate them first:\n\n"
            "```\npython -m src.predict.gameweek\n```\n\n"
            "Between seasons the fixture feed is empty, so use the last round that was "
            "actually played:\n\n"
            "```\npython -m src.predict.gameweek --replay\n```"
        )
        return

    summary = summarise(predictions)

    if summary["mode"] == "replay":
        st.warning(
            f"**These are past matches, not upcoming ones.** This is a replay of the round "
            f"of {summary['date_range']}, re-predicted using only what was known "
            f"beforehand. The fixture feed carries no Premier League matches between "
            f"seasons, so this is the only way to exercise the report out of season. "
            f"To predict real fixtures, list them in "
            f"`data/manual/upcoming_fixtures.csv` and re-run without `--replay`."
        )

    left, middle, right = st.columns(3)
    left.metric("Fixtures", summary["fixtures"])
    middle.metric("Model", summary["model"])
    right.metric("With a market line", f"{summary['with_odds']}/{summary['fixtures']}")
    st.caption(summary["date_range"])

    st.markdown(
        "Exact scorelines top out around 12% probability, so these are the *most likely* "
        "results rather than confident calls. The bookmaker's line is shown alongside "
        "because it is the only honest reference for whether a prediction is worth much."
    )

    share = modal_scoreline_share(predictions)
    if share >= 0.5:
        st.caption(
            f"⚠️ The same scoreline tops {share:.0%} of these cards. That is mostly the "
            f"sport rather than a fault: with both sides expected to score around 1.3, "
            f"1-1 stays the single most likely result until one team is expected to "
            f"score about 2.4. Read the home/draw/away split for the real differences "
            f"between fixtures — those vary far more than the top scoreline does."
        )

    st.divider()

    for match in predictions:
        render_match(match)


main()
