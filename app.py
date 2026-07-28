"""Streamlit report: this round's fixtures with predicted scorelines.

Run with:
    streamlit run app.py

Reads ``data/final/predictions.json``, so produce that first:
    python -m src.predict.gameweek --replay

Layout only. The shaping lives in ``src/report/view.py``, the card markup in
``render.py`` and the styling in ``theme.py``, so all three can be checked without a
browser.
"""

from __future__ import annotations

import streamlit as st

from src.report.render import legend_html, match_card, summary_bar
from src.report.theme import CSS
from src.report.view import load_predictions, modal_scoreline_share, summarise

# Three across is what the card was sized for. Streamlit stacks columns on narrow
# screens by itself, so no media query is needed.
COLUMNS = 3

st.set_page_config(page_title="Premier League predictions", page_icon="⚽", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def render_empty() -> None:
    st.markdown(
        '<div class="pl-notice"><b>NO PREDICTIONS YET.</b><br><br>'
        "Generate them with <code>python -m src.predict.gameweek</code>.<br>"
        "Between seasons the fixture feed is empty, so use the last round that was "
        "actually played: <code>python -m src.predict.gameweek --replay</code>."
        "</div>",
        unsafe_allow_html=True,
    )


def render_notices(summary: dict, repeated: float) -> None:
    if summary["mode"] == "replay":
        st.markdown(
            f'<div class="pl-notice"><b>REPLAY — THESE ARE PAST MATCHES.</b><br><br>'
            f"The round of {summary['date_range']}, re-predicted from only what was "
            f"known beforehand. The fixture feed carries no Premier League matches "
            f"between seasons. To predict real fixtures, list them in "
            f"<code>data/manual/upcoming_fixtures.csv</code> and re-run without "
            f"<code>--replay</code>.</div>",
            unsafe_allow_html=True,
        )

    if repeated >= 0.5:
        st.markdown(
            f'<div class="pl-notice">The same scoreline tops <b>{repeated:.0%}</b> of '
            f"these cards. That is mostly the sport: with both sides expected to score "
            f"around 1.3, 1-1 stays the single most likely result until one team is "
            f"expected to score about 2.4. The home/draw/away split is where fixtures "
            f"genuinely differ.</div>",
            unsafe_allow_html=True,
        )


def render_grid(predictions: list[dict]) -> None:
    for start in range(0, len(predictions), COLUMNS):
        row = predictions[start : start + COLUMNS]
        columns = st.columns(COLUMNS, gap="medium")
        for column, match in zip(columns, row, strict=False):
            with column:
                st.markdown(match_card(match), unsafe_allow_html=True)


def main() -> None:
    st.markdown("# PREMIER LEAGUE ⚽ PREDICTIONS")

    predictions = load_predictions()
    if not predictions:
        st.markdown('<div class="pl-sub">no round loaded</div>', unsafe_allow_html=True)
        render_empty()
        return

    summary = summarise(predictions)
    repeated = modal_scoreline_share(predictions)

    st.markdown(
        f'<div class="pl-sub">{summary["date_range"]} &nbsp;·&nbsp; scorelines with '
        f"honest probabilities, shown against the bookmaker</div>",
        unsafe_allow_html=True,
    )

    render_notices(summary, repeated)

    # Collapsed by default: the page should stay a scoreboard for anyone who already
    # knows how to read it, and explain itself on request for anyone who does not.
    with st.expander("❔  HOW TO READ THIS", expanded=False):
        st.markdown(legend_html(), unsafe_allow_html=True)

    st.markdown(summary_bar(summary, repeated), unsafe_allow_html=True)
    render_grid(predictions)


main()
