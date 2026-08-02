"""Streamlit report: this round's fixtures with predicted scorelines.

Run with, from the project folder:
    .venv\\Scripts\\python.exe -m streamlit run app.py

Reads the stored rounds under ``data/final/rounds/``, so produce one first:
    .venv\\Scripts\\python.exe -m src.predict.gameweek --replay

Layout only. The shaping lives in ``src/report/view.py``, the card markup in
``render.py`` and the styling in ``theme.py``, so all three can be checked without a
browser.
"""

from __future__ import annotations

import streamlit as st

from src.report.render import (
    empty_notice,
    legend_html,
    match_card,
    scorecard_bar,
    summary_bar,
)
from src.report.theme import CSS
from src.report.view import (
    load_round_predictions,
    modal_scoreline_share,
    round_options,
    season_scorecard,
    summarise,
)

# Three across is what the card was sized for. Streamlit stacks columns on narrow
# screens by itself, so no media query is needed.
COLUMNS = 3

st.set_page_config(page_title="Premier League predictions", page_icon="⚽", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def render_empty() -> None:
    st.markdown(empty_notice(), unsafe_allow_html=True)


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


def choose_round(options: list[dict]) -> dict:
    """Pick a stored round: season along the top, then the rounds within it.

    Page-level Streamlit widgets, which is fine and is the point: the rule that
    interactivity must be CSS applies to a *card*, because a card is one block of markup
    and a rerun would reflow the grid. Rerunning to change which round is shown is exactly
    what should happen.

    This was a dropdown and it was the wrong control twice over. A ``selectbox`` filters
    as you type, which is baffling with two options; and it hides what exists until you
    open it, when "which rounds have we predicted?" is a question the page should answer
    without being asked. Laid out flat, the archive is legible at a glance.

    Each level disappears when it has one choice, because a control with a single option
    is furniture rather than a control.
    """
    seasons = list(dict.fromkeys(option["season_slug"] for option in options))

    season = seasons[0]
    if len(seasons) > 1:
        st.markdown('<div class="pl-picker-key">SEASON</div>', unsafe_allow_html=True)
        picked = st.segmented_control(
            "Season",
            seasons,
            default=seasons[0],
            format_func=lambda slug: slug.replace("_", "/"),
            label_visibility="collapsed",
        )
        # These controls are deselectable, so a second click on the active choice returns
        # None. Falling back to the newest season keeps the page showing something.
        season = picked or seasons[0]

    rounds = [option for option in options if option["season_slug"] == season]
    if len(rounds) == 1:
        return rounds[0]

    # Ascending, so the strip reads like a season running left to right. The options
    # arrive newest-first, which is right for the season control and for choosing a
    # default, but backwards for a row of round numbers.
    gameweeks = sorted(option["gameweek"] for option in rounds)
    st.markdown('<div class="pl-picker-key">ROUND</div>', unsafe_allow_html=True)
    chosen = st.pills(
        "Round",
        gameweeks,
        # The latest round is the one you almost always want.
        default=gameweeks[-1],
        format_func=lambda number: f"GW {number}",
        label_visibility="collapsed",
        # Keyed by season so switching seasons does not carry a gameweek across that the
        # new season has never stored.
        key=f"round-{season}",
    )
    chosen = gameweeks[-1] if chosen is None else chosen
    return next(option for option in rounds if option["gameweek"] == chosen)


def main() -> None:
    st.markdown("# PREMIER LEAGUE ⚽ PREDICTIONS")

    options = round_options()
    if not options:
        st.markdown('<div class="pl-sub">no round loaded</div>', unsafe_allow_html=True)
        render_empty()
        return

    selected = choose_round(options)
    predictions = load_round_predictions(selected["season_slug"], selected["gameweek"])

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

    # The record comes before the fixtures deliberately. A page that shows predictions
    # and hides how they turned out is marketing; this is the number that qualifies
    # everything below it.
    card = season_scorecard(selected["season_slug"])
    st.markdown(
        scorecard_bar(card, selected["season_slug"].replace("_", "/")),
        unsafe_allow_html=True,
    )

    render_grid(predictions)


main()
