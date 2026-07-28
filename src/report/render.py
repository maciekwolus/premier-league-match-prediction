"""HTML for a fixture card.

Streamlit's own widgets stack vertically and cannot be packed three to a row at a
readable density, so a card is emitted as one block of markup instead. Keeping that
markup here rather than in ``app.py`` means the structure can be asserted in tests -
a card that silently loses its bookmaker row still renders perfectly well.
"""

from __future__ import annotations

from html import escape

from src.report.badges import badge_data_uri
from src.report.view import (
    as_percent,
    disagreement,
    most_likely_outcome,
    outcome_rows,
    scoreline_rows,
)

OUTCOME_LABELS = {"home": "H", "draw": "D", "away": "A"}

# Below this the model and the market agree closely enough that flagging it would be
# noise rather than information.
MEANINGFUL_EDGE = 0.02


def _badge(team: str) -> str:
    return (
        f'<img class="pl-badge" src="{badge_data_uri(team)}" '
        f'alt="{escape(team)} kit" width="40" height="40">'
    )


def _scoreline_bars(match: dict) -> str:
    rows = []
    for index, row in enumerate(scoreline_rows(match)):
        lead = " pl-lead" if index == 0 else ""
        rows.append(
            f'<div class="pl-score-row">'
            f'<span class="pl-score{lead}">{escape(row["score"])}</span>'
            f'<span class="pl-bar-track">'
            f'<span class="pl-bar{lead}" style="width:{row["width"] * 100:.0f}%"></span>'
            f"</span>"
            f'<span class="pl-pct">{row["label"]}</span>'
            f"</div>"
        )
    return "".join(rows)


def _outcome_grid(match: dict) -> str:
    cells = []
    for row in outcome_rows(match):
        edge = row["edge"]
        delta = ""
        if edge is not None and abs(edge) >= MEANINGFUL_EDGE:
            sign = "pl-up" if edge > 0 else "pl-down"
            delta = f'<span class="pl-edge {sign}">{edge * 100:+.0f}</span>'
        cells.append(
            f'<div class="pl-outcome">'
            f'<div class="pl-outcome-key">{OUTCOME_LABELS[row["outcome"]]}</div>'
            f'<div class="pl-outcome-val">{row["model_label"]}</div>'
            f"{delta}"
            f"</div>"
        )
    return f'<div class="pl-outcomes">{"".join(cells)}</div>'


def _market_row(match: dict) -> str:
    rows = outcome_rows(match)
    if rows[0]["bookmaker"] is None:
        return '<div class="pl-market pl-market-empty">NO MARKET YET</div>'

    parts = " ".join(f"{OUTCOME_LABELS[row['outcome']]} {row['bookmaker_label']}" for row in rows)
    return f'<div class="pl-market">BOOKMAKER &nbsp;{parts}</div>'


def _flag(match: dict) -> str:
    gap = disagreement(match)
    if not gap:
        return ""
    outcome, edge = gap
    direction = "OVER" if edge > 0 else "UNDER"
    return (
        f'<div class="pl-flag">{direction}RATES {OUTCOME_LABELS[outcome]} '
        f"BY {abs(edge) * 100:.0f} PTS vs MARKET</div>"
    )


def match_card(match: dict) -> str:
    """One fixture as a self-contained block of HTML."""
    home, away = escape(match["home_team"]), escape(match["away_team"])
    verdict = most_likely_outcome(match).upper()
    expected = match.get("expected_goals", {})
    goals = (
        f'<span class="pl-xg">xG {expected.get("home", 0):.2f} - '
        f"{expected.get('away', 0):.2f}</span>"
        if expected
        else ""
    )

    return (
        '<div class="pl-card">'
        f'<div class="pl-date">{escape(match["date"])}{goals}</div>'
        '<div class="pl-teams">'
        f'<div class="pl-team">{_badge(match["home_team"])}'
        f'<span class="pl-name">{home}</span></div>'
        '<div class="pl-v">v</div>'
        f'<div class="pl-team">{_badge(match["away_team"])}'
        f'<span class="pl-name">{away}</span></div>'
        "</div>"
        f'<div class="pl-verdict">{verdict}</div>'
        f'<div class="pl-scores">{_scoreline_bars(match)}</div>'
        f"{_outcome_grid(match)}"
        f"{_market_row(match)}"
        f"{_flag(match)}"
        "</div>"
    )


def summary_bar(summary: dict, repeated: float) -> str:
    """The strip of headline numbers above the grid."""
    items = [
        ("FIXTURES", str(summary["fixtures"])),
        ("MODEL", summary["model"].upper()),
        ("WITH MARKET", f"{summary['with_odds']}/{summary['fixtures']}"),
        ("TOP SCORE REPEATS", as_percent(repeated)),
    ]
    cells = "".join(
        f'<div class="pl-stat"><span class="pl-stat-key">{key}</span>'
        f'<span class="pl-stat-val">{escape(value)}</span></div>'
        for key, value in items
    )
    return f'<div class="pl-statbar">{cells}</div>'
