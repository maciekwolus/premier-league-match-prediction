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


def _eleven(players: list[dict]) -> str:
    """One side's expected XI as rows of name, position and rating."""
    if not players:
        return '<div class="pl-xi-empty">no recent history</div>'

    rows = []
    for player in players:
        overall = player.get("overall")
        rating = (
            f'<span class="pl-xi-rating">{overall}</span>'
            if overall is not None
            else '<span class="pl-xi-rating pl-xi-unrated" title="no rating found">–</span>'
        )
        rows.append(
            f'<div class="pl-xi-row">'
            f'<span class="pl-xi-pos pl-line-{escape(player.get("line", "unknown"))}">'
            f"{escape(player.get('position', ''))}</span>"
            f'<span class="pl-xi-name">{escape(player["player"])}</span>'
            f"{rating}</div>"
        )
    return "".join(rows)


def _lineups(match: dict) -> str:
    """The expected XIs, collapsed behind a toggle.

    A ``details`` element rather than a Streamlit widget: the card is one block of
    markup, and opening it must not rerun the script or reflow the grid.

    Labelled for what it actually is. These are the players a club has started most
    often recently, not a team sheet - nobody knows the real one until an hour before
    kick-off, and in August it is last season's side.
    """
    lineups = match.get("lineups") or {}
    home, away = lineups.get("home", []), lineups.get("away", [])
    if not home and not away:
        return ""

    def mean(players: list[dict]) -> str:
        rated = [p["overall"] for p in players if p.get("overall") is not None]
        return f"{sum(rated) / len(rated):.1f}" if rated else "–"

    return (
        '<details class="pl-lineups">'
        "<summary>EXPECTED XI</summary>"
        '<div class="pl-xi-note">Most-used eleven from each club\'s last 6 matches, '
        "with FIFA overall ratings. Not a team sheet.</div>"
        '<div class="pl-xi-grid">'
        f'<div class="pl-xi-side"><div class="pl-xi-head">'
        f'{escape(match["home_team"])}<span class="pl-xi-mean">{mean(home)}</span></div>'
        f"{_eleven(home)}</div>"
        f'<div class="pl-xi-side"><div class="pl-xi-head">'
        f'{escape(match["away_team"])}<span class="pl-xi-mean">{mean(away)}</span></div>'
        f"{_eleven(away)}</div>"
        "</div></details>"
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
        f"{_lineups(match)}"
        "</div>"
    )


# Each row pairs a sample of the real thing with what it means. Showing the actual
# markup rather than describing it means the legend cannot drift out of step with the
# cards: both are built from the same CSS classes.
LEGEND_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "SCORELINES",
        '<div class="pl-score-row">'
        '<span class="pl-score pl-lead">1-1</span>'
        '<span class="pl-bar-track"><span class="pl-bar pl-lead" style="width:100%"></span></span>'
        '<span class="pl-pct">13%</span></div>'
        '<div class="pl-score-row">'
        '<span class="pl-score">2-1</span>'
        '<span class="pl-bar-track"><span class="pl-bar" style="width:70%"></span></span>'
        '<span class="pl-pct">9%</span></div>',
        "The three most likely exact scores, longest bar first. Even a perfect model "
        "tops out near <b>12%</b> — football is that random — so read these as "
        "<i>most likely</i>, never as a prediction.",
    ),
    (
        "H / D / A",
        '<div class="pl-outcomes">'
        '<div class="pl-outcome"><div class="pl-outcome-key">H</div>'
        '<div class="pl-outcome-val">47%</div>'
        '<span class="pl-edge pl-up">+8</span></div>'
        '<div class="pl-outcome"><div class="pl-outcome-key">D</div>'
        '<div class="pl-outcome-val">27%</div></div>'
        '<div class="pl-outcome"><div class="pl-outcome-key">A</div>'
        '<div class="pl-outcome-val">26%</div>'
        '<span class="pl-edge pl-down">-7</span></div></div>',
        "Home win, draw, away win. The small <b class='pl-up'>green</b> or "
        "<b class='pl-down'>red</b> number is how many percentage points the model sits "
        "<i>above or below the bookmaker</i>. No number means they broadly agree.",
    ),
    (
        "BOOKMAKER",
        '<div class="pl-market">BOOKMAKER &nbsp;H 39% D 27% A 33%</div>',
        "The market's own view, with the bookmaker's margin stripped out. It is the "
        "benchmark, not decoration: across 2,280 matches the closing line still beats "
        "every model here.",
    ),
    (
        "DISAGREEMENT",
        '<div class="pl-flag">UNDERRATES A BY 12 PTS vs MARKET</div>',
        "Shown only when the model differs from the market by more than ten points. "
        "Once odds exist, <i>where the model disagrees</i> is the only thing it can add — "
        "the odds already answer who wins.",
    ),
    (
        "xG",
        '<div class="pl-date"><span>2026-05-24</span>'
        '<span class="pl-xg">xG 1.50 - 1.06</span></div>',
        "Expected goals: how many each side is predicted to score. Everything else on "
        "the card is derived from these two numbers.",
    ),
    (
        "EXPECTED XI",
        '<details class="pl-lineups" open><summary>EXPECTED XI</summary>'
        '<div class="pl-xi-grid"><div class="pl-xi-side">'
        '<div class="pl-xi-head">Arsenal<span class="pl-xi-mean">84.7</span></div>'
        '<div class="pl-xi-row"><span class="pl-xi-pos pl-line-gk">GK</span>'
        '<span class="pl-xi-name">David Raya</span>'
        '<span class="pl-xi-rating">87</span></div>'
        '<div class="pl-xi-row"><span class="pl-xi-pos pl-line-def">DC</span>'
        '<span class="pl-xi-name">Gabriel</span>'
        '<span class="pl-xi-rating">88</span></div>'
        "</div></div></details>",
        "Open a card's <b>EXPECTED XI</b> for both sides, with each player's FIFA "
        "overall and the team average. This is the eleven a club has started most often "
        "in its last six matches — <i>not</i> a team sheet, which nobody has until an "
        "hour before kick-off. In August it is last season's side.",
    ),
    (
        "KITS",
        f'<div style="display:flex;gap:0.5rem">{_badge("Newcastle")}'
        f"{_badge('Aston Villa')}{_badge('West Ham')}</div>",
        "Pixel kits, <b>not club badges</b> — real crests are trademarked. Colours and "
        "pattern follow each club's actual shirt, so Newcastle are striped and West Ham "
        "wear a sash.",
    ),
)


def legend_html() -> str:
    """A guide to the card, built from the same pieces the card is."""
    rows = "".join(
        f'<div class="pl-legend-row">'
        f'<div class="pl-legend-key">{key}</div>'
        f'<div class="pl-legend-sample">{sample}</div>'
        f'<div class="pl-legend-text">{text}</div>'
        f"</div>"
        for key, sample, text in LEGEND_ROWS
    )
    return (
        '<div class="pl-legend">'
        '<div class="pl-legend-lead">Every card is one fixture. The model is trained on '
        "seven seasons of results, lineups and player ratings, and never sees anything "
        "that happened after kick-off.</div>"
        f"{rows}"
        '<div class="pl-legend-foot">The honest headline: over 2,280 matches the '
        "bookmaker scores <b>0.1965</b> and the best model here <b>0.2027</b> — lower is "
        "better. Nothing beats the market, which is the expected result.</div>"
        "</div>"
    )


def summary_bar(summary: dict, repeated: float) -> str:
    """The strip of headline numbers above the grid.

    Each carries a hover explanation, because "TOP SCORE REPEATS 80%" means nothing to
    someone seeing the page for the first time.
    """
    items = [
        ("FIXTURES", str(summary["fixtures"]), "Matches in this round."),
        (
            "MODEL",
            summary["model"].upper(),
            "Which model produced these numbers. dixon-coles-squad estimates each "
            "club's attack and defence from results, then adjusts for how strong "
            "today's expected XI is against that club's usual one.",
        ),
        (
            "WITH MARKET",
            f"{summary['with_odds']}/{summary['fixtures']}",
            "How many fixtures have bookmaker odds to compare against. A market does "
            "not exist until close to kick-off.",
        ),
        (
            "TOP SCORE REPEATS",
            as_percent(repeated),
            "How often the same scoreline heads the list. 1-1 dominates because it "
            "stays the single most likely result until one side is expected to score "
            "about 2.4 goals — so this is high in any round without a mismatch.",
        ),
    ]
    cells = "".join(
        f'<div class="pl-stat" title="{escape(hint)}">'
        f'<span class="pl-stat-key">{key}</span>'
        f'<span class="pl-stat-val">{escape(value)}</span></div>'
        for key, value, hint in items
    )
    return f'<div class="pl-statbar">{cells}</div>'
