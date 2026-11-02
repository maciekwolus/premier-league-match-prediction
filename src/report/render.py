"""HTML for a fixture card.

Streamlit's own widgets stack vertically and cannot be packed three to a row at a
readable density, so a card is emitted as one block of markup instead. Keeping that
markup here rather than in ``app.py`` means the structure can be asserted in tests -
a card that silently loses its bookmaker row still renders perfectly well.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from src.betting.prices import BOOKMAKER
from src.betting.simulate import MARKETS, STAKE, STAKE_TAX
from src.report.badges import badge_data_uri
from src.report.results import verdict
from src.report.view import (
    as_percent,
    disagreement,
    most_likely_outcome,
    outcome_rows,
    scoreline_rows,
)

OUTCOME_LABELS = {"home": "H", "draw": "D", "away": "A"}

MARKET_NAMES = {"outcome": "home/draw/away", "score": "exact score"}

# On the page wherever a balance is, so a figure cannot be read away from what it rests
# on. This was meant to be a Polish bookmaker, and the reader should know it is not.
PRICE_NOTE = (
    f"<b>Simulation, and not Polish prices.</b> Home/draw/away uses {BOOKMAKER} closing odds. "
    "Exact-score odds are estimated from that line with a 25% bookmaker margin, because no "
    "source we may use records them. Each stake loses the 12% Polish tax first."
)

# Shown in the empty state, so the reader can paste a cd that is right for their machine
# rather than one copied from whoever wrote the docs.
REPO_ROOT_HINT = str(Path(__file__).resolve().parents[2])

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


LINES = ("gk", "def", "mid", "att")


def _slug(value: str) -> str:
    """A value safe to use as an HTML id."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _mean_rating(players: list[dict]) -> str:
    rated = [p["overall"] for p in players if p.get("overall") is not None]
    return f"{sum(rated) / len(rated):.1f}" if rated else "–"


def _player_token(player: dict, team: str) -> str:
    """One player on the pitch: kit, name, rating."""
    overall = player.get("overall")
    rating = (
        f'<span class="pl-token-rating">{overall}</span>'
        if overall is not None
        else '<span class="pl-token-rating pl-xi-unrated" title="no rating found">–</span>'
    )
    surname = player["player"].split()[-1] if player["player"].split() else player["player"]
    return (
        f'<div class="pl-token" title="{escape(player["player"])} '
        f'({escape(player.get("position", ""))})">'
        f'<img class="pl-token-kit" src="{badge_data_uri(team, scale=3)}" '
        f'alt="" width="26" height="26">'
        f'<span class="pl-token-name">{escape(surname)}</span>'
        f"{rating}</div>"
    )


def _formation(players: list[dict], team: str, invert: bool) -> str:
    """A side laid out by line, as it would stand on a pitch.

    ``invert`` reverses the order for the away side, so both teams face each other the
    way a broadcast graphic shows them - attackers meeting at the halfway line.
    """
    if not players:
        return '<div class="pl-xi-empty">no recent history for this club</div>'

    by_line = {line: [p for p in players if p.get("line") == line] for line in LINES}
    spare = [p for p in players if p.get("line") not in LINES]
    if spare:
        by_line["mid"] = by_line["mid"] + spare

    order = LINES if not invert else tuple(reversed(LINES))
    rows = [
        f'<div class="pl-pitch-row">'
        f"{''.join(_player_token(player, team) for player in by_line[line])}"
        f"</div>"
        for line in order
        if by_line[line]
    ]

    shape = "-".join(str(len(by_line[line])) for line in ("def", "mid", "att") if by_line[line])
    return (
        f'<div class="pl-side {"pl-side-away" if invert else "pl-side-home"}">'
        f'<div class="pl-side-head">'
        f'<span class="pl-side-name">{escape(team)}</span>'
        f'<span class="pl-side-shape">{shape}</span>'
        f'<span class="pl-side-mean">{_mean_rating(players)}</span>'
        f"</div>"
        f"{''.join(rows)}</div>"
    )


def _lineups(match: dict) -> str:
    """The expected XIs, shown on a pitch in an overlay.

    The toggle is a hidden checkbox and the button a ``label``, because a card is one
    block of markup: a Streamlit widget would rerun the script and reflow the grid, and
    a ``details`` element pushed the rest of the page down when it opened. This way the
    overlay floats above everything and the grid never moves.

    Labelled for what it actually is. These are the players a club has started most
    often recently, not a team sheet - nobody knows the real one until an hour before
    kick-off, and in August it is last season's side.
    """
    lineups = match.get("lineups") or {}
    home, away = lineups.get("home", []), lineups.get("away", [])
    if not home and not away:
        return ""

    toggle = f"xi-{_slug(match['match_id'])}"
    title = f"{escape(match['home_team'])} v {escape(match['away_team'])}"

    return (
        f'<input type="checkbox" id="{toggle}" class="pl-modal-toggle" hidden>'
        f'<label for="{toggle}" class="pl-xi-button" role="button" tabindex="0">'
        f"⚽ EXPECTED XI</label>"
        f'<div class="pl-modal">'
        f'<label for="{toggle}" class="pl-modal-backdrop" aria-label="Close"></label>'
        f'<div class="pl-modal-box" role="dialog">'
        f'<div class="pl-modal-head">'
        f'<span class="pl-modal-title">{title}</span>'
        f'<label for="{toggle}" class="pl-modal-close" role="button" '
        f'aria-label="Close">✕</label>'
        f"</div>"
        f'<div class="pl-pitch">'
        f"{_formation(away, match['away_team'], invert=True)}"
        f'<div class="pl-halfway"></div>'
        f"{_formation(home, match['home_team'], invert=False)}"
        f"</div>"
        f'<div class="pl-modal-note">{_xi_note(home, away)}</div>'
        f"</div></div>"
    )


def _xi_note(home: list[dict], away: list[dict]) -> str:
    """What the eleven actually is, which is not the same for every club.

    A promoted side has no history in this division, so its XI comes from ratings — who
    its best players are — rather than from who has been starting. Describing that as a
    most-used eleven would be a quiet lie about exactly the club a reader knows least
    about, so the note changes when the source does.
    """
    from_ratings = any(
        player.get("source") == "ratings" for side in (home, away) for player in side
    )
    caveat = (
        " <b>Where a club is newly promoted</b> there is no history in this division to "
        "read an eleven from, so its side is its best-rated players instead — who they "
        "are, not who will start."
        if from_ratings
        else ""
    )

    return (
        "Most-used eleven from each club's last 6 matches, with FIFA overall ratings and "
        "the team average. <b>Not a team sheet</b> — nobody has one until an hour before "
        f"kick-off.{caveat}"
    )


def _result_row(match: dict) -> str:
    """What actually happened, on a card for a match that has been played.

    Both claims are shown and they are not the same size. Getting the outcome right is
    ordinary - the favourite wins most weeks. Getting the exact scoreline right is a
    one-in-eight shot at best, so it is marked when it lands and passed over quietly when
    it does not, rather than displayed as a failure.
    """
    result = verdict(match)
    if not result:
        return ""

    if result["exact"] and result["outcome"]:
        note = '<span class="pl-hit">EXACT SCORE CALLED</span>'
    elif result["exact"]:
        # Both facts, because they disagree and the flattering one is not the whole
        # story. A 1-1 leading scoreline sits happily on a card whose headline verdict is
        # HOME WIN - the draw is the most likely single score while the home win is the
        # most likely outcome - so an exact hit here came with the wrong call above it.
        # Showing only the green half would be advertising.
        note = '<span class="pl-hit">EXACT SCORE</span><span class="pl-miss">WRONG CALL</span>'
    elif result["outcome"]:
        note = '<span class="pl-hit pl-hit-soft">OUTCOME RIGHT</span>'
    else:
        note = '<span class="pl-miss">MISSED</span>'

    return (
        f'<div class="pl-result">'
        f'<span class="pl-result-key">FINAL</span>'
        f'<span class="pl-result-score">{escape(result["score"])}</span>'
        f"{note}"
        f"</div>"
    )


def match_card(match: dict) -> str:
    """One fixture as a self-contained block of HTML."""
    home, away = escape(match["home_team"]), escape(match["away_team"])
    most_likely = most_likely_outcome(match).upper()
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
        f'<div class="pl-verdict">{most_likely}</div>'
        f"{_result_row(match)}"
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
        "FINAL",
        '<div class="pl-result">'
        '<span class="pl-result-key">FINAL</span>'
        '<span class="pl-result-score">2-1</span>'
        '<span class="pl-hit">EXACT SCORE CALLED</span></div>'
        '<div class="pl-result">'
        '<span class="pl-result-key">FINAL</span>'
        '<span class="pl-result-score">0-3</span>'
        '<span class="pl-miss">MISSED</span></div>',
        "On a round that has been played, what actually happened — and whether the "
        "prediction stood up. <b>EXACT SCORE CALLED</b> is the leading scoreline landing, "
        "which is roughly a one-in-eight shot. <b>OUTCOME RIGHT</b> is the weaker claim "
        "that the most likely of home/draw/away happened, which the favourite manages "
        "about half the time anyway.",
    ),
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
        '<span class="pl-xi-button" style="cursor:default">⚽ EXPECTED XI</span>',
        "The button on every card. It opens both sides laid out on a pitch, each player "
        "with their FIFA overall, plus the shape and the team average. This is the "
        "eleven a club has started most often in its last six matches — <i>not</i> a "
        "team sheet, which nobody has until an hour before kick-off. In August it is "
        "last season's side.",
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


def empty_notice(repo_root: str = REPO_ROOT_HINT) -> str:
    """What the page shows with no predictions on disk.

    This is the one message a stuck reader is guaranteed to see, so it carries commands
    that must work as pasted. Both spell out the interpreter inside the virtual
    environment rather than a bare ``python``, which only resolves once the environment
    is activated - and they lead with ``cd``, because running them from the wrong folder
    is what actually happens. PowerShell reports that as "the module '.venv' could not be
    loaded", which names neither the directory nor the real problem.
    """
    return (
        '<div class="pl-notice"><b>NO PREDICTIONS YET.</b><br><br>'
        "The report reads <code>data/final/rounds/</code> and no round has been "
        "stored yet. Predict one — from the project folder, in a terminal:<br><br>"
        f"<code>cd {escape(repo_root)}</code><br>"
        "<code>.venv\\Scripts\\python.exe -m src.predict.gameweek --replay</code>"
        "<br><br>Then reload this page. <code>--replay</code> predicts the last round "
        "that was actually played; between seasons the fixture feed is empty, so drop "
        "the flag only once real fixtures exist. On macOS or Linux the interpreter is "
        "<code>.venv/bin/python</code>."
        "</div>"
    )


def scorecard_bar(card: dict, season_label: str) -> str:
    """How the stored predictions have actually done, season to date.

    Deliberately leads with the comparison rather than the hit count. "6 of 10 outcomes
    right" sounds like a result and is not one - the favourite wins about that often.
    Our RPS beside the bookmaker's over the same fixtures is the only line here that
    carries information, and when the sample is too small to read, the bar says so
    instead of letting a good week look like skill.
    """
    if not card["played"]:
        return ""

    cells = [
        ("SEASON", escape(season_label), "Which season these stored predictions cover."),
        (
            "SCORED",
            f"{card['played']}",
            "Matches predicted before kick-off that have since been played.",
        ),
        (
            "OUTCOME RIGHT",
            f"{card['outcome']}/{card['played']}",
            "How often the most likely of home/draw/away happened. The favourite wins "
            "roughly half the time, so this is a floor to beat, not an achievement.",
        ),
        (
            "EXACT SCORE",
            f"{card['exact']}/{card['played']}",
            "How often the leading scoreline was the final score. One in eight is about "
            "the ceiling for anyone.",
        ),
    ]

    if card["market_rps"] is not None:
        gap = card["rps"] - card["market_rps"]
        verdict_text = "ahead" if gap < 0 else "behind"
        cells.append(
            (
                "RPS vs MARKET",
                f"{card['rps']:.4f} / {card['market_rps']:.4f}",
                f"Ranked Probability Score, ours then the bookmaker's, over the "
                f"{card['compared']} fixtures both of us priced. Lower is better, so we "
                f"are {verdict_text} by {abs(gap):.4f}.",
            )
        )

    rendered = "".join(
        f'<div class="pl-stat" title="{escape(hint)}">'
        f'<span class="pl-stat-key">{key}</span>'
        f'<span class="pl-stat-val">{value}</span></div>'
        for key, value, hint in cells
    )

    caveat = ""
    if card["small_sample"]:
        ahead = card["market_rps"] is not None and card["rps"] < card["market_rps"]
        beating = (
            " <b>This sample currently shows us ahead of the market — do not believe it.</b> "
            "The walk-forward backtest over 2,280 matches has the closing line winning by "
            "a clear margin, and a run of fixtures that says otherwise is the noise, not "
            "the signal."
            if ahead
            else ""
        )
        caveat = (
            '<div class="pl-notice pl-notice-quiet">'
            f"<b>{card['played']} matches is not a sample.</b> These numbers are here "
            "because the record should be visible from the first round, not because a "
            "round proves anything — over ten fixtures the gap between a good model and "
            f"a bad one is mostly luck.{beating}"
            "</div>"
        )

    return f'<div class="pl-statbar pl-scorecard">{rendered}</div>{caveat}'


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


def _money(amount: float) -> str:
    """A signed amount, coloured by which side of zero it landed on."""
    tone = "pl-gain" if amount > 0 else "pl-loss" if amount < 0 else ""
    return f'<span class="{tone}">{amount:+,.2f}&nbsp;zł</span>'


def betting_bar(record: dict, past: dict | None) -> str:
    """Where ten zloty a match would stand this season, in both markets.

    Written to be hard to misread, because a page showing a profit is the one most likely
    to be believed without a second look. It therefore always says the prices are not
    Polish and that the exact-score ones are estimated, and any balance above zero is
    contradicted out loud with the backtest, the same rule the scorecard follows when a
    small sample happens to beat the market.
    """
    total = record["total"]
    outcome, score = total["outcome"], total["score"]
    if not total["matches"]:
        return ""

    cells = [
        (
            "IF WE HAD BET",
            f"{STAKE:.0f} ZŁ × {total['matches']}",
            "Ten zloty on every settled match this season, in each market. Rounds "
            "predicted before kick-off only, and every fixture, so nothing is hand-picked.",
        ),
        (
            "HOME/DRAW/AWAY",
            _money(outcome["balance"]),
            f"Won {outcome['won']} of {outcome['bets']} at {BOOKMAKER} closing odds, after "
            f"the {STAKE_TAX:.0%} stake tax. Without the tax: "
            f"{outcome['untaxed_balance']:+,.2f} zł.",
        ),
        (
            "EXACT SCORE (EST.)",
            _money(score["balance"]),
            f"Won {score['won']} of {score['bets']} at estimated prices, after the "
            f"{STAKE_TAX:.0%} stake tax. Without the tax: "
            f"{score['untaxed_balance']:+,.2f} zł.",
        ),
        (
            "STAKED EACH",
            f"{outcome['staked']:,.0f} ZŁ",
            "What each market has put on so far. The two are separate bets, so the "
            "combined outlay is double this.",
        ),
    ]
    rendered = "".join(
        f'<div class="pl-stat" title="{escape(hint)}">'
        f'<span class="pl-stat-key">{key}</span>'
        f'<span class="pl-stat-val">{value}</span></div>'
        for key, value, hint in cells
    )

    contradiction = ""
    ahead = [market for market in MARKETS if total[market]["balance"] > 0]
    if ahead and past:
        losses = " and ".join(
            f"{MARKET_NAMES[market]} lost {abs(past['total'][market]['balance']):,.0f} zł"
            for market in ahead
        )
        contradiction = (
            " <b>A balance above zero here should not be believed:</b> over "
            f"{past['total']['matches']:,} backtested matches, {losses}."
        )

    caveat = (
        '<div class="pl-notice pl-notice-quiet">'
        f"<b>{total['matches']} bets is not a sample.</b> Luck alone moves these balances by "
        f"about {outcome['luck']:,.0f} zł (home/draw/away) and {score['luck']:,.0f} zł "
        f"(exact score).{contradiction} {PRICE_NOTE}</div>"
    )
    return f'<div class="pl-statbar pl-scorecard pl-betting">{rendered}</div>{caveat}'


def _money_table(headers: list[str], rows: list[list[str]], total: list[str] | None = None) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    foot = ""
    if total:
        foot = '<tr class="pl-money-total">' + "".join(f"<td>{cell}</td>" for cell in total)
        foot += "</tr>"
    return (
        '<div class="pl-money-scroll"><table class="pl-money">'
        f"<tr>{head}</tr>{body}{foot}</table></div>"
    )


def _season_line(label: str, row: dict) -> list[str]:
    """One season of the backtest: hit rate, balance and return for each market."""
    outcome, score = row["outcome"], row["score"]
    return [
        label,
        f"{row['matches']:,}",
        f"{outcome['won'] / outcome['bets']:.1%}",
        _money(outcome["balance"]),
        f"{outcome['roi']:+.1%}",
        f"{score['won'] / score['bets']:.1%}",
        _money(score["balance"]),
        f"{score['roi']:+.1%}",
    ]


def betting_detail(record: dict, past: dict | None) -> str:
    """Every round of the season, then the same two bets over the backtest."""
    sections = []

    if record["rounds"]:
        rows = [
            [
                f"GW {row['gameweek']}",
                str(row["matches"]),
                f"{row['outcome']['won']}/{row['outcome']['bets']}",
                _money(row["outcome"]["balance"]),
                _money(row["outcome"]["running"]),
                f"{row['score']['won']}/{row['score']['bets']}",
                _money(row["score"]["balance"]),
                _money(row["score"]["running"]),
            ]
            for row in record["rounds"]
        ]
        outcome, score = record["total"]["outcome"], record["total"]["score"]
        total = [
            "TOTAL",
            str(record["total"]["matches"]),
            f"{outcome['won']}/{outcome['bets']}",
            _money(outcome["balance"]),
            "",
            f"{score['won']}/{score['bets']}",
            _money(score["balance"]),
            "",
        ]
        heading = f"{escape(record['season'])}, ROUND BY ROUND"
        sections.append(f'<div class="pl-money-key">{heading}</div>')
        sections.append(
            _money_table(
                ["ROUND", "BETS", "1X2 WON", "1X2", "SO FAR", "EXACT WON", "EXACT", "SO FAR"],
                rows,
                total,
            )
        )
        if record["pending"]:
            sections.append(
                f'<div class="pl-money-key">{record["pending"]} MORE AWAITING A RESULT</div>'
            )

    if past:
        sections.append(
            '<div class="pl-money-key">THE SAME BETS, WALKED FORWARD THROUGH THE BACKTEST</div>'
        )
        sections.append(
            _money_table(
                ["SEASON", "BETS", "1X2 HIT", "1X2", "RETURN", "EXACT HIT", "EXACT", "RETURN"],
                [_season_line(row["season"], row) for row in past["seasons"]],
                _season_line("TOTAL", past["total"]),
            )
        )
        margins = ", ".join(
            f"{entry['balance']:+,.0f} zł at {entry['overround'] - 1:.0%}"
            for entry in past["overrounds"]
        )
        sections.append(
            '<div class="pl-notice pl-notice-quiet">The exact-score column rests on an assumed '
            f"bookmaker margin, so here is what that assumption is worth: {margins}. The "
            "backtest uses the XI that really started, which a live round has to guess, so "
            "it is slightly kinder than the real record above.</div>"
        )

    sections.append(f'<div class="pl-notice pl-notice-quiet">{PRICE_NOTE}</div>')
    return "".join(sections)
