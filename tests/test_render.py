"""Tests for the fixture card markup.

A card that quietly drops its bookmaker row, or renders a team name as markup, still
looks perfectly fine on the page. These assert the parts are present and the text is
escaped.
"""

from src.report.render import match_card, summary_bar
from tests.test_report import make_match


def test_card_shows_both_teams_and_their_badges():
    card = match_card(make_match())

    assert "Arsenal" in card
    assert "Chelsea" in card
    assert card.count("pl-badge") == 2
    assert card.count("data:image/svg+xml;base64,") == 2


def test_card_lists_every_scoreline():
    card = match_card(make_match())
    assert card.count("pl-score-row") == 3


def test_the_leading_scoreline_is_marked():
    """The top row is highlighted, so exactly one bar carries the lead class."""
    card = match_card(make_match())
    assert card.count("pl-bar pl-lead") == 1


def test_card_shows_the_market_when_there_is_one():
    card = match_card(make_match())
    assert "BOOKMAKER" in card


def test_card_says_so_when_there_is_no_market():
    """Odds do not exist until close to kickoff; the card must still make sense."""
    card = match_card(make_match(bookmaker=None))

    assert "NO MARKET YET" in card
    assert "BOOKMAKER" not in card


def test_a_small_edge_is_not_labelled():
    """A rounded "+0" reads as a finding when it is agreement."""
    card = match_card(make_match(outcome=(0.45, 0.27, 0.28), bookmaker=(0.45, 0.27, 0.28)))
    assert "pl-edge" not in card


def test_a_real_edge_is_labelled_with_its_direction():
    card = match_card(make_match(outcome=(0.60, 0.20, 0.20), bookmaker=(0.40, 0.30, 0.30)))

    assert "pl-up" in card
    assert "pl-down" in card


def test_a_large_disagreement_gets_a_flag():
    card = match_card(make_match(outcome=(0.35, 0.30, 0.35), bookmaker=(0.24, 0.26, 0.50)))
    assert "pl-flag" in card


def test_close_agreement_gets_no_flag():
    card = match_card(make_match(outcome=(0.46, 0.27, 0.27), bookmaker=(0.45, 0.27, 0.28)))
    assert "pl-flag" not in card


def test_team_names_are_escaped():
    """A club name is data, and data does not get to inject markup."""
    card = match_card(make_match(home="<script>alert(1)</script>"))

    assert "<script>" not in card
    assert "&lt;script&gt;" in card


def test_expected_goals_are_shown():
    match = make_match()
    match["expected_goals"] = {"home": 1.62, "away": 1.14}

    assert "1.62" in match_card(match)


def test_a_card_without_expected_goals_still_renders():
    match = make_match()
    match.pop("expected_goals", None)

    assert "pl-card" in match_card(match)


def test_summary_bar_reports_the_headline_numbers():
    bar = summary_bar(
        {"fixtures": 10, "model": "dixon-coles-squad", "with_odds": 10, "mode": "replay"},
        0.8,
    )

    assert "10" in bar
    assert "DIXON-COLES-SQUAD" in bar
    assert "80%" in bar
