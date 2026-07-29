"""Tests for the fixture card markup.

A card that quietly drops its bookmaker row, or renders a team name as markup, still
looks perfectly fine on the page. These assert the parts are present and the text is
escaped.
"""

from src.report.render import LEGEND_ROWS, legend_html, match_card, summary_bar
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


def with_lineups(match: dict | None = None, home=None, away=None) -> dict:
    match = match or make_match()
    match["lineups"] = {
        "home": home
        if home is not None
        else [
            {"player": "A Keeper", "position": "GK", "line": "gk", "starts": 6, "overall": 82},
            {"player": "A Defender", "position": "DC", "line": "def", "starts": 5, "overall": 79},
        ],
        "away": away
        if away is not None
        else [
            {"player": "B Keeper", "position": "GK", "line": "gk", "starts": 6, "overall": 87},
        ],
    }
    return match


def test_the_lineup_button_looks_like_a_button():
    """The previous version was small grey text and nobody would guess it did anything."""
    card = match_card(with_lineups())

    assert 'class="pl-xi-button"' in card
    assert "EXPECTED XI" in card


def test_the_overlay_starts_closed():
    """An unchecked toggle: the modal is hidden until the button is pressed."""
    card = match_card(with_lineups())

    assert 'class="pl-modal-toggle" hidden>' in card
    assert "checked" not in card


def test_the_overlay_can_be_dismissed_two_ways():
    """A close button and a click on the backdrop, both plain labels - no JavaScript."""
    card = match_card(with_lineups())

    assert "pl-modal-close" in card
    assert "pl-modal-backdrop" in card


def test_each_fixture_gets_its_own_toggle():
    """Duplicate ids would make one card's button open another card's overlay."""
    first = match_card(with_lineups())
    second = match_card(with_lineups(make_match(home="Everton", away="Fulham")))
    second = second.replace("2026_27_20260815_arsenal_chelsea", "2026_27_20260815_everton_fulham")

    assert "xi-2026-27-20260815-arsenal-chelsea" in first


def test_lineups_list_players_with_their_ratings():
    card = match_card(with_lineups())

    assert "Keeper" in card
    assert ">82<" in card
    assert card.count("pl-token") >= 3


def test_lineups_show_each_side_average():
    """The whole point: seeing why one side is favoured."""
    card = match_card(with_lineups())

    assert "80.5" in card  # (82 + 79) / 2
    assert "87.0" in card


def test_an_unrated_player_still_appears():
    """A player we could not match must not silently vanish from the eleven."""
    card = match_card(
        with_lineups(home=[{"player": "Unmatched", "position": "FW", "line": "att", "starts": 3}])
    )

    assert "Unmatched" in card
    assert "pl-xi-unrated" in card


def test_a_side_with_no_history_says_so():
    """A promoted club has no recent Premier League matches to read an XI from."""
    card = match_card(with_lineups(home=[]))
    assert "no recent history" in card


def test_a_card_without_lineups_omits_the_button():
    card = match_card(make_match())

    assert "pl-xi-button" not in card
    assert "EXPECTED XI" not in card


def test_players_are_laid_out_by_line():
    """Goalkeeper, defence, midfield, attack - one pitch row each."""
    card = match_card(
        with_lineups(
            home=[
                {"player": "A Keeper", "position": "GK", "line": "gk", "overall": 80},
                {"player": "A Back", "position": "DC", "line": "def", "overall": 80},
                {"player": "B Back", "position": "DC", "line": "def", "overall": 80},
                {"player": "A Mid", "position": "MC", "line": "mid", "overall": 80},
                {"player": "A Striker", "position": "FW", "line": "att", "overall": 80},
            ],
            away=[],
        )
    )

    assert card.count("pl-pitch-row") == 4
    assert ">2-1-1<" in card  # defence-midfield-attack


def test_the_away_side_is_inverted():
    """Both sides face each other, so the away eleven runs attack-first."""
    card = match_card(with_lineups())

    assert "pl-side-away" in card
    assert "pl-side-home" in card


def test_only_surnames_appear_on_the_pitch():
    """Full names would not fit a token; the full one is in the hover title."""
    card = match_card(
        with_lineups(
            home=[{"player": "Trent Alexander-Arnold", "position": "DR", "line": "def"}],
            away=[],
        )
    )

    assert ">Alexander-Arnold<" in card
    assert 'title="Trent Alexander-Arnold' in card


def test_lineups_say_they_are_not_a_team_sheet():
    """Calling this a predicted lineup would overstate it, especially in August."""
    assert "Not a team sheet" in match_card(with_lineups())


def test_player_names_are_escaped():
    card = match_card(
        with_lineups(home=[{"player": "<img src=x>", "position": "FW", "line": "att", "starts": 1}])
    )

    assert "<img src=x>" not in card
    assert "&lt;img" in card


def test_legend_explains_every_part_of_a_card():
    """Anything on a card that a newcomer cannot decode needs a row here."""
    legend = legend_html()

    for key in ("SCORELINES", "H / D / A", "BOOKMAKER", "DISAGREEMENT", "xG", "KITS"):
        assert key in legend


def test_legend_shows_real_samples_not_descriptions():
    """The samples reuse the card's own classes, so the two cannot drift apart."""
    legend = legend_html()

    for css_class in ("pl-bar", "pl-outcome", "pl-market", "pl-flag", "pl-badge", "pl-xg"):
        assert css_class in legend


def test_legend_states_the_scoreline_ceiling():
    """The single most misleading thing about the page is that 12% looks low."""
    assert "12%" in legend_html()


def test_legend_admits_the_market_wins():
    """A report that hid this would be selling something."""
    legend = legend_html()

    assert "0.1965" in legend
    assert "0.2027" in legend


def test_legend_says_the_kits_are_not_badges():
    assert "not club badges" in legend_html()


def test_legend_has_a_row_for_each_entry():
    assert legend_html().count("pl-legend-row") == len(LEGEND_ROWS)


def test_stat_bar_carries_hover_explanations():
    """ "TOP SCORE REPEATS 80%" means nothing without one."""
    bar = summary_bar(
        {"fixtures": 10, "model": "dixon-coles-squad", "with_odds": 10, "mode": "replay"},
        0.8,
    )
    assert bar.count("title=") == 4


def test_summary_bar_reports_the_headline_numbers():
    bar = summary_bar(
        {"fixtures": 10, "model": "dixon-coles-squad", "with_odds": 10, "mode": "replay"},
        0.8,
    )

    assert "10" in bar
    assert "DIXON-COLES-SQUAD" in bar
    assert "80%" in bar
