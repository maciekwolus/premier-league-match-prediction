"""Tests for the generated club badges.

These are kit patterns rather than crests - real badges are trademarked and not ours to
ship - so the checks are that every club gets one, that clubs are told apart, and that
nothing needs a network request to draw.
"""

import base64

import pytest

from src.report.badges import (
    CLUB_KITS,
    DEFAULT_KIT,
    GRID,
    SHIRT,
    badge_data_uri,
    badge_svg,
    kit_for,
)

# Every club that has appeared across the seven seasons in the match table.
ALL_CLUBS = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich",
    "Leeds",
    "Leicester",
    "Liverpool",
    "Luton",
    "Man City",
    "Man United",
    "Newcastle",
    "Norwich",
    "Nott'm Forest",
    "Sheffield United",
    "Southampton",
    "Sunderland",
    "Tottenham",
    "Watford",
    "West Brom",
    "West Ham",
    "Wolves",
}


def test_every_club_has_a_kit():
    """A club without one falls back to grey, which looks like a bug on the page."""
    assert ALL_CLUBS - set(CLUB_KITS) == set()


def test_club_names_match_the_match_table_spelling():
    """These keys are looked up by football-data's spelling, not a tidied version."""
    assert "Nott'm Forest" in CLUB_KITS
    assert "Man United" in CLUB_KITS
    assert "Nottingham Forest" not in CLUB_KITS


def test_an_unknown_club_falls_back_rather_than_raising():
    """A newly promoted side must not break the report before its kit is added."""
    assert kit_for("Newly Promoted FC") == DEFAULT_KIT


def test_shirt_grid_is_square_and_the_declared_size():
    assert len(SHIRT) == GRID
    assert all(len(row) == GRID for row in SHIRT)


@pytest.mark.parametrize("team", sorted(ALL_CLUBS))
def test_every_badge_renders(team):
    svg = badge_svg(team)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "<rect" in svg


def test_badges_are_drawn_crisply():
    """Smoothing a twelve-pixel shirt would defeat the point."""
    assert 'shape-rendering="crispEdges"' in badge_svg("Arsenal")


def test_different_clubs_look_different():
    """Two clubs sharing a badge would be worse than having none."""
    rendered = {team: badge_svg(team) for team in sorted(ALL_CLUBS)}
    assert len(set(rendered.values())) == len(ALL_CLUBS)


def test_a_striped_club_uses_both_colours():
    primary, secondary, _ = kit_for("Newcastle")
    svg = badge_svg("Newcastle")

    assert primary in svg
    assert secondary in svg


def test_a_plain_club_uses_its_primary_colour():
    primary, _, _ = kit_for("Liverpool")
    assert primary in badge_svg("Liverpool")


def test_scale_changes_the_rendered_size():
    small, large = badge_svg("Arsenal", scale=2), badge_svg("Arsenal", scale=8)

    assert f'width="{GRID * 2}"' in small
    assert f'width="{GRID * 8}"' in large


def test_data_uri_is_self_contained():
    """The report must draw without fetching anything - no files, no network."""
    uri = badge_data_uri("Chelsea")

    assert uri.startswith("data:image/svg+xml;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
    assert decoded == badge_svg("Chelsea")


def test_badges_are_deterministic():
    assert badge_svg("Everton") == badge_svg("Everton")


def test_every_club_of_the_upcoming_season_has_a_kit():
    """A promoted club falls back to grey, which is correct as a default and wrong as a
    permanent state once it is actually in the division - two of the three promoted for
    2026/27 were rendering identically on the pitch view."""
    for club in ("Coventry", "Hull", "Ipswich"):
        assert club in CLUB_KITS
        assert CLUB_KITS[club] != DEFAULT_KIT
