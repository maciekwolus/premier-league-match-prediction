"""Tests for team name reconciliation between sources."""

import pytest

from src.matching.team_names import (
    UNDERSTAT_TO_FOOTBALL_DATA,
    UnknownTeamError,
    understat_to_football_data,
)

KNOWN = {
    "Arsenal",
    "Man City",
    "Man United",
    "Newcastle",
    "Nott'm Forest",
    "West Brom",
    "Wolves",
}


@pytest.mark.parametrize(
    ("understat", "football_data"),
    [
        ("Manchester City", "Man City"),
        ("Manchester United", "Man United"),
        ("Newcastle United", "Newcastle"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("West Bromwich Albion", "West Brom"),
        ("Wolverhampton Wanderers", "Wolves"),
    ],
)
def test_maps_the_names_that_differ(understat, football_data):
    assert understat_to_football_data(understat, KNOWN) == football_data


def test_names_that_already_agree_pass_through():
    assert understat_to_football_data("Arsenal", KNOWN) == "Arsenal"


def test_unknown_team_raises_rather_than_passing_through():
    """A new or renamed club must fail loudly, not silently drop its fixtures."""
    with pytest.raises(UnknownTeamError, match="Rushden"):
        understat_to_football_data("Rushden & Diamonds", KNOWN)


def test_no_validation_when_known_teams_omitted():
    assert understat_to_football_data("Anything At All") == "Anything At All"


def test_mapping_targets_are_distinct():
    """Two Understat names collapsing onto one club would corrupt the join."""
    targets = list(UNDERSTAT_TO_FOOTBALL_DATA.values())
    assert len(targets) == len(set(targets))


def test_mapping_is_not_an_identity():
    for understat, football_data in UNDERSTAT_TO_FOOTBALL_DATA.items():
        assert understat != football_data
