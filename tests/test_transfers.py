"""Tests for detecting who has left a club.

The error that matters here is asymmetric. Leaving a departed player in the XI makes one
prediction slightly stale. Wrongly deciding a player has left *deletes him from the side*,
and it happens silently - so the matching is deliberately generous, and these pin the two
spellings that would otherwise cost Man United its best two players.
"""

from __future__ import annotations

import pandas as pd

from src.predict.transfers import arrivals, departures, fpl_squads, is_in_squad, rating_index


def bootstrap(squads: dict[str, list[tuple[str, str, str]]]) -> dict:
    """An FPL payload. Each player is (first_name, second_name, web_name)."""
    teams = [{"id": index + 1, "name": name} for index, name in enumerate(squads)]
    elements = []
    for index, players in enumerate(squads.values(), start=1):
        for first, second, web in players:
            elements.append(
                {"team": index, "first_name": first, "second_name": second, "web_name": web}
            )
    return {"teams": teams, "elements": elements}


MAN_UNITED = {
    "Man Utd": [
        ("Bruno", "Borges Fernandes", "B.Fernandes"),
        ("Amad", "Diallo", "Amad"),
        ("Matheus", "Cunha", "Cunha"),
        ("Harry", "Maguire", "Maguire"),
    ]
}


def test_squads_are_keyed_by_football_data_names():
    """FPL says "Man Utd"; everything else in this project says "Man United"."""
    squads = fpl_squads(bootstrap(MAN_UNITED))

    assert "Man United" in squads
    assert len(squads["Man United"]) == 4


def test_a_player_still_at_the_club_is_found():
    squads = fpl_squads(bootstrap(MAN_UNITED))
    assert is_in_squad("Harry Maguire", squads["Man United"])


def test_a_longer_understat_name_still_matches():
    """Understat says "Amad Diallo Traore"; FPL says "Amad Diallo" and "Amad".

    A naive comparison calls him departed and removes a starter.
    """
    squads = fpl_squads(bootstrap(MAN_UNITED))
    assert is_in_squad("Amad Diallo Traore", squads["Man United"])


def test_a_shorter_understat_name_still_matches():
    """The mirror case: Understat "Bruno Fernandes" against FPL "Bruno Borges Fernandes"."""
    squads = fpl_squads(bootstrap(MAN_UNITED))
    assert is_in_squad("Bruno Fernandes", squads["Man United"])


def test_a_departed_player_is_not_found():
    """The case that started this: Casemiro played all season and has since left."""
    squads = fpl_squads(bootstrap(MAN_UNITED))
    assert not is_in_squad("Casemiro", squads["Man United"])


def test_departures_names_only_those_who_left():
    squads = fpl_squads(bootstrap(MAN_UNITED))
    xi = ["Bruno Fernandes", "Amad Diallo Traore", "Casemiro", "Harry Maguire"]

    assert departures(xi, "Man United", squads) == ["Casemiro"]


def test_an_unknown_club_reports_nobody_rather_than_everybody():
    """If a club name fails to map, the safe answer is "no information", not "all gone" -
    the other direction would empty a whole side on a mapping slip."""
    squads = fpl_squads(bootstrap(MAN_UNITED))

    assert departures(["Anyone"], "Wrexham", squads) == []


def test_an_empty_name_is_not_treated_as_present():
    squads = fpl_squads(bootstrap(MAN_UNITED))
    assert not is_in_squad("", squads["Man United"])


def test_accents_do_not_break_matching():
    """Understat and FPL disagree on accents constantly."""
    squad = fpl_squads(bootstrap({"Man Utd": [("Lisandro", "Martínez", "Martínez")]}))
    assert is_in_squad("Lisandro Martinez", squad["Man United"])


# ------------------------------------------------------------------------ arrivals


def test_arrivals_are_players_with_no_appearance_history():
    squads = fpl_squads(bootstrap(MAN_UNITED))
    known = {"Bruno Fernandes", "Amad Diallo Traore", "Harry Maguire"}

    names = [row["player"] for row in arrivals("Man United", squads, known)]

    assert names == ["Cunha"]


def test_arrivals_carry_a_rating_where_one_can_be_found():
    """A signing is only useful to report if we can say how good they are."""
    squads = fpl_squads(bootstrap(MAN_UNITED))
    fifa = pd.DataFrame(
        [
            {
                "season": "2025/26",
                "club_fd": "Man United",
                "player_name": "Matheus Cunha",
                "overall": 83,
            },
        ]
    )
    index = rating_index(fifa, "Man United", "2025/26")

    found = arrivals("Man United", squads, {"Harry Maguire"}, fifa=index)
    cunha = [row for row in found if row["player"] == "Cunha"]

    assert cunha and cunha[0]["overall"] == 83


def test_arrivals_are_ranked_best_first():
    squads = fpl_squads(bootstrap({"Man Utd": [("A", "Good", "Good"), ("B", "Better", "Better")]}))
    fifa = pd.DataFrame(
        [
            {"season": "2025/26", "club_fd": "Man United", "player_name": "A Good", "overall": 70},
            {
                "season": "2025/26",
                "club_fd": "Man United",
                "player_name": "B Better",
                "overall": 88,
            },
        ]
    )
    index = rating_index(fifa, "Man United", "2025/26")

    found = arrivals("Man United", squads, set(), fifa=index)

    assert [row["player"] for row in found] == ["Better", "Good"]


def test_an_unrated_arrival_is_still_reported():
    """A signing FIFA has never heard of is exactly the one worth telling a human about."""
    squads = fpl_squads(bootstrap({"Man Utd": [("New", "Signing", "Signing")]}))
    index = rating_index(
        pd.DataFrame(columns=["season", "club_fd", "player_name", "overall"]),
        "Man United",
        "2025/26",
    )

    found = arrivals("Man United", squads, set(), fifa=index)

    assert found == [{"player": "Signing", "overall": None}]
