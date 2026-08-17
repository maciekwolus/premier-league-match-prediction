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


# ------------------------------------------------------- differently spelled name parts


def test_a_transliterated_surname_still_matches():
    """Understat writes "Yarmolyuk", FPL writes "Yarmoliuk" - the same Ukrainian name
    through two transliterations. Token comparison sees two unrelated words, and Brentford
    silently loses a midfielder."""
    squad = fpl_squads(bootstrap({"Brentford": [("Yehor", "Yarmoliuk", "Yarmoliuk")]}))
    assert is_in_squad("Yehor Yarmolyuk", squad["Brentford"])


def test_a_forename_spelled_differently_still_matches():
    """Understat "Yeremi Pino" against FPL "Yéremy Pino Santos": one letter in the
    forename, plus a surname part Understat omits."""
    squad = fpl_squads(bootstrap({"Crystal Palace": [("Yéremy", "Pino Santos", "Yeremy")]}))
    assert is_in_squad("Yeremi Pino", squad["Crystal Palace"])


def test_a_lone_name_never_matches_on_spelling_alone():
    """The case that makes this rule dangerous, and why two tokens are required.

    "Casemiro" scores 75 against the "Carneiro" inside "Matheus Santos Carneiro da Cunha".
    Matching those would keep Casemiro at Man United forever, which is the precise bug
    this module was written to fix.
    """
    squad = fpl_squads(bootstrap({"Man Utd": [("Matheus", "Santos Carneiro da Cunha", "Cunha")]}))

    assert not is_in_squad("Casemiro", squad["Man United"])
    assert departures(["Casemiro"], "Man United", squad) == ["Casemiro"]


def test_two_different_players_are_not_merged_by_a_shared_forename():
    """Every token has to find a counterpart, so a matching forename is not enough."""
    squad = fpl_squads(bootstrap({"Ipswich": [("Sam", "Szmodics", "Szmodics")]}))
    assert not is_in_squad("Sam Morsy", squad["Ipswich"])


def test_a_genuinely_different_player_is_still_a_departure():
    """Christian Nørgaard against Cristhian Mosquera: the forenames are one letter apart
    and the players are unrelated."""
    squad = fpl_squads(bootstrap({"Arsenal": [("Cristhian", "Mosquera", "Mosquera")]}))
    assert departures(["Christian Nørgaard"], "Arsenal", squad) == ["Christian Nørgaard"]


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


# ----------------------------------------------- recruiting is not the departure check


def test_a_shared_surname_never_recruits_a_player():
    """The bug that made the first version of signing-promotion unusable.

    ``is_in_squad`` is generous on purpose: it answers "has this player left?", where a
    false match harmlessly keeps someone and a false miss deletes a real player. Reused
    to *recruit*, that generosity put Lautaro Martinez into Aston Villa's eleven off
    Emiliano Martinez, and Davinson Sanchez into two clubs at once.
    """
    from src.predict.squads import _signing_candidates

    villa = fpl_squads(bootstrap({"Aston Villa": [("Emiliano", "Martinez", "Martinez")]}))
    pool = pd.DataFrame(
        [
            {"player_name": "Lautaro Martinez", "positions": "ST", "overall": 88},
            {"player_name": "Emiliano Martinez", "positions": "GK", "overall": 84},
        ]
    )

    found = set(_signing_candidates(pool, villa["Aston Villa"])["player_name"])

    assert "Lautaro Martinez" not in found
    assert "Emiliano Martinez" in found


def test_a_genuine_signing_is_found_under_his_old_club():
    """Ratings are a September snapshot, so a summer signing is filed at the club he
    left - Tielemans is in FC 26 at Aston Villa and in FPL at Man United."""
    from src.predict.squads import _signing_candidates

    united = fpl_squads(bootstrap({"Man Utd": [("Youri", "Tielemans", "Tielemans")]}))
    pool = pd.DataFrame([{"player_name": "Youri Tielemans", "positions": "CM", "overall": 85}])

    found = _signing_candidates(pool, united["Man United"])

    assert list(found["player_name"]) == ["Youri Tielemans"]
    assert list(found["line"]) == ["mid"]


def test_a_single_name_player_is_never_recruited():
    """One token is not enough evidence in this direction, whatever it matches."""
    from src.predict.squads import _signing_candidates

    squad = fpl_squads(bootstrap({"Man Utd": [("", "Casemiro", "Casemiro")]}))
    pool = pd.DataFrame([{"player_name": "Casemiro", "positions": "CDM", "overall": 84}])

    assert _signing_candidates(pool, squad["Man United"]).empty
