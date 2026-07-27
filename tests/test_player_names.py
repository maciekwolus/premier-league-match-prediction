"""Tests for Understat-to-FIFA player name matching.

Synthetic squads throughout, so these run offline and each rule can be exercised in
isolation - which matters here, because the cascade's whole design is that a cheap safe
rule fires before an expensive risky one.
"""

import pandas as pd
import pytest

from src.matching.player_names import (
    coverage,
    match_one,
    normalise,
    subset_match,
    to_initials,
)


def squad(*players) -> pd.DataFrame:
    """Build an indexed FIFA squad from (short_name, long_name) pairs."""
    df = pd.DataFrame(players, columns=["player_name", "long_name"])
    df["norm_short"] = df["player_name"].map(normalise)
    df["norm_long"] = df["long_name"].map(normalise)
    df["tokens"] = [
        frozenset(s.split()) | frozenset(long_.split())
        for s, long_ in zip(df["norm_short"], df["norm_long"], strict=True)
    ]
    return df


EMPTY = squad()


# ------------------------------------------------------------------ normalising


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Virgil van Dijk", "virgil van dijk"),
        ("V. van Dijk", "v van dijk"),
        # ø, ł and ß are distinct letters that NFKD cannot decompose, so they are
        # transliterated explicitly rather than silently dropped.
        ("Martin Ødegaard", "martin odegaard"),
        ("Łukasz Fabiański", "lukasz fabianski"),
        ("N'Golo Kanté", "n golo kante"),
        ("Trent Alexander-Arnold", "trent alexander arnold"),
        ("  Extra   Spaces  ", "extra spaces"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_normalise_handles_non_strings():
    assert normalise(None) == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Virgil van Dijk", "v van dijk"),
        ("Mohamed Salah", "m salah"),
        ("Fabinho", "fabinho"),  # single names are left alone
    ],
)
def test_to_initials(raw, expected):
    assert to_initials(raw) == expected


# --------------------------------------------------------------------- cascade


def test_exact_long_name_wins_first():
    scope = squad(("V. van Dijk", "Virgil van Dijk"))
    name, method, score = match_one("Virgil van Dijk", scope, EMPTY)

    assert (name, method, score) == ("V. van Dijk", "exact_long_club", 100.0)


def test_matches_abbreviated_short_name():
    """Understat's full name against FIFA's "V. van Dijk"."""
    scope = squad(("V. van Dijk", ""))
    name, method, _ = match_one("Virgil van Dijk", scope, EMPTY)

    assert name == "V. van Dijk"
    assert method == "initials_club"


def test_matches_full_birth_name_by_subset():
    """EA FC 25 records "Mohamed Salah Hamed Ghaly" for Understat's "Mohamed Salah"."""
    scope = squad(("Mohamed Salah Hamed Ghaly", "Mohamed Salah Hamed Ghaly"))
    name, method, _ = match_one("Mohamed Salah", scope, EMPTY)

    assert name == "Mohamed Salah Hamed Ghaly"
    assert method == "subset_club"


def test_accents_do_not_block_a_match():
    scope = squad(("M. Ødegaard", "Martin Ødegaard"))
    name, _, _ = match_one("Martin Odegaard", scope, EMPTY)

    assert name == "M. Ødegaard"


def test_club_scope_is_preferred_over_season_scope():
    club = squad(("J. Henderson", "Jordan Henderson"))
    season = squad(("D. Henderson", "Dean Henderson"))
    name, method, _ = match_one("Jordan Henderson", club, season)

    assert name == "J. Henderson"
    assert method.endswith("_club")


# ------------------------------------------------------- ambiguity is not a match


def test_ambiguous_subset_is_refused():
    """Arsenal 2024/25 had both "Gabriel" and "Gabriel Martinelli"."""
    scope = squad(
        ("Gabriel", "Gabriel dos Santos Magalhães"),
        ("Gabriel Martinelli", "Gabriel Teodoro Martinelli Silva"),
    )
    assert subset_match("Gabriel", scope) is None


def test_unambiguous_subset_is_accepted():
    scope = squad(
        ("Gabriel", "Gabriel dos Santos Magalhães"),
        ("B. Saka", "Bukayo Saka"),
    )
    assert subset_match("Gabriel", scope) == "Gabriel"


def test_completely_unknown_name_is_unmatched():
    scope = squad(("B. Saka", "Bukayo Saka"))
    name, method, score = match_one("Someone Entirely Else", scope, EMPTY)

    assert name is None
    assert method == "unmatched"
    assert score == 0.0


# ------------------------------------------------------------------- world scope


def test_world_scope_finds_a_mid_season_signing():
    """Ratings are a September snapshot, so January arrivals sit at their old club."""
    world = squad(("R. Dias", "Rúben Dias"))
    name, method, _ = match_one("Rúben Dias", EMPTY, EMPTY, world)

    assert name == "R. Dias"
    assert method == "exact_world"


def test_world_scope_refuses_an_ambiguous_name():
    """18,000 names contain repeats; only a unique hit is safe."""
    world = squad(("Rodrigo", "Rodrigo"), ("Rodrigo", "Rodrigo"))
    name, method, _ = match_one("Rodrigo", EMPTY, EMPTY, world)

    assert name is None
    assert method == "unmatched"


def test_world_scope_is_only_consulted_last():
    club = squad(("B. Saka", "Bukayo Saka"))
    world = squad(("B. Saka", "Bukayo Saka"))
    _, method, _ = match_one("Bukayo Saka", club, EMPTY, world)

    assert method.endswith("_club")


# ------------------------------------------------------------------- coverage


def test_coverage_weights_by_starts_not_by_name():
    """A missed regular starter matters; a missed bench player barely does."""
    player_map = pd.DataFrame(
        [
            {"season": "2019/20", "starts": 38, "fifa_player_name": "A"},
            {"season": "2019/20", "starts": 0, "fifa_player_name": None},
            {"season": "2019/20", "starts": 0, "fifa_player_name": None},
        ]
    )
    report = coverage(player_map)

    assert report.loc["2019/20", "name_pct"] == pytest.approx(33.3, abs=0.1)
    assert report.loc["2019/20", "starter_pct"] == 100.0
