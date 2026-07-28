"""Tests for predicting fixtures that have not been played.

The risk here is different from the rest of the pipeline. Historical code fails loudly
when a join goes wrong, because the row counts are known; prediction code can quietly
produce a plausible-looking number for the wrong team, or several for the same fixture.
"""

import pandas as pd
import pytest

from src.predict import squads
from src.predict.fixtures import as_matches
from src.predict.squads import apply_squad_changes, most_used_eleven, unmatched_changes

ELEVEN = [
    ("Keeper", "GK"),
    *[(f"Def {i}", "DC") for i in range(4)],
    *[(f"Mid {i}", "MC") for i in range(4)],
    *[(f"Att {i}", "FW") for i in range(2)],
]


def make_lineups(team="Arsenal", matches=6, players=None, start_date="2026-01-01"):
    """Lineup rows for several matches, in the shape the XI picker expects."""
    players = players or ELEVEN
    rows = []
    date = pd.Timestamp(start_date)

    for match in range(matches):
        for name, position in players:
            rows.append(
                {
                    "match_id": f"m{match}",
                    "date": date,
                    "team": team,
                    "player": name,
                    "position": position,
                    "minutes": 90,
                    "is_starter": True,
                }
            )
        date += pd.Timedelta(days=7)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ expected XI


def test_expected_eleven_has_eleven_players():
    lineups = make_lineups()
    eleven = most_used_eleven(lineups, "Arsenal", pd.Timestamp("2026-06-01"))

    assert len(eleven) == 11


def test_expected_eleven_has_exactly_one_goalkeeper():
    """Two keepers sharing a season would otherwise both make the eleven, or neither."""
    players = [*ELEVEN, ("Reserve Keeper", "GK")]
    lineups = make_lineups(players=players)
    eleven = most_used_eleven(lineups, "Arsenal", pd.Timestamp("2026-06-01"))

    assert (eleven["line"] == "gk").sum() == 1
    assert len(eleven) == 11


def test_expected_eleven_ignores_matches_after_the_fixture():
    """Picking a team sheet from matches that have not happened is leakage."""
    lineups = make_lineups(matches=6)
    early = most_used_eleven(lineups, "Arsenal", pd.Timestamp("2026-01-08"))

    assert len(early) <= 11
    assert lineups[lineups["date"] >= pd.Timestamp("2026-01-08")].shape[0] > 0


def test_unknown_team_yields_an_empty_eleven_rather_than_raising():
    lineups = make_lineups()
    eleven = most_used_eleven(lineups, "Newly Promoted", pd.Timestamp("2026-06-01"))

    assert eleven.empty


def test_the_most_used_players_are_chosen():
    """A player who started once loses to the ten who started every week."""
    lineups = make_lineups(matches=5)

    occasional = lineups[(lineups["match_id"] == "m0") & (lineups["player"] == "Mid 0")].copy()
    occasional["player"] = "Squad Filler"
    lineups = pd.concat([lineups, occasional], ignore_index=True)

    eleven = most_used_eleven(lineups, "Arsenal", pd.Timestamp("2026-06-01"))

    assert len(eleven) == 11
    assert "Squad Filler" not in set(eleven["player"])
    assert "Mid 0" in set(eleven["player"])


# --------------------------------------------------------------- squad changes


def make_squad():
    return pd.DataFrame(
        [
            {"season": "2026/27", "fifa_player_name": "A. Player", "team": "Arsenal"},
            {"season": "2026/27", "fifa_player_name": "B. Player", "team": "Chelsea"},
        ]
    )


def test_a_transfer_moves_one_player():
    changes = pd.DataFrame(
        [{"season": "2026/27", "fifa_player_name": "A. Player", "team": "Chelsea", "note": ""}]
    )
    updated = apply_squad_changes(make_squad(), changes, "2026/27")

    assert updated[updated["fifa_player_name"] == "A. Player"]["team"].item() == "Chelsea"
    assert len(updated) == 2


def test_a_blank_team_removes_the_player():
    """A player who leaves the league has no club, and must not linger in a squad."""
    changes = pd.DataFrame(
        [{"season": "2026/27", "fifa_player_name": "A. Player", "team": "", "note": "abroad"}]
    )
    updated = apply_squad_changes(make_squad(), changes, "2026/27")

    assert "A. Player" not in set(updated["fifa_player_name"])
    assert len(updated) == 1


def test_changes_for_another_season_are_ignored():
    changes = pd.DataFrame(
        [{"season": "2025/26", "fifa_player_name": "A. Player", "team": "Chelsea", "note": ""}]
    )
    updated = apply_squad_changes(make_squad(), changes, "2026/27")

    assert updated[updated["fifa_player_name"] == "A. Player"]["team"].item() == "Arsenal"


def test_an_empty_change_file_changes_nothing():
    empty = pd.DataFrame(columns=["season", "fifa_player_name", "team", "note"])
    updated = apply_squad_changes(make_squad(), empty, "2026/27")

    pd.testing.assert_frame_equal(updated, make_squad())


def test_a_misspelled_name_is_reported():
    """A typo would silently do nothing, so it has to be surfaced."""
    changes = pd.DataFrame(
        [{"season": "2026/27", "fifa_player_name": "A. Playr", "team": "Chelsea", "note": ""}]
    )
    unmatched = unmatched_changes(changes, {"A. Player", "B. Player"}, "2026/27")

    assert unmatched == ["A. Playr"]


def test_a_correct_name_is_not_reported():
    changes = pd.DataFrame(
        [{"season": "2026/27", "fifa_player_name": "A. Player", "team": "Chelsea", "note": ""}]
    )
    assert unmatched_changes(changes, {"A. Player"}, "2026/27") == []


def test_missing_change_file_is_not_an_error(tmp_path, monkeypatch):
    """A fresh clone has no transfers recorded yet."""
    monkeypatch.setattr(squads, "SQUAD_CHANGES_CSV", tmp_path / "absent.csv")
    assert squads.load_squad_changes().empty


def test_change_file_missing_a_column_raises(tmp_path, monkeypatch):
    path = tmp_path / "squad_changes.csv"
    path.write_text("season,team\n2026/27,Arsenal\n", encoding="utf-8")
    monkeypatch.setattr(squads, "SQUAD_CHANGES_CSV", path)

    with pytest.raises(ValueError, match="fifa_player_name"):
        squads.load_squad_changes()


# -------------------------------------------------------- squad quality for fixtures


def make_player_map(team="Arsenal", season="2025/26", players=None):
    players = players or [name for name, _ in ELEVEN]
    return pd.DataFrame(
        [
            {
                "season": season,
                "team": team,
                "understat_player": name,
                "fifa_player_name": f"FIFA {name}",
            }
            for name in players
        ]
    )


def make_fifa_ratings(season="2025/26", players=None, overall=78):
    players = players or [name for name, _ in ELEVEN]
    return pd.DataFrame(
        [
            {
                "season": season,
                "player_name": f"FIFA {name}",
                "overall": overall,
                "age": 27,
                "potential": overall + 2,
                "value_eur": 1_000_000,
                "pace": 70,
                "shooting": 65,
                "passing": 68,
                "dribbling": 71,
                "defending": 55,
                "physic": 66,
            }
            for name in players
        ]
    )


def make_fixture(home="Arsenal", away="Chelsea", season="2026/27"):
    return pd.DataFrame(
        [
            {
                "match_id": "2026_27_20260815_arsenal_chelsea",
                "season": season,
                "date": pd.Timestamp("2026-08-15"),
                "home_team": home,
                "away_team": away,
            }
        ]
    )


def test_upcoming_fixtures_get_real_squad_ratings(monkeypatch):
    """The regression this guards: for a while these were all filled with the league
    median, so every team looked identical and half the feature table said nothing."""
    monkeypatch.setattr(squads, "load_squad_changes", lambda: pd.DataFrame())

    lineups = make_lineups(team="Arsenal", start_date="2026-01-01")
    features, problems = squads.expected_squad_features(
        make_fixture(), lineups, make_player_map(), make_fifa_ratings(), "2025/26"
    )

    assert problems == []
    arsenal = features[features["team"] == "Arsenal"].iloc[0]
    assert arsenal["squad_overall_mean"] == pytest.approx(78)
    assert arsenal["starters_rated"] == 11


def test_a_stronger_squad_scores_higher(monkeypatch):
    """Two teams must not come out identical, which is what the bug looked like."""
    monkeypatch.setattr(squads, "load_squad_changes", lambda: pd.DataFrame())

    strong = make_lineups(team="Arsenal", start_date="2026-01-01")
    weak = make_lineups(team="Chelsea", start_date="2026-01-01")
    weak["player"] = weak["player"] + " (C)"
    lineups = pd.concat([strong, weak], ignore_index=True)

    player_map = pd.concat(
        [
            make_player_map("Arsenal"),
            make_player_map("Chelsea", players=[f"{n} (C)" for n, _ in ELEVEN]),
        ],
        ignore_index=True,
    )
    ratings = pd.concat(
        [
            make_fifa_ratings(overall=84),
            make_fifa_ratings(players=[f"{n} (C)" for n, _ in ELEVEN], overall=71),
        ],
        ignore_index=True,
    )

    features, _ = squads.expected_squad_features(
        make_fixture(), lineups, player_map, ratings, "2025/26"
    )
    by_team = features.set_index("team")["squad_overall_mean"]

    assert by_team["Arsenal"] == pytest.approx(84)
    assert by_team["Chelsea"] == pytest.approx(71)


def test_a_transfer_is_reflected_in_the_squad(monkeypatch):
    """The promise in the README: one line in the change file and a re-run."""
    changes = pd.DataFrame(
        [{"season": "2026/27", "fifa_player_name": "FIFA Unknown", "team": "Arsenal", "note": ""}]
    )
    monkeypatch.setattr(squads, "load_squad_changes", lambda: changes)

    lineups = make_lineups(team="Arsenal", start_date="2026-01-01")
    _, problems = squads.expected_squad_features(
        make_fixture(), lineups, make_player_map(), make_fifa_ratings(), "2025/26"
    )

    # The player is not in the map, so the run reports it rather than doing nothing.
    assert any("FIFA Unknown" in problem for problem in problems)


def test_a_recognised_transfer_raises_no_complaint(monkeypatch):
    changes = pd.DataFrame(
        [
            {
                "season": "2026/27",
                "fifa_player_name": "FIFA Keeper",
                "team": "Arsenal",
                "note": "",
            }
        ]
    )
    monkeypatch.setattr(squads, "load_squad_changes", lambda: changes)

    lineups = make_lineups(team="Arsenal", start_date="2026-01-01")
    _, problems = squads.expected_squad_features(
        make_fixture(), lineups, make_player_map(), make_fifa_ratings(), "2025/26"
    )

    assert problems == []


# -------------------------------------------------------------------- fixtures


def test_fixtures_become_match_shaped_rows_without_results():
    fixtures = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-15")],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    rows = as_matches(fixtures)

    assert rows["match_id"].item() == "2026_27_20260815_arsenal_chelsea"
    assert pd.isna(rows["home_goals"].item())
    assert pd.isna(rows["away_goals"].item())


def test_no_fixtures_yields_an_empty_frame():
    assert as_matches(pd.DataFrame()).empty
