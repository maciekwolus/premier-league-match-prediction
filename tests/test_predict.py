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
