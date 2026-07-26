"""Tests for roster flattening and lineup validation.

Rosters are written to a temporary directory rather than read from the scraped cache,
so these run offline.
"""

import json

import pandas as pd
import pytest

from src.data import clean_lineups
from src.data.clean_lineups import build_lineups, roster_rows


def make_player(name, position, minutes=90, **overrides):
    player = {
        "player": name,
        "player_id": str(abs(hash(name)) % 100000),
        "position": position,
        "positionOrder": "1",
        "time": str(minutes),
        "goals": "0",
        "own_goals": "0",
        "assists": "0",
        "shots": "1",
        "key_passes": "0",
        "xG": "0.1",
        "xA": "0.05",
        "xGChain": "0.2",
        "xGBuildup": "0.1",
        "yellow_card": "0",
        "red_card": "0",
    }
    player.update(overrides)
    return player


def make_roster(home_starters=11, away_starters=11, subs=3):
    """A roster in Understat's shape: keyed by an opaque id, with 'Sub' marking bench."""

    def side(prefix, starters):
        players = {}
        for i in range(starters):
            players[f"{prefix}{i}"] = make_player(f"{prefix} Starter {i}", "MC")
        for i in range(subs):
            players[f"{prefix}s{i}"] = make_player(f"{prefix} Sub {i}", "Sub", minutes=10)
        return players

    return {"h": side("H", home_starters), "a": side("A", away_starters)}


@pytest.fixture
def roster_dir(tmp_path, monkeypatch):
    """Point roster lookups at a temporary directory."""
    monkeypatch.setattr(
        clean_lineups, "roster_path", lambda match_id: tmp_path / f"{match_id}.json"
    )
    return tmp_path


def write_roster(roster_dir, understat_id, roster):
    (roster_dir / f"{understat_id}.json").write_text(json.dumps(roster), encoding="utf-8")


def make_linked(understat_id="11643", match_id="2019_20_20190809_liverpool_norwich"):
    return pd.DataFrame(
        [
            {
                "understat_match_id": understat_id,
                "match_id": match_id,
                "home_team": "Liverpool",
                "away_team": "Norwich",
            }
        ]
    )


# ------------------------------------------------------------------ flattening


def test_roster_rows_covers_both_teams(roster_dir):
    write_roster(roster_dir, "11643", make_roster())
    rows = roster_rows("11643", "match_1", "Liverpool", "Norwich")

    assert len(rows) == 28  # (11 + 3) per side
    assert {r["team"] for r in rows} == {"Liverpool", "Norwich"}
    assert {r["side"] for r in rows} == {"home", "away"}


def test_starters_flagged_by_position(roster_dir):
    write_roster(roster_dir, "11643", make_roster())
    rows = roster_rows("11643", "match_1", "Liverpool", "Norwich")

    assert sum(r["is_starter"] for r in rows) == 22
    assert all(r["position"] != "Sub" for r in rows if r["is_starter"])


def test_understat_fields_are_renamed(roster_dir):
    write_roster(roster_dir, "11643", make_roster())
    row = roster_rows("11643", "match_1", "Liverpool", "Norwich")[0]

    # Understat's camelCase becomes our snake_case
    assert "xg" in row and "xG" not in row
    assert "minutes" in row and "time" not in row
    assert row["match_id"] == "match_1"


# ------------------------------------------------------------------ validation


def test_build_lineups_accepts_a_complete_match(roster_dir):
    write_roster(roster_dir, "11643", make_roster())
    lineups, problems = build_lineups(make_linked())

    assert problems == []
    assert len(lineups) == 28
    assert lineups["minutes"].dtype.kind in "if"  # coerced to numeric


def test_detects_wrong_number_of_starters(roster_dir):
    """Ten starters means the roster is incomplete, not that someone was sent off."""
    write_roster(roster_dir, "11643", make_roster(home_starters=10))
    _, problems = build_lineups(make_linked())

    assert any("starters" in p for p in problems)


def test_missing_roster_file_is_reported_not_raised(roster_dir):
    _, problems = build_lineups(make_linked(understat_id="does_not_exist"))

    assert any("no roster file" in p for p in problems)
    assert any("fetch_lineups" in p for p in problems)
