"""Tests for the prediction archive.

The archive exists to answer one question: what did the model say *before* those matches
were played? Every property here defends that. A stored round that can be silently
rewritten answers a different question and looks identical while doing it.
"""

from __future__ import annotations

import json

import pytest

from src.predict.archive import (
    RoundAlreadyStored,
    available_rounds,
    group_by_gameweek,
    latest_round,
    load_round,
    round_path,
    save_round,
)


def prediction(match_id: str = "m1", season_slug: str = "2026_27", gameweek: int = 1) -> dict:
    return {
        "match_id": match_id,
        "season_slug": season_slug,
        "gameweek": gameweek,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "outcome": {"home": 0.5, "draw": 0.25, "away": 0.25},
    }


def test_a_round_round_trips(tmp_path):
    save_round([prediction()], "2026_27", 1, root=tmp_path)

    assert load_round("2026_27", 1, root=tmp_path) == [prediction()]


def test_an_unpredicted_round_is_empty_not_an_error(tmp_path):
    assert load_round("2026_27", 7, root=tmp_path) == []


def test_storing_a_round_twice_raises(tmp_path):
    """The central guarantee. Without it the archive is just a cache."""
    save_round([prediction()], "2026_27", 1, root=tmp_path)

    with pytest.raises(RoundAlreadyStored):
        save_round([prediction()], "2026_27", 1, root=tmp_path)


def test_a_refused_write_leaves_the_original_untouched(tmp_path):
    """A failed overwrite must not half-write. The record has to survive the attempt."""
    save_round([prediction(match_id="original")], "2026_27", 1, root=tmp_path)

    with pytest.raises(RoundAlreadyStored):
        save_round([prediction(match_id="replacement")], "2026_27", 1, root=tmp_path)

    assert load_round("2026_27", 1, root=tmp_path)[0]["match_id"] == "original"


def test_force_replaces_deliberately(tmp_path):
    """Overwriting stays possible - it just cannot happen by accident."""
    save_round([prediction(match_id="original")], "2026_27", 1, root=tmp_path)
    save_round([prediction(match_id="corrected")], "2026_27", 1, root=tmp_path, force=True)

    assert load_round("2026_27", 1, root=tmp_path)[0]["match_id"] == "corrected"


def test_the_refusal_says_what_to_do_about_it(tmp_path):
    save_round([prediction()], "2026_27", 1, root=tmp_path)

    with pytest.raises(RoundAlreadyStored, match="force"):
        save_round([prediction()], "2026_27", 1, root=tmp_path)


def test_rounds_are_padded_so_they_sort_as_they_read(tmp_path):
    """gw2 sorting after gw10 would put the archive in the wrong order everywhere."""
    assert round_path("2026_27", 2, root=tmp_path).name == "gw02.json"
    assert round_path("2026_27", 10, root=tmp_path).name == "gw10.json"


def test_available_rounds_are_ordered_oldest_first(tmp_path):
    for season, gameweek in [("2026_27", 10), ("2025_26", 38), ("2026_27", 2)]:
        save_round([prediction()], season, gameweek, root=tmp_path)

    assert available_rounds(tmp_path) == [("2025_26", 38), ("2026_27", 2), ("2026_27", 10)]


def test_available_rounds_is_empty_before_anything_is_stored(tmp_path):
    assert available_rounds(tmp_path / "nothing here") == []


def test_latest_round_crosses_a_season_boundary(tmp_path):
    """Gameweek 1 of a new season is later than gameweek 38 of the old one."""
    save_round([prediction(match_id="last season", gameweek=38)], "2025_26", 38, root=tmp_path)
    save_round([prediction(match_id="new season", gameweek=1)], "2026_27", 1, root=tmp_path)

    assert latest_round(tmp_path)[0]["match_id"] == "new season"


def test_latest_round_is_empty_before_anything_is_stored(tmp_path):
    assert latest_round(tmp_path / "nothing here") == []


def test_a_stray_file_is_ignored_rather_than_guessed_at(tmp_path):
    (tmp_path / "2026_27").mkdir()
    (tmp_path / "2026_27" / "gwNOTES.json").write_text("{}", encoding="utf-8")
    save_round([prediction()], "2026_27", 1, root=tmp_path)

    assert available_rounds(tmp_path) == [("2026_27", 1)]


def test_a_run_covering_two_rounds_is_split(tmp_path):
    """The replay window is a date range, so it can straddle a rescheduled boundary.
    Each stored file has to be a real round, not whatever was predicted together."""
    predictions = [
        prediction(match_id="a", gameweek=1),
        prediction(match_id="b", gameweek=1),
        prediction(match_id="c", gameweek=2),
    ]

    grouped = group_by_gameweek(predictions)

    assert set(grouped) == {("2026_27", 1), ("2026_27", 2)}
    assert len(grouped[("2026_27", 1)]) == 2


def test_stored_json_is_readable_by_anything(tmp_path):
    """The archive outlives this code; it must not need it to be read."""
    path = save_round([prediction()], "2026_27", 1, root=tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))[0]["gameweek"] == 1
