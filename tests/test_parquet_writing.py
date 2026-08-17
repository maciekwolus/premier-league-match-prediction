"""Tests that a committed parquet does not depend on which machine wrote it.

The failure this prevents is invisible in the data and loud in the repository. pandas 3
defaults datetimes to microseconds where pandas 2 used nanoseconds, so the scheduled job
(Linux, pandas 3) and a laptop (pandas 2) wrote byte-different files holding *identical*
values. The job saw a change every single day, committed "Update results" with 0
insertions and 0 deletions, redeployed the site for nothing, and the next local rebuild
flipped it straight back.
"""

from __future__ import annotations

import pandas as pd

from src.config import PARQUET_TIME_UNIT, write_parquet


def frame(unit: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-21", "2026-08-22"]).as_unit(unit),
            "home_team": ["Arsenal", "Hull"],
            "home_goals": pd.array([2, 1], dtype="Int64"),
        }
    )


def test_the_written_resolution_does_not_depend_on_the_incoming_one(tmp_path):
    """The whole point: two environments, same values, same file."""
    nanoseconds, microseconds = tmp_path / "ns.parquet", tmp_path / "us.parquet"

    write_parquet(frame("ns"), nanoseconds)
    write_parquet(frame("us"), microseconds)

    assert nanoseconds.read_bytes() == microseconds.read_bytes()


def test_the_written_resolution_is_the_declared_one(tmp_path):
    path = tmp_path / "m.parquet"
    write_parquet(frame("ns"), path)

    assert str(pd.read_parquet(path)["date"].dtype) == f"datetime64[{PARQUET_TIME_UNIT}]"


def test_the_values_survive_unchanged(tmp_path):
    """Normalising resolution must not move a single timestamp."""
    path = tmp_path / "m.parquet"
    original = frame("ns")
    write_parquet(original, path)

    written = pd.read_parquet(path)

    assert list(written["date"].astype("datetime64[ns]")) == list(original["date"])
    assert list(written["home_team"]) == list(original["home_team"])


def test_a_frame_without_dates_is_written_unchanged(tmp_path):
    path = tmp_path / "m.parquet"
    write_parquet(pd.DataFrame({"player": ["a"], "overall": [80]}), path)

    assert pd.read_parquet(path)["overall"].tolist() == [80]


def test_a_timezone_aware_column_keeps_its_zone(tmp_path):
    """FPL kickoff times arrive in UTC. Changing resolution must not drop the zone."""
    path = tmp_path / "m.parquet"
    aware = pd.DataFrame({"kickoff": pd.to_datetime(["2026-08-21T19:00Z"]).as_unit("ns")})

    write_parquet(aware, path)
    written = pd.read_parquet(path)["kickoff"]

    assert str(written.dtype) == f"datetime64[{PARQUET_TIME_UNIT}, UTC]"
    assert written.iloc[0] == pd.Timestamp("2026-08-21T19:00Z")


def test_writing_creates_the_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "m.parquet"
    write_parquet(frame("ns"), path)

    assert path.exists()


# ------------------------------------------- leaving an unchanged file alone


def test_an_unchanged_rebuild_does_not_touch_the_file(tmp_path):
    """The version-proof half. pyarrow stamps its own version into every file it writes,
    so identical data still produces different bytes across environments - and git
    compares bytes. Rewriting is what produced a daily empty commit."""
    path = tmp_path / "m.parquet"
    write_parquet(frame("us"), path)
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    wrote = write_parquet(frame("ns"), path)

    assert wrote is False
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamp


def test_a_genuine_change_is_written(tmp_path):
    path = tmp_path / "m.parquet"
    write_parquet(frame("us"), path)

    changed = frame("us")
    changed.loc[0, "home_goals"] = 5

    assert write_parquet(changed, path) is True
    assert pd.read_parquet(path)["home_goals"].tolist() == [5, 1]


def test_a_new_row_is_written(tmp_path):
    """The case that matters in the season: results arriving one round at a time."""
    path = tmp_path / "m.parquet"
    write_parquet(frame("us"), path)

    grown = pd.concat([frame("us"), frame("us").head(1)], ignore_index=True)

    assert write_parquet(grown, path) is True
    assert len(pd.read_parquet(path)) == 3


def test_an_unreadable_existing_file_is_replaced_rather_than_trusted(tmp_path):
    path = tmp_path / "m.parquet"
    path.write_bytes(b"not a parquet file")

    assert write_parquet(frame("us"), path) is True
    assert len(pd.read_parquet(path)) == 2


def test_the_first_write_reports_that_it_wrote(tmp_path):
    assert write_parquet(frame("us"), tmp_path / "new.parquet") is True
