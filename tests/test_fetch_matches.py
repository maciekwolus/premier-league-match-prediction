"""Tests for the match downloader. No network access required."""

import pytest

from src.config import SEASONS_BY_LABEL
from src.data.fetch_matches import _count_data_rows, main, raw_path


def test_raw_path_is_named_by_season():
    assert raw_path(SEASONS_BY_LABEL["2025/26"]).name == "E0_2025_26.csv"


def test_count_data_rows_excludes_header():
    csv = "Div,Date,HomeTeam\nE0,15/08/2025,Liverpool\nE0,16/08/2025,Arsenal\n"
    assert _count_data_rows(csv) == 2


def test_count_data_rows_ignores_trailing_blank_lines():
    csv = "Div,Date\nE0,15/08/2025\n\n\n"
    assert _count_data_rows(csv) == 1


def test_count_data_rows_handles_header_only():
    assert _count_data_rows("Div,Date,HomeTeam\n") == 0


def test_unknown_season_is_rejected():
    """A typo should fail immediately, not silently download nothing."""
    with pytest.raises(SystemExit):
        main(["--season", "2099/00"])
