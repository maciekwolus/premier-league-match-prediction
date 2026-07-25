"""Sanity checks on the season table - a typo here would silently break every phase."""

from src.config import MATCHES_PER_SEASON, SEASONS, SEASONS_BY_CODE, Season


def test_seven_seasons_defined():
    assert len(SEASONS) == 7


def test_codes_and_labels_are_unique():
    assert len({s.code for s in SEASONS}) == len(SEASONS)
    assert len({s.label for s in SEASONS}) == len(SEASONS)


def test_seasons_are_consecutive():
    years = [int(s.understat) for s in SEASONS]
    assert years == list(range(years[0], years[0] + len(SEASONS)))


def test_code_matches_label():
    """The football-data code is the two-digit start and end year, e.g. 2019/20 -> 1920."""
    for season in SEASONS:
        start, end = season.label.split("/")
        assert season.code == start[2:] + end


def test_understat_key_is_start_year():
    for season in SEASONS:
        assert season.understat == season.label.split("/")[0]


def test_slug_is_filename_safe():
    assert Season("2019/20", "1920", "2019", "FIFA 20").slug == "2019_20"
    assert all("/" not in s.slug for s in SEASONS)


def test_matches_url_is_well_formed():
    url = SEASONS_BY_CODE["2526"].matches_url
    assert url == "https://www.football-data.co.uk/mmz4281/2526/E0.csv"


def test_matches_per_season():
    assert MATCHES_PER_SEASON == 380
