"""Tests for the FIFA / EA FC ratings loader.

The real CSVs are hand-downloaded from Kaggle and gitignored, so these build synthetic
editions instead - which also lets us cover the schema variation the loader exists to
absorb.
"""

import pandas as pd
import pytest

from src.config import SEASONS
from src.data import load_fifa
from src.data.load_fifa import load_edition, resolve_columns, validate_edition
from src.matching.team_names import FIFA_TO_FOOTBALL_DATA

SEASON = SEASONS[0]

# The 20 clubs of 2019/20, under their FIFA spellings.
CLUBS_2019_20 = [
    "Arsenal",
    "Aston Villa",
    "AFC Bournemouth",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Leicester City",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Norwich City",
    "Sheffield United",
    "Southampton",
    "Tottenham Hotspur",
    "Watford",
    "West Ham United",
    "Wolverhampton Wanderers",
]


def make_edition_csv(path, clubs=CLUBS_2019_20, squad_size=25, columns="modern"):
    """Write a synthetic ratings CSV using one of the real-world column layouts."""
    rows = []
    for club in clubs:
        for i in range(squad_size):
            rows.append(
                {
                    "short_name": f"{club[:3]} Player {i}",
                    "long_name": f"{club} Full Name {i}",
                    "club_name": club,
                    "age": 20 + (i % 15),
                    "overall": 60 + (i % 30),
                    "potential": 70 + (i % 25),
                    "player_positions": "CM",
                    "value_eur": 1_000_000,
                    "pace": 70,
                    "shooting": 65,
                    "passing": 68,
                    "dribbling": 71,
                    "defending": 55,
                    "physic": 66,
                }
            )

    df = pd.DataFrame(rows)

    if columns == "alternative":
        # A different author's spelling of the same fields.
        df = df.rename(
            columns={
                "short_name": "Name",
                "club_name": "Team",
                "overall": "OVR",
                "potential": "POT",
                "player_positions": "Position",
                "physic": "Physicality",
            }
        )
    elif columns == "minimal":
        df = df[["short_name", "club_name", "overall"]]

    df.to_csv(path, index=False)
    return df


@pytest.fixture
def fifa_dir(tmp_path, monkeypatch):
    """Point ratings lookups at a temporary directory."""
    monkeypatch.setattr(load_fifa, "raw_path", lambda season: tmp_path / f"{season.fifa_slug}.csv")
    return tmp_path


# ------------------------------------------------------------------ file naming


def test_edition_slugs_are_filename_safe():
    slugs = [s.fifa_slug for s in SEASONS]
    assert slugs == ["fifa20", "fifa21", "fifa22", "fifa23", "fc24", "fc25", "fc26"]


# --------------------------------------------------------------- column mapping


def test_resolves_the_common_layout():
    resolved = resolve_columns(["short_name", "club_name", "overall", "pace"])
    assert resolved["player_name"] == "short_name"
    assert resolved["club"] == "club_name"


def test_resolves_an_alternative_layout():
    """Editions come from different Kaggle authors, so spellings differ."""
    resolved = resolve_columns(["Name", "Team", "OVR", "Physicality"])
    assert resolved["player_name"] == "Name"
    assert resolved["club"] == "Team"
    assert resolved["overall"] == "OVR"
    assert resolved["physic"] == "Physicality"


def test_column_matching_ignores_case_and_padding():
    resolved = resolve_columns(["  Short_Name  ", "CLUB_NAME", "Overall"])
    assert set(resolved) >= {"player_name", "club", "overall"}


def test_unmatched_columns_are_simply_absent():
    resolved = resolve_columns(["short_name", "club_name", "overall"])
    assert "pace" not in resolved


# ----------------------------------------------------------------- loading


def test_loads_and_filters_to_premier_league(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv")
    df, problems = load_edition(SEASON)

    assert problems == []
    assert len(df) == 20 * 25
    assert df["club_fd"].nunique() == 20
    assert "Man United" in set(df["club_fd"])  # translated, not raw FIFA spelling


def test_non_premier_league_clubs_are_dropped(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv", clubs=[*CLUBS_2019_20, "FC Barcelona", "Juventus"])
    df, _ = load_edition(SEASON)

    assert df["club_fd"].nunique() == 20
    assert "FC Barcelona" not in set(df["club"])


def test_alternative_layout_loads_too(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv", columns="alternative")
    df, _ = load_edition(SEASON)

    assert len(df) == 20 * 25
    assert df["overall"].notna().all()


def test_missing_optional_columns_are_reported_not_fatal(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv", columns="minimal")
    df, problems = load_edition(SEASON)

    assert len(df) == 20 * 25
    assert df["pace"].isna().all()
    assert any("no column found" in p for p in problems)


def test_missing_required_column_raises(fifa_dir):
    pd.DataFrame({"short_name": ["A"], "club_name": ["Arsenal"]}).to_csv(
        fifa_dir / "fifa20.csv", index=False
    )
    with pytest.raises(ValueError, match="missing required column"):
        load_edition(SEASON)


def test_absent_file_names_the_expected_path(fifa_dir):
    with pytest.raises(FileNotFoundError, match="fifa20.csv"):
        load_edition(SEASON)


# ----------------------------------------------------------------- validation


def test_valid_edition_has_no_problems(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv")
    df, _ = load_edition(SEASON)
    assert validate_edition(df, SEASON) == []


def test_detects_an_unmapped_club(fifa_dir):
    """A club whose FIFA spelling we lack drops out silently - the count must catch it."""
    make_edition_csv(fifa_dir / "fifa20.csv", clubs=CLUBS_2019_20[:19])
    df, _ = load_edition(SEASON)
    problems = validate_edition(df, SEASON)

    assert any("matched 19 Premier League clubs" in p for p in problems)


def test_detects_implausible_squad_size(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv", squad_size=3)
    df, _ = load_edition(SEASON)
    problems = validate_edition(df, SEASON)

    assert any("implausible squad sizes" in p for p in problems)


def test_detects_ratings_outside_the_valid_range(fifa_dir):
    make_edition_csv(fifa_dir / "fifa20.csv")
    df, _ = load_edition(SEASON)
    df.loc[df.index[0], "overall"] = 140
    problems = validate_edition(df, SEASON)

    assert any("outside 40-99" in p for p in problems)


# -------------------------------------------------------- combined edition file


@pytest.fixture
def combined_dir(tmp_path, monkeypatch):
    """Point both per-edition and combined lookups at a temporary directory."""
    monkeypatch.setattr(load_fifa, "raw_path", lambda season: tmp_path / f"{season.fifa_slug}.csv")
    monkeypatch.setattr(load_fifa, "RAW_FIFA_DIR", tmp_path)
    load_fifa.read_combined.cache_clear()
    return tmp_path


def make_combined_csv(path, versions=(20, 21, 22, 23, 24), version_column="fifa_version"):
    """A single file covering several editions, as the FC 24 Kaggle dataset ships."""
    frames = []
    for version in versions:
        df = make_edition_csv(path.parent / "_scratch.csv")
        df[version_column] = version
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    (path.parent / "_scratch.csv").unlink()
    combined.to_csv(path, index=False)
    return combined


def test_edition_versions_are_derived_correctly():
    assert [s.fifa_version for s in SEASONS] == [20, 21, 22, 23, 24, 25, 26]


def test_loads_one_edition_out_of_a_combined_file(combined_dir):
    make_combined_csv(combined_dir / "male_players.csv")
    df, problems = load_edition(SEASON)  # FIFA 20

    assert problems == []
    assert len(df) == 20 * 25
    assert df["fifa_edition"].unique().tolist() == ["FIFA 20"]


def test_combined_file_satisfies_only_the_editions_it_contains(combined_dir):
    make_combined_csv(combined_dir / "male_players.csv", versions=(20, 21))
    absent = {s.fifa_edition for s in load_fifa.missing_editions()}

    assert "FIFA 20" not in absent
    assert "FIFA 21" not in absent
    assert {"FIFA 22", "FIFA 23", "EA FC 24", "EA FC 25", "EA FC 26"} <= absent


def test_version_labels_with_text_are_understood(combined_dir):
    """Authors write the version as 24, "24" or "EA FC 24"."""
    make_combined_csv(combined_dir / "male_players.csv", versions=("FIFA 20", "FIFA 21"))
    df, _ = load_edition(SEASON)

    assert len(df) == 20 * 25


def test_per_edition_file_wins_over_the_combined_one(combined_dir):
    make_combined_csv(combined_dir / "male_players.csv")
    make_edition_csv(combined_dir / "fifa20.csv", clubs=CLUBS_2019_20[:20], squad_size=30)
    df, _ = load_edition(SEASON)

    assert len(df) == 20 * 30  # the dedicated file, not the combined one


def test_combined_file_without_a_version_column_is_ignored(combined_dir):
    make_combined_csv(combined_dir / "male_players.csv", version_column="irrelevant")
    with pytest.raises(FileNotFoundError):
        load_edition(SEASON)


# ------------------------------------------------------------------- mapping


def test_every_football_data_club_is_reachable():
    """All 28 clubs appearing in seven seasons need a FIFA spelling."""
    football_data_clubs = {
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Burnley",
        "Chelsea",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Ipswich",
        "Leeds",
        "Leicester",
        "Liverpool",
        "Luton",
        "Man City",
        "Man United",
        "Newcastle",
        "Norwich",
        "Nott'm Forest",
        "Sheffield United",
        "Southampton",
        "Sunderland",
        "Tottenham",
        "Watford",
        "West Brom",
        "West Ham",
        "Wolves",
    }
    assert football_data_clubs - set(FIFA_TO_FOOTBALL_DATA.values()) == set()
