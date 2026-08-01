"""Tests for squad-quality aggregation from starting XIs."""

import pandas as pd
import pytest

from src.features.squad import line_of, squad_features, starting_ratings


@pytest.mark.parametrize(
    ("position", "line"),
    [
        ("GK", "gk"),
        ("DC", "def"),
        ("DL", "def"),
        ("DR", "def"),
        ("MC", "mid"),
        ("ML", "mid"),
        ("AMC", "mid"),
        ("AMR", "att"),
        ("AML", "att"),
        ("FW", "att"),
        ("FWL", "att"),
        ("Sub", "unknown"),
    ],
)
def test_line_of(position, line):
    assert line_of(position) == line


def test_defensive_midfielders_are_midfielders():
    """ "DMC" begins with D, so prefix order decides whether it lands in defence."""
    assert line_of("DMC") == "mid"
    assert line_of("DML") == "mid"
    assert line_of("DMR") == "mid"


def test_wide_attacking_midfielders_are_attackers():
    """Understat names AML and AMR for a slot in a formation grid, but the players in
    them are wingers - Mbeumo is an AMR, Cunha an AML.

    Reading them as midfielders produced sides with four defenders, six midfielders and
    *no attacker*, which is not a formation anybody has played: Man United's most-used XI
    rendered as 4-6-0 while genuinely being a 4-3-3. Across all 5,320 starting XIs on
    record this change makes 4-3-3 the most common shape and leaves none with an empty
    attack.
    """
    assert line_of("AML") == "att"
    assert line_of("AMR") == "att"


def test_a_number_ten_is_still_a_midfielder():
    """The distinction that makes the rule worth stating: AMC is a central creator, a
    different job from a winger, and prefix order has to catch it first."""
    assert line_of("AMC") == "mid"


def make_lineup(match_id="m1", team="Arsenal", players=None) -> pd.DataFrame:
    players = players or [
        ("Keeper", "GK"),
        *[(f"Def {i}", "DC") for i in range(4)],
        *[(f"Mid {i}", "MC") for i in range(4)],
        *[(f"Att {i}", "FW") for i in range(2)],
    ]
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "team": team,
                "player": name,
                "position": position,
                "is_starter": True,
            }
            for name, position in players
        ]
    )


def make_map(lineup, season="2019/20", unmatched=()) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "team": row.team,
                "understat_player": row.player,
                "fifa_player_name": None if row.player in unmatched else f"FIFA {row.player}",
            }
            for row in lineup.itertuples()
        ]
    )


def make_fifa(lineup, season="2019/20", overall=75, gk_face_null=True) -> pd.DataFrame:
    rows = []
    for row in lineup.itertuples():
        is_keeper = row.position == "GK"
        rows.append(
            {
                "season": season,
                "player_name": f"FIFA {row.player}",
                "overall": overall,
                "age": 27,
                "potential": overall + 3,
                "value_eur": 1_000_000,
                # FIFA leaves outfield attributes null for goalkeepers by design.
                "pace": None if (is_keeper and gk_face_null) else 70,
                "shooting": None if (is_keeper and gk_face_null) else 65,
                "passing": None if (is_keeper and gk_face_null) else 68,
                "dribbling": None if (is_keeper and gk_face_null) else 71,
                "defending": None if (is_keeper and gk_face_null) else 55,
                "physic": None if (is_keeper and gk_face_null) else 66,
            }
        )
    return pd.DataFrame(rows)


# The match_id encodes the season, which starting_ratings parses out of it.
MATCH_ID = "2019_20_20190801_arsenal_chelsea"


def test_every_starter_is_rated():
    lineup = make_lineup(MATCH_ID)
    rated = starting_ratings(lineup, make_map(lineup), make_fifa(lineup))

    assert len(rated) == 11
    assert rated["overall"].notna().all()


def test_squad_aggregates_over_the_eleven():
    lineup = make_lineup(MATCH_ID)
    features = squad_features(lineup, make_map(lineup), make_fifa(lineup))

    assert len(features) == 1
    row = features.iloc[0]
    assert row["squad_overall_mean"] == pytest.approx(75)
    assert row["starters_rated"] == 11
    assert row["rated_share"] == pytest.approx(1.0)
    assert row["squad_value_total"] == pytest.approx(11_000_000)


def test_line_averages_are_reported_separately():
    lineup = make_lineup(MATCH_ID)
    features = squad_features(lineup, make_map(lineup), make_fifa(lineup)).iloc[0]

    for column in (
        "squad_gk_overall",
        "squad_def_overall",
        "squad_mid_overall",
        "squad_att_overall",
    ):
        assert features[column] == pytest.approx(75)


def test_unmatched_players_lower_the_rated_share():
    """An unmatched starter must be visible, not silently averaged away."""
    lineup = make_lineup(MATCH_ID)
    player_map = make_map(lineup, unmatched={"Mid 0", "Mid 1"})
    features = squad_features(lineup, player_map, make_fifa(lineup)).iloc[0]

    assert features["starters_rated"] == 9
    assert features["rated_share"] == pytest.approx(9 / 11)
    assert features["squad_overall_mean"] == pytest.approx(75)  # over those we do have


def test_face_stats_average_over_outfielders_only():
    """Goalkeepers have no pace by design; including them as zero would be wrong."""
    lineup = make_lineup(MATCH_ID)
    features = squad_features(lineup, make_map(lineup), make_fifa(lineup)).iloc[0]

    assert features["squad_pace_mean"] == pytest.approx(70)


def test_substitutes_are_excluded():
    lineup = make_lineup(MATCH_ID)
    bench = lineup.iloc[:3].copy()
    bench["player"] = ["Bench A", "Bench B", "Bench C"]
    bench["position"] = "Sub"
    bench["is_starter"] = False
    combined = pd.concat([lineup, bench], ignore_index=True)

    features = squad_features(combined, make_map(combined), make_fifa(combined)).iloc[0]
    assert features["starters_rated"] == 11
