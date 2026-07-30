"""Tests for deriving suspensions from cards.

An off-by-one here is silent: banning a player for the match they were sent off in, or
for one match too few, produces an expected XI that looks entirely reasonable. These pin
the arithmetic to the club's own fixture list rather than to dates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.predict.suspensions import (
    RED_CARD_BAN,
    YELLOW_TIERS,
    club_matches,
    load_unavailable,
    manually_unavailable,
    suspended_for,
    unavailable_for,
)

SEASON = "2025/26"
TEAM = "Arsenal"


def season_lineups(cards: dict[int, dict[str, tuple[int, int]]], matches: int = 25):
    """A club's season, with (yellow, red) cards for named players in given matches.

    ``cards`` maps a match index (0-based) to {player: (yellows, reds)}. Every match has
    a full row for each named player so the fixture list is complete either way.
    """
    players = sorted({player for entry in cards.values() for player in entry})
    players = players or ["Filler"]

    rows = []
    for index in range(matches):
        date = pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * index)
        for player in players:
            yellows, reds = cards.get(index, {}).get(player, (0, 0))
            rows.append(
                {
                    "match_id": f"m{index}",
                    "season": SEASON,
                    "team": TEAM,
                    "player": player,
                    "date": date,
                    "is_starter": True,
                    "yellow_cards": yellows,
                    "red_cards": reds,
                }
            )
    # Columns even when there are no rows: a real lineups table always has them, and a
    # club simply has no rows in it until it has played.
    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "season",
            "team",
            "player",
            "date",
            "is_starter",
            "yellow_cards",
            "red_cards",
        ],
    )


def match_date(index: int) -> pd.Timestamp:
    return pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * index)


def test_the_fixture_list_is_the_clock():
    lineups = season_lineups({}, matches=5)
    assert club_matches(lineups, TEAM, SEASON) == [match_date(n) for n in range(5)]


def test_a_red_card_bans_the_next_match():
    lineups = season_lineups({2: {"Rice": (0, 1)}})

    assert suspended_for(lineups, TEAM, SEASON, match_date(3)) == {"Rice"}


def test_a_red_card_does_not_ban_the_match_it_was_shown_in():
    """The player was on the pitch. Banning them retroactively would rewrite history."""
    lineups = season_lineups({2: {"Rice": (0, 1)}})

    assert suspended_for(lineups, TEAM, SEASON, match_date(2)) == set()


def test_a_red_card_ban_is_served_and_ends():
    lineups = season_lineups({2: {"Rice": (0, 1)}})

    assert suspended_for(lineups, TEAM, SEASON, match_date(3)) == {"Rice"}
    assert suspended_for(lineups, TEAM, SEASON, match_date(4)) == set()


def test_every_red_costs_exactly_one_match():
    """Deliberate floor: the offence is not in the data, and guessing high would remove
    players who are actually available."""
    assert RED_CARD_BAN == 1


def test_five_yellows_bans_the_next_match():
    lineups = season_lineups({index: {"Rice": (1, 0)} for index in range(5)})

    assert suspended_for(lineups, TEAM, SEASON, match_date(5)) == {"Rice"}


def test_four_yellows_bans_nothing():
    lineups = season_lineups({index: {"Rice": (1, 0)} for index in range(4)})

    assert suspended_for(lineups, TEAM, SEASON, match_date(4)) == set()


def test_the_five_yellow_ban_is_one_match_only():
    lineups = season_lineups({index: {"Rice": (1, 0)} for index in range(5)})

    assert suspended_for(lineups, TEAM, SEASON, match_date(6)) == set()


def test_ten_yellows_costs_two_matches():
    lineups = season_lineups({index: {"Rice": (1, 0)} for index in range(10)})

    assert suspended_for(lineups, TEAM, SEASON, match_date(10)) == {"Rice"}
    assert suspended_for(lineups, TEAM, SEASON, match_date(11)) == {"Rice"}
    assert suspended_for(lineups, TEAM, SEASON, match_date(12)) == set()


def test_the_count_does_not_reset_after_a_ban_is_served():
    """A player banned at five is banned again at ten, not at fifteen."""
    lineups = season_lineups({index: {"Rice": (1, 0)} for index in range(10)})

    assert suspended_for(lineups, TEAM, SEASON, match_date(5)) == {"Rice"}  # five
    assert suspended_for(lineups, TEAM, SEASON, match_date(10)) == {"Rice"}  # ten


def test_a_threshold_reached_after_its_deadline_carries_no_ban():
    """Five yellows inside the first 19 matches is a ban; the twentieth match is not."""
    late = {index: {"Rice": (1, 0)} for index in (15, 16, 17, 18, 19)}
    lineups = season_lineups(late, matches=25)

    assert suspended_for(lineups, TEAM, SEASON, match_date(20)) == set()


def test_a_threshold_reached_on_its_deadline_still_counts():
    """Reaching five in the club's 19th match - index 18 - is inside the window."""
    on_time = {index: {"Rice": (1, 0)} for index in (14, 15, 16, 17, 18)}
    lineups = season_lineups(on_time, matches=25)

    assert suspended_for(lineups, TEAM, SEASON, match_date(19)) == {"Rice"}


def test_two_yellows_in_one_match_can_step_over_a_threshold():
    """Thresholds are crossed, not landed on - a player can go from four to six."""
    cards = {index: {"Rice": (1, 0)} for index in range(4)}
    cards[4] = {"Rice": (2, 0)}
    lineups = season_lineups(cards)

    assert suspended_for(lineups, TEAM, SEASON, match_date(5)) == {"Rice"}


def test_a_red_and_an_accumulation_ban_can_overlap_without_double_counting():
    cards = {index: {"Rice": (1, 0)} for index in range(5)}
    cards[4] = {"Rice": (1, 1)}
    lineups = season_lineups(cards)

    assert suspended_for(lineups, TEAM, SEASON, match_date(5)) == {"Rice"}
    assert suspended_for(lineups, TEAM, SEASON, match_date(6)) == set()


def test_players_are_kept_apart():
    lineups = season_lineups({2: {"Rice": (0, 1), "Saka": (1, 0)}})

    assert suspended_for(lineups, TEAM, SEASON, match_date(3)) == {"Rice"}


def test_another_club_is_not_affected():
    lineups = season_lineups({2: {"Rice": (0, 1)}})

    assert suspended_for(lineups, "Chelsea", SEASON, match_date(3)) == set()


def test_a_promoted_club_with_no_matches_yet_yields_nothing():
    """Gameweek 1: the club is in the table but has played nothing, so has no clock."""
    assert suspended_for(season_lineups({}, matches=0), TEAM, SEASON, match_date(1)) == set()


def test_the_third_tier_exists_even_though_it_has_never_fired():
    """Encoded because the rule exists. No player reached 15 yellows in seven seasons."""
    assert YELLOW_TIERS[-1] == (15, 38, 3)


# ------------------------------------------------------------------- hand-flagged absences


def unavailable_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["season", "team", "player", "from_date", "until_date", "reason", "note"]
    )
    for column in ("from_date", "until_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def test_a_missing_file_is_an_empty_frame_not_an_error(tmp_path):
    assert load_unavailable(tmp_path / "absent.csv").empty


def test_a_file_missing_a_column_raises(tmp_path):
    path = tmp_path / "unavailable.csv"
    path.write_text("season,team,player\n", encoding="utf-8")

    with pytest.raises(ValueError, match="from_date"):
        load_unavailable(path)


def test_a_hand_flagged_player_is_out_within_the_window():
    frame = unavailable_frame(
        [
            {
                "season": SEASON,
                "team": TEAM,
                "player": "Saka",
                "from_date": "2025-09-01",
                "until_date": "2025-10-01",
                "reason": "injury",
                "note": "hamstring",
            }
        ]
    )

    assert manually_unavailable(frame, TEAM, SEASON, pd.Timestamp("2025-09-15")) == {"Saka"}
    assert manually_unavailable(frame, TEAM, SEASON, pd.Timestamp("2025-10-15")) == set()
    assert manually_unavailable(frame, TEAM, SEASON, pd.Timestamp("2025-08-15")) == set()


def test_an_open_ended_absence_has_no_end():
    """An injury with no announced return is the normal case, not a malformed row."""
    frame = unavailable_frame(
        [
            {
                "season": SEASON,
                "team": TEAM,
                "player": "Saka",
                "from_date": "2025-09-01",
                "until_date": None,
                "reason": "injury",
                "note": "",
            }
        ]
    )

    assert manually_unavailable(frame, TEAM, SEASON, pd.Timestamp("2026-03-01")) == {"Saka"}


def test_suspensions_and_hand_flags_combine():
    lineups = season_lineups({2: {"Rice": (0, 1)}})
    frame = unavailable_frame(
        [
            {
                "season": SEASON,
                "team": TEAM,
                "player": "Saka",
                "from_date": "2025-08-01",
                "until_date": None,
                "reason": "injury",
                "note": "",
            }
        ]
    )

    out = unavailable_for(lineups, TEAM, SEASON, match_date(3), unavailable=frame)

    assert out == {"Rice", "Saka"}
