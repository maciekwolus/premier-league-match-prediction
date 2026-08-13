"""Tests for deciding, unattended, whether a round should be predicted now.

This is the judgement a scheduled job makes with nobody watching, and both ways of
getting it wrong are silent. Predicting too early commits to a team sheet a fortnight
out - and a stored round is never rewritten, so that decision cannot be taken back.
Predicting after kickoff produces a "forecast" of a match already under way, which is
the one thing the archive exists to rule out.
"""

from __future__ import annotations

import pandas as pd

from src.predict.fixtures import DEFAULT_LEAD_DAYS, due_round

NOW = pd.Timestamp("2026-08-18 09:00")


def schedule(*rounds: tuple[int, str]) -> pd.DataFrame:
    """A schedule as ``(gameweek, first kickoff)``, two fixtures each."""
    rows = []
    for gameweek, first in rounds:
        start = pd.Timestamp(first)
        rows += [
            {"gameweek": gameweek, "date": start, "home_team": "A", "away_team": "B"},
            {
                "gameweek": gameweek,
                "date": start + pd.Timedelta(days=1),
                "home_team": "C",
                "away_team": "D",
            },
        ]
    return pd.DataFrame(rows)


def test_a_round_inside_the_window_is_due():
    fixtures = schedule((1, "2026-08-20 19:00"))
    assert due_round(fixtures, now=NOW, within_days=DEFAULT_LEAD_DAYS) == 1


def test_the_window_counts_hours_not_calendar_days():
    """A Friday 19:00 kickoff is three days *and ten hours* from Tuesday morning, so it
    is not yet due. This is why a daily job needs a window of days rather than a fixed
    weekday: the round still gets three attempts, they just start on the Wednesday.
    """
    friday_evening = schedule((1, "2026-08-21 19:00"))

    assert due_round(friday_evening, now=NOW, within_days=3) is None
    assert due_round(friday_evening, now=NOW + pd.Timedelta(days=1), within_days=3) == 1


def test_a_round_beyond_the_window_waits():
    """Predicting a fortnight early is a decision that cannot be taken back, because the
    archive refuses to rewrite what it stored."""
    fixtures = schedule((1, "2026-09-05 15:00"))
    assert due_round(fixtures, now=NOW, within_days=DEFAULT_LEAD_DAYS) is None


def test_a_round_that_has_already_kicked_off_is_never_due():
    """If the job was down for the whole window the honest outcome is a missing round.
    A prediction written after kickoff is not a prediction."""
    fixtures = schedule((1, "2026-08-18 08:00"))
    assert due_round(fixtures, now=NOW) is None


def test_the_earliest_upcoming_round_is_the_one_chosen():
    fixtures = schedule((1, "2026-08-20 19:00"), (2, "2026-08-27 19:00"))
    assert due_round(fixtures, now=NOW, within_days=30) == 1


def test_a_finished_round_is_skipped_for_the_next_one():
    fixtures = schedule((1, "2026-08-10 19:00"), (2, "2026-08-20 19:00"))
    assert due_round(fixtures, now=NOW) == 2


def test_the_window_is_measured_from_the_first_kickoff_not_the_last():
    """A round spans a weekend. Measuring from its last fixture would let the job predict
    a round whose opening match had already been played."""
    fixtures = pd.DataFrame(
        [
            {
                "gameweek": 1,
                "date": pd.Timestamp("2026-08-17 19:00"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                "gameweek": 1,
                "date": pd.Timestamp("2026-08-19 19:00"),
                "home_team": "C",
                "away_team": "D",
            },
        ]
    )
    assert due_round(fixtures, now=NOW) is None


def test_an_empty_schedule_is_not_due():
    assert due_round(pd.DataFrame()) is None


def test_a_schedule_without_gameweeks_is_not_due():
    """Between seasons the fixture feed has no rounds. Nothing to predict beats guessing."""
    fixtures = pd.DataFrame([{"date": pd.Timestamp("2026-08-20"), "home_team": "A"}])
    assert due_round(fixtures, now=NOW) is None


def test_a_round_with_no_announced_kickoff_is_not_due():
    """Televised fixtures are listed before they are scheduled. A round with no date
    cannot be judged against a window, so it waits rather than being predicted blind."""
    fixtures = pd.DataFrame([{"gameweek": 1, "date": pd.NaT, "home_team": "A", "away_team": "B"}])
    assert due_round(fixtures, now=NOW) is None


def test_the_window_boundary_is_inclusive():
    """Exactly three days out is due - otherwise a round could fall between two daily
    runs and never be predicted at all."""
    fixtures = schedule((1, "2026-08-21 09:00"))
    assert due_round(fixtures, now=NOW, within_days=3) == 1
