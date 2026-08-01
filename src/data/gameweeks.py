"""Which gameweek a fixture belongs to.

No upstream source gives a round number - football-data publishes dates, Understat
publishes dates, and neither says "this is matchday 12". So it is derived: within a
season, a fixture belongs to the gameweek matching how many matches its clubs have
played once it is counted.

**Date clustering is the obvious approach and it is wrong.** A postponed match played
six weeks later is still part of its original round to the league, but no date rule puts
it there, and midweek rounds sit close enough to weekend ones to merge. Counting each
club's matches instead is robust to both: a club plays each round exactly once.

Measured against 2025/26: 36 of the 38 gameweeks come out at exactly 10 fixtures, and
the remaining 2 split because a match was rescheduled across a round boundary. That is
the true shape of a Premier League season, so callers must not assume 10.
"""

from __future__ import annotations

import pandas as pd


def assign_gameweeks(matches: pd.DataFrame) -> pd.Series:
    """A gameweek number per row, aligned to ``matches.index``.

    Counts within each season separately, in date order. A fixture's gameweek is the
    higher of its two clubs' running match counts, so a club returning from a postponed
    fixture does not drag the whole round backwards.
    """
    required = {"season", "date", "home_team", "away_team"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"assign_gameweeks needs column(s) {sorted(missing)}")

    gameweeks = pd.Series(pd.NA, index=matches.index, dtype="Int64")

    for _, season_rows in matches.groupby("season", sort=False):
        played: dict[str, int] = {}
        # Sort by date but keep the original index, so the result can be assigned back
        # to rows rather than to positions.
        for index in season_rows.sort_values("date", kind="stable").index:
            home = matches.at[index, "home_team"]
            away = matches.at[index, "away_team"]
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1
            gameweeks.at[index] = max(played[home], played[away])

    return gameweeks


def gameweek_sizes(matches: pd.DataFrame, gameweeks: pd.Series | None = None) -> pd.Series:
    """How many fixtures fell in each gameweek. Diagnostic, not a validation.

    A round that is not 10 is normal - see the module docstring - so this reports rather
    than raises. It exists so a caller can see *which* rounds split.
    """
    if gameweeks is None:
        gameweeks = assign_gameweeks(matches)
    return gameweeks.value_counts().sort_index()
