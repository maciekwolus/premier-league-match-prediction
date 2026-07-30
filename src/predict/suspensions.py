"""Who is banned for an upcoming fixture, derived from cards already in the data.

``lineups.parquet`` records yellow and red cards per player per match, so suspensions
need no new source: 318 reds and 9,575 yellows across seven seasons are already on disk.
That makes this the one availability signal that is free and exact.

**What cannot be seen: the offence.** A straight red for violent conduct is a three-match
ban and a second yellow is one, and Understat records both as a dismissal with no reason
attached - only 7 of 318 reds carry a yellow on the same row, so the two cannot be told
apart even by inference. Every red is therefore treated as one match, which under-counts
the serious ones. That is a deliberate floor: banning a player for three matches when it
was really one would remove a starter who is actually available, and the wrong direction
of error is the one that changes a prediction on no evidence.

Injuries are not here. They cannot be derived from anything we hold and there is no
usable free feed - see ``data/manual/unavailable.csv`` for the hand-maintained answer.
"""

from __future__ import annotations

import pandas as pd

from src.config import MANUAL_DIR

UNAVAILABLE_CSV = MANUAL_DIR / "unavailable.csv"

# Every dismissal costs one match. See the module docstring: the alternative is guessing
# at the offence, and guessing high removes available players.
RED_CARD_BAN = 1

# Premier League yellow-card accumulation. Each tier is (cards, by which club match, ban).
# The deadlines matter: reaching five yellows in match 25 carries no ban, because the
# tier only applies within the club's first 19 matches. Counts do not reset when a ban is
# served - a player banned at five is banned again at ten.
#
# The third tier has never fired in the seven seasons on disk: 783 player-seasons reach
# five yellows, 69 reach ten, none reach fifteen. It is encoded because the rule exists,
# not because it is load-bearing.
YELLOW_TIERS: tuple[tuple[int, int, int], ...] = (
    (5, 19, 1),
    (10, 32, 2),
    (15, 38, 3),
)


def load_unavailable(path=None) -> pd.DataFrame:
    """Hand-recorded absences - injuries, illness, compassionate leave, anything else.

    Records *changes*, never whole squads, in keeping with every other manual file here:
    a row is one player out for one window, and an empty file means nobody is flagged.
    """
    path = path or UNAVAILABLE_CSV
    columns = ["season", "team", "player", "from_date", "until_date", "reason", "note"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    for column in ("from_date", "until_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def club_matches(lineups: pd.DataFrame, team: str, season: str) -> list[pd.Timestamp]:
    """The dates this club played in this season, in order.

    Bans are counted in matches, not days, so the club's own fixture list is the clock.
    """
    missing = {"team", "season", "date"} - set(lineups.columns)
    if missing:
        # Loudly, rather than returning an empty set: a caller passing lineups without a
        # season would otherwise see suspensions silently stop being applied, which is
        # indistinguishable from nobody being banned.
        raise ValueError(f"suspensions need column(s) {sorted(missing)} on lineups")

    played = lineups[(lineups["team"] == team) & (lineups["season"] == season)]
    return sorted(played["date"].drop_duplicates().tolist())


def _ban_windows(cards: pd.DataFrame, fixtures: list[pd.Timestamp]) -> dict[str, set[int]]:
    """Match indices each player is banned for, keyed by player.

    Index 0 is the club's first match of the season. A ban triggered in match *i* covers
    matches *i+1* onward, because the offence is served from the next fixture.
    """
    banned: dict[str, set[int]] = {}
    position = {date: index for index, date in enumerate(fixtures)}

    for player, rows in cards.groupby("player"):
        rows = rows.sort_values("date")
        out: set[int] = set()
        running_yellows = 0
        reached: set[int] = set()

        for row in rows.itertuples():
            index = position.get(row.date)
            if index is None:
                continue

            if row.red_cards:
                out.update(range(index + 1, index + 1 + RED_CARD_BAN))

            running_yellows += int(row.yellow_cards)
            for threshold, deadline, length in YELLOW_TIERS:
                # ``>=`` rather than ``==`` because a player can pick up two yellows in
                # one match and step over a threshold rather than onto it.
                if threshold in reached or running_yellows < threshold:
                    continue
                reached.add(threshold)
                # The deadline is on the match in which the threshold is reached, counted
                # from one: reaching five in the club's 20th match carries no ban.
                if index + 1 <= deadline:
                    out.update(range(index + 1, index + 1 + length))

        if out:
            banned[player] = out
    return banned


def suspended_for(lineups: pd.DataFrame, team: str, season: str, before: pd.Timestamp) -> set[str]:
    """Players banned for this club's next fixture on or after ``before``.

    ``lineups`` must carry ``season`` and ``date``; the caller already joins those on for
    the expected XI, so nothing extra is loaded here.
    """
    fixtures = club_matches(lineups, team, season)
    if not fixtures:
        return set()

    # The fixture being predicted sits after everything played, so its index is however
    # many the club has already played. A replayed round lands on its own index.
    target = sum(1 for date in fixtures if date < before)

    cards = lineups[
        (lineups["team"] == team)
        & (lineups["season"] == season)
        & ((lineups["yellow_cards"] > 0) | (lineups["red_cards"] > 0))
    ]
    if cards.empty:
        return set()

    windows = _ban_windows(cards, fixtures)
    return {player for player, indices in windows.items() if target in indices}


def manually_unavailable(
    unavailable: pd.DataFrame, team: str, season: str, on: pd.Timestamp
) -> set[str]:
    """Players hand-flagged as out on this date.

    A blank ``until_date`` means open-ended, which is the honest default for an injury
    with no announced return.
    """
    if unavailable.empty:
        return set()

    rows = unavailable[
        (unavailable["team"] == team)
        & (unavailable["season"].astype(str) == str(season))
        & (unavailable["from_date"] <= on)
    ]
    open_ended = rows["until_date"].isna()
    return set(rows[open_ended | (rows["until_date"] >= on)]["player"])


def unavailable_for(
    lineups: pd.DataFrame,
    team: str,
    season: str,
    before: pd.Timestamp,
    unavailable: pd.DataFrame | None = None,
) -> set[str]:
    """Everyone this club cannot pick: suspended, plus anyone flagged by hand."""
    out = suspended_for(lineups, team, season, before)
    if unavailable is None:
        unavailable = load_unavailable()
    return out | manually_unavailable(unavailable, team, season, before)
