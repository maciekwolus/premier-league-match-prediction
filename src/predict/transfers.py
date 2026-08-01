"""Who is actually at a club right now, from the Fantasy Premier League squad lists.

Ratings are a September snapshot and appearances are last season's, so both go stale the
moment a transfer window opens. Casemiro started for Man United through 2025/26, has since
left, and every source this project held still believed he was there - the expected XI put
him in midfield for the opening round of 2026/27.

**FPL is the one source that knows.** It lists every club's current squad, free and
updated weekly, because it has to: people pick those players. Diffing our expected XI
against it turns "somebody has to notice a transfer" into "the data notices".

The cost is name matching, and it is the Phase 4 problem again rather than a lookup.
Understat says ``Bruno Fernandes``, FPL says ``Bruno Borges Fernandes``; Understat says
``Amad Diallo Traore``, FPL says ``Amad Diallo``. A naive comparison calls both departed,
which would delete two of Man United's best players. So the same cascade is used here:
exact on normalised names, then subset matching within the club, and only a name with no
candidate at all counts as gone.

**A departure is acted on; an arrival is only reported.** Removing a player we can show is
no longer at the club is safe - the next-most-used player steps up, exactly as with a
suspension. Deciding that a signing will *start*, when they have never played for the club
and we have no appearances to rank them by, is a guess this project has no basis for. They
are surfaced for a human instead.
"""

from __future__ import annotations

import pandas as pd

from src.data.clean_fpl import club_names
from src.data.fetch_fpl import fetch
from src.matching.player_names import normalise

# A club's squad is small, so containment is safe here in a way it would not be against a
# world-wide pool: "Amad" fits inside "Amad Diallo Traore" and nothing else in the squad.
MIN_TOKENS_FOR_SUBSET = 1


def fpl_squads(bootstrap: dict | None = None) -> dict[str, list[dict]]:
    """Every club's current squad, keyed by football-data club name.

    Each player carries every spelling FPL offers - full name, short name and surname -
    because which one matches Understat varies by player.
    """
    bootstrap = bootstrap if bootstrap is not None else fetch("bootstrap-static/")
    clubs = club_names(bootstrap)

    squads: dict[str, list[dict]] = {name: [] for name in clubs.values()}
    for element in bootstrap["elements"]:
        club = clubs.get(element["team"])
        if club is None:
            continue
        full = f"{element.get('first_name', '')} {element.get('second_name', '')}".strip()
        squads[club].append(
            {
                "full_name": full,
                "web_name": element.get("web_name", ""),
                "second_name": element.get("second_name", ""),
                "status": element.get("status", "a"),
            }
        )
    return squads


def _spellings(player: dict) -> set[frozenset[str]]:
    """Every name FPL offers for a player, as normalised token sets."""
    names = (player["full_name"], player["web_name"], player["second_name"])
    return {frozenset(normalise(name).split()) for name in names if normalise(name)}


def is_in_squad(player_name: str, squad: list[dict]) -> bool:
    """Whether this player appears in the club's current squad, by any spelling.

    Matching is deliberately generous: a false "not in squad" removes a real player from
    the expected XI, which is a worse error than leaving a departed one in. Containment in
    either direction counts, which is what rescues ``Amad Diallo Traore`` against FPL's
    ``Amad Diallo`` and ``Bruno Fernandes`` against ``Bruno Borges Fernandes``.
    """
    target = frozenset(normalise(player_name).split())
    if not target:
        return False

    for candidate in squad:
        for tokens in _spellings(candidate):
            if not tokens:
                continue
            if target == tokens or target <= tokens or tokens <= target:
                return True
    return False


def departures(players: list[str], team: str, squads: dict[str, list[dict]]) -> list[str]:
    """Players in our XI who are no longer in the club's squad.

    An unknown club returns nothing rather than declaring the whole side departed - the
    safe direction when a club name fails to map.
    """
    squad = squads.get(team)
    if not squad:
        return []
    return [player for player in players if not is_in_squad(player, squad)]


def arrivals(
    team: str,
    squads: dict[str, list[dict]],
    known_players: set[str],
    fifa: pd.DataFrame | None = None,
) -> list[dict]:
    """Squad members with no appearance history for this club, best-rated first.

    Reported rather than selected. A signing has never played for the club, so there is
    nothing to rank them by against the players who have - putting them straight into the
    XI would be asserting a team sheet rather than describing one.
    """
    squad = squads.get(team) or []
    new = [
        player for player in squad if not any(is_in_squad(name, [player]) for name in known_players)
    ]

    if fifa is None or fifa.empty:
        return [{"player": player["web_name"], "overall": None} for player in new]

    rated = []
    for player in new:
        overall = _rating_for(player, fifa)
        rated.append({"player": player["web_name"], "overall": overall})
    return sorted(rated, key=lambda row: (row["overall"] is None, -(row["overall"] or 0)))


def _rating_for(player: dict, fifa: pd.DataFrame) -> int | None:
    """The FIFA overall for an FPL player, or None when no single name matches."""
    for name in (player["full_name"], player["second_name"], player["web_name"]):
        target = frozenset(normalise(name).split())
        if not target:
            continue
        contains = fifa["_tokens"].map(lambda tokens, want=target: bool(tokens) and want <= tokens)
        hits = fifa[contains]
        if len(hits) == 1:
            return int(hits.iloc[0]["overall"])
    return None


def rating_index(fifa: pd.DataFrame, club: str, season: str) -> pd.DataFrame:
    """One club's ratings with a token column, ready for ``_rating_for``."""
    squad = fifa[(fifa.get("club_fd") == club) & (fifa["season"] == season)].copy()
    if squad.empty:
        return squad
    squad["_tokens"] = squad["player_name"].map(lambda name: frozenset(normalise(name).split()))
    return squad
