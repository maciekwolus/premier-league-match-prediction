"""Team name reconciliation between data sources.

football-data.co.uk names are canonical, because `match_id` is built from them in
:mod:`src.data.clean_matches` and every later join hangs off that id.

Across the seven seasons in scope, Understat and football-data agree on 22 of 28 team
names; only the six below differ. Mapping them explicitly - rather than fuzzy matching -
means an unexpected name raises instead of silently dropping a fixture.
"""

from __future__ import annotations

# Understat name -> football-data name
UNDERSTAT_TO_FOOTBALL_DATA = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}


class UnknownTeamError(KeyError):
    """Raised when a source produces a team name we have no mapping for."""


def understat_to_football_data(name: str, known_teams: set[str] | None = None) -> str:
    """Translate an Understat team name to its football-data equivalent.

    Names that already agree pass through untouched. When ``known_teams`` is supplied
    the result is checked against it, so a name that is neither mapped nor recognised
    raises rather than quietly entering the pipeline.
    """
    mapped = UNDERSTAT_TO_FOOTBALL_DATA.get(name, name)

    if known_teams is not None and mapped not in known_teams:
        raise UnknownTeamError(
            f"Understat team {name!r} mapped to {mapped!r}, which is not a known "
            f"football-data team. Add it to UNDERSTAT_TO_FOOTBALL_DATA."
        )

    return mapped
