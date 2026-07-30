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


# Fantasy Premier League club name -> football-data name.
#
# Only the differences are listed; the other fifteen agree. The three promoted clubs are
# spelled by football-data as they appear in its Championship files - `Coventry`, `Hull`,
# `Ipswich` - which was checked against the source rather than guessed, and Ipswich is
# already in `matches.parquet` from 2024/25 under exactly that name.
FPL_TO_FOOTBALL_DATA = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}


# FIFA / EA FC club name -> football-data name.
#
# Unlike Understat, the ratings files cover every club in the world, so a name that is
# absent from this table is simply not a Premier League club and is dropped. That makes
# a *missing* mapping silent, so the loader instead asserts that each edition yields
# exactly 20 Premier League clubs - a club we failed to map shows up as 19.
#
# Several clubs appear under more than one name across editions, usually for licensing
# reasons, so this mapping is deliberately many-to-one.
FIFA_TO_FOOTBALL_DATA = {
    # Promoted for 2026/27. They have never been in this project's Premier League window,
    # so their ratings were previously dropped as any other foreign club would be - which
    # left three clubs of the new season with no squad quality at all.
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    # EA FC 26 abbreviates where earlier editions spelled clubs out. Listing the short
    # forms explicitly beats normalising, because the abbreviations are not derivable
    # ("Spurs", "Man Utd") and near-misses exist that must NOT match - "Newcastle Jets"
    # and "Notts County" both live in the same file.
    "Brighton": "Brighton",
    "Man Utd": "Man United",
    "Man City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "West Ham": "West Ham",
    "West Brom": "West Brom",
    "Wolves": "Wolves",
    "Tottenham": "Tottenham",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Norwich": "Norwich",
    "Ipswich": "Ipswich",
    "Luton": "Luton",
    "Sheffield Utd": "Sheffield United",
}

PREMIER_LEAGUE_CLUBS_PER_SEASON = 20


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


def fpl_to_football_data(name: str) -> str:
    """Translate a Fantasy Premier League club name to its football-data equivalent.

    Raises on anything unrecognised rather than passing it through. Unlike the ratings
    files, the FPL club list is exactly the twenty clubs in the division, so a name we
    cannot place is a real problem - a promoted club we have not mapped - and silently
    accepting it would put a club into the pipeline under a name nothing else uses.
    """
    mapped = FPL_TO_FOOTBALL_DATA.get(name, name)

    if mapped not in set(FIFA_TO_FOOTBALL_DATA.values()) | set(FPL_TO_FOOTBALL_DATA.values()):
        raise UnknownTeamError(
            f"FPL team {name!r} mapped to {mapped!r}, which is not a football-data team "
            f"this project knows. Add it to FPL_TO_FOOTBALL_DATA."
        )

    return mapped


def fifa_to_football_data(name: str) -> str | None:
    """Translate a FIFA club name, or ``None`` if it is not a Premier League club.

    Returning ``None`` rather than raising is deliberate: the ratings files list every
    club in the world, and dropping the other ~700 is the intended behaviour. The
    safety net against a *mis-mapped* Premier League club is the caller's check that
    each edition yields exactly 20 clubs.
    """
    return FIFA_TO_FOOTBALL_DATA.get(name.strip() if isinstance(name, str) else name)
