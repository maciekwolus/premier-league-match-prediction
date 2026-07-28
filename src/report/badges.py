"""Pixel-art club badges, generated rather than downloaded.

Real club crests are trademarked and not ours to redistribute, so these are **kit
patterns instead of logos**: a shirt silhouette in the club's actual colours, wearing
the stripe, hoop or halved pattern that club is known for. Newcastle come out black and
white striped, Wolves gold on black, Norwich yellow and green - recognisable at a glance
without pretending to be anyone's badge.

Everything is drawn as an inline SVG data URI, so the report needs no image files, no
network, and no build step.
"""

from __future__ import annotations

import base64

# Kit patterns. `plain` is a single colour, `stripes` vertical bars, `hoops` horizontal
# bands, `halves` a split down the middle, `sash` a diagonal band.
PLAIN, STRIPES, HOOPS, HALVES, SASH = "plain", "stripes", "hoops", "halves", "sash"

# (primary, secondary, pattern). Colours are the clubs' familiar home kits.
CLUB_KITS: dict[str, tuple[str, str, str]] = {
    "Arsenal": ("#EF0107", "#FFFFFF", PLAIN),
    "Aston Villa": ("#670E36", "#95BFE5", HALVES),
    "Bournemouth": ("#DA291C", "#000000", STRIPES),
    "Brentford": ("#D20000", "#FFFFFF", STRIPES),
    "Brighton": ("#0057B8", "#FFFFFF", STRIPES),
    "Burnley": ("#6C1D45", "#99D6EA", PLAIN),
    "Chelsea": ("#034694", "#FFFFFF", PLAIN),
    "Crystal Palace": ("#1B458F", "#C4122E", STRIPES),
    "Everton": ("#003399", "#FFFFFF", PLAIN),
    "Fulham": ("#FFFFFF", "#000000", PLAIN),
    "Ipswich": ("#3A64A3", "#FFFFFF", PLAIN),
    "Leeds": ("#FFFFFF", "#FFCD00", PLAIN),
    "Leicester": ("#003090", "#FDBE11", PLAIN),
    "Liverpool": ("#C8102E", "#00B2A9", PLAIN),
    "Luton": ("#F78F1E", "#002D62", PLAIN),
    "Man City": ("#6CABDD", "#FFFFFF", PLAIN),
    "Man United": ("#DA291C", "#FBE122", PLAIN),
    "Newcastle": ("#241F20", "#FFFFFF", STRIPES),
    "Norwich": ("#FFF200", "#00A650", PLAIN),
    "Nott'm Forest": ("#DD0000", "#FFFFFF", PLAIN),
    "Sheffield United": ("#EE2737", "#FFFFFF", STRIPES),
    "Southampton": ("#D71920", "#FFFFFF", STRIPES),
    "Sunderland": ("#EB172B", "#FFFFFF", STRIPES),
    "Tottenham": ("#FFFFFF", "#132257", PLAIN),
    "Watford": ("#FBEE23", "#ED2127", HOOPS),
    "West Brom": ("#122F67", "#FFFFFF", STRIPES),
    "West Ham": ("#7A263A", "#1BB1E7", SASH),
    "Wolves": ("#FDB913", "#231F20", PLAIN),
}

# A club we have no kit for - a newly promoted side, most likely.
DEFAULT_KIT = ("#7A7A7A", "#D8D8D8", PLAIN)

# The shirt occupies a 12x12 grid. Small enough to read as pixel art, large enough for
# four stripes to be distinguishable.
GRID = 12

# 1 is shirt body, 2 is sleeve, 0 is background. Drawn once and reused for every club:
# the silhouette is the same, only the colours and the pattern on top of it change.
SHIRT = [
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]


def kit_for(team: str) -> tuple[str, str, str]:
    return CLUB_KITS.get(team, DEFAULT_KIT)


def _cell_colour(row: int, column: int, primary: str, secondary: str, pattern: str) -> str:
    """Which of the two colours this pixel of the shirt body takes."""
    if pattern == STRIPES:
        return secondary if (column // 2) % 2 else primary
    if pattern == HOOPS:
        return secondary if (row // 3) % 2 else primary
    if pattern == HALVES:
        return secondary if column >= GRID // 2 else primary
    if pattern == SASH:
        # A diagonal band two pixels wide, running shoulder to hip.
        return secondary if 0 <= (column - row + 3) <= 1 else primary
    return primary


def badge_svg(team: str, scale: int = 4) -> str:
    """The club's kit as a pixel-art SVG."""
    primary, secondary, pattern = kit_for(team)
    size = GRID * scale

    rects = []
    for row, line in enumerate(SHIRT):
        for column, cell in enumerate(line):
            if cell == 0:
                continue
            colour = (
                secondary if cell == 2 else _cell_colour(row, column, primary, secondary, pattern)
            )
            rects.append(
                f'<rect x="{column * scale}" y="{row * scale}" '
                f'width="{scale}" height="{scale}" fill="{colour}"/>'
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" shape-rendering="crispEdges">'
        f"{''.join(rects)}</svg>"
    )


def badge_data_uri(team: str, scale: int = 4) -> str:
    """The badge as a data URI, ready for an ``img`` tag."""
    encoded = base64.b64encode(badge_svg(team, scale).encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
