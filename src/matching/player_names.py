"""Match Understat player names to FIFA / EA FC player names.

Understat uses display names (``Virgil van Dijk``). FIFA carries both a short form
(``V. van Dijk``) and a long form (``Virgil van Dijk``), and which of the two looks like
the Understat name varies by edition. Accents, hyphens and one-name Brazilians make the
rest messy.

Matching runs as a cascade, cheapest and safest first, and stops at the first hit:

1. ``exact_long``    normalised Understat name == normalised FIFA long name
2. ``exact_short``   == normalised FIFA short name
3. ``initials``      Understat reduced to ``v van dijk`` == FIFA short name
4. ``subset``        one name's words contained in the other's, uniquely
5. ``fuzzy_club``    best rapidfuzz score within the same club and season
6. ``fuzzy_season``  best score anywhere in the season, at a higher threshold

**Scope is what makes fuzzy matching safe here.** Steps 1-4 compare against one club's
squad - roughly 30 candidates - rather than the 2,483 names in the file. Step 5 widens
to the season only because players move clubs mid-season, and pays for it with a higher
threshold.

Anything still unmatched belongs in ``data/manual/player_name_overrides.csv``, which is
consulted before the cascade runs and always wins.

Usage:
    python -m src.matching.player_names
"""

from __future__ import annotations

import re
import sys
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from src.config import MANUAL_DIR, PROCESSED_DIR
from src.data.clean_lineups import LINEUPS_PARQUET
from src.data.load_fifa import FIFA_PLAYERS_PARQUET

PLAYER_MAP_PARQUET = PROCESSED_DIR / "player_map.parquet"
OVERRIDES_CSV = MANUAL_DIR / "player_name_overrides.csv"

# Below these, a "best match" is more likely to be a different player than a spelling
# variant. Season-wide matching is stricter because it has ~20x more chances to be wrong.
FUZZY_CLUB_THRESHOLD = 82
FUZZY_SEASON_THRESHOLD = 92

# Phase 5 reads squad quality off the starting XI, so that is what coverage must measure.
TARGET_STARTER_COVERAGE = 95.0


# Letters that NFKD cannot decompose, because they are distinct letters rather than a
# base plus a mark. Without these, "Ødegaard" normalises to "degaard" - the character is
# dropped outright - and never matches Understat's "Odegaard".
TRANSLITERATIONS = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "ł": "l",
        "Ł": "L",
        "þ": "th",
        "Þ": "Th",
        "ß": "ss",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
        "ı": "i",
    }
)


def normalise(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    "Virgil van Dijk" -> "virgil van dijk";  "V. van Dijk" -> "v van dijk"
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name.translate(TRANSLITERATIONS))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_only.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def to_initials(name: str) -> str:
    """Reduce a full name to the abbreviated form FIFA uses.

    "Virgil van Dijk" -> "v van dijk", matching FIFA's "V. van Dijk".
    """
    parts = normalise(name).split()
    if len(parts) < 2:
        return " ".join(parts)
    return " ".join([parts[0][0], *parts[1:]])


def load_overrides() -> dict[tuple[str, str], str]:
    """Hand-written decisions, keyed by (season, understat name)."""
    if not OVERRIDES_CSV.exists():
        return {}

    df = pd.read_csv(OVERRIDES_CSV)
    required = {"season", "understat_player", "fifa_player_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{OVERRIDES_CSV} is missing column(s) {sorted(missing)}")

    df = df[df["fifa_player_name"].notna()]
    return {(row.season, row.understat_player): row.fifa_player_name for row in df.itertuples()}


def understat_players(lineups: pd.DataFrame) -> pd.DataFrame:
    """Distinct Understat players per season and club, with minutes as their weight."""
    lineups = lineups.copy()
    lineups["season"] = lineups["match_id"].str.slice(0, 7).str.replace("_", "/", regex=False)

    grouped = (
        lineups.groupby(["season", "team", "player"], as_index=False)
        .agg(
            minutes=("minutes", "sum"),
            starts=("is_starter", "sum"),
            appearances=("match_id", "count"),
        )
        .sort_values(["season", "team", "minutes"], ascending=[True, True, False])
    )
    return grouped


def _index_fifa(fifa: pd.DataFrame) -> pd.DataFrame:
    """Add the normalised forms the cascade compares against."""
    fifa = fifa.copy()
    fifa["norm_short"] = fifa["player_name"].map(normalise)
    fifa["norm_long"] = fifa["long_name"].map(normalise)
    fifa["tokens"] = [
        frozenset(short.split()) | frozenset(long_.split())
        for short, long_ in zip(fifa["norm_short"], fifa["norm_long"], strict=True)
    ]
    return fifa


def subset_match(understat_name: str, scope: pd.DataFrame) -> str | None:
    """Match when one name's words are contained in the other's, and only one fits.

    EA FC 25 records full birth names - "Mohamed Salah Hamed Ghaly" for Understat's
    "Mohamed Salah" - which defeats both exact comparison and token_sort_ratio, since
    the length gap drags the score below any safe threshold.

    Containment alone is not enough: Arsenal's 2024/25 squad contains both "Gabriel"
    and "Gabriel Martinelli", so a lone "Gabriel" is genuinely ambiguous. Requiring
    exactly one candidate turns those into unmatched rows for a human to settle, rather
    than a coin flip recorded as fact.
    """
    target = frozenset(normalise(understat_name).split())
    if not target:
        return None

    hits = {
        row.player_name
        for row in scope.itertuples()
        if row.tokens and (target <= row.tokens or row.tokens <= target)
    }
    return hits.pop() if len(hits) == 1 else None


def match_one(
    understat_name: str,
    club_squad: pd.DataFrame,
    season_squad: pd.DataFrame,
    world_squad: pd.DataFrame | None = None,
) -> tuple[str | None, str, float]:
    """Resolve one name. Returns (fifa player_name, method, score).

    ``world_squad`` is every club in that edition, not just the Premier League. Ratings
    are a September snapshot, so a January signing is still listed at their old club and
    is invisible to the first two scopes. It is searched last and with exact rules only -
    fuzzy matching against 18,000 names would invent matches rather than find them.
    """
    target = normalise(understat_name)
    initials = to_initials(understat_name)

    for scope, scope_name in ((club_squad, "club"), (season_squad, "season")):
        if scope.empty:
            continue

        hit = scope[scope["norm_long"] == target]
        if not hit.empty:
            return hit.iloc[0]["player_name"], f"exact_long_{scope_name}", 100.0

        hit = scope[scope["norm_short"] == target]
        if not hit.empty:
            return hit.iloc[0]["player_name"], f"exact_short_{scope_name}", 100.0

        hit = scope[scope["norm_short"] == initials]
        if not hit.empty:
            return hit.iloc[0]["player_name"], f"initials_{scope_name}", 100.0

        unique = subset_match(understat_name, scope)
        if unique is not None:
            return unique, f"subset_{scope_name}", 100.0

    for scope, scope_name, threshold in (
        (club_squad, "club", FUZZY_CLUB_THRESHOLD),
        (season_squad, "season", FUZZY_SEASON_THRESHOLD),
    ):
        if scope.empty:
            continue

        choices = {}
        for row in scope.itertuples():
            for candidate in (row.norm_long, row.norm_short):
                if candidate:
                    choices.setdefault(candidate, row.player_name)

        best = process.extractOne(target, list(choices), scorer=fuzz.token_sort_ratio)
        if best and best[1] >= threshold:
            return choices[best[0]], f"fuzzy_{scope_name}", float(best[1])

    if world_squad is not None and not world_squad.empty:
        for column in ("norm_long", "norm_short"):
            hit = world_squad[world_squad[column] == target]
            if len(hit) == 1:
                return hit.iloc[0]["player_name"], "exact_world", 100.0

    return None, "unmatched", 0.0


def build_map(lineups: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Resolve every Understat player-season to a FIFA player."""
    fifa = _index_fifa(fifa)
    overrides = load_overrides()
    players = understat_players(lineups)

    # The first two scopes see Premier League squads only; the third sees every club.
    premier_league = fifa[fifa["in_premier_league"]] if "in_premier_league" in fifa else fifa
    by_club = {key: group for key, group in premier_league.groupby(["season", "club_fd"])}
    by_season = {key: group for key, group in premier_league.groupby("season")}
    by_world = {key: group for key, group in fifa.groupby("season")}

    records = []
    for row in players.itertuples():
        override = overrides.get((row.season, row.player))
        if override:
            records.append((row.season, row.team, row.player, override, "override", 100.0))
            continue

        empty = fifa.iloc[0:0]
        name, method, score = match_one(
            row.player,
            by_club.get((row.season, row.team), empty),
            by_season.get(row.season, empty),
            by_world.get(row.season, empty),
        )
        records.append((row.season, row.team, row.player, name, method, score))

    resolved = pd.DataFrame(
        records,
        columns=["season", "team", "understat_player", "fifa_player_name", "method", "score"],
    )
    return players.merge(
        resolved,
        left_on=["season", "team", "player"],
        right_on=["season", "team", "understat_player"],
    ).drop(columns="player")


def coverage(player_map: pd.DataFrame) -> pd.DataFrame:
    """Per-season match rate, weighted by starting appearances rather than by name.

    A missed squad player costs almost nothing; a missed regular starter distorts every
    match that player featured in, so starts are the honest denominator.
    """
    player_map = player_map.copy()
    player_map["matched"] = player_map["fifa_player_name"].notna()

    return player_map.groupby("season").apply(
        lambda g: pd.Series(
            {
                "players": len(g),
                "matched": int(g["matched"].sum()),
                "name_pct": round(100 * g["matched"].mean(), 1),
                "starter_pct": round(
                    100 * g.loc[g["matched"], "starts"].sum() / max(g["starts"].sum(), 1), 1
                ),
            }
        ),
        include_groups=False,
    )


def build(strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, hint in (
        (LINEUPS_PARQUET, "python -m src.data.clean_lineups"),
        (FIFA_PLAYERS_PARQUET, "python -m src.data.load_fifa"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {hint}")

    lineups = pd.read_parquet(LINEUPS_PARQUET)
    fifa = pd.read_parquet(FIFA_PLAYERS_PARQUET)

    player_map = build_map(lineups, fifa)
    report = coverage(player_map)

    print(report.to_string())
    print("\nby method:")
    print(player_map["method"].value_counts().to_string())

    short = report[report["starter_pct"] < TARGET_STARTER_COVERAGE]
    if not short.empty:
        message = (
            f"{len(short)} season(s) below {TARGET_STARTER_COVERAGE}% starter coverage:\n"
            f"{short.to_string()}\n"
            f"Add the unresolved names to {OVERRIDES_CSV}."
        )
        if strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    return player_map, report


def main() -> int:
    player_map, _ = build(strict=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    player_map.to_parquet(PLAYER_MAP_PARQUET, index=False)

    unmatched = player_map[player_map["fifa_player_name"].isna()]
    print(f"\n{len(player_map)} player-seasons, {len(unmatched)} unmatched")
    print(f"written to {PLAYER_MAP_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
