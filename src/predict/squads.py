"""Expected starting XIs, and the squad edits that keep them current.

You do not know a lineup until an hour before kickoff, so the default is the eleven a
club has actually used most across its recent matches. That is a decent guess in
mid-season and a poor one in August, which is what the manual files are for.

**Two committed files record squad changes, and both record *changes* rather than whole
squads.** A file restating twenty full squads would go stale within a week of a transfer
window opening, and a stale squad file that looks authoritative is worse than none:

``data/manual/squad_changes.csv``
    ``season, fifa_player_name, team, note`` - one row per move. A blank ``team`` means
    the player has left the league. This is what corrects the September-snapshot problem:
    ratings are published once a year, so a July signing is still listed at their old
    club. The rating is right; only the club is wrong.

``data/manual/player_ratings_manual.csv``
    ``season, fifa_player_name, overall, age, position, note`` - for a signing with no
    FIFA entry at all, from a league outside the dataset or straight out of an academy.
    An overall rating alone is enough to compute squad quality.
"""

from __future__ import annotations

import pandas as pd

from src.config import MANUAL_DIR
from src.features.squad import STARTERS_PER_TEAM, line_of
from src.predict.suspensions import load_unavailable, unavailable_for

SQUAD_CHANGES_CSV = MANUAL_DIR / "squad_changes.csv"
MANUAL_RATINGS_CSV = MANUAL_DIR / "player_ratings_manual.csv"

# How much history to read a club's preferred eleven from. Long enough to see through
# rotation and a cup week, short enough to follow a change of shape.
RECENT_MATCHES = 6


def load_squad_changes() -> pd.DataFrame:
    """Hand-recorded transfers. Empty frame when nothing has been entered."""
    columns = ["season", "fifa_player_name", "team", "note"]
    if not SQUAD_CHANGES_CSV.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(SQUAD_CHANGES_CSV)
    missing = {"season", "fifa_player_name", "team"} - set(df.columns)
    if missing:
        raise ValueError(f"{SQUAD_CHANGES_CSV} is missing column(s) {sorted(missing)}")

    return df


def load_manual_ratings() -> pd.DataFrame:
    """Ratings typed by hand for players FIFA has never heard of."""
    columns = ["season", "fifa_player_name", "overall", "age", "position", "note"]
    if not MANUAL_RATINGS_CSV.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(MANUAL_RATINGS_CSV)
    missing = {"season", "fifa_player_name", "overall"} - set(df.columns)
    if missing:
        raise ValueError(f"{MANUAL_RATINGS_CSV} is missing column(s) {sorted(missing)}")

    return df


def most_used_eleven(
    lineups: pd.DataFrame,
    team: str,
    before: pd.Timestamp,
    matches: int = RECENT_MATCHES,
    unavailable: set[str] | None = None,
) -> pd.DataFrame:
    """The eleven this club has started most often in its last few matches.

    Selection is by starts, then by minutes, and only positions the club actually used
    are represented - taking the top eleven by appearances alone would happily field
    four goalkeepers.

    ``unavailable`` names players who cannot play - suspended, or flagged by hand. They
    are removed *before* the eleven is picked rather than after, so the next-most-used
    player steps up and the side is still eleven strong. Dropping them afterwards would
    field ten and understate the squad, which is a bigger error than the absence itself.
    """
    history = lineups[(lineups["team"] == team) & (lineups["date"] < before)]
    if history.empty:
        return pd.DataFrame(columns=["player", "position", "line", "starts"])

    recent_ids = history.drop_duplicates("match_id").nlargest(matches, "date")["match_id"].tolist()
    recent = history[history["match_id"].isin(recent_ids) & history["is_starter"]]
    if unavailable:
        recent = recent[~recent["player"].isin(unavailable)]
    if recent.empty:
        return pd.DataFrame(columns=["player", "position", "line", "starts"])

    tally = (
        recent.groupby("player", as_index=False)
        .agg(
            starts=("match_id", "count"),
            minutes=("minutes", "sum"),
            position=("position", lambda s: s.mode().iloc[0]),
        )
        .sort_values(["starts", "minutes"], ascending=False)
    )
    tally["line"] = tally["position"].map(line_of)

    # One goalkeeper, then the ten outfielders with the most starts. Without the split a
    # rotating keeper pair can push a striker out of the eleven, or contribute two.
    keepers = tally[tally["line"] == "gk"].head(1)
    outfield = tally[tally["line"] != "gk"].head(STARTERS_PER_TEAM - len(keepers))

    return pd.concat([keepers, outfield], ignore_index=True)


def apply_squad_changes(
    squad_players: pd.DataFrame, changes: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Move players between clubs according to the hand-recorded transfers.

    Rows whose ``team`` is blank are departures and are dropped outright.
    """
    if changes.empty or "season" not in changes.columns:
        return squad_players

    relevant = changes[changes["season"] == season]
    if relevant.empty:
        return squad_players

    updated = squad_players.copy()
    for change in relevant.itertuples():
        name = change.fifa_player_name
        leaving = pd.isna(change.team) or str(change.team).strip() == ""

        updated = updated[updated["fifa_player_name"] != name]
        if not leaving:
            updated = pd.concat(
                [
                    updated,
                    pd.DataFrame(
                        [{"fifa_player_name": name, "team": change.team, "season": season}]
                    ),
                ],
                ignore_index=True,
            )

    return updated


def _with_manual_ratings(fifa: pd.DataFrame, lookup_season: str, season: str) -> pd.DataFrame:
    """Add hand-written ratings to the pool, labelled so the squad join finds them."""
    manual = load_manual_ratings()
    if manual.empty:
        return fifa

    relevant = manual[manual["season"].isin({season, lookup_season})]
    if relevant.empty:
        return fifa

    added = pd.DataFrame(
        {
            "season": lookup_season,
            "player_name": relevant["fifa_player_name"],
            "overall": pd.to_numeric(relevant["overall"], errors="coerce"),
            "age": pd.to_numeric(relevant.get("age"), errors="coerce"),
        }
    )
    return pd.concat([fifa, added], ignore_index=True)


def expected_squad_players(
    fixtures: pd.DataFrame,
    lineups: pd.DataFrame,
    player_map: pd.DataFrame,
    fifa: pd.DataFrame,
    lookup_season: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Every expected starter, with the ratings we could find for them.

    Builds each side's expected XI, applies the hand-recorded transfers, and runs the
    result through the same aggregation the training table used - so an upcoming fixture
    is described in exactly the same terms as the matches the model learned from.

    ``lookup_season`` is the season whose ratings and name map to use. A new season has
    neither of its own: its ratings edition is published weeks after it starts, and no
    player has been mapped to it yet, so the most recent completed season stands in.

    Returns (features, problems). Problems name any transfer whose player could not be
    found, since a typo there would otherwise do nothing at all.
    """
    season = fixtures["season"].iloc[0]

    # Hand-written ratings join the pool before anything is looked up, so a signing FIFA
    # has never heard of counts towards squad quality like any other player.
    fifa = _with_manual_ratings(fifa, lookup_season, season)

    changes = load_squad_changes()
    known = set(player_map["fifa_player_name"].dropna()) | set(fifa["player_name"])
    problems = [
        f"squad change for an unknown player: {name!r}"
        for name in unmatched_changes(changes, known, season)
    ]

    hand_flagged = load_unavailable()

    rows = []
    for fixture in fixtures.itertuples():
        for team in (fixture.home_team, fixture.away_team):
            # Bans are counted against the season being played, not the season whose
            # ratings are being borrowed - a card shown last May is served last May.
            out = unavailable_for(lineups, team, season, fixture.date, unavailable=hand_flagged)
            eleven = most_used_eleven(lineups, team, fixture.date, unavailable=out)
            for player in eleven.itertuples():
                rows.append(
                    {
                        "match_id": fixture.match_id,
                        "season": lookup_season,
                        "team": team,
                        "player": player.player,
                        "position": player.position,
                        "starts": int(player.starts),
                        "is_starter": True,
                    }
                )

    if not rows:
        return pd.DataFrame(), problems

    from src.features.squad import starting_ratings

    return starting_ratings(pd.DataFrame(rows), player_map, fifa), problems


def expected_squad_features(
    fixtures: pd.DataFrame,
    lineups: pd.DataFrame,
    player_map: pd.DataFrame,
    fifa: pd.DataFrame,
    lookup_season: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Squad quality per (match_id, team), aggregated from the expected XIs."""
    from src.features.squad import aggregate_ratings

    rated, problems = expected_squad_players(fixtures, lineups, player_map, fifa, lookup_season)
    if rated.empty:
        return pd.DataFrame(), problems

    return aggregate_ratings(rated), problems


# Order a team sheet reads in, rather than by rating - a lineup with the keeper in the
# middle looks wrong however accurate it is.
LINE_ORDER = {"gk": 0, "def": 1, "mid": 2, "att": 3, "unknown": 4}


def lineups_by_side(rated: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, dict[str, list]]:
    """The expected XIs as plain data, keyed by match_id then home/away."""
    if rated.empty:
        return {}

    sides = {
        fixture.match_id: {"home": fixture.home_team, "away": fixture.away_team}
        for fixture in fixtures.itertuples()
    }

    rated = rated.copy()
    rated["order"] = rated["line"].map(LINE_ORDER).fillna(len(LINE_ORDER))

    result: dict[str, dict[str, list]] = {}
    for (match_id, team), group in rated.groupby(["match_id", "team"], sort=False):
        teams = sides.get(match_id)
        if not teams:
            continue
        side = "home" if teams["home"] == team else "away"

        players = [
            {
                "player": row.player,
                "position": row.position,
                "line": row.line,
                "starts": int(row.starts) if pd.notna(row.starts) else 0,
                "overall": int(row.overall) if pd.notna(row.overall) else None,
            }
            for row in group.sort_values(["order", "starts"], ascending=[True, False]).itertuples()
        ]
        result.setdefault(match_id, {})[side] = players

    return result


def unmatched_changes(changes: pd.DataFrame, known_players: set[str], season: str) -> list[str]:
    """Names in the change file that match no known player.

    A typo here would silently do nothing, so the caller reports these loudly rather
    than letting a transfer quietly fail to apply.
    """
    if changes.empty or "season" not in changes.columns:
        return []

    relevant = changes[changes["season"] == season]
    return sorted(
        {
            name
            for name in relevant["fifa_player_name"]
            if isinstance(name, str) and name not in known_players
        }
    )
