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
    lineups: pd.DataFrame, team: str, before: pd.Timestamp, matches: int = RECENT_MATCHES
) -> pd.DataFrame:
    """The eleven this club has started most often in its last few matches.

    Selection is by starts, then by minutes, and only positions the club actually used
    are represented - taking the top eleven by appearances alone would happily field
    four goalkeepers.
    """
    history = lineups[(lineups["team"] == team) & (lineups["date"] < before)]
    if history.empty:
        return pd.DataFrame(columns=["player", "position", "line", "starts"])

    recent_ids = history.drop_duplicates("match_id").nlargest(matches, "date")["match_id"].tolist()
    recent = history[history["match_id"].isin(recent_ids) & history["is_starter"]]
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


def unmatched_changes(changes: pd.DataFrame, known_players: set[str], season: str) -> list[str]:
    """Names in the change file that match no known player.

    A typo here would silently do nothing, so the caller reports these loudly rather
    than letting a transfer quietly fail to apply.
    """
    relevant = changes[changes["season"] == season]
    return sorted(
        {
            name
            for name in relevant["fifa_player_name"]
            if isinstance(name, str) and name not in known_players
        }
    )
