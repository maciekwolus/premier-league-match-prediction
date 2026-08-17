"""Expected starting XIs, and the squad edits that keep them current.

You do not know a lineup until an hour before kickoff, so the default is the eleven a
club has actually used most across its recent matches. That is a decent guess in
mid-season and a poor one in August, which is what the manual files are for.

**Who has left is detected rather than typed.** ``predict.transfers`` diffs each club's
recent starters against its current FPL squad, so a departed player drops out without
anyone recording the transfer. A ``squad_changes.csv`` once existed for that job and never
worked - the function that would have applied it was never called - so it is gone rather
than left looking functional.

One committed file still records squad changes, and it records *changes* rather than whole
squads, because a file restating twenty squads goes stale within a week of a window opening
and a stale squad file that looks authoritative is worse than none:

``data/manual/player_ratings_manual.csv``
    ``season, fifa_player_name, overall, age, position, note`` - for a signing with no
    FIFA entry at all, from a league outside the dataset or straight out of an academy.
    An overall rating alone is enough to compute squad quality.
"""

from __future__ import annotations

import pandas as pd

from src.config import MANUAL_DIR
from src.features.squad import STARTERS_PER_TEAM, line_of
from src.matching.player_names import normalise
from src.predict.suspensions import load_unavailable, unavailable_for
from src.predict.transfers import departures

MANUAL_RATINGS_CSV = MANUAL_DIR / "player_ratings_manual.csv"

# How much history to read a club's preferred eleven from. Long enough to see through
# rotation, short enough to follow a change of shape.
#
# **Six was too few, and it failed hardest exactly where it matters.** Predicting the
# opening round of a season means the last six matches are all end-of-season fixtures,
# where sides rotate and dead rubbers are common - so the "most-used eleven" was the
# rotation side, not the first choice. Measured on 2025/26: Tottenham fielded reserve
# keeper Kinsky over Vicario and Liverpool fielded Mamardashvili over Alisson, and every
# club's mean squad rating came out below its real one (Liverpool 83.5 against 86.0,
# Tottenham 79.7 against 80.5).
#
# Half a season fixes it and 19 against 38 makes almost no difference, so the signal is
# real rather than a longer window flattering itself: enough starts to separate a
# first-choice player from a rotation one, recent enough to follow a change mid-season.
RECENT_MATCHES = 19


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


def recent_starters(
    lineups: pd.DataFrame, team: str, before: pd.Timestamp, matches: int = RECENT_MATCHES
) -> pd.DataFrame:
    """Rows for everyone who started for this club in its last few matches.

    The pool the expected XI is drawn from, and the only sensible scope for asking who
    has left: a club's full appearance history reaches back seven seasons, so checking
    that against a current squad reports Cristiano Ronaldo as a Man United departure.
    """
    history = lineups[(lineups["team"] == team) & (lineups["date"] < before)]
    if history.empty:
        return history

    recent_ids = history.drop_duplicates("match_id").nlargest(matches, "date")["match_id"].tolist()
    return history[history["match_id"].isin(recent_ids) & history["is_starter"]]


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
    recent = recent_starters(lineups, team, before, matches)
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
    return _pick_a_shape(tally)


# The bounds every Premier League formation lives inside. A side always has one keeper,
# never fewer than three defenders or more than five, and always at least one forward.
FORMATION_MINIMUM = {"gk": 1, "def": 3, "mid": 2, "att": 1}
FORMATION_MAXIMUM = {"gk": 1, "def": 5, "mid": 6, "att": 4}


def _pick_a_shape(tally: pd.DataFrame) -> pd.DataFrame:
    """The most-used eleven that is also a team.

    Ranking purely by appearances does not produce a formation. Arsenal's most-used ten
    outfielders were five defenders and five midfielders, with six available forwards all
    sitting a start or two behind - so the XI came out 5-5-0, which nobody has played.
    The goalkeeper was already protected this way, for exactly the same reason; every
    other line needed it too.

    Each line takes its minimum first, by starts and then minutes, and the remaining
    places go to whoever has played most within the maxima. A club whose pool genuinely
    lacks a line - no forward left after departures, say - gets what exists rather than an
    invented player, and the shape says so.
    """
    picked = []
    for line, minimum in FORMATION_MINIMUM.items():
        picked.append(tally[tally["line"] == line].head(minimum))

    chosen = pd.concat(picked) if picked else tally.head(0)

    # Outfield places are capped whether or not a keeper was found. Leeds' only goalkeeper
    # in the pool had left the club, and without this the free places went to outfielders
    # instead - eleven of them, on a pitch with nobody in goal.
    outfield_cap = STARTERS_PER_TEAM - FORMATION_MINIMUM["gk"]
    remaining = min(STARTERS_PER_TEAM, outfield_cap + len(chosen[chosen["line"] == "gk"])) - len(
        chosen
    )
    if remaining > 0:
        counts = chosen["line"].value_counts().to_dict()
        for row in tally.drop(chosen.index).itertuples():
            if remaining == 0:
                break
            line = row.line
            if counts.get(line, 0) >= FORMATION_MAXIMUM.get(line, STARTERS_PER_TEAM):
                continue
            chosen = pd.concat([chosen, tally.loc[[row.Index]]])
            counts[line] = counts.get(line, 0) + 1
            remaining -= 1

    return chosen.reset_index(drop=True)


# FIFA's primary position -> the four lines. A promoted club has no appearance history to
# read a shape from, so the fallback XI is built by position instead.
FIFA_POSITION_LINES = {
    "GK": "gk",
    "CB": "def",
    "LB": "def",
    "RB": "def",
    "CDM": "mid",
    "CM": "mid",
    "CAM": "mid",
    "LM": "mid",
    "RM": "mid",
    "LW": "att",
    "RW": "att",
    "ST": "att",
}

# A plain 4-4-2. Any shape is a guess for a club we have never seen play in this division;
# this one is the least surprising, and squad quality is a mean so the exact split matters
# far less than which eleven players are in it.
FALLBACK_SHAPE = {"gk": 1, "def": 4, "mid": 4, "att": 2}


def ratings_eleven(fifa: pd.DataFrame, club: str, season: str) -> pd.DataFrame:
    """The best-rated eleven a club has, taken from ratings rather than appearances.

    For a promoted club there is no Premier League history to read a most-used eleven
    from, and the alternative to this is what the pipeline did before: no squad quality
    at all, silently filled with the training median, so a newly promoted side was
    described to the model as an average Premier League squad.

    This is a *different kind of guess* from the appearance-based XI and is labelled as
    such by the caller. It says who a club's best players are, not who will start.
    """
    columns = ["player", "position", "line", "starts"]
    if fifa.empty or "club_fd" not in fifa.columns:
        return pd.DataFrame(columns=columns)

    squad = fifa[(fifa["club_fd"] == club) & (fifa["season"] == season)].copy()
    if squad.empty:
        return pd.DataFrame(columns=columns)

    squad["position"] = squad["positions"].astype(str).str.split(",").str[0].str.strip()
    squad["line"] = squad["position"].map(FIFA_POSITION_LINES)
    squad = squad.dropna(subset=["line", "overall"]).sort_values("overall", ascending=False)

    picked = [squad[squad["line"] == line].head(count) for line, count in FALLBACK_SHAPE.items()]
    eleven = pd.concat(picked, ignore_index=True) if picked else squad.head(0)

    return pd.DataFrame(
        {
            "player": eleven["player_name"],
            "position": eleven["position"],
            "line": eleven["line"],
            # Zero starts is honest: this club has started nobody in this division.
            "starts": 0,
        }
    ).reset_index(drop=True)


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
    squads: dict[str, list[dict]] | None = None,
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

    problems: list[str] = []
    hand_flagged = load_unavailable()

    # Scoped to the lookup season *and* the club, because that is exactly how the ratings
    # join is keyed. A club relegated and promoted back has players all over the map from
    # its earlier spell, and a global name check would see them and wrongly conclude the
    # appearance-based XI can be rated.
    for_lookup = player_map[player_map["season"] == lookup_season]
    rated_pairs = set(zip(for_lookup["team"], for_lookup["understat_player"], strict=False))

    rows = []
    for fixture in fixtures.itertuples():
        for team in (fixture.home_team, fixture.away_team):
            # Bans are counted against the season being played, not the season whose
            # ratings are being borrowed - a card shown last May is served last May.
            out = unavailable_for(lineups, team, season, fixture.date, unavailable=hand_flagged)

            # Players who have left the club since we last saw them play. Appearances are
            # last season's, so without this a departed regular keeps his place in the XI
            # indefinitely - Casemiro started for Man United through 2025/26 and was still
            # being picked for the opening round of 2026/27, a year after leaving.
            if squads is not None:
                pool = recent_starters(lineups, team, fixture.date)
                gone = departures(sorted(set(pool["player"])), team, squads)
                if gone:
                    problems.append(f"{team}: left the club - {', '.join(gone)}")
                out = out | set(gone)

            eleven = most_used_eleven(lineups, team, fixture.date, unavailable=out)

            # A promoted club fails this two ways, and both end in the same place: no
            # squad quality, quietly replaced downstream by the training median, so a
            # newly promoted side is described to the model as an average Premier League
            # squad. Either it has no history in this division at all, or it has history
            # from an earlier spell whose players are absent from the current name map.
            # Falling back to ratings is a worse guess than appearances and a much better
            # one than nothing.
            from_ratings = eleven.empty or not any(
                (team, player) in rated_pairs for player in eleven["player"]
            )
            if from_ratings:
                eleven = ratings_eleven(fifa, team, lookup_season)
            if not eleven.empty:
                eleven = eleven.assign(source="ratings" if from_ratings else "appearances")

            if not from_ratings and not eleven.empty and "gk" not in set(eleven["line"]):
                # A club can lose every goalkeeper it has recently started - Leeds lost
                # theirs to a transfer - and no side takes the field without one. The
                # ratings know who the club's keepers are even when appearances do not,
                # so the best-rated one fills the shirt rather than leaving it empty.
                keeper = ratings_eleven(fifa, team, lookup_season)
                keeper = keeper[keeper["line"] == "gk"].head(1)
                if not keeper.empty:
                    problems.append(f"{team}: no goalkeeper has started recently; using ratings")
                    keeper = keeper.assign(source="ratings")
                    eleven = pd.concat([keeper, eleven.head(STARTERS_PER_TEAM - 1)])

            for player in eleven.itertuples():
                rows.append(
                    {
                        "match_id": fixture.match_id,
                        "season": lookup_season,
                        "team": team,
                        "player": player.player,
                        "position": player.position,
                        # Carried rather than recomputed downstream: a ratings XI uses
                        # FIFA's position codes, which the Understat mapping cannot read.
                        "line": player.line,
                        "starts": int(player.starts),
                        "is_starter": True,
                        # Recorded so the report can say which kind of guess this is,
                        # rather than calling a ratings XI a most-used eleven.
                        "xi_source": player.source,
                        # A name that came from the ratings is already spelled as FIFA
                        # spells it, so it carries its own mapping and skips the Understat
                        # name map. That is true of a backfilled keeper as much as a whole
                        # ratings eleven.
                        "fifa_player_name": (player.player if player.source == "ratings" else None),
                    }
                )

    if not rows:
        return pd.DataFrame(), problems

    from src.features.squad import starting_ratings

    rated = starting_ratings(pd.DataFrame(rows), player_map, fifa)
    if squads is not None:
        rated, signed = promote_signings(rated, fifa, squads, lookup_season)
        problems.extend(signed)
    return rated, problems


# How much better than the man he displaces a signing has to be. Not zero: ratings carry
# a point or two of noise, and swapping a whole XI around on that would churn the side
# every time the ratings edition changed.
SIGNING_MARGIN = 2


def promote_signings(
    rated: pd.DataFrame,
    fifa: pd.DataFrame,
    squads: dict[str, list[dict]],
    lookup_season: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Let a signing take a place from a weaker player in the same line.

    Appearances cannot see a transfer in: a player who has just joined has started
    nothing for this club, so the appearance-based eleven fields whoever he replaced.
    Youri Tielemans moved to Man United rated 85 and the expected eleven kept picking
    around him.

    **A rating is a weaker claim than a start, so it only wins by a clear margin** and
    only within the same line - a signing does not displace a goalkeeper by being a
    better midfielder. The result is still a description rather than a team sheet: an
    average signing sits on the bench where he belongs, and only a clearly better one
    takes a shirt.
    """
    if rated.empty or "overall" not in rated.columns:
        return rated, []

    pool = fifa[fifa["season"] == lookup_season]
    notes: list[str] = []
    replaced: list[int] = []
    additions: list[dict] = []

    # Matching a whole league's ratings against a squad list is the expensive step here,
    # and it depends only on the club - so it is done once per club rather than once per
    # fixture, which is the difference between seconds and minutes.
    by_club = {team: _signing_candidates(pool, squad) for team, squad in squads.items()}

    # A name two clubs both claim belongs to neither. FIFA lists one "Joao Pedro", at
    # Chelsea; Brighton's squad contains "Joao Pedro Loureiro da Costa", a different
    # player entirely, and containment handed Chelsea's man to Brighton as well. Any name
    # that resolves to more than one club is ambiguous by definition, and the safe answer
    # for a signing is to leave the appearance-based eleven alone.
    claims: dict[str, set[str]] = {}
    for team, frame in by_club.items():
        for name in frame["player_name"]:
            claims.setdefault(name, set()).add(team)
    contested = {name for name, teams in claims.items() if len(teams) > 1}
    if contested:
        by_club = {
            team: frame[~frame["player_name"].isin(contested)] for team, frame in by_club.items()
        }

    for (_match_id, team), side in rated.groupby(["match_id", "team"], sort=False):
        if team not in by_club or by_club[team].empty:
            continue
        already = set(side["player"]) | set(side["fifa_player_name"].dropna())
        candidates = by_club[team]
        candidates = candidates[~candidates["player_name"].isin(already)]

        for line, group in side.groupby("line"):
            weakest = group.dropna(subset=["overall"]).nsmallest(1, "overall")
            if weakest.empty:
                continue
            incumbent = weakest.iloc[0]
            better = candidates[
                (candidates["line"] == line)
                & (candidates["overall"] > incumbent["overall"] + SIGNING_MARGIN)
            ]
            if better.empty:
                continue

            signing = better.nlargest(1, "overall").iloc[0]
            candidates = candidates.drop(signing.name)
            replaced.append(weakest.index[0])
            row = dict(incumbent)
            row.update(
                {
                    "player": signing["player_name"],
                    "position": signing["position"],
                    "overall": signing["overall"],
                    "fifa_player_name": signing["player_name"],
                    "xi_source": "signing",
                }
            )
            additions.append(row)
            notes.append(
                f"{team}: {signing['player_name']} ({int(signing['overall'])}) comes in for "
                f"{incumbent['player']} ({int(incumbent['overall'])})"
            )

    if not additions:
        return rated, notes

    kept = rated.drop(index=replaced)
    return pd.concat([kept, pd.DataFrame(additions)], ignore_index=True), sorted(set(notes))


def _signing_candidates(pool: pd.DataFrame, squad: list[dict]) -> pd.DataFrame:
    """Every rated player in a club's current squad, however the ratings file them.

    Looked up across the whole league rather than by club: ratings are a September
    snapshot, so a summer signing is still filed under the club he left. Youri Tielemans
    is in FC 26 at Aston Villa and in the FPL squad list at Man United.
    """
    if pool.empty:
        return pool.head(0).assign(line=None, position=None)

    # **Not ``is_in_squad``, and that is the whole difficulty here.** That function is
    # deliberately generous because it answers the *departure* question, where a false
    # match harmlessly keeps a player and a false miss silently deletes a real one. Run
    # backwards to recruit, the same generosity is a disaster: a lone surname token
    # matches any player on earth who shares it. The first version of this put Lautaro
    # Martinez into Aston Villa off Emiliano Martinez, Scott McTominay into Bournemouth,
    # and Davinson Sanchez into two clubs at once.
    #
    # Recruiting demands full-name agreement instead: every token of the shorter name
    # present in the longer, and never on a single token.
    squad_names = [
        frozenset(normalise(player["full_name"]).split())
        for player in squad
        if normalise(player["full_name"])
    ]

    def signed_here(fifa_name) -> bool:
        tokens = frozenset(normalise(str(fifa_name)).split())
        if len(tokens) < 2:
            return False
        return any(
            len(squad_name) >= 2 and (tokens <= squad_name or squad_name <= tokens)
            for squad_name in squad_names
        )

    candidates = pool[pool["player_name"].map(signed_here)].copy()
    if candidates.empty:
        return candidates.assign(line=None, position=None)

    candidates["position"] = candidates["positions"].astype(str).str.split(",").str[0].str.strip()
    candidates["line"] = candidates["position"].map(FIFA_POSITION_LINES)
    return candidates.dropna(subset=["line", "overall"])


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
                # Carried through so the report can say which kind of guess this is.
                # Calling a ratings XI a most-used eleven would be a quiet lie about a
                # promoted club, which is precisely the side a reader knows least about.
                "source": getattr(row, "xi_source", "appearances"),
            }
            for row in group.sort_values(["order", "starts"], ascending=[True, False]).itertuples()
        ]
        result.setdefault(match_id, {})[side] = players

    return result
