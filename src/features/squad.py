"""Squad quality for each starting XI, from FIFA ratings.

One row per (match_id, team). The route is deliberately indirect - lineups name players
as Understat spells them, ratings as FIFA does - so everything goes through
``player_map.parquet`` rather than joining names directly.

Nothing here is post-match: a starting XI and its ratings are known once the team sheet
is announced, and Phase 7 predicts with an expected XI.
"""

from __future__ import annotations

import pandas as pd

STARTERS_PER_TEAM = 11

# Understat position codes -> the four lines. Order matters twice over: "DMC" must be
# read as a midfielder before the plain "D" prefix claims it as a defender, and the wide
# attacking codes must be read before the general "AM" one.
#
# **AML and AMR are wingers, not midfielders.** Understat names them for the slot they
# occupy in a formation grid, but in the modern game a left or right attacking midfielder
# is a forward - Mbeumo and Cunha are AMR and AML. Counting them as midfielders produced
# sides with four defenders, six midfielders and *no attacker at all*, which is not a
# formation anybody has played. AMC stays a midfielder: that is a number ten.
#
# Measured across all 5,320 starting XIs on record, this makes 4-3-3 the most common shape
# by a wide margin, which is what the modern Premier League actually looks like, and takes
# the count of XIs with zero attackers from 1 to 0.
LINE_PREFIXES = (
    ("GK", "gk"),
    ("DM", "mid"),
    ("AMC", "mid"),
    ("AML", "att"),
    ("AMR", "att"),
    ("AM", "mid"),
    ("M", "mid"),
    ("D", "def"),
    ("FW", "att"),
)

# Present in every edition, so these are safe to build features on.
CORE_RATINGS = ("overall", "age")

# Absent for at least one season each; see CLAUDE.md. Kept because they are informative
# where present, but a model must tolerate whole seasons of nulls.
OPTIONAL_RATINGS = (
    "potential",
    "value_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
)


def line_of(position: str) -> str:
    """Map an Understat position code to gk / def / mid / att."""
    if not isinstance(position, str):
        return "unknown"
    for prefix, line in LINE_PREFIXES:
        if position.startswith(prefix):
            return line
    return "unknown"


def starting_ratings(
    lineups: pd.DataFrame, player_map: pd.DataFrame, fifa: pd.DataFrame
) -> pd.DataFrame:
    """Every starter with the FIFA ratings we could find for them.

    Rows survive even when the player could not be matched, so that the share of a
    starting XI we actually have ratings for becomes a feature rather than a silent gap.
    """
    starters = lineups[lineups["is_starter"]].copy()
    # Historical rows carry the season inside match_id. Expected XIs for an upcoming
    # fixture set it explicitly instead, because the ratings and name map they need to
    # look up belong to the most recent season on record, not to the one being played.
    if "season" not in starters.columns:
        starters["season"] = starters["match_id"].str.slice(0, 7).str.replace("_", "/", regex=False)
    # ``line_of`` reads Understat's codes. An XI built from ratings uses FIFA's instead
    # (CB, LB, ST), which fall through to "unknown" - so a supplied line is kept rather
    # than recomputed. Without this every promoted club's whole outfield reads as unknown
    # and the pitch view stacks all ten in midfield.
    if "line" in starters.columns:
        starters["line"] = starters["line"].fillna(starters["position"].map(line_of))
    else:
        starters["line"] = starters["position"].map(line_of)

    # An XI built from ratings rather than appearances already names players as FIFA
    # does, so it arrives with the mapping filled in. Keep it aside and restore it after
    # the join, rather than letting the merge overwrite it with a null.
    supplied = starters.pop("fifa_player_name") if "fifa_player_name" in starters else None

    linked = starters.merge(
        player_map[["season", "team", "understat_player", "fifa_player_name"]],
        left_on=["season", "team", "player"],
        right_on=["season", "team", "understat_player"],
        how="left",
    )
    if supplied is not None:
        linked["fifa_player_name"] = linked["fifa_player_name"].fillna(
            supplied.reset_index(drop=True)
        )

    ratings = fifa[["season", "player_name", *CORE_RATINGS, *OPTIONAL_RATINGS]]
    # A FIFA name can repeat across clubs within a season, so collapse to one row per
    # (season, name) before joining - otherwise a single starter would fan out into
    # several rows and quietly reweight the squad average.
    ratings = ratings.drop_duplicates(["season", "player_name"])

    return linked.merge(
        ratings,
        left_on=["season", "fifa_player_name"],
        right_on=["season", "player_name"],
        how="left",
    )


def squad_features(
    lineups: pd.DataFrame, player_map: pd.DataFrame, fifa: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate each starting XI into one row of squad-quality features."""
    return aggregate_ratings(starting_ratings(lineups, player_map, fifa))


def aggregate_ratings(rated: pd.DataFrame) -> pd.DataFrame:
    """Collapse rated players into one row per (match_id, team).

    Split out from ``squad_features`` so a caller that already has the rated rows - the
    report, which lists the individual players - can aggregate them without joining
    everything a second time.
    """
    grouped = rated.groupby(["match_id", "team"], sort=False)

    features = grouped.agg(
        squad_overall_mean=("overall", "mean"),
        squad_overall_max=("overall", "max"),
        squad_overall_std=("overall", "std"),
        squad_age_mean=("age", "mean"),
        squad_potential_mean=("potential", "mean"),
        squad_value_total=("value_eur", "sum"),
        starters_rated=("overall", "count"),
    )
    features["rated_share"] = features["starters_rated"] / STARTERS_PER_TEAM

    # Mean overall per line. A team missing a line entirely (a back three recorded
    # without wing-backs, say) yields NaN rather than a misleading zero.
    by_line = (
        rated.pivot_table(
            index=["match_id", "team"], columns="line", values="overall", aggfunc="mean"
        )
        .rename(columns=lambda line: f"squad_{line}_overall")
        .drop(columns=["squad_unknown_overall"], errors="ignore")
    )
    features = features.join(by_line)

    # Face stats are null for goalkeepers by design, so the mean is over outfielders.
    # That is the intended quantity, not an accident of skipping nulls.
    face = grouped[list(OPTIONAL_RATINGS[2:])].mean()
    face.columns = [f"squad_{column}_mean" for column in face.columns]
    features = features.join(face)

    return features.reset_index()
