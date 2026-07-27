"""Assemble the model-ready feature table: one row per match, nothing post-kickoff.

Reads every processed table, builds per-team features, then pivots the two team rows of
each fixture into a single row with ``home_``/``away_`` columns and their differences.

**Differences carry most of the signal.** A model cares far less that a squad averages 78
overall than that it is six points better than the opposition, so every paired feature
also appears as ``diff_``.

Usage:
    python -m src.features.build
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import FINAL_DIR
from src.data.clean_lineups import LINEUPS_PARQUET, UNDERSTAT_MATCHES_PARQUET
from src.data.clean_matches import MATCHES_PARQUET
from src.data.load_fifa import FIFA_PLAYERS_PARQUET
from src.features.form import build_team_matches
from src.features.squad import squad_features
from src.matching.player_names import PLAYER_MAP_PARQUET

FEATURES_PARQUET = FINAL_DIR / "features.parquet"

# Carried through from the match table. Everything else there - shots, cards, half-time
# scores - is a post-match fact and must not appear except as a rolling average of
# previous matches.
MATCH_COLUMNS = [
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "odds_close_home",
    "odds_close_draw",
    "odds_close_away",
    "odds_close_avg_home",
    "odds_close_avg_draw",
    "odds_close_avg_away",
]

# Post-match columns, named so the leakage check can look for them by name rather than
# trusting that we remembered to exclude them.
FORBIDDEN = (
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "home_fouls",
    "away_fouls",
    "ht_home_goals",
    "ht_away_goals",
    "ht_result",
    "xg_home",
    "xg_away",
    "total_goals",
    "goal_difference",
)

# Per-team features that become home_/away_/diff_ triples.
TEAM_FEATURES = (
    "elo_before",
    "rest_days",
    "matches_played",
    "season_matches_played",
    "goals_for_last5",
    "goals_against_last5",
    "xg_for_last5",
    "xg_against_last5",
    "shots_for_last5",
    "shots_on_target_for_last5",
    "corners_for_last5",
    "points_last5",
    "squad_overall_mean",
    "squad_overall_max",
    "squad_overall_std",
    "squad_age_mean",
    "squad_potential_mean",
    "squad_value_total",
    "squad_gk_overall",
    "squad_def_overall",
    "squad_mid_overall",
    "squad_att_overall",
    "squad_pace_mean",
    "squad_shooting_mean",
    "squad_passing_mean",
    "squad_dribbling_mean",
    "squad_defending_mean",
    "squad_physic_mean",
    "rated_share",
)

# Differences are meaningless for these: a count of matches played or the share of a
# squad we have ratings for is context, not strength.
NO_DIFFERENCE = {"matches_played", "season_matches_played", "rated_share", "rest_days"}


def load_tables() -> tuple[pd.DataFrame, ...]:
    paths = {
        MATCHES_PARQUET: "python -m src.data.clean_matches",
        UNDERSTAT_MATCHES_PARQUET: "python -m src.data.clean_lineups",
        LINEUPS_PARQUET: "python -m src.data.clean_lineups",
        FIFA_PLAYERS_PARQUET: "python -m src.data.load_fifa",
        PLAYER_MAP_PARQUET: "python -m src.matching.player_names",
    }
    for path, hint in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {hint}")

    return tuple(pd.read_parquet(path) for path in paths)


def build_features() -> pd.DataFrame:
    matches, understat, lineups, fifa, player_map = load_tables()

    team_matches = build_team_matches(matches, understat)
    squads = squad_features(lineups, player_map, fifa)
    team_matches = team_matches.merge(squads, on=["match_id", "team"], how="left")

    available = [column for column in TEAM_FEATURES if column in team_matches.columns]
    home = team_matches[team_matches["is_home"]].set_index("match_id")[available]
    away = team_matches[~team_matches["is_home"]].set_index("match_id")[available]

    features = matches[MATCH_COLUMNS].set_index("match_id")
    features = features.join(home.add_prefix("home_")).join(away.add_prefix("away_"))

    for column in available:
        if column not in NO_DIFFERENCE:
            features[f"diff_{column}"] = features[f"home_{column}"] - features[f"away_{column}"]

    features["is_promoted_home"] = features["home_matches_played"] == 0
    features["is_promoted_away"] = features["away_matches_played"] == 0

    return features.reset_index().sort_values("date").reset_index(drop=True)


def validate(features: pd.DataFrame, team_matches: pd.DataFrame | None = None) -> list[str]:
    """Return a list of problems. Empty list means clean."""
    problems: list[str] = []

    if len(features) != 2660:
        problems.append(f"{len(features)} rows, expected 2660")

    leaked = sorted(set(FORBIDDEN) & set(features.columns))
    if leaked:
        problems.append(
            f"post-match columns present in the feature table: {leaked}. "
            f"These are not knowable before kickoff."
        )

    if features["match_id"].duplicated().any():
        problems.append("duplicate match_id")

    # Every match must have Elo for both sides; a null means a team slipped through the
    # chronological pass.
    for column in ("home_elo_before", "away_elo_before"):
        nulls = features[column].isna().sum()
        if nulls:
            problems.append(f"{nulls} null values in {column}")

    # Squad ratings are allowed to be sparse, but not mostly missing.
    rated = features["home_squad_overall_mean"].notna().mean()
    if rated < 0.98:
        problems.append(f"only {rated:.1%} of matches have home squad ratings")

    return problems


def build(strict: bool = True) -> pd.DataFrame:
    features = build_features()
    problems = validate(features)

    seasons = features.groupby("season").size()
    print(f"{len(features)} matches, {len(features.columns)} columns")
    print(f"seasons: {seasons.to_dict()}")

    coverage = features[[c for c in features.columns if c.startswith("home_squad")]].notna().mean()
    print(f"\nsquad feature coverage:\n{(coverage * 100).round(1).to_string()}")

    if problems:
        message = "Validation failed:\n" + "\n".join(f"  {p}" for p in problems)
        if strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    return features


def main() -> int:
    features = build()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PARQUET, index=False)

    print(f"\nwritten to {FEATURES_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
