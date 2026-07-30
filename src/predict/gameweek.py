"""Predict a round of fixtures and write the report's input.

The upcoming fixtures are appended to the historical match table with their results left
blank, and the existing feature machinery runs over the whole thing. That is the point of
the team-match table shape: a fixture that has not happened has no result to leak, and
the rolling windows already only look backwards, so nothing needs a separate code path.

Usage:
    python -m src.predict.gameweek                       # next round
    python -m src.predict.gameweek --replay              # last known round, as a check
    python -m src.predict.gameweek --model poisson-glm
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import UPCOMING_SEASON
from src.data.clean_lineups import LINEUPS_PARQUET, UNDERSTAT_MATCHES_PARQUET
from src.data.clean_matches import MATCHES_PARQUET
from src.data.gameweeks import assign_gameweeks
from src.data.load_fifa import FIFA_PLAYERS_PARQUET
from src.evaluate.metrics import implied_probabilities
from src.features.build import FEATURES_PARQUET, NO_DIFFERENCE, TEAM_FEATURES
from src.features.form import build_team_matches
from src.features.squad import aggregate_ratings
from src.matching.player_names import PLAYER_MAP_PARQUET
from src.models.score_matrix import outcome_probabilities, score_matrix, top_scorelines
from src.predict.archive import (
    ROUNDS_DIR,
    RoundAlreadyStored,
    group_by_gameweek,
    save_round,
)
from src.predict.fixtures import as_matches, upcoming_fixtures
from src.predict.squads import expected_squad_players, lineups_by_side

MODELS = {
    "poisson-glm": ("src.models.poisson_glm", "PoissonRegressionModel"),
    "dixon-coles-squad": ("src.models.dixon_coles_squad", "SquadDixonColesModel"),
    "dixon-coles": ("src.models.dixon_coles", "DixonColesModel"),
    "baseline-elo": ("src.models.baselines", "EloModel"),
}
# Dixon-Coles rather than the model with the best RPS, deliberately.
#
# poisson-glm scores better on outcome probabilities (0.2040 against 0.2118) because it
# hedges towards the average, and RPS rewards hedging. That same caution makes it call
# 1-1 in 74% of matches and produce only nine distinct top scorelines across a season -
# for Crystal Palace against Arsenal it predicts 1.27 vs 1.63 goals where the market has
# the away side at 51%, and reports 1-1.
#
# Dixon-Coles estimates each club's attack and defence directly, so it commits: 1-1 tops
# 60% of matches, eleven distinct scorelines appear, and that same fixture comes out at
# 0.92 vs 1.68 with 0-1 most likely, which is what the bookmakers show.
#
# Since the product is a scoreline with a probability, a model that says 1-1 for three
# matches in four is failing at the job even when its RPS is better. Use --model to
# switch: poisson-glm remains the more accurate outcome predictor.
DEFAULT_MODEL = "dixon-coles-squad"


def load_model(name: str):
    module_name, class_name = MODELS[name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)()


def lineups_with_dates() -> pd.DataFrame:
    """Lineups joined to their match dates, which the XI picker needs."""
    lineups = pd.read_parquet(LINEUPS_PARQUET)
    matches = pd.read_parquet(MATCHES_PARQUET)[["match_id", "date"]]
    return lineups.merge(matches, on="match_id", how="left")


def features_for(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Build the feature row for each upcoming fixture.

    History and fixtures go through the same builder, so form, Elo and rest are computed
    exactly as they were for every match the models trained on, and squad quality comes
    from each side's expected XI.

    Returns (features, problems, expected XIs keyed by match_id).
    """
    matches = pd.read_parquet(MATCHES_PARQUET)
    understat = pd.read_parquet(UNDERSTAT_MATCHES_PARQUET)

    # Drop any history for the fixtures being predicted before appending them. Without
    # this the match_id appears twice and the join below fans out, producing several
    # differing predictions for one fixture. It matters for --replay, where the round is
    # by definition already in the table, and it makes a re-run harmless in general.
    history = matches[~matches["match_id"].isin(fixtures["match_id"])]

    # Line the fixtures up with the match table's columns before concatenating, so the
    # result keeps its dtypes rather than being widened by all-NA columns.
    aligned = fixtures.reindex(columns=matches.columns)
    combined = pd.concat([history, aligned], ignore_index=True).sort_values("date")
    team_matches = build_team_matches(combined, understat)

    # Squad quality for the fixtures themselves. Without this the model is handed the
    # league median for every rating column and sees two indistinguishable, average
    # teams - which is not a missing nicety but half the feature table.
    lineups = lineups_with_dates()
    player_map = pd.read_parquet(PLAYER_MAP_PARQUET)
    fifa = pd.read_parquet(FIFA_PLAYERS_PARQUET)
    lookup_season = str(player_map["season"].max())

    rated, problems = expected_squad_players(fixtures, lineups, player_map, fifa, lookup_season)
    if not rated.empty:
        team_matches = team_matches.merge(
            aggregate_ratings(rated), on=["match_id", "team"], how="left"
        )

    available = [column for column in TEAM_FEATURES if column in team_matches.columns]
    home = team_matches[team_matches["is_home"]].set_index("match_id")[available]
    away = team_matches[~team_matches["is_home"]].set_index("match_id")[available]

    rows = fixtures.set_index("match_id")[["season", "date", "home_team", "away_team"]]
    rows = rows.join(home.add_prefix("home_")).join(away.add_prefix("away_"))

    for column in available:
        if column not in NO_DIFFERENCE:
            rows[f"diff_{column}"] = rows[f"home_{column}"] - rows[f"away_{column}"]

    rows["is_promoted_home"] = rows["home_matches_played"] == 0
    rows["is_promoted_away"] = rows["away_matches_played"] == 0

    return rows.reset_index(), problems, lineups_by_side(rated, fixtures)


def gameweeks_for(fixtures: pd.DataFrame) -> dict[str, int]:
    """The gameweek each fixture belongs to, keyed by ``match_id``.

    A fixture's round depends on how many matches its clubs have already played, so this
    has to be computed against the season's full schedule rather than the handful being
    predicted. Existing rows for these fixtures are dropped first for the same reason
    ``features_for`` does it: with ``--replay`` the round is already in the table, and
    counting it twice would put every later fixture a round out.
    """
    # Only the schedule matters here, so both sides are narrowed to those columns before
    # concatenating. Reindexing the fixtures to the full match table would drag in all-NA
    # result columns and the dtype-coercion warning that comes with them.
    columns = ["match_id", "season", "date", "home_team", "away_team"]
    matches = pd.read_parquet(MATCHES_PARQUET, columns=columns)
    history = matches[~matches["match_id"].isin(fixtures["match_id"])]
    combined = pd.concat([history, fixtures[columns]], ignore_index=True)

    combined["gameweek"] = assign_gameweeks(combined)
    predicted = combined[combined["match_id"].isin(fixtures["match_id"])]
    return dict(zip(predicted["match_id"], predicted["gameweek"], strict=True))


def predict(
    fixtures: pd.DataFrame, model_name: str = DEFAULT_MODEL, mode: str = "upcoming"
) -> list[dict]:
    """Train on everything known, then predict the given fixtures."""
    history = pd.read_parquet(FEATURES_PARQUET)

    model = load_model(model_name)
    model.fit(history)

    upcoming, problems, elevens = features_for(fixtures)
    for problem in problems:
        print(f"  warning: {problem}", file=sys.stderr)

    # Anything still absent is filled with the training median so the shapes match. This
    # should now be rare - squad quality is built above - and a column landing here means
    # the model sees an average team, so it is worth knowing about rather than silent.
    filled = [column for column in history.columns if column not in upcoming.columns]
    for column in filled:
        upcoming[column] = (
            history[column].median() if pd.api.types.is_numeric_dtype(history[column]) else np.nan
        )
    if filled:
        print(f"  {len(filled)} feature(s) unavailable, filled with the training median")

    lambda_home, lambda_away = model.predict(upcoming)

    # The bookmaker's view of the same fixtures, where the feed supplied one. Showing it
    # beside the model is the whole point of the report: it is the only reference that
    # says whether a prediction is worth anything.
    market = market_probabilities(fixtures)

    # Which round each fixture belongs to, and when this was decided. Both go into every
    # record because the archive is keyed on the first and only trustworthy with the
    # second: a prediction without a timestamp cannot be shown to predate its result.
    gameweeks = gameweeks_for(fixtures)
    predicted_at = datetime.now(UTC).isoformat(timespec="seconds")

    report = []
    for index, fixture in enumerate(upcoming.itertuples()):
        matrix = score_matrix(lambda_home[index], lambda_away[index])
        home_probability, draw_probability, away_probability = outcome_probabilities(matrix)

        report.append(
            {
                "bookmaker": market.get(fixture.match_id),
                "lineups": elevens.get(fixture.match_id, {}),
                "match_id": fixture.match_id,
                "date": str(pd.Timestamp(fixture.date).date()),
                "season_slug": str(fixture.season).replace("/", "_"),
                "gameweek": int(gameweeks[fixture.match_id]),
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "model": model_name,
                "predicted_at": predicted_at,
                # Whether these are fixtures still to be played or a round being
                # re-predicted. Without it the report cannot tell the reader which it is
                # showing, and a replayed round reads as next week's matches.
                "mode": mode,
                "expected_goals": {
                    "home": round(float(lambda_home[index]), 2),
                    "away": round(float(lambda_away[index]), 2),
                },
                "outcome": {
                    "home": round(home_probability, 3),
                    "draw": round(draw_probability, 3),
                    "away": round(away_probability, 3),
                },
                "scorelines": [
                    {"score": f"{home}-{away}", "probability": round(probability, 3)}
                    for home, away, probability in top_scorelines(matrix, 3)
                ],
            }
        )

    return report


def market_probabilities(fixtures: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Bookmaker odds turned into probabilities, keyed by match_id.

    Returns an empty mapping when the fixtures carry no odds, which is normal well
    before kickoff - the market has not formed yet.
    """
    columns = ("odds_close_avg_home", "odds_close_avg_draw", "odds_close_avg_away")
    if not all(column in fixtures.columns for column in columns):
        return {}

    priced = fixtures.dropna(subset=list(columns))
    if priced.empty:
        return {}

    probabilities = implied_probabilities(
        priced["odds_close_avg_home"],
        priced["odds_close_avg_draw"],
        priced["odds_close_avg_away"],
    )
    return {
        match_id: {
            "home": round(float(row[0]), 3),
            "draw": round(float(row[1]), 3),
            "away": round(float(row[2]), 3),
        }
        for match_id, row in zip(priced["match_id"], probabilities, strict=True)
    }


def replay_fixtures() -> pd.DataFrame:
    """The most recent round of *known* matches, reshaped as if it were upcoming.

    A dry run that can be checked against reality, which matters because the real path
    cannot be exercised outside a season.
    """
    matches = pd.read_parquet(MATCHES_PARQUET)
    last_date = matches["date"].max()
    window = matches[matches["date"] >= last_date - pd.Timedelta(days=3)]

    keep = [
        "date",
        "home_team",
        "away_team",
        "season",
        "match_id",
        "odds_close_avg_home",
        "odds_close_avg_draw",
        "odds_close_avg_away",
    ]
    fixtures = window[[column for column in keep if column in window.columns]].copy()
    for column in ("home_goals", "away_goals"):
        fixtures[column] = pd.NA
    return fixtures.reset_index(drop=True)


def format_report(report: list[dict]) -> str:
    lines = []
    for match in report:
        scorelines = " · ".join(
            f"{entry['score']} ({entry['probability']:.0%})" for entry in match["scorelines"]
        )
        outcome = match["outcome"]
        lines.append(
            f"  {match['home_team']:>16} vs {match['away_team']:<16} {match['date']}\n"
            f"      {scorelines}\n"
            f"      Home {outcome['home']:.0%} | Draw {outcome['draw']:.0%} | "
            f"Away {outcome['away']:.0%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="which model to use"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="predict the last known round instead, as an end-to-end check",
    )
    parser.add_argument("--offline", action="store_true", help="never download fixtures")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an already-stored round (by default storing one is a one-time act)",
    )
    args = parser.parse_args(argv)

    if args.replay:
        fixtures = replay_fixtures()
        source = "replay of the last known round"
    else:
        raw, source = upcoming_fixtures(allow_download=not args.offline)
        fixtures = as_matches(raw)

    if fixtures.empty:
        print(
            f"No fixtures found ({source}).\n"
            f"The Premier League feed is empty outside a season - add rows to "
            f"data/manual/upcoming_fixtures.csv, or run with --replay to check the "
            f"pipeline against the last round that was played.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(fixtures)} fixtures from {source}\n")
    report = predict(fixtures, args.model, mode="replay" if args.replay else "upcoming")

    print(format_report(report))

    print()
    for (season_slug, gameweek), round_predictions in sorted(group_by_gameweek(report).items()):
        try:
            path = save_round(round_predictions, season_slug, gameweek, force=args.force)
        except RoundAlreadyStored as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"stored {len(round_predictions)} fixture(s) as {path.relative_to(ROUNDS_DIR)}")

    if not args.replay and UPCOMING_SEASON.fifa_edition != "EA FC 27":
        print(
            f"\nNote: {UPCOMING_SEASON.label} is using {UPCOMING_SEASON.fifa_edition} ratings, "
            f"the newest edition that exists. Record transfers in "
            f"data/manual/squad_changes.csv until the new edition is published."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
