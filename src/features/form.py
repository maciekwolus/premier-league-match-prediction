"""Form, Elo and rest, computed from previous matches only.

Everything here is built on a *team-match* table: two rows per fixture, one per side.
That shape is what makes the leakage rule enforceable - a team's history is a single
chronological series regardless of whether it played home or away, so one ``shift(1)``
covers every rolling feature.

**The rule this module exists to obey:** a match may never contribute to its own
features. Rolling windows shift before they aggregate; Elo is recorded as it stood
*before* the match and updated afterwards.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 5

# Elo. K controls how fast ratings move; HOME_ADVANTAGE is in rating points and is
# roughly the long-run home edge in this league. Between seasons ratings are pulled
# part-way back to the mean, because squads turn over in the summer.
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_REGRESSION = 0.25

# Rolled over previous matches. Each becomes {stat}_last5 for the team.
ROLLING_STATS = (
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_on_target_for",
    "corners_for",
    "points",
)


def team_match_table(matches: pd.DataFrame, understat: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fixture, one per team, in chronological order."""
    xg = understat[["match_id", "xg_home", "xg_away"]]
    df = matches.merge(xg, on="match_id", how="left")

    def side(is_home: bool) -> pd.DataFrame:
        us, them = ("home", "away") if is_home else ("away", "home")
        return pd.DataFrame(
            {
                "match_id": df["match_id"],
                "season": df["season"],
                "date": df["date"],
                "team": df[f"{us}_team"],
                "opponent": df[f"{them}_team"],
                "is_home": is_home,
                "goals_for": df[f"{us}_goals"],
                "goals_against": df[f"{them}_goals"],
                "xg_for": df[f"xg_{us}"],
                "xg_against": df[f"xg_{them}"],
                "shots_for": df[f"{us}_shots"],
                "shots_on_target_for": df[f"{us}_shots_target"],
                "corners_for": df[f"{us}_corners"],
            }
        )

    both = pd.concat([side(True), side(False)], ignore_index=True)

    both["points"] = 3 * (both["goals_for"] > both["goals_against"]) + 1 * (
        both["goals_for"] == both["goals_against"]
    )

    return both.sort_values(["date", "match_id", "team"]).reset_index(drop=True)


def add_rolling(team_matches: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling means over the previous ``window`` matches for each team.

    The window deliberately crosses season boundaries: a team's last five matches means
    their last five, so the opening weekend uses the previous May rather than starting
    blind. Roughly a quarter of all matches fall in the first five rounds, so treating
    that as an edge case would be a mistake.
    """
    df = team_matches.sort_values(["team", "date"]).copy()
    grouped = df.groupby("team", sort=False)

    for stat in ROLLING_STATS:
        # shift(1) first: without it a match lands inside its own average and the model
        # scores brilliantly in backtests and fails on Saturday.
        df[f"{stat}_last{window}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    df["matches_played"] = grouped.cumcount()
    df["season_matches_played"] = df.groupby(["team", "season"], sort=False).cumcount()

    return df.sort_values(["date", "match_id", "team"]).reset_index(drop=True)


def add_rest_days(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Days since this team's previous match, whenever that was."""
    df = team_matches.sort_values(["team", "date"]).copy()
    df["rest_days"] = df.groupby("team", sort=False)["date"].diff().dt.days

    return df.sort_values(["date", "match_id", "team"]).reset_index(drop=True)


def add_elo(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's Elo rating *as it stood before* the match.

    One chronological pass over fixtures. Both sides' pre-match ratings are recorded
    before either is updated, so a match can never influence its own features.
    """
    fixtures = (
        team_matches[team_matches["is_home"]]
        .loc[:, ["match_id", "season", "date", "team", "opponent", "goals_for", "goals_against"]]
        .sort_values(["date", "match_id"])
    )

    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}
    before: dict[tuple[str, str], float] = {}

    def current(team: str, season: str) -> float:
        """This team's rating, regressed towards the mean if a new season has started."""
        rating = ratings.get(team, ELO_START)
        if last_season.get(team) not in (None, season):
            rating = ELO_START + (1 - ELO_SEASON_REGRESSION) * (rating - ELO_START)
        ratings[team] = rating
        last_season[team] = season
        return rating

    for match in fixtures.itertuples():
        home_elo = current(match.team, match.season)
        away_elo = current(match.opponent, match.season)

        before[(match.match_id, match.team)] = home_elo
        before[(match.match_id, match.opponent)] = away_elo

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400))
        if match.goals_for > match.goals_against:
            actual_home = 1.0
        elif match.goals_for == match.goals_against:
            actual_home = 0.5
        else:
            actual_home = 0.0

        change = ELO_K * (actual_home - expected_home)
        ratings[match.team] = home_elo + change
        ratings[match.opponent] = away_elo - change

    df = team_matches.copy()
    df["elo_before"] = [
        before.get((match_id, team))
        for match_id, team in zip(df["match_id"], df["team"], strict=True)
    ]
    return df


def build_team_matches(matches: pd.DataFrame, understat: pd.DataFrame) -> pd.DataFrame:
    """The full team-match table with every pre-match feature attached."""
    df = team_match_table(matches, understat)
    df = add_rolling(df)
    df = add_rest_days(df)
    return add_elo(df)
