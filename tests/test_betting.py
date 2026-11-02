"""Tests for the simulated betting record.

A balance is the number on this page most likely to be believed without checking, so these
pin what it may claim: the tax is really deducted, nothing is staked twice, the money
describes the same picks as the scorecard, a backtest cannot see the round it is betting
on, and a profit is never shown without saying what it rests on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.betting.backtest import walk_forward
from src.betting.prices import (
    CORRECT_SCORE_OVERROUND,
    MINIMUM_PRICE,
    closing_prices,
    estimated_score_price,
    market_matrix,
)
from src.betting.simulate import PICK_COLUMNS, both_markets, by_round, priced, settle
from src.betting.tracker import format_summary, history, stored_bets, tracker
from src.evaluate.metrics import implied_probabilities
from src.models.score_matrix import outcome_probabilities
from src.predict.archive import save_round
from src.report.render import betting_bar, betting_detail
from src.report.results import attach_results, fixture_key, scorecard

SEASON = "2026_27"
LINE = (2.0, 3.5, 4.0)


def bets_frame(*bets: tuple[float, bool]) -> pd.DataFrame:
    """A frame of (price, won), the same for both markets."""
    return pd.DataFrame(
        {
            "gameweek": range(1, len(bets) + 1),
            "price_outcome": [price for price, _ in bets],
            "price_score": [price for price, _ in bets],
            "won_outcome": [won for _, won in bets],
            "won_score": [won for _, won in bets],
        }
    )


def stored(
    away: str,
    gameweek: int = 1,
    score: str = "1-1",
    outcome: tuple[float, float, float] = (0.5, 0.3, 0.2),
    mode: str = "upcoming",
    predicted_at: str = "2026-08-13T06:00:00+00:00",
) -> dict:
    return {
        "match_id": f"{SEASON}_{gameweek}_arsenal_{away}",
        "season_slug": SEASON,
        "gameweek": gameweek,
        "home_team": "Arsenal",
        "away_team": away,
        "mode": mode,
        "predicted_at": predicted_at,
        "scorelines": [{"score": score, "probability": 0.12}],
        "outcome": {"home": outcome[0], "draw": outcome[1], "away": outcome[2]},
    }


def result(away: str, home_goals: int, away_goals: int) -> dict:
    return {
        fixture_key(SEASON, "Arsenal", away): {
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    }


def line_for(*aways: str) -> dict:
    return {fixture_key(SEASON, "Arsenal", away): LINE for away in aways}


# ---------------------------------------------------------------- settling


def test_a_winner_pays_on_the_stake_left_after_tax():
    settled = settle(bets_frame((2.0, True)), "outcome")
    assert settled["returned"] == pytest.approx(17.60)
    assert settled["balance"] == pytest.approx(7.60)


def test_a_loser_costs_the_whole_stake():
    settled = settle(bets_frame((2.0, False)), "outcome")
    assert settled["returned"] == 0
    assert settled["balance"] == -10


def test_the_untaxed_balance_is_reported_beside_the_real_one():
    summary = both_markets(bets_frame((2.0, True)))
    assert summary["outcome"]["balance"] == pytest.approx(7.60)
    assert summary["outcome"]["untaxed_balance"] == pytest.approx(10.0)


def test_no_bets_settle_to_nothing_rather_than_dividing_by_zero():
    settled = settle(bets_frame(), "score")
    assert (settled["bets"], settled["balance"], settled["roi"]) == (0, 0, None)


def test_rounds_carry_a_running_balance_that_ends_at_the_total():
    bets = bets_frame((2.0, True), (3.0, False), (4.0, True))
    rounds = by_round(bets)

    assert [row["gameweek"] for row in rounds] == [1, 2, 3]
    assert rounds[-1]["outcome"]["running"] == pytest.approx(
        both_markets(bets)["outcome"]["balance"]
    )


def test_luck_grows_with_the_number_of_bets():
    few = settle(bets_frame(*[(8.0, index % 8 == 0) for index in range(16)]), "score")
    many = settle(bets_frame(*[(8.0, index % 8 == 0) for index in range(160)]), "score")
    assert many["luck"] > few["luck"] > 0


# ---------------------------------------------------------------- prices


def test_the_market_matrix_reproduces_the_line_it_came_from():
    matrix = market_matrix(*LINE)
    target = implied_probabilities([LINE[0]], [LINE[1]], [LINE[2]])[0]

    assert matrix.sum() == pytest.approx(1.0)
    assert outcome_probabilities(matrix) == pytest.approx(tuple(target), abs=0.01)


def test_a_heavier_favourite_makes_its_scorelines_shorter():
    even = estimated_score_price(2.6, 3.4, 2.6, 3, 0)
    lopsided = estimated_score_price(1.2, 7.0, 13.0, 3, 0)
    assert lopsided < even


def test_a_bigger_margin_pays_less():
    fair = estimated_score_price(*LINE, 1, 1, overround=1.0)
    usual = estimated_score_price(*LINE, 1, 1)
    assert usual == pytest.approx(fair / CORRECT_SCORE_OVERROUND, abs=0.02)


def test_no_estimated_price_falls_below_the_minimum():
    assert estimated_score_price(*LINE, 1, 1, overround=50.0) >= MINIMUM_PRICE


def test_a_fixture_without_a_complete_price_is_absent(tmp_path):
    table = pd.DataFrame(
        {
            "season": ["2026/27", "2026/27"],
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Hull", "Coventry"],
            "odds_close_home": [2.0, None],
            "odds_close_draw": [3.5, 3.5],
            "odds_close_away": [4.0, 4.0],
        }
    )
    path = tmp_path / "matches.parquet"
    table.to_parquet(path)

    prices = closing_prices(path)
    assert prices == {fixture_key(SEASON, "Arsenal", "Hull"): (2.0, 3.5, 4.0)}


# ---------------------------------------------------------------- picks


def test_the_picks_are_settled_on_the_outcome_and_the_score():
    picks = pd.DataFrame(
        [
            {
                "season_slug": SEASON,
                "gameweek": 1,
                "home_team": "Arsenal",
                "away_team": "Hull",
                "pick_outcome": "H",
                "pick_score": "1-1",
                "home_goals": 2,
                "away_goals": 0,
            }
        ],
        columns=PICK_COLUMNS,
    )
    row = priced(picks, line_for("Hull")).iloc[0]

    assert row["price_outcome"] == 2.0
    assert bool(row["won_outcome"]) is True
    assert bool(row["won_score"]) is False


def test_a_replayed_round_is_not_a_bet_anyone_could_have_placed(tmp_path):
    save_round([stored("Hull", mode="replay")], SEASON, 1, tmp_path)

    bets, pending, _ = stored_bets(
        SEASON, tmp_path, results=result("Hull", 1, 1), prices=line_for("Hull")
    )
    assert bets.empty
    assert pending == 0


def test_an_unplayed_fixture_waits_and_costs_nothing(tmp_path):
    save_round([stored("Hull")], SEASON, 1, tmp_path)

    bets, pending, unpriced = stored_bets(SEASON, tmp_path, results={}, prices={})
    assert (len(bets), pending, unpriced) == (0, 1, 0)


def test_a_played_match_without_a_price_carries_no_bet_and_is_counted(tmp_path):
    save_round([stored("Hull")], SEASON, 1, tmp_path)

    bets, pending, unpriced = stored_bets(SEASON, tmp_path, results=result("Hull", 1, 1), prices={})
    assert (len(bets), pending, unpriced) == (0, 0, 1)


def test_a_postponed_fixture_is_staked_once(tmp_path):
    save_round([stored("Hull", gameweek=1)], SEASON, 1, tmp_path)
    later = stored("Hull", gameweek=2, predicted_at="2026-08-20T06:00:00+00:00")
    save_round([later], SEASON, 2, tmp_path)

    bets, _, _ = stored_bets(
        SEASON, tmp_path, results=result("Hull", 1, 1), prices=line_for("Hull")
    )
    assert len(bets) == 1
    assert bets.iloc[0]["gameweek"] == 1


def test_the_money_and_the_scorecard_grade_the_same_picks(tmp_path):
    matches = [
        stored("Hull", score="2-1"),
        stored("Coventry", score="1-1", outcome=(0.2, 0.3, 0.5)),
        stored("Wolves", score="1-0", outcome=(0.3, 0.4, 0.3)),
    ]
    save_round(matches, SEASON, 1, tmp_path)
    results = {
        **result("Hull", 2, 1),
        **result("Coventry", 1, 1),
        **result("Wolves", 0, 0),
    }

    record = tracker(SEASON, tmp_path, results, line_for("Hull", "Coventry", "Wolves"))
    card = scorecard(attach_results(matches, results))

    assert record["total"]["outcome"]["won"] == card["outcome"]
    assert record["total"]["score"]["won"] == card["exact"]


# ---------------------------------------------------------------- the backtest


def synthetic_features() -> pd.DataFrame:
    """Two complete, tiny seasons: every pairing of four clubs, home and away."""
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    fixtures = [
        pair
        for round_of_pairs in (
            [(0, 1), (2, 3)],
            [(2, 0), (3, 1)],
            [(0, 3), (1, 2)],
            [(1, 0), (3, 2)],
            [(0, 2), (1, 3)],
            [(3, 0), (2, 1)],
        )
        for pair in round_of_pairs
    ]
    rows = []
    for year, season in ((2023, "2023/24"), (2024, "2024/25")):
        for index, (home, away) in enumerate(fixtures):
            rows.append(
                {
                    "match_id": f"{season}_{index}",
                    "season": season,
                    "date": pd.Timestamp(year, 8, 1) + pd.Timedelta(days=7 * (index // 2)),
                    "home_team": clubs[home],
                    "away_team": clubs[away],
                    "home_goals": index % 3,
                    "away_goals": index % 2,
                    "home_squad_overall_mean": 80.0,
                }
            )
    return pd.DataFrame(rows)


class RecordingModel:
    """Remembers what it was trained on, and predicts from those results alone."""

    fits: list[pd.DataFrame] = []

    def fit(self, train: pd.DataFrame) -> None:
        type(self).fits.append(train)
        self.level = 1.0 + float(train["home_goals"].mean())

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(test), self.level), np.full(len(test), 1.0)


@pytest.fixture
def recorded():
    RecordingModel.fits = []
    return RecordingModel


def test_the_first_season_is_never_bet_on(recorded):
    picks = walk_forward(synthetic_features(), recorded)
    assert set(picks["season_slug"]) == {"2024_25"}
    assert len(picks) == 12


def test_a_round_is_never_trained_on_itself_or_anything_later(recorded):
    features = synthetic_features()
    walk_forward(features, recorded)

    test_season = features[features["season"] == "2024/25"]
    round_starts = sorted(test_season["date"].unique())
    assert len(recorded.fits) == len(round_starts)

    for train, start in zip(recorded.fits, round_starts, strict=True):
        assert train["date"].max() < start


def test_rewriting_a_rounds_results_cannot_change_its_own_picks(recorded):
    features = synthetic_features()
    original = walk_forward(features, recorded)

    final_round = features["date"] == features["date"].max()
    rewritten = features.copy()
    rewritten.loc[final_round, ["home_goals", "away_goals"]] = [9, 0]
    changed = walk_forward(rewritten, recorded)

    columns = ["pick_outcome", "pick_score", "prob_home"]
    pd.testing.assert_frame_equal(original[columns], changed[columns])


def test_the_season_in_progress_trains_without_squad_ratings(recorded):
    """The live path has no ratings for it, and using them scored 0.2025 for a model nobody
    can run, against the 0.2033 deployed at the time."""
    walk_forward(synthetic_features(), recorded)

    last = recorded.fits[-1]
    in_progress = last[last["season"] == "2024/25"]
    complete = last[last["season"] == "2023/24"]

    assert len(in_progress) > 0
    assert in_progress["home_squad_overall_mean"].isna().all()
    assert complete["home_squad_overall_mean"].notna().all()


# ---------------------------------------------------------------- what the page says


def backtest_file(tmp_path: Path) -> Path:
    frame = bets_frame((2.0, True), (2.0, False), (8.0, False), (8.0, False)).assign(
        season_slug=["2024_25", "2024_25", "2025_26", "2025_26"]
    )
    path = tmp_path / "backtest.parquet"
    frame.to_parquet(path)
    return path


def lucky_record(tmp_path: Path) -> dict:
    save_round([stored("Hull", score="1-1")], SEASON, 1, tmp_path)
    return tracker(SEASON, tmp_path, result("Hull", 1, 1), line_for("Hull"))


def test_a_checkout_without_the_backtest_still_works(tmp_path):
    assert history(tmp_path / "missing.parquet") is None


def test_history_totals_every_season(tmp_path):
    past = history(backtest_file(tmp_path))
    assert [row["season"] for row in past["seasons"]] == ["2024/25", "2025/26"]
    assert past["total"]["matches"] == 4


def test_a_thinner_margin_never_makes_the_exact_score_balance_worse(tmp_path):
    balances = [entry["balance"] for entry in history(backtest_file(tmp_path))["overrounds"]]
    assert balances == sorted(balances, reverse=True)


def test_the_bar_says_these_are_not_polish_prices_and_are_estimated(tmp_path):
    bar = betting_bar(lucky_record(tmp_path), None)
    assert "not Polish prices" in bar
    assert "estimated" in bar


def test_a_profit_is_contradicted_by_the_backtest(tmp_path):
    record = lucky_record(tmp_path)
    assert record["total"]["score"]["balance"] > 0

    past = history(backtest_file(tmp_path))
    assert "should not be believed" in betting_bar(record, past)


def test_a_loss_is_not_contradicted(tmp_path):
    save_round([stored("Hull")], SEASON, 1, tmp_path)
    record = tracker(SEASON, tmp_path, result("Hull", 0, 3), line_for("Hull"))
    past = history(backtest_file(tmp_path))

    assert record["total"]["score"]["balance"] < 0
    assert "should not be believed" not in betting_bar(record, past)


def test_a_season_with_nothing_settled_shows_no_bar(tmp_path):
    assert betting_bar(tracker(SEASON, tmp_path, {}, {}), None) == ""


def test_the_detail_lists_every_round_and_the_backtest(tmp_path):
    detail = betting_detail(lucky_record(tmp_path), history(backtest_file(tmp_path)))
    assert "GW 1" in detail
    assert "2025/26" in detail
    assert "not Polish prices" in detail


def test_the_run_summary_says_what_the_prices_are(tmp_path):
    summary = format_summary(lucky_record(tmp_path), None)
    assert "not a Polish bookmaker" in summary
    assert "estimated" in summary
    assert "| GW 1 |" in summary


# ---------------------------------------------------------------- connected


def test_the_tracker_is_actually_called():
    """Three things in this project have shipped tested and connected to nothing."""
    root = Path(__file__).resolve().parents[1]
    assert "src.betting.tracker" in (root / ".github/workflows/predict-round.yml").read_text(
        encoding="utf-8"
    )
    app = (root / "app.py").read_text(encoding="utf-8")
    assert "betting_bar(" in app
    assert "betting_detail(" in app
