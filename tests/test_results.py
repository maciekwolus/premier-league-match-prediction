"""Tests for scoring stored predictions against what happened.

This is the code most able to lie in our favour, and to do it quietly: scoring our model
over one set of fixtures and the bookmaker over another produces a flattering number that
looks perfectly ordinary. These pin the comparisons to the same matches, and pin the
claims to what was actually predicted.
"""

from __future__ import annotations

from src.predict.archive import save_round
from src.report.render import match_card, scorecard_bar
from src.report.results import (
    MEANINGFUL_SAMPLE,
    attach_results,
    outcome_of,
    scorecard,
    verdict,
)
from src.report.view import load_round_predictions, round_options, season_scorecard


def prediction(
    match_id: str = "m1",
    scoreline: str = "1-1",
    outcome: tuple[float, float, float] = (0.5, 0.3, 0.2),
    bookmaker: tuple[float, float, float] | None = (0.45, 0.3, 0.25),
) -> dict:
    match = {
        "match_id": match_id,
        "date": "2026-08-15",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "scorelines": [{"score": scoreline, "probability": 0.12}],
        "outcome": {"home": outcome[0], "draw": outcome[1], "away": outcome[2]},
    }
    if bookmaker:
        match["bookmaker"] = {
            "home": bookmaker[0],
            "draw": bookmaker[1],
            "away": bookmaker[2],
        }
    return match


def results_for(**scores: tuple[int, int]) -> dict[str, dict]:
    return {
        match_id: {"home_goals": home, "away_goals": away}
        for match_id, (home, away) in scores.items()
    }


def test_outcome_labels_match_the_metrics():
    assert outcome_of(2, 1) == "H"
    assert outcome_of(1, 1) == "D"
    assert outcome_of(0, 3) == "A"


def test_an_unplayed_fixture_gets_no_result():
    attached = attach_results([prediction()], results={})

    assert "actual" not in attached[0]
    assert verdict(attached[0]) is None


def test_a_played_fixture_carries_its_score():
    attached = attach_results([prediction()], results_for(m1=(2, 1)))

    assert attached[0]["actual"]["score"] == "2-1"
    assert attached[0]["actual"]["outcome"] == "H"


def test_attaching_results_does_not_mutate_the_archive():
    """The stored record is evidence; a display concern must not write into it."""
    stored = prediction()
    attach_results([stored], results_for(m1=(2, 1)))

    assert "actual" not in stored


def test_an_exact_scoreline_is_recognised():
    attached = attach_results([prediction(scoreline="2-1")], results_for(m1=(2, 1)))
    result = verdict(attached[0])

    assert result["exact"]
    assert result["outcome"]


def test_the_right_outcome_with_the_wrong_score_is_not_an_exact_hit():
    """The distinction the card depends on: 1-1 predicted, 2-2 played.

    The draw has to be the *most likely outcome* for this to be an outcome hit, which is
    not implied by 1-1 leading the scorelines - see the test below.
    """
    draw_favourite = prediction(scoreline="1-1", outcome=(0.30, 0.45, 0.25))
    attached = attach_results([draw_favourite], results_for(m1=(2, 2)))
    result = verdict(attached[0])

    assert not result["exact"]
    assert result["outcome"]


def test_a_miss_is_a_miss():
    home_favourite = prediction(scoreline="2-1", outcome=(0.6, 0.25, 0.15))
    attached = attach_results([home_favourite], results_for(m1=(0, 3)))
    result = verdict(attached[0])

    assert not result["exact"]
    assert not result["outcome"]


def test_the_predicted_outcome_comes_from_the_probabilities_not_the_scoreline():
    """A leading scoreline of 1-1 does not mean a draw was the most likely outcome -
    the draw share is usually smaller than a win share spread over many scorelines."""
    draw_scoreline_home_favourite = prediction(scoreline="1-1", outcome=(0.55, 0.25, 0.20))
    attached = attach_results([draw_scoreline_home_favourite], results_for(m1=(3, 0)))

    assert verdict(attached[0])["outcome"] is True


def test_an_empty_scorecard_when_nothing_has_been_played():
    card = scorecard(attach_results([prediction()], results={}))

    assert card["played"] == 0
    assert card["rps"] is None


def test_the_scorecard_counts_both_kinds_of_hit():
    predictions = [
        prediction("m1", scoreline="2-1"),  # exact
        prediction("m2", scoreline="1-0"),  # outcome only
        prediction("m3", scoreline="1-0", outcome=(0.6, 0.25, 0.15)),  # miss
    ]
    attached = attach_results(predictions, results_for(m1=(2, 1), m2=(3, 0), m3=(0, 2)))

    card = scorecard(attached)

    assert card["played"] == 3
    assert card["exact"] == 1
    assert card["outcome"] == 2


def test_our_score_and_the_market_are_taken_over_the_same_fixtures():
    """The bug this guards against is silent and flattering: scoring ourselves on every
    match and the bookmaker only on the ones it priced makes the two incomparable."""
    predictions = [
        prediction("priced", outcome=(0.5, 0.3, 0.2)),
        prediction("unpriced", outcome=(0.9, 0.05, 0.05), bookmaker=None),
    ]
    attached = attach_results(predictions, results_for(priced=(1, 1), unpriced=(0, 4)))

    card = scorecard(attached)

    assert card["played"] == 2
    assert card["compared"] == 1

    only_priced = scorecard(attach_results([predictions[0]], results_for(priced=(1, 1))))
    assert card["rps"] == only_priced["rps"]


def test_a_perfect_prediction_scores_zero():
    certain_home_win = prediction(outcome=(1.0, 0.0, 0.0), bookmaker=(1.0, 0.0, 0.0))
    card = scorecard(attach_results([certain_home_win], results_for(m1=(3, 0))))

    assert card["rps"] == 0.0


def test_a_confident_wrong_prediction_scores_worse_than_the_market():
    ours = prediction("m1", outcome=(0.95, 0.03, 0.02), bookmaker=(0.4, 0.3, 0.3))
    card = scorecard(attach_results([ours], results_for(m1=(0, 2))))

    assert card["rps"] > card["market_rps"]


def test_one_round_is_flagged_as_too_small_to_read():
    """Ten matches is a round, not evidence. The page has to say so."""
    predictions = [prediction(f"m{n}") for n in range(10)]
    attached = attach_results(predictions, results_for(**{f"m{n}": (1, 1) for n in range(10)}))

    assert scorecard(attached)["small_sample"] is True


def test_a_season_of_rounds_is_not_flagged():
    count = MEANINGFUL_SAMPLE
    predictions = [prediction(f"m{n}") for n in range(count)]
    attached = attach_results(predictions, results_for(**{f"m{n}": (1, 1) for n in range(count)}))

    assert scorecard(attached)["small_sample"] is False


def test_a_half_played_round_scores_only_what_was_played():
    """Midweek fixtures mean a round is often partly played. The rest must not count."""
    predictions = [prediction("played"), prediction("not yet")]
    attached = attach_results(predictions, results_for(played=(1, 1)))

    assert scorecard(attached)["played"] == 1


def test_a_lucky_round_that_beats_the_market_is_called_out(tmp_path):
    """The one number on this page that could genuinely mislead.

    Over ten fixtures a mediocre model beats the closing line often enough, and a page
    reporting that without contradiction would be claiming something the 2,280-match
    backtest denies.
    """
    sharp = prediction("m1", outcome=(0.9, 0.05, 0.05), bookmaker=(0.4, 0.3, 0.3))
    card = scorecard(attach_results([sharp], results_for(m1=(3, 0))))
    assert card["rps"] < card["market_rps"]

    bar = scorecard_bar(card, "2026/27")

    assert "do not believe it" in bar
    assert "2,280" in bar


def test_a_losing_round_gets_the_caveat_without_the_rebuttal():
    """No need to argue with a number that already agrees with the backtest."""
    poor = prediction("m1", outcome=(0.05, 0.05, 0.9), bookmaker=(0.4, 0.3, 0.3))
    card = scorecard(attach_results([poor], results_for(m1=(3, 0))))

    bar = scorecard_bar(card, "2026/27")

    assert "is not a sample" in bar
    assert "do not believe it" not in bar


def test_an_exact_score_with_the_wrong_call_admits_both():
    """The real case from 2025/26 gameweek 38, and the flattering one to get wrong.

    A 1-1 leading scoreline sits on a card whose headline verdict is HOME WIN, because
    the draw is the most likely single score while the home win is the most likely
    outcome. When it finishes 1-1 the scoreline was right and the call above it was
    wrong, and showing only the green half would be advertising.
    """
    home_favourite_drawn = prediction("m1", scoreline="1-1", outcome=(0.47, 0.27, 0.26))
    card = match_card(attach_results([home_favourite_drawn], results_for(m1=(1, 1)))[0])

    assert "EXACT SCORE" in card
    assert "WRONG CALL" in card


def test_an_exact_score_that_also_called_it_says_so_plainly():
    both_right = prediction("m1", scoreline="2-1", outcome=(0.55, 0.25, 0.20))
    card = match_card(attach_results([both_right], results_for(m1=(2, 1)))[0])

    assert "EXACT SCORE CALLED" in card
    assert "WRONG CALL" not in card


def test_an_unplayed_fixture_gets_no_result_row():
    assert "pl-result" not in match_card(prediction())


def test_no_scorecard_before_anything_has_been_played():
    assert scorecard_bar(scorecard([]), "2026/27") == ""


# --------------------------------------------------------------- the round selector


def test_rounds_are_offered_newest_first(tmp_path):
    """In April you want gameweek 33, not gameweek 1."""
    for gameweek in (1, 12, 7):
        save_round([prediction(f"m{gameweek}")], "2026_27", gameweek, root=tmp_path)

    options = round_options(tmp_path)

    assert [option["gameweek"] for option in options] == [12, 7, 1]
    assert options[0]["label"] == "2026/27  ·  GW 12"


def test_a_new_season_is_offered_above_the_old_one(tmp_path):
    save_round([prediction("old")], "2025_26", 38, root=tmp_path)
    save_round([prediction("new")], "2026_27", 1, root=tmp_path)

    assert round_options(tmp_path)[0]["season_slug"] == "2026_27"


def test_no_stored_rounds_offers_nothing(tmp_path):
    assert round_options(tmp_path / "empty") == []


def test_a_selected_round_arrives_with_its_results(tmp_path):
    save_round([prediction("m1", scoreline="2-1")], "2026_27", 3, root=tmp_path)

    loaded = load_round_predictions("2026_27", 3, root=tmp_path, results=results_for(m1=(2, 1)))

    assert loaded[0]["actual"]["score"] == "2-1"


def test_the_season_scorecard_spans_every_stored_round(tmp_path):
    """The point of season-to-date: one round is noise, a season is a record."""
    save_round([prediction("m1", scoreline="1-0")], "2026_27", 1, root=tmp_path)
    save_round([prediction("m2", scoreline="2-0")], "2026_27", 2, root=tmp_path)

    card = season_scorecard("2026_27", root=tmp_path, results=results_for(m1=(1, 0), m2=(3, 0)))

    assert card["played"] == 2
    assert card["exact"] == 1


def test_the_season_scorecard_ignores_other_seasons(tmp_path):
    save_round([prediction("this season")], "2026_27", 1, root=tmp_path)
    save_round([prediction("last season")], "2025_26", 38, root=tmp_path)

    card = season_scorecard(
        "2026_27",
        root=tmp_path,
        results=results_for(**{"this season": (1, 1), "last season": (1, 1)}),
    )

    assert card["played"] == 1
