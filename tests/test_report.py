"""Tests for how predictions are shaped for display.

Presentation bugs are quiet: a bar scaled wrongly or a sign flipped still renders, and
the page looks authoritative either way. These check the claims the report makes rather
than that it renders.
"""

import pytest

from src.report.view import (
    NOTABLE_DISAGREEMENT,
    as_percent,
    disagreement,
    modal_scoreline_share,
    most_likely_outcome,
    outcome_rows,
    scoreline_rows,
    summarise,
)


def make_match(
    home="Arsenal",
    away="Chelsea",
    outcome=(0.48, 0.26, 0.26),
    bookmaker=(0.45, 0.27, 0.28),
    scorelines=((("2-1"), 0.11), (("1-1"), 0.10), (("2-0"), 0.09)),
) -> dict:
    match = {
        "match_id": "2026_27_20260815_arsenal_chelsea",
        "date": "2026-08-15",
        "home_team": home,
        "away_team": away,
        "model": "poisson-glm",
        "outcome": {"home": outcome[0], "draw": outcome[1], "away": outcome[2]},
        "scorelines": [
            {"score": score, "probability": probability} for score, probability in scorelines
        ],
    }
    if bookmaker is not None:
        match["bookmaker"] = {
            "home": bookmaker[0],
            "draw": bookmaker[1],
            "away": bookmaker[2],
        }
    return match


# ------------------------------------------------------------------- scorelines


def test_scoreline_bars_scale_to_the_leader():
    """Scaled against 100% every bar would be a sliver, since the best is around 12%."""
    rows = scoreline_rows(make_match())

    assert rows[0]["width"] == pytest.approx(1.0)
    assert rows[1]["width"] == pytest.approx(0.10 / 0.11)
    assert all(0 < row["width"] <= 1 for row in rows)


def test_scoreline_labels_are_percentages():
    rows = scoreline_rows(make_match())
    assert rows[0]["label"] == "11%"
    assert rows[0]["score"] == "2-1"


def test_a_match_with_no_scorelines_yields_nothing():
    assert scoreline_rows({"scorelines": []}) == []
    assert scoreline_rows({}) == []


# --------------------------------------------------------------------- outcomes


def test_outcome_rows_cover_home_draw_away_in_order():
    rows = outcome_rows(make_match())
    assert [row["outcome"] for row in rows] == ["home", "draw", "away"]


def test_edge_is_model_minus_market():
    """Sign matters: a flipped edge would credit the model for the market's opinion."""
    rows = outcome_rows(make_match(outcome=(0.60, 0.20, 0.20), bookmaker=(0.50, 0.25, 0.25)))

    assert rows[0]["edge"] == pytest.approx(0.10)
    assert rows[1]["edge"] == pytest.approx(-0.05)


def test_a_fixture_with_no_market_still_renders():
    """Odds do not exist until a market forms, which is normal well before kickoff."""
    rows = outcome_rows(make_match(bookmaker=None))

    assert all(row["bookmaker"] is None for row in rows)
    assert all(row["bookmaker_label"] == "—" for row in rows)
    assert all(row["edge"] is None for row in rows)


def test_most_likely_outcome_reads_the_highest():
    assert most_likely_outcome(make_match(outcome=(0.5, 0.3, 0.2))) == "Home win"
    assert most_likely_outcome(make_match(outcome=(0.2, 0.5, 0.3))) == "Draw"
    assert most_likely_outcome(make_match(outcome=(0.2, 0.3, 0.5))) == "Away win"


def test_most_likely_outcome_handles_an_empty_match():
    assert most_likely_outcome({}) == "—"


# ----------------------------------------------------------------- disagreement


def test_close_agreement_is_not_reported():
    """Two points apart is agreement, and calling it a finding would be misleading."""
    assert (
        disagreement(make_match(outcome=(0.48, 0.26, 0.26), bookmaker=(0.47, 0.27, 0.26))) is None
    )


def test_a_real_gap_is_reported_with_its_direction():
    """Both sets sum to 1, so the draw must differ for home and away not to tie."""
    match = make_match(outcome=(0.35, 0.30, 0.35), bookmaker=(0.24, 0.26, 0.50))
    outcome, edge = disagreement(match)

    assert outcome == "away"
    assert edge == pytest.approx(-0.15)


def test_the_largest_gap_wins_regardless_of_sign():
    match = make_match(outcome=(0.60, 0.20, 0.20), bookmaker=(0.40, 0.25, 0.35))
    outcome, edge = disagreement(match)

    assert outcome == "home"
    assert edge == pytest.approx(0.20)


def test_no_market_means_no_disagreement():
    assert disagreement(make_match(bookmaker=None)) is None


def test_the_threshold_is_what_it_claims():
    just_under = make_match(
        outcome=(0.50 + NOTABLE_DISAGREEMENT - 0.001, 0.25, 0.25), bookmaker=(0.50, 0.25, 0.25)
    )
    assert disagreement(just_under) is None


# -------------------------------------------------------------------- summary


def test_summary_counts_fixtures_and_market_coverage():
    predictions = [make_match(), make_match(bookmaker=None)]
    summary = summarise(predictions)

    assert summary["fixtures"] == 2
    assert summary["with_odds"] == 1
    assert summary["model"] == "poisson-glm"


def test_summary_of_a_single_date_is_that_date():
    assert summarise([make_match()])["date_range"] == "2026-08-15"


def test_summary_of_several_dates_is_a_range():
    later = make_match()
    later["date"] = "2026-08-17"
    assert summarise([make_match(), later])["date_range"] == "2026-08-15 to 2026-08-17"


def test_an_empty_report_does_not_crash():
    summary = summarise([])
    assert summary["fixtures"] == 0
    assert summary["mode"] == "upcoming"  # every caller reads this key


def test_a_replayed_round_is_labelled_as_one():
    """Without this the page shows past matches as though they were next week's."""
    replayed = make_match()
    replayed["mode"] = "replay"

    assert summarise([replayed])["mode"] == "replay"
    assert summarise([make_match()])["mode"] == "upcoming"


def test_modal_scoreline_share_measures_repetition():
    """1-1 leading nearly every card is mostly the sport, but the page should say so."""
    same = [make_match(scorelines=(("1-1", 0.12), ("2-1", 0.09))) for _ in range(4)]
    assert modal_scoreline_share(same) == pytest.approx(1.0)

    mixed = [
        make_match(scorelines=(("1-1", 0.12),)),
        make_match(scorelines=(("2-0", 0.11),)),
    ]
    assert modal_scoreline_share(mixed) == pytest.approx(0.5)


def test_modal_scoreline_share_of_nothing_is_zero():
    assert modal_scoreline_share([]) == 0.0
    assert modal_scoreline_share([{"scorelines": []}]) == 0.0


# ----------------------------------------------------------------- loading


def test_as_percent_rounds_to_whole_numbers():
    assert as_percent(0.485) == "48%"
    assert as_percent(0.0) == "0%"
    assert as_percent(1.0) == "100%"
