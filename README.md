# Premier League Match Prediction

Predicts the **scoreline** of upcoming Premier League fixtures, with probabilities:

```
Arsenal vs Chelsea    2-1 (11%) · 1-1 (10%) · 2-0 (9%)
                      Home 48% | Draw 26% | Away 26%
```

Exact scorelines in football top out around 12% probability even for a perfect model, so
the output is the *most likely* scorelines with honest probabilities rather than a single
confident guess. Across the 2,660 matches gathered here, the most common result — 1-1 —
occurs 11.2% of the time, which is roughly the ceiling any model can approach. Every
prediction is shown next to the bookmaker's line, so it is obvious whether the model is
actually adding anything.

## How it works

| Input | Source |
|---|---|
| Match results, shots, cards, odds | [football-data.co.uk](https://www.football-data.co.uk/englandm.php) |
| Lineups and expected goals (xG) | [Understat](https://understat.com) |
| Player ratings | FIFA 20-23 / EA Sports FC 24-26 |

Player ratings are matched to the players who actually started each match, giving a
squad-quality signal per team per fixture. Those, plus rolling form, xG, Elo and rest
days, feed two models — a classical Dixon-Coles goals model and a gradient-boosting
model — which produce a scoreline probability matrix.

Seasons covered: **2019/20 through 2025/26** (7 seasons, 2,660 matches).

## Setup

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Data pipeline

Downloads are cached, so re-running is cheap; pass `--force` to re-fetch.

```bash
python -m src.data.fetch_matches
```

```bash
python -m src.data.clean_matches
```

This produces `data/processed/matches.parquet` — one row per match with the result,
match statistics, referee, and both opening and closing bookmaker odds. Cleaning
validates every season before writing (380 matches, 20 teams, 19 home and 19 away per
team, results agreeing with scores) and raises rather than emitting a suspect table.

Downloaded and generated data is gitignored; the pipeline rebuilds it from scratch.

## Development

```bash
pytest
```

```bash
pytest tests/test_clean_matches.py -k slugify
```

```bash
ruff check . && ruff format .
```

Tests build synthetic seasons instead of reading downloaded files, so they run offline
and stay meaningful if an upstream source changes.

## Project status

Phases 0–1 complete — skeleton, configuration, and match-results ingestion.
See [PLAN.md](PLAN.md) for the full ten-phase build plan and where things stand.

## Layout

```
src/config.py           seasons, paths, source URLs - the single place seasons are defined
src/data/               acquisition and cleaning, one module per source
tests/                  unit tests, no network access
data/raw/               downloaded source data (gitignored)
data/processed/         cleaned and joined data (gitignored)
data/final/             model-ready feature table (gitignored)
data/manual/            hand-written name-override files (committed)
```
