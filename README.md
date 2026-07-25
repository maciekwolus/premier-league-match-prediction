# Premier League Match Prediction

Predicts the **scoreline** of upcoming Premier League fixtures, with probabilities:

```
Arsenal vs Chelsea    2-1 (11%) · 1-1 (10%) · 2-0 (9%)
                      Home 48% | Draw 26% | Away 26%
```

Exact scorelines in football top out around 12% probability even for a perfect model,
so the output is the *most likely* scorelines with honest probabilities rather than a
single confident guess. Every prediction is shown next to the bookmaker's line, which
makes it obvious whether the model is actually adding anything.

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

Seasons covered: **2019/20 through 2025/26** (7 seasons, ~2,660 matches).

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

## Development

Run the tests:

```bash
pytest
```

Lint and format:

```bash
ruff check . && ruff format .
```

## Project status

Phases 0–1 complete — skeleton, configuration, and match-results ingestion
(2,660 matches across 7 seasons, fully validated).
See [PLAN.md](PLAN.md) for the full ten-phase build plan and where things stand.

Rebuild the match data from scratch:

```bash
python -m src.data.fetch_matches && python -m src.data.clean_matches
```

## Layout

```
src/config.py    seasons, paths, data source URLs - the single place seasons are defined
tests/           unit tests
data/raw/        downloaded source data (gitignored)
data/processed/  cleaned and joined data (gitignored)
data/final/      model-ready feature table (gitignored)
data/manual/     hand-written name-override files (committed)
```
