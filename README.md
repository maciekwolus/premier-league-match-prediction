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
days, feed a set of goals models — from a naive league average through Dixon-Coles and
Poisson regression to gradient boosting — each producing the two expected-goals numbers
that become a scoreline probability matrix.

Seasons covered: **2019/20 through 2025/26** (7 seasons, 2,660 matches).

## Getting started

**Almost no data is stored in this repository** — it is gitignored and rebuilt from
source, the exception being the hand-written override files in `data/manual/`. A fresh
clone therefore needs steps 5 to 8 before anything works, and step 9 to reproduce the
results below.

### Prerequisites

| | |
|---|---|
| Python | 3.12 or newer (`python --version`) |
| Disk | ~870 MB — 700 MB virtual environment (AutoGluon is most of it), 165 MB data |
| Network | Required for the initial data build |

### 1. Clone

```bash
git clone https://github.com/maciekwolus/premier-league-match-prediction.git
```

```bash
cd premier-league-match-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

macOS and Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies and verify

```bash
pip install -r requirements.txt
```

```bash
pytest
```

The tests need neither network nor data, so a green run here confirms the environment is
sound before the slow step.

**AutoGluon is around 700 MB of that install and is optional.** It is imported only when
a gradient-boosting model is actually built, so the whole data pipeline, the tests, and
`compare --fast` all work without it. Comment it out of `requirements.txt` if you would
rather not download it.

### 5. Build the data

Run in order — each stage reads the previous one's output.

```bash
python -m src.data.fetch_matches
```

```bash
python -m src.data.clean_matches
```

```bash
python -m src.data.fetch_lineups
```

```bash
python -m src.data.clean_lineups
```

| Stage | Time | Produces |
|---|---|---|
| `fetch_matches` | ~10 s | 7 season CSVs |
| `clean_matches` | ~5 s | `matches.parquet` — 2,660 matches with results, stats, referee, opening and closing odds |
| `fetch_lineups` | **~35 min** | ~2,670 Understat responses |
| `clean_lineups` | ~30 s | `understat_matches.parquet` (match xG) and `lineups.parquet` (77,278 player appearances) |

`fetch_lineups` is slow on purpose: it spaces roughly 2,660 requests to stay polite
towards a small free site. Every response is cached, so **interrupting it is safe** —
re-running resumes where it stopped rather than starting over. Pass `--delay` to change
the spacing, or `--season "2025/26"` to rebuild one season.

Every download stage caches; `--force` re-fetches.

### 6. Add the player ratings (manual)

Kaggle requires an account, so these files cannot be fetched automatically.

**Three downloads cover it.** Some datasets bundle every edition into one file with a
version column, and the loader reads that shape directly:

1. [EA Sports FC 24 complete player dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
   — `male_players.csv` spans FIFA 15 to FC 24, covering **five** of the seven editions.
   Save it unchanged as `data/raw/fifa/male_players.csv`.
2. FC 25 — [aniss7/fifa-player-data-from-sofifa-2025-06-03](https://www.kaggle.com/datasets/aniss7/fifa-player-data-from-sofifa-2025-06-03).
   Save `player-data-full-2025-june.csv` as `data/raw/fifa/fc25.csv`.
3. FC 26 — [flynn28/eafc26-player-database](https://www.kaggle.com/datasets/flynn28/eafc26-player-database).
   Save the **men's** file (`EAFC26-Men.csv`) as `data/raw/fifa/fc26.csv`.

> **Not every Kaggle dataset is usable.** Some are raw web scrapes whose columns are CSS
> class names (`odd href`, `swapHeader`) and which identify clubs only by numeric id.
> A dataset is usable only if it has a column of club **names** —
> `mexwell/ea-fc25-player-database` does not, so avoid it.

Otherwise download each edition separately — FIFA
[20](https://www.kaggle.com/datasets/stefanoleone992/fifa-20-complete-player-dataset) ·
[21](https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset) ·
[22](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) ·
[23](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
— and save them in `data/raw/fifa/` under **exactly** these names:

| Season | Edition | Save as |
|---|---|---|
| 2019/20 | FIFA 20 | `data/raw/fifa/fifa20.csv` |
| 2020/21 | FIFA 21 | `data/raw/fifa/fifa21.csv` |
| 2021/22 | FIFA 22 | `data/raw/fifa/fifa22.csv` |
| 2022/23 | FIFA 23 | `data/raw/fifa/fifa23.csv` |
| 2023/24 | EA FC 24 | `data/raw/fifa/fc24.csv` |
| 2024/25 | EA FC 25 | `data/raw/fifa/fc25.csv` |
| 2025/26 | EA FC 26 | `data/raw/fifa/fc26.csv` |

A per-edition file takes precedence over the combined one, so you can mix the two — use
the bundle for the older editions and dedicated files for anything it misses.

Each file needs at least a player name, a club and an overall rating. Age, potential,
value, positions and the six attribute scores (pace, shooting, passing, dribbling,
defending, physical) are used when present. Column *names* do not matter — the loader
recognises the common spellings and reports anything it cannot place.

Then:

```bash
python -m src.data.load_fifa
```

Running it before the files exist prints exactly which ones are missing and where they
belong, so it is safe to use as a checklist. It writes
`data/processed/fifa_players.parquet` — player ratings per season, already filtered to
the 20 clubs that actually played that season.

Pass `--allow-missing` to build from the editions you have while still tracking down a
source for another. Seasons it skips will have no squad-quality features, so it is not
the default.

Goalkeepers carry null pace/shooting/passing/dribbling/defending/physical — FIFA rates
them on separate diving and handling attributes instead. That accounts for roughly 11%
of rows and is expected, not missing data.

Coverage differs by edition. Player name, club, overall, potential and age are present
throughout; the six face stats are missing for 2024/25, and potential and value for
2025/26. Overall rating — the strongest squad-quality signal — is complete for all seven
seasons.

### 7. Match players across the sources

```bash
python -m src.matching.player_names
```

Understat writes `Mohamed Salah`; FIFA writes `M. Salah` or `Mohamed Salah Hamed Ghaly`.
This resolves the two, reaching **98.6% of starting appearances**. Unresolved names go in
`data/manual/player_name_overrides.csv`, which is committed and always takes precedence —
add a row there rather than loosening a matching threshold.

### 8. Build the feature table

```bash
python -m src.features.build
```

Produces `data/final/features.parquet` — 2,660 rows, 99 columns, one per match. Squad
quality from each starting XI, rolling form and expected goals over previous matches,
Elo, rest days, and the home-minus-away difference of each.

**Everything here is knowable before kickoff.** Shots, cards, half-time scores and the
match's own expected goals are deliberately excluded; they appear only as rolling
averages of *earlier* matches.

### 9. Train and compare the models

```bash
python -m src.evaluate.compare --fast
```

Trains every model family and scores it against the bookmaker's closing line, using
walk-forward validation — train on seasons 1..n, test on n+1, never a random split.
`--fast` skips AutoGluon and finishes in seconds; the full run takes around 15 minutes.

| Model | RPS | Accuracy |
|---|---|---|
| **bookmaker (closing)** | **0.1965** | 55.3% |
| gbm-with-odds | 0.2027 | 53.3% |
| poisson-glm-with-odds | 0.2029 | 53.6% |
| poisson-glm | 0.2040 | 53.5% |
| baseline-elo | 0.2051 | 53.1% |
| gbm | 0.2068 | 52.9% |
| dixon-coles | 0.2118 | 50.2% |
| baseline-team-average | 0.2199 | 48.6% |
| baseline-league-average | 0.2346 | 43.1% |

Ranked Probability Score is the headline metric, not accuracy — football outcomes are
*ordered*, and RPS is the only common metric that knows predicting a draw when the home
side wins is a smaller error than predicting an away win. Lower is better.

**Nothing beats the market.** That is the expected outcome, and a model that did beat it
would be worth suspecting before celebrating. The best model lands 0.006 RPS behind a
number anyone can read off a screen for free.

### What you end up with

Five tables in `data/processed/` plus the feature table, roughly 160 MB of source data
behind them:

| Table | Rows | Contents |
|---|---|---|
| `matches.parquet` | 2,660 | Results, shots, cards, referee, opening and closing odds |
| `understat_matches.parquet` | 2,660 | Match-level expected goals |
| `lineups.parquet` | 77,278 | Player appearances — position, minutes, xG, xA, cards |
| `fifa_players.parquet` | 127,930 | Player ratings per season; `in_premier_league` flags the season's 20 clubs |
| `player_map.parquet` | 3,874 | Each Understat player linked to their FIFA entry |
| `final/features.parquet` | 2,660 | The model-ready table — one row per match, 99 columns |

Every stage validates before writing and raises rather than emitting a suspect table:
380 matches per season, 20 teams, 19 home and 19 away each, 11 starters per side, both
sources agreeing on every final score, and 20 flagged clubs per ratings edition.

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

Phases 0–6 complete — all three sources ingested and joined, players matched at 98.6% of
starting appearances, the feature table built, and eight models trained and benchmarked
against the closing line.
See [PLAN.md](PLAN.md) for the full ten-phase build plan and where things stand.

## Layout

```
src/config.py           seasons, paths, source URLs - the single place seasons are defined
src/data/               acquisition and cleaning, one module per source
src/matching/           name reconciliation between sources
src/models/             goals models and the scoreline matrix
src/evaluate/           scoring, walk-forward backtesting, the bookmaker benchmark
src/features/           squad quality, form, Elo, and the feature table
tests/                  unit tests, no network access
data/raw/               downloaded source data (gitignored)
data/processed/         cleaned and joined data (gitignored)
data/final/             model-ready feature table (gitignored)
data/manual/            hand-written name-override files (committed)
```
