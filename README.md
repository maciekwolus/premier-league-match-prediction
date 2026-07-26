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

## Getting started

**No data is stored in this repository** — it is all gitignored and rebuilt from source.
A fresh clone therefore needs step 5 before anything works.

### Prerequisites

| | |
|---|---|
| Python | 3.12 or newer (`python --version`) |
| Disk | ~500 MB — 420 MB virtual environment, 50 MB data |
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

Kaggle requires an account, so these seven files cannot be fetched automatically.
Download the player database for each edition and save it in `data/raw/fifa/` under
**exactly** these names:

| Season | Edition | Save as |
|---|---|---|
| 2019/20 | FIFA 20 | `data/raw/fifa/fifa20.csv` |
| 2020/21 | FIFA 21 | `data/raw/fifa/fifa21.csv` |
| 2021/22 | FIFA 22 | `data/raw/fifa/fifa22.csv` |
| 2022/23 | FIFA 23 | `data/raw/fifa/fifa23.csv` |
| 2023/24 | EA FC 24 | `data/raw/fifa/fc24.csv` |
| 2024/25 | EA FC 25 | `data/raw/fifa/fc25.csv` |
| 2025/26 | EA FC 26 | `data/raw/fifa/fc26.csv` |

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
Premier League squads.

Every stage validates before writing and raises rather than emitting a suspect table:
380 matches per season, 20 teams, 19 home and 19 away each, 11 starters per side, and
both sources agreeing on every final score.

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

Phases 0–3 complete — skeleton, match results, lineups with expected goals, and the
player-ratings loader (which stays idle until the Kaggle CSVs are added in step 6).
See [PLAN.md](PLAN.md) for the full ten-phase build plan and where things stand.

## Layout

```
src/config.py           seasons, paths, source URLs - the single place seasons are defined
src/data/               acquisition and cleaning, one module per source
src/matching/           name reconciliation between sources
tests/                  unit tests, no network access
data/raw/               downloaded source data (gitignored)
data/processed/         cleaned and joined data (gitignored)
data/final/             model-ready feature table (gitignored)
data/manual/            hand-written name-override files (committed)
```
