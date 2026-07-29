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

Trained on **2019/20 through 2025/26** — 7 seasons, 2,660 matches — and used to predict
whatever is played next.

## Running it locally

Once the data is built, this is all you need. From the repository root — on Windows:

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS and Linux:

```bash
.venv/bin/python -m streamlit run app.py
```

It opens on **http://localhost:8501**. `Ctrl+C` in that terminal stops it.

Those name the interpreter directly, so they work whether or not the virtual environment
is activated. If you *have* activated it ([step 3](#3-activate-it)), the short form does
the same thing:

```bash
streamlit run app.py
```

To refresh the predictions the page shows:

```bash
.venv\Scripts\python.exe -m src.predict.gameweek --replay
```

Then reload the browser. `--replay` re-predicts the last round that was actually played,
which is the only thing available between seasons; drop it once real fixtures exist.

> **Two things that will otherwise cost you ten minutes.**
> `streamlit` is not on your PATH — it lives in the virtual environment, so without
> activating you get *"command not found"*.
> And Streamlit re-runs `app.py` on save but **does not reload imported modules**, so
> after editing anything under `src/` you must restart the server or you will get an
> `ImportError` for a function that plainly exists.

**Never built the data?** Start with *Getting started* below — a fresh clone has no data
at all, and the page will tell you so rather than showing an empty grid.

## Getting started

**Almost no data is stored in this repository** — it is gitignored and rebuilt from
source, the exception being the hand-written override files in `data/manual/`. A fresh
clone therefore needs steps 5 to 8 before anything works, then 9 to 11 to reproduce the
results below, predict a round, and read it.

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
| dixon-coles-squad | 0.2087 | 51.7% |
| dixon-coles | 0.2118 | 50.2% |
| baseline-team-average | 0.2199 | 48.6% |
| baseline-league-average | 0.2346 | 43.1% |

Ranked Probability Score is the headline metric, not accuracy — football outcomes are
*ordered*, and RPS is the only common metric that knows predicting a draw when the home
side wins is a smaller error than predicting an away win. Lower is better.

**Nothing beats the market.** That is the expected outcome, and a model that did beat it
would be worth suspecting before celebrating. The best model lands 0.006 RPS behind a
number anyone can read off a screen for free.

#### Why the report does not use the best model

RPS scores the home / draw / away split, and it rewards hedging. The models that hedge
best also produce the least interesting scorelines:

| Model | RPS | 1-1 most likely | Distinct top scorelines |
|---|---|---|---|
| poisson-glm | **0.2040** | 74% of matches | 9 |
| gbm | 0.2068 | 65% | 11 |
| **dixon-coles-squad** | 0.2087 | **54%** | **11** |
| dixon-coles | 0.2118 | 60% | 11 |

Take Crystal Palace against Arsenal, with the market pricing the away win at 51%.
`poisson-glm` expects 1.27 goals to 1.63 and calls it **1-1**. The Dixon-Coles models
expect around 0.9 to 1.7 and call it **0-1** — which is what the bookmakers show. Lowering
the regularisation does not close the gap; the fixture stays 1-1 even at a quarter of the
penalty, so it is the model's structure rather than its tuning.

Dixon-Coles estimates each club's attack and defence strength directly, so it commits. The
GLM spreads the same information across 85 correlated features and retreats to the average.
**Since the output here is a scoreline with a probability, the report accepts the worse RPS
for a model that says something.**

#### How the player ratings reach the prediction

Plain Dixon-Coles reads results and nothing else, so the FIFA work would not touch the
report at all. `dixon-coles-squad` adds one term to the likelihood — **not** the squad
rating itself, which merely restates what the attack parameter already knows, but how far
today's XI sits from the one that club usually fields:

```
delta = (this XI's mean overall − the club's usual mean) / scale
```

That deviation is the part results cannot see: whether a side is stronger or weaker than
normal *today*. Fitted across seven seasons, a stronger-than-usual XI lifts its own scoring
(`+0.086`) and suppresses the opponent's (`+0.121`), and the model beats plain Dixon-Coles
on every metric while repeating itself less. A club with no history is measured against the
league rather than itself, so a weak promoted side starts below average instead of at it.

### 10. Predict the next round

```bash
python -m src.predict.gameweek
```

Trains on everything known, then predicts fixtures that have not been played, writing
`data/final/predictions.json`. Uses `dixon-coles-squad` by default — see
[why the report does not use the best model](#why-the-report-does-not-use-the-best-model)
— and `--model poisson-glm` switches to the more accurate outcome predictor:

```
        Man City vs Aston Villa      2026-05-24
      2-0 (10%) · 2-1 (10%) · 1-1 (10%)
      Home 64% | Draw 20% | Away 16%
```

Between seasons the Premier League fixture feed is empty, so there is nothing to predict.
Two ways round that — list fixtures by hand in `data/manual/upcoming_fixtures.csv`, or:

```bash
python -m src.predict.gameweek --replay
```

which predicts the most recent round that *was* played, so the output can be checked
against what actually happened.

#### Keeping squads current

Lineups are not known until an hour before kickoff, so the expected XI defaults to the
eleven a club has started most often recently. During a transfer window that goes stale
quickly, and ratings make it worse: they are published once a year, so a July signing is
still listed at their old club — the rating is right, the club is wrong.

Two committed files fix that, and both record **changes, not whole squads**:

| File | One row per |
|---|---|
| `data/manual/squad_changes.csv` | Transfer. A blank `team` means the player left the league |
| `data/manual/player_ratings_manual.csv` | Player with no FIFA entry at all |

Moving a player is one line and a re-run — no code change, and an unrecognised name is
reported rather than silently ignored. Both files feed the expected XI that squad-quality
features are built from, so an edit changes the prediction.

A new season also starts before its own ratings edition exists: 2026/27 begins in August,
EA FC 27 arrives in late September. `UPCOMING_SEASON` in `src/config.py` therefore points
at the newest edition that does exist. Change that one line when the new one is published.

### 11. Read the report

```bash
streamlit run app.py
```

See [Running it locally](#running-it-locally) for the shortcuts and the two things that
otherwise waste ten minutes.

**Three fixtures to a row**, styled after a printed league table and teletext results
page — hard edges, a monospaced grid, and a palette narrow enough to have been printed
with three inks. Cards stack to one column on a phone.

Each card carries the most likely scorelines, the home / draw / away split, and the
bookmaker's line beside it. Where the model and the market disagree by more than ten
points, the card says so — the only genuinely interesting thing a model can offer once a
market exists. Not *who wins*, since the odds already answer that, but *where do I
disagree*.

Each card has an **⚽ EXPECTED XI** button that opens both sides laid out on a pitch —
attackers meeting at the halfway line, the shape (3-5-2) and each player's FIFA rating
shown, with the team average. That is where the squad-quality signal becomes visible
rather than staying an input to the model.

The overlay floats above the page and closes on the ✕ or the backdrop, so the grid never
moves and nothing reruns. It is labelled as what it is: the players a club has started
most often lately, not a team sheet.

A collapsed **HOW TO READ THIS** panel explains every part of a card, using live samples
of the real components rather than descriptions of them — so the guide cannot drift out of
step with what is on screen. The stat bar carries hover explanations for the same reason.

Club badges are **pixel kit patterns, not crests**: real badges are trademarked and not
ours to redistribute, so each club gets a twelve-pixel shirt in its own colours wearing
the pattern it is known for — Newcastle striped black and white, Wolves gold on black,
West Ham with a claret sash. They are generated as inline SVG, so the page needs no image
files and no network.

```
Crystal Palace vs Arsenal                      Most likely: Away win
  0-1  ████████████  14%
  0-2  ██████████    12%
  1-1  ██████████    12%

  Home 17%  (-7 vs market)   Draw 25%   Away 58%  (+7 vs market)
  Bookmaker: Home 24% · Draw 26% · Away 51%
```

The page also says so when it is showing a replay rather than upcoming fixtures, and when
one scoreline tops most of the cards — 1-1 recurs because it stays the single most likely
result until one side is expected to score around 2.4 goals, so it dominates every round
that lacks a mismatch. The home / draw / away split is where fixtures actually differ.

**Editing anything under `src/` needs a server restart.** Streamlit re-runs `app.py` but
does not reload imported modules, so a stale copy raises `ImportError` for a function that
exists and whose tests pass.

### What you end up with

Five tables in `data/processed/`, then the feature table and the predictions in
`data/final/`, with roughly 160 MB of source data behind them:

| File | Rows | Contents |
|---|---|---|
| `processed/matches.parquet` | 2,660 | Results, shots, cards, referee, opening and closing odds |
| `processed/understat_matches.parquet` | 2,660 | Match-level expected goals |
| `processed/lineups.parquet` | 77,278 | Player appearances — position, minutes, xG, xA, cards |
| `processed/fifa_players.parquet` | 127,930 | Player ratings per season; `in_premier_league` flags the season's 20 clubs |
| `processed/player_map.parquet` | 3,874 | Each Understat player linked to their FIFA entry |
| `final/features.parquet` | 2,660 | The model-ready table — one row per match, 99 columns |
| `final/predictions.json` | one round | Scorelines and outcome probabilities for the fixtures predicted |

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

**Complete.** All ten phases: raw downloads through player matching, features and eight
benchmarked models to a report of predicted scorelines. 247 tests, none needing network
or data.
See [PLAN.md](PLAN.md) for the full ten-phase build plan and where things stand.

## Layout

```
src/config.py           seasons, paths, source URLs - the single place seasons are defined
src/data/               acquisition and cleaning, one module per source
src/matching/           name reconciliation between sources
src/models/             goals models and the scoreline matrix
src/evaluate/           scoring, walk-forward backtesting, the bookmaker benchmark
src/features/           squad quality, form, Elo, and the feature table
src/predict/            fixtures, expected XIs, and predictions for unplayed matches
src/report/             shaping predictions for display
app.py                  the Streamlit report
tests/                  unit tests, no network access
data/raw/               downloaded source data (gitignored)
data/processed/         cleaned and joined data (gitignored)
data/final/             feature table and predictions (gitignored)
data/manual/            hand-written overrides: names, squad changes, fixtures (committed)
```
