# Commands

Every command this project accepts, and what it does. For *why* any of it is built the
way it is, read [README.md](README.md); for the rules a change has to respect, read
[CLAUDE.md](CLAUDE.md).

**Every command here must be run from the project folder.** Not some of them — all of
them. `cd` there first, every time you open a new terminal:

```bash
cd C:\repositories\premier-league-match-prediction
```

> **If you skip that, PowerShell blames the wrong thing.** You get
> *"The module '.venv' could not be loaded"* and a suggestion to run `Import-Module .venv`,
> which is nonsense — there is no module. It means only that there is no `.venv` folder
> **here**, because you are in the wrong directory. `cd` and re-run.

**Every command below names the interpreter directly** (`.venv\Scripts\python.exe`), so it
works whether or not the virtual environment is activated. If you have activated it, plain
`python` and `streamlit` do the same thing. On macOS or Linux swap the prefix for
`.venv/bin/python`.

---

## Run the site

```bash
cd C:\repositories\premier-league-match-prediction
```
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Opens **http://localhost:8501**. `Ctrl+C` in that terminal stops it.

Once more than one round is stored, a **gameweek selector** appears at the top of the page.
Rounds that have been played show the final score on each card and a season-to-date
scorecard: how often the call was right, and our RPS beside the bookmaker's.

That is the whole thing — the report opens on the newest round stored under
`data/final/rounds/`, which is already on disk, so nothing needs rebuilding first. To
predict another round:

```bash
.venv\Scripts\python.exe -m src.predict.gameweek --replay
```

Then reload the browser.

> **Two things that otherwise cost ten minutes.**
> `streamlit` is not on your PATH — it lives in the virtual environment, which is why the
> commands here spell out the interpreter.
> Streamlit re-runs `app.py` when you save it but **does not reload imported modules**, so
> after editing anything under `src/` you must restart the server, or you get an
> `ImportError` for a function that is plainly there and whose tests pass.

---

## Predict

```bash
.venv\Scripts\python.exe -m src.predict.gameweek
```

Predicts the next round of fixtures and stores it under `data/final/rounds/`, which is what
the report renders.

| Flag | Effect |
|---|---|
| `--replay` | Predict the last round that was actually **played**, so the output can be checked against reality |
| `--model NAME` | `dixon-coles-squad` (default), `poisson-glm`, `dixon-coles`, `baseline-elo` |
| `--offline` | Never download fixtures; use `data/manual/upcoming_fixtures.csv` only |
| `--force` | Replace a round already stored. Without it, re-running a stored round exits 1 |

**A stored round is not rewritten.** Predicting a gameweek that already has a file fails
rather than overwriting it, because the archive exists to show what the model said *before*
those matches were played — and a record that can be quietly replaced afterwards proves
nothing. `--force` is there for when you genuinely mean it.

**`--replay` is the only way to run this between seasons.** The Premier League fixture
feed is empty from June to August, so without it there is nothing to predict. The report
says which mode produced what you are looking at, because the cards look identical either
way.

**The default model is not the best-scoring model, deliberately.** `poisson-glm` has the
better RPS (0.2040 against 0.2087) but calls 1-1 in 74% of matches, because RPS rewards
hedging and the GLM hedges. When the product is a scoreline, that is a failure whatever
the score says. The reasoning is in full at `DEFAULT_MODEL` in
[src/predict/gameweek.py](src/predict/gameweek.py).

---

## Compare the models

```bash
.venv\Scripts\python.exe -m src.evaluate.compare --fast
```

Backtests every model walk-forward — train on seasons 1..n, test on n+1, never a random
split — and prints a table ranked by Ranked Probability Score, lower being better.

| Flag | Effect |
|---|---|
| `--fast` | Skip AutoGluon, which dominates the runtime. Seconds instead of ~15 minutes |
| `--save` | Also write per-match predictions to disk |

Without `--fast` it runs all nine models plus the bookmaker benchmark and takes about
fifteen minutes.

**Expect the bookmaker to win.** It scores 0.1965 and no model here beats it. A model that
did should be suspected of leakage before it is believed.

---

## Rebuild the data

Only needed on a fresh clone, or when a new season's results land. Each stage writes to
disk, so you can re-run one without redoing the ones before it. **Run them in this order** —
each depends on the last.

```bash
.venv\Scripts\python.exe -m src.data.fetch_matches
```
```bash
.venv\Scripts\python.exe -m src.data.clean_matches
```
```bash
.venv\Scripts\python.exe -m src.data.fetch_lineups
```
```bash
.venv\Scripts\python.exe -m src.data.clean_lineups
```
```bash
.venv\Scripts\python.exe -m src.data.load_fifa
```
```bash
.venv\Scripts\python.exe -m src.matching.player_names
```
```bash
.venv\Scripts\python.exe -m src.features.build
```

| Stage | What it does | Flags |
|---|---|---|
| `fetch_matches` | Download results and odds | `--season "2025/26"` (repeatable), `--force` |
| `clean_matches` | Validate and join into `matches.parquet` | — |
| `fetch_lineups` | Download Understat rosters | `--season`, `--force`, `--stage {matches,rosters,all}`, `--delay N` |
| `clean_lineups` | Build `lineups.parquet` | — |
| `load_fifa` | Read the hand-placed ratings CSVs | `--allow-missing` |
| `player_names` | Match Understat names to FIFA names | — |
| `build` | Assemble `data/final/features.parquet` | — |

**`fetch_lineups` makes about 2,660 requests** and takes a while — run it in the background
and leave it. Downloads are cached, so a re-run is cheap unless you pass `--force`.

**`load_fifa` will not run without files you have to place by hand.** The ratings come from
Kaggle, which needs an account, so they cannot be downloaded here — see
[README.md](README.md) step 6 for which files and where. Run without them it prints the
missing editions and exits 1. `--allow-missing` builds from whatever is present, which is
useful for a partial check but produces a feature table with holes.

---

## Keep the squads current

Four hand-edited files under `data/manual/`. They are the only committed data, because each
one records a **decision** rather than a fact that could be re-derived.

**Each records a *change*, never whole state.** A file that restates twenty squads to move
one player goes stale, and a stale file that looks authoritative is worse than no file.
Adding a transfer should be one line plus a rebuild.

| File | Columns | Use it when |
|---|---|---|
| `upcoming_fixtures.csv` | `date,home_team,away_team` | The fixture feed has nothing (between seasons) and you want to predict specific matches |
| `squad_changes.csv` | `season,fifa_player_name,team,note` | A player transferred and the ratings file still lists their old club |
| `player_ratings_manual.csv` | `season,fifa_player_name,overall,age,position,note` | A player has no FIFA entry at all — a new signing from another league |
| `player_name_overrides.csv` | `season,understat_player,fifa_player_name,confidence,reason` | The name matcher missed a pair, e.g. `Chicharito` → `J. Hernández` |

After editing any of them:

```bash
.venv\Scripts\python.exe -m src.features.build
```
```bash
.venv\Scripts\python.exe -m src.predict.gameweek --replay
```

**Add a row to `player_name_overrides.csv` rather than loosening a matching threshold.**
It is consulted before the cascade runs and always wins, so it fixes one player without
putting every other match at risk.

---

## Development

```bash
.venv\Scripts\python.exe -m pytest
```

| Command | Scope |
|---|---|
| `-m pytest` | All 387 tests, about fifteen seconds |
| `-m pytest tests/test_config.py` | One file |
| `-m pytest tests/test_config.py::test_slug` | One test |
| `-m pytest -k slugify` | Everything matching a keyword |
| `-m ruff check .` | Lint |
| `-m ruff format .` | Format |

### Refresh the README screenshots

With the report already running in another terminal:

```bash
.venv\Scripts\python.exe tools/screenshots.py docs/screenshots
```

Drives the Chrome already installed on the machine over the DevTools Protocol and rewrites
both images in `docs/screenshots/`. Run it after any change to the report's look — the
alternative is screenshots that quietly stop matching the page.

**The tests never touch the network and never read `data/`** — they build synthetic
seasons. So they stay meaningful when an upstream source changes, and a green run on a
fresh clone tells you the environment is right before you spend an hour on the data build.

Run lint, format and the full suite before opening a PR.

---

## Getting unstuck

| Symptom | Cause |
|---|---|
| `The module '.venv' could not be loaded` | **You are in the wrong folder.** PowerShell's message is misleading — there is no module. `cd` to the project folder and re-run |
| `streamlit: command not found` | The venv is not activated. Use `.venv\Scripts\python.exe -m streamlit run app.py` |
| `ImportError` for a function that exists | Streamlit did not reload the module. Restart the server |
| `FileNotFoundError` on a parquet | That build stage has not run. See the order above |
| `load_fifa` exits 1 | The Kaggle CSVs are not in `data/raw/fifa/` |
| The report shows nothing | No round is stored under `data/final/rounds/`. Run `src.predict.gameweek --replay` |
| Most cards say 1-1 | Largely real — 1-1 tops about 60% of matches on the default model, and more than that in a small round. If it is *literally* every card, check `--model`: `poisson-glm` hedges to 74% |
| The report shows last season | That is `--replay`, and the banner says so |
