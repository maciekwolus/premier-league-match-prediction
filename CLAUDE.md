# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicts Premier League **scorelines with probabilities** (`1-1 (11%) · 2-1 (9%)`), not
just win/draw/loss. Pipeline: match results + lineups + FIFA player ratings → per-match
feature table → goals models → scoreline probability matrix → Streamlit report.

`PLAN.md` is the source of truth for scope and what comes next. It defines ten phases;
read it before starting work. **Phases 0–7 are complete: the pipeline predicts fixtures
that have not been played.** Next is Phase 8, the Streamlit report.

Backtest results, walk-forward over 2,280 matches (RPS, lower is better):

| | RPS |
|---|---|
| bookmaker (closing) | **0.1965** |
| gbm-with-odds | 0.2027 |
| poisson-glm-with-odds | 0.2029 |
| poisson-glm | 0.2040 |
| baseline-elo | 0.2051 |
| gbm | 0.2068 |
| dixon-coles | 0.2118 |
| baseline-team-average | 0.2199 |
| baseline-league-average | 0.2346 |

**Nothing beats the market, and that is the expected result** — a model that did should
be suspected of leakage before being believed. Note that plain Elo beats AutoGluon on 85
features: treat the baselines as the bar any new model must clear before it has earned
its complexity.

What exists in `data/processed/` after a full build:

| Table | Rows | Contents |
|---|---|---|
| `matches.parquet` | 2,660 | results, match statistics, referee, opening and closing odds |
| `understat_matches.parquet` | 2,660 | match-level expected goals |
| `lineups.parquet` | 77,278 | player appearances: position, minutes, xG, xA, cards |
| `fifa_players.parquet` | 127,930 | player ratings per season, every club, `in_premier_league` flags the season's 20 |
| `player_map.parquet` | 3,874 | Understat player-season → FIFA player, with the rule that matched it |

And `data/final/features.parquet`: 2,660 rows, 99 columns — one per match, nothing
post-kickoff.

## Commands

Use the venv interpreter directly; activation is unnecessary.

```
.venv/Scripts/python.exe -m pytest                                   # all tests
.venv/Scripts/python.exe -m pytest tests/test_config.py              # one file
.venv/Scripts/python.exe -m pytest tests/test_config.py::test_slug   # one test
.venv/Scripts/python.exe -m pytest -k slugify                        # by keyword
.venv/Scripts/python.exe -m ruff check .                             # lint
.venv/Scripts/python.exe -m ruff format .                            # format
```

Rebuild the match data (downloads are cached; `--force` re-fetches):

```
.venv/Scripts/python.exe -m src.data.fetch_matches
.venv/Scripts/python.exe -m src.data.clean_matches
.venv/Scripts/python.exe -m src.data.fetch_lineups   # ~2,660 requests, run in background
.venv/Scripts/python.exe -m src.data.clean_lineups
.venv/Scripts/python.exe -m src.data.load_fifa       # needs hand-placed CSVs, see below
.venv/Scripts/python.exe -m src.matching.player_names
.venv/Scripts/python.exe -m src.features.build
.venv/Scripts/python.exe -m src.evaluate.compare          # all models, ~15 min
.venv/Scripts/python.exe -m src.evaluate.compare --fast   # skips AutoGluon, seconds
.venv/Scripts/python.exe -m src.predict.gameweek          # next round
.venv/Scripts/python.exe -m src.predict.gameweek --replay # last known round, as a check
```

**`--replay` is how the prediction path gets exercised out of season.** It predicts the
most recent round of matches that were actually played, so the output can be scored
against reality. The Premier League fixture feed is empty between seasons, which means
the live path cannot otherwise be run at all from June to August.

**FIFA ratings are not downloadable here.** Kaggle requires an account, and credentials
are the user's to handle. Files are placed by hand in `data/raw/fifa/`, either one per
edition named `fifa20…fifa23`, `fc24…fc26` (`Season.fifa_slug`), or a multi-edition
`male_players.csv` carrying a `fifa_version` column — the FC 24 dataset bundles FIFA 15
through FC 24 that way. A per-edition file wins over the bundle, so the two can be mixed.
`load_fifa` run without them prints which editions are missing and exits 1.

Tests never hit the network and never read `data/` — they build synthetic seasons — so
they stay meaningful when upstream sources change, and a green run verifies a fresh
environment before the slow data build.

## Architecture

**Data flows one way through three directories**, each stage written to disk so any
stage can be rebuilt without redoing the ones before it:

```
data/raw/        as downloaded, never modified in place
data/processed/  cleaned and joined, one parquet per source
data/final/      the model-ready feature table
data/manual/     hand-written override files (the only committed data)
```

`.gitignore` excludes all of `data/` except `data/manual/*.csv`. Those files record
decisions rather than data — losing them would mean re-deriving each one by hand — so
they are committed deliberately. `player_name_overrides.csv` is consulted before the
matching cascade runs and always wins; add a row there rather than loosening a threshold.

**Hand-maintained files record *changes*, never whole state.** Phase 7 adds squad edits for
the open transfer window, and the same rule applies: a file that restates twenty full
squads to move one player will go stale, and a stale file that looks authoritative is worse
than no file. Adding a transfer should be one line and a rebuild.

**`src/config.py` is the only place seasons are defined.** Each upstream source names
seasons differently, so the `Season` dataclass carries every identifier at once —
`code` for football-data (`1920`), `understat` (`2019`), `fifa_edition` (`FIFA 20`).
Adding a season means adding one row to `SEASONS` and touching nothing else. Note EA
renamed the game series mid-range: FIFA 20–23, then EA FC 24–26.

**`match_id` is the contract between phases.** Built in `src/data/clean_matches.py` as
`{season}_{YYYYMMDD}_{home}_{away}` (e.g. `2019_20_20190809_liverpool_norwich`). Every
later source — lineups, player ratings, features — joins onto it. It is deliberately
human-readable so failed joins can be diagnosed by eye. Changing its format invalidates
every downstream parquet.

**Cross-source joins key on `(season, home_team, away_team)`, not on date.** A home/away
pairing occurs exactly once per season, so it is unique, and it avoids the timezone and
date-format fragility of date joins. Dates and final scores are then cross-checked
*afterwards* as independent evidence the join is right — 2,660 matching scores is not
something a wrong join produces by accident. Use this shape for any new source.

**`player_map.parquet` is how a lineup reaches a rating.** `lineups.parquet` names players
as Understat spells them; `fifa_players.parquet` as FIFA does. The map joins
`(season, team, understat_player)` to a `fifa_player_name`, so squad-quality features go
lineups → player_map → fifa_players, never lineups → fifa_players directly. Around 1.4% of
starting appearances have no match and land as nulls, which aggregations must expect.

**Cleaners validate before writing and raise rather than emit a suspect table.**
`clean_matches.build()` is strict by default; `validate_season` returns a list of
problems and an empty list means clean. Follow this shape for new sources.

**Every model implements the same two-method contract** in `src/models/base.py`:
`fit(train)` then `predict(test) -> (lambda_home, lambda_away)`. Nothing downstream knows
which model produced a prediction, so the scoreline matrix, scoring and the report are
identical for all of them — which is what makes the comparison fair. A new model needs
one file and one entry in `evaluate/compare.build_models`.

`tests/test_model_contract.py` runs every model through the same checks, the important
one being that **rewriting the test set's results must not change any prediction**. That
is the only check that catches a leaking model, because a leaking model's output is
otherwise perfectly well-formed.

**Upcoming fixtures are appended to history with blank results**, then run through the
same feature builder. A fixture that has not happened has no result to leak and the
rolling windows already only look backwards, so prediction needs no separate code path.
Two consequences worth remembering: `add_elo` must record a pre-match rating and skip the
update when goals are missing, and `predict.gameweek.features_for` drops any history for
the fixtures being predicted before appending them — otherwise a `match_id` appears twice
and the join fans out into several conflicting predictions for one fixture.

**`UPCOMING_SEASON` in `config.py` is deliberately outside `SEASONS`.** Every ingestion
stage treats that tuple as "seasons with complete data" and would demand a results file
and a ratings edition that do not exist yet. Move the season into `SEASONS` once it has
finished.

**Per-team features are computed on a team-match table**, two rows per fixture rather
than one. `features/form.py` builds it, and `features/build.py` pivots it back to one row
per match with `home_`/`away_`/`diff_` columns. The shape is what makes the leakage rule
enforceable: a team's history is a single chronological series whether it played home or
away, so one `.shift(1)` covers every rolling feature instead of each needing its own
correct handling. Phase 7 must build the same table for upcoming fixtures.

The Streamlit app is created when its phase begins, not as an empty stub ahead of time.

## Rules that matter

**No data leakage.** Features may only use information available *before* kickoff. Shots,
cards and half-time scores from the match being predicted are post-match facts. They are
kept in `matches.parquet` because rolling averages over *previous* matches are valuable,
but using them for their own match produces a model that scores brilliantly in backtests
and fails on Saturday. Concretely: rolling features must shift before the window
(`.shift(1).rolling(n)`), or every match lands inside its own average.

This is enforced two ways, and the second is the one that matters.
`features.build.FORBIDDEN` names post-match columns so a careless passthrough is caught
by name — cheap, but blind to a window that includes its own row, because that column
has an innocent name. So `test_form.py` **changes a match's score and asserts that
match's own features do not move**, with a mirror test that later matches' features
*do*. Any new feature belongs under both.

**Healthy correlation with goal difference is about 0.43**, from squad-quality
difference — roughly what the closing odds themselves manage. A feature correlating much
higher is leaking, not clever.

**Validate walk-forward, never randomly.** Train on seasons 1..n, test on n+1. A random
train/test split lets the model see the future.

**Bookmaker odds are the benchmark, not a free feature.** The headline metric is Ranked
Probability Score against closing odds (`odds_close_*` — closing, not opening; they
absorb team news). A variant using odds as features is trained separately to measure
market signal.

**Fail loudly on data joins.** Every season must yield exactly 380 matches, 20 teams,
19 home and 19 away per team. Team names differ between sources (`Man United` vs
`Manchester United`), so joins go through explicit mapping tables and assert their row
counts. Silently dropped rows corrupt everything downstream and are nearly impossible
to trace later.

## Data quirks worth knowing

- **2019/20 ran to 26 July 2020** (Covid suspension). Rest-day features will look absurd
  across that gap, and the restart weakened home advantage. Flag the season rather than
  treating it as normal.
- **Raw CSVs grow every season** — 106 columns in 2019/20, 132 in 2025/26, as
  football-data adds bookmakers. `COLUMN_MAP` takes only the subset present in all
  seasons; a missing column raises immediately rather than producing silent nulls.
- **football-data team names are internally consistent** (28 distinct across 7 seasons),
  so no mapping is needed within this source, and they are the canonical spelling every
  other source is translated into.
- **`understatapi` pins old transitive deps** (urllib3 1.26.5, idna 2.10), which suggests
  light maintenance. If it breaks, the fallbacks are the `soccerdata` library or scraping
  Understat's embedded JSON directly. As of July 2026 it works and returned all 2,660
  rosters without a single failure.
- **Understat and football-data agree on 22 of 28 team names.** Only the six long-form
  names differ, mapped explicitly in `src/matching/team_names.py`. An unmapped name
  raises `UnknownTeamError` rather than dropping the fixture.
- **Understat starters are `position != "Sub"`** and come to exactly 11 per side on all
  2,660 matches. Its `time` field caps at 90, so stoppage time is not counted.
- **Ratings CSVs come from different Kaggle authors per edition**, so column names vary.
  `load_fifa.COLUMN_ALIASES` lists every known spelling and matches case-insensitively;
  a missing *required* column raises, a missing *optional* one is reported and left null.
- **Ratings cover every club in the world and none are dropped.** They are a September
  snapshot, so a January signing is still listed at their old club — Jarrod Bowen sits at
  Hull City in FIFA 20. An earlier version filtered each edition to that season's twenty
  clubs, which deleted those players outright and left them unmatchable in Phase 4.
  `in_premier_league` flags the season's twenty instead; squad-quality features filter on
  the flag, name matching uses the full pool.
- **The season's twenty come from `matches.parquet`, not from "clubs ever in the Premier
  League".** Leeds and Sunderland appear in every edition regardless of division, so the
  looser test yields 28. An unmapped club would be invisible — it just looks like one of
  the many non-Premier-League rows — so the guard is asserting exactly 20 flagged clubs
  per edition. A club whose spelling we missed shows up as 19.
- **Understat serves names HTML-escaped** — `Dara O&#039;Shea`. Unescaped in
  `clean_lineups`; 12 names across 406 rows were affected.
- **`Ø`, `Ł`, `ß` and friends survive NFKD normalisation**, because they are distinct
  letters rather than a base plus a mark, and then get stripped as non-ASCII. `Ødegaard`
  became `degaard` and matched nothing. `player_names.TRANSLITERATIONS` handles them
  explicitly; add to it rather than reaching for a fuzzier threshold.
- **EA FC 26 abbreviates club names** (`Man Utd`, `Newcastle Utd`, `Spurs`, bare
  `Brighton`) where earlier editions spell them out, and the same file contains
  `Newcastle Jets` and `Notts County` — so the mapping is exact-match, never fuzzy.
- **Goalkeepers have null pace/shooting/etc.** by design; FIFA rates them on separate
  `gk_` attributes. About 11% of rows, and not missing data. `features/squad.py` averages
  face stats over outfielders only, which is the intended quantity — any new aggregation
  should do the same rather than filling zeros.
- **Attribute coverage is uneven across editions and features must tolerate it.**
  `overall`, `potential`, `age` and `club` exist everywhere. The six face stats are
  absent for 2024/25, and `potential`/`value_eur` are absent for 2025/26. Squad-quality
  features should lean on `overall`, which is complete, and treat the rest as optional.
- **The face stats are taken as a complete set or not at all.** Some SoFIFA exports
  carry *detailed* skills instead — a column named `dribbling` meaning ball control,
  alongside `acceleration`/`sprint_speed` rather than `pace`. Accepting that single
  column would put a different quantity in the same field for one season.
- **A new season starts before its ratings edition exists.** 2026/27 kicks off in August
  2026; EA FC 27 arrives in late September. Carry the previous edition forward by pointing
  `Season.fifa_edition` at it, and swap when the new one lands. Meanwhile the summer
  transfer window is open, so signings are listed at their old club — ratings right, club
  wrong — which is what the Phase 7 squad-change file exists to correct.
- **2024/25 ratings are an end-of-season snapshot (June 2025), not a release-time one.**
  Every other season uses ratings published at kickoff. A June snapshot partly reflects
  how players performed *during* 2024/25, so it is mildly leaky for that season alone.
  Watch for 2024/25 scoring anomalously well in the Phase 6 walk-forward backtest; if it
  does, that is the cause.
- **SoFIFA returns 403 and Kaggle needs auth**, so there is no unattended path to this
  data. FIFA Index (fifaindex.com) is reachable and covers all seven editions, but its
  list pages carry only overall/potential — attributes need one request per player.

## Workflow

- **One branch per phase, pushed, then a PR** — the user reviews before merge. Do not
  commit directly to `main`.
- `gh` is installed and authenticated, but may be missing from `PATH` in an
  already-running session; call `C:\Program Files\GitHub CLI\gh.exe` if `gh` is not found.
- Run lint, format and the full test suite before opening a PR.
