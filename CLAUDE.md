# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicts Premier League **scorelines with probabilities** (`1-1 (11%) · 2-1 (9%)`), not
just win/draw/loss. Pipeline: match results + lineups + FIFA player ratings → per-match
feature table → goals models → scoreline probability matrix → Streamlit report.

`PLAN.md` is the source of truth for scope and what comes next. It defines ten phases;
read it before starting work. **Phases 0–4 are complete: all three sources are joined,
and players are matched across them at 98.6% of starting appearances.** Next is Phase 5,
the feature table.

What exists in `data/processed/` after a full build:

| Table | Rows | Contents |
|---|---|---|
| `matches.parquet` | 2,660 | results, match statistics, referee, opening and closing odds |
| `understat_matches.parquet` | 2,660 | match-level expected goals |
| `lineups.parquet` | 77,278 | player appearances: position, minutes, xG, xA, cards |
| `fifa_players.parquet` | 127,930 | player ratings per season, every club, `in_premier_league` flags the season's 20 |
| `player_map.parquet` | 3,874 | Understat player-season → FIFA player, with the rule that matched it |

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
```

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
something a wrong join produces by accident. Follow this shape for the FIFA join too.

**Cleaners validate before writing and raise rather than emit a suspect table.**
`clean_matches.build()` is strict by default; `validate_season` returns a list of
problems and an empty list means clean. Follow this shape for new sources.

Packages (`src/features/`, `src/models/`, …) are created when their phase begins, not
as empty stubs ahead of time.

## Rules that matter

**No data leakage.** Features may only use information available *before* kickoff. Shots,
cards and half-time scores from the match being predicted are post-match facts. They are
kept in `matches.parquet` because rolling averages over *previous* matches are valuable,
but using them for their own match produces a model that scores brilliantly in backtests
and fails on Saturday. Concretely: rolling features must shift before the window
(`.shift(1).rolling(n)`), or every match lands inside its own average.

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
  so no mapping is needed within this source. Understat and FIFA will each need one.
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
- **An unmapped FIFA club would vanish silently**, because the ratings files list every
  club in the world and non-Premier-League ones are dropped on purpose. The guard is
  asserting exactly 20 clubs per edition — a club we failed to map shows up as 19.
- **Filter ratings to the clubs that played *that season***, taken from
  `matches.parquet`, not to "clubs ever in the Premier League". Leeds and Sunderland
  appear in every edition regardless of division, so the looser filter yields 28 clubs.
  This is a *flag* (`in_premier_league`), not a filter — see the next point.
- **Ratings are a September snapshot, so January signings sit at their old club.**
  Jarrod Bowen is at Hull City in FIFA 20. Dropping non-Premier-League rows deleted those
  players outright and made them unmatchable; `fifa_players.parquet` therefore keeps every
  club and flags the Premier League subset. Squad-quality features filter on the flag.
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
  `gk_` attributes. About 11% of rows. Aggregations in Phase 5 must not treat this as
  missing data.
- **Attribute coverage is uneven across editions and features must tolerate it.**
  `overall`, `potential`, `age` and `club` exist everywhere. The six face stats are
  absent for 2024/25, and `potential`/`value_eur` are absent for 2025/26. Squad-quality
  features should lean on `overall`, which is complete, and treat the rest as optional.
- **The face stats are taken as a complete set or not at all.** Some SoFIFA exports
  carry *detailed* skills instead — a column named `dribbling` meaning ball control,
  alongside `acceleration`/`sprint_speed` rather than `pace`. Accepting that single
  column would put a different quantity in the same field for one season.
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
