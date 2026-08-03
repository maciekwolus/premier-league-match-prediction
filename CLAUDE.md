# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicts Premier League **scorelines with probabilities** (`1-1 (11%) · 2-1 (9%)`), not
just win/draw/loss. Pipeline: match results + lineups + FIFA player ratings → per-match
feature table → goals models → scoreline probability matrix → Streamlit report.

**Five docs, five jobs — put a change in the right one.** `README.md` is the shop window:
what this is, why it exists, screenshots, results. `SETUP.md` is the end-to-end install and
run guide. `COMMANDS.md` is the flag-by-flag reference and the error-message table.
`SKILLS.md` indexes the project skills and says when to reach for each. This file is the
operating manual for working *on* it. A run command belongs in SETUP.md and COMMANDS.md,
not in the README, which links to them instead.

`PLAN.md` is the source of truth for scope and what comes next; read it before starting
work. It runs in two stages. **Stage one (Phases 0–9) built the pipeline** from raw
downloads to a Streamlit report. **Stage two (Phases 11–15) made it survive a live
season**: predictions archived per gameweek and never rewritten, a browsable history
scored against reality, suspensions derived from cards already on disk, squads for a
season whose ratings edition does not exist yet, and the skills. Both are complete and
441 tests cover them. Phase 16 is planned and deliberately optional. There is no Phase 10
— the numbering skips it so the two stages stay visually distinct.

**The project now predicts a season it has no results for.** Training runs on 2019/20
through 2025/26; the live target is 2026/27, whose twenty clubs and 380 fixtures come from
the Fantasy Premier League API rather than from `matches.parquet`, which is empty for a
season nobody has played.

Backtest results, walk-forward over 2,280 matches (RPS, lower is better):

| | RPS |
|---|---|
| bookmaker (closing) | **0.1965** |
| gbm-with-odds | 0.2027 |
| poisson-glm-with-odds | 0.2029 |
| poisson-glm | 0.2039 |
| baseline-elo | 0.2051 |
| gbm | 0.2068 |
| dixon-coles-squad | 0.2087 |
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
| `fpl_fixtures.parquet` | 380 | the upcoming season's schedule with *official* gameweek numbers |

And in `data/final/`: `features.parquet` — 2,660 rows, 99 columns, one per match and
nothing post-kickoff — plus `rounds/<season>/gwNN.json`, the prediction archive.

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
.venv/Scripts/python.exe -m src.data.fetch_fpl            # clubs, fixtures, availability
.venv/Scripts/python.exe -m src.data.clean_fpl            # the upcoming season's 380 fixtures
.venv/Scripts/python.exe -m src.evaluate.compare          # all models, ~15 min
.venv/Scripts/python.exe -m src.evaluate.compare --fast   # skips AutoGluon, seconds
.venv/Scripts/python.exe -m src.predict.gameweek          # next round
.venv/Scripts/python.exe -m src.predict.gameweek --replay # last known round, as a check
.venv/Scripts/python.exe -m streamlit run app.py          # the report
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
data/processed/  cleaned and joined, one parquet per source (incl. fpl_fixtures.parquet)
data/final/      the model-ready feature table, and rounds/ - the prediction archive
data/manual/     hand-written override files (the only committed data)
```

`.gitignore` excludes all of `data/` except `data/manual/*.csv`. Those files record
decisions rather than data — losing them would mean re-deriving each one by hand — so
they are committed deliberately. `player_name_overrides.csv` is consulted before the
matching cascade runs and always wins; add a row there rather than loosening a threshold.

**Hand-maintained files record *changes*, never whole state.** A file that restates twenty
full squads to move one player goes stale within a week of a window opening, and a stale
file that looks authoritative is worse than no file. Four remain — name overrides, manual
ratings, absences, and hand-typed fixtures — and each is one line per decision. Transfers
are *not* among them: who has left a club is detected from the FPL squad lists, which is
why `squad_changes.csv` was removed rather than kept.

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

**Squad ratings enter the goals models as a deviation, never as a level.**
`dixon_coles_squad.py` adds `(this XI's mean overall − the club's usual mean)`, because a
club's attack parameter and its average squad rating explain the same variation and
compete if both are levels. The deviation is the part results cannot see: whether a side
is fielding better or worse than it normally does. It earns its place — RPS 0.2118 to
0.2087 over plain Dixon-Coles, with fewer repeated scorelines — and it is the only model
that uses the FIFA pipeline *and* commits to a scoreline. A club with no history is
measured against the league instead, which is where absolute rating is the right quantity.

**Every model implements the same two-method contract** in `src/models/base.py`:
`fit(train)` then `predict(test) -> (lambda_home, lambda_away)`. Nothing downstream knows
which model produced a prediction, so the scoreline matrix, scoring and the report are
identical for all of them — which is what makes the comparison fair. A new model needs
one file and one entry in `evaluate/compare.build_models`.

`tests/test_model_contract.py` runs every model through the same checks, the important
one being that **rewriting the test set's results must not change any prediction**. That
is the only check that catches a leaking model, because a leaking model's output is
otherwise perfectly well-formed.

**A feature that nothing calls is worse than a missing one**, because it reads as done.
The squad-change files were written, tested and documented for a phase before anyone
noticed the prediction path never called them — and in the same blind spot, upcoming
fixtures were receiving none of the 48 squad-quality columns, just the training median for
each, so every team looked average. Both had passing tests. When adding anything to the
prediction path, check what calls it and assert the output actually varies.

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

**The report keeps its logic out of `app.py`.** `src/report/view.py` holds the shaping
decisions — bar widths, percentages, the model-minus-market edge — so they can be tested
without a browser, and `app.py` is layout only. Presentation bugs are quiet: a flipped
sign or a mis-scaled bar still renders, and the page looks authoritative either way.

**Streamlit re-runs `app.py` but does not reload imported modules.** After editing
anything under `src/`, restart the server — otherwise the page raises `ImportError` for a
function that exists and whose tests pass, which is a confusing few minutes.

**The round picker is a page-level widget on purpose, and styled to disappear into the
page.** Season segments and round pills, laid out flat: a `selectbox` filtered as you typed
— baffling with two options — and hid which rounds exist behind a click, when "what have we
predicted?" is a question the page should answer unasked. Style them through
`[data-testid="stButtonGroup"]` and the `data-variant` / `data-selected` attributes rather
than the emotion class hashes sitting beside them, which change on every Streamlit upgrade.

**Interactivity in a card is CSS, never a Streamlit widget.** A card is one block of
markup so it can be packed three to a row, and raw HTML cannot trigger a rerun. The
expected-XI overlay is therefore a hidden checkbox with `<label>`s as its button and its
close control: no JavaScript, no rerun, and the grid does not move when it opens. Reach
for the same pattern before considering a custom component.

**Verify the report by measuring the rendered page, not by reading the code.** Every
front-end bug in this project was invisible in the source and obvious in the DOM: a
grid that silently stacked because the pane was 393px wide, a pixel font that never
reached the expander label, an icon that would have rendered as the word
`keyboard_arrow_right`. `read_page` and `javascript_tool` for computed styles and
bounding boxes.

**Commands in user-facing text spell out `.venv\Scripts\python.exe` and lead with `cd`.**
A bare `python` assumes an activated environment, which is the step people skip. Running
from the wrong folder is the more common failure and PowerShell reports it as *"the module
'.venv' could not be loaded"* — naming neither the directory nor the real cause. The
report's empty state is the one message a stuck reader is guaranteed to reach, so it
carries both, with the repo path derived at runtime rather than hardcoded.

**A stored round is never rewritten, and that refusal is the feature.**
`predict.archive.save_round` raises `RoundAlreadyStored` unless passed `force=True`; the
CLI exits 1. The archive answers one question — what did the model say *before* those
matches were played — and a file that can be silently replaced after the result is known
answers nothing. It also cannot be reconstructed: re-running a past round predicts it with
today's model and today's data, and that output is indistinguishable from the original.
Written at the time or not at all. The report reads the newest stored round and there is
deliberately no "latest" file beside the archive, because two copies of one round invite
them to disagree and the page would show whichever is wrong.

**An official gameweek beats a derived one, and for an unstarted season the derivation
cannot work at all.** `predict.gameweeks_for` uses the fixture's own `gameweek` column when
the FPL schedule supplies one. Without that check every 2026/27 fixture came out as
gameweek 1 — the derivation counts each club's matches in `matches.parquet`, which is empty
for a season nobody has played, so every round would have tried to overwrite the last. The
archive's refusal is what surfaced it.

**Gameweeks are derived when nothing publishes them, which is every historical season.** `data/gameweeks.py`
counts each club's matches within a season rather than clustering dates: a club plays each
round exactly once, so a postponed fixture played six weeks later still lands correctly and
midweek rounds do not merge into the weekend. **Do not assume 10 fixtures to a round** —
measured on 2025/26, 36 of the 38 come out at exactly 10 and 2 split because a match
crossed a round boundary. `gameweek_sizes` reports rather than validates for that reason.

**The expected XI is picked to a shape, not just by appearances.** Ranking the outfield
purely by starts does not produce a formation: Arsenal came out **5-5-0** for gameweek 1,
five defenders and five midfielders with six available forwards all a start behind. Each
line takes a minimum first (`FORMATION_MINIMUM`, 1/3/2/1) and the free places go to whoever
has played most within `FORMATION_MAXIMUM` (1/5/6/4), so every side lands inside the range
real formations occupy. The goalkeeper was already protected this way; every other line
needed it too.

**A club can lose every goalkeeper it has recently started.** Leeds' only keeper in the
pool had transferred, and the free places then went to outfielders — eleven of them, with
nobody in goal. Outfield places are capped at ten whether or not a keeper was found, and a
missing keeper is filled from the ratings, which know a club's keepers even when
appearances no longer do. That row is labelled `ratings` individually, so the overlay's
wording stays true for a side that is otherwise appearance-based.

**Unavailable players are removed before the XI is picked, never after.** `most_used_eleven`
filters suspended and hand-flagged players out of the pool, so the next-most-used player
steps up and the side is still eleven. Dropping them afterwards fields ten and understates
the squad, which is a larger error than the absence. A club whose recent history shows
exactly eleven names has nobody to promote and genuinely yields ten — that is honest, and
`rated_share` falls to say so.

**Every red card is one match, deliberately.** The offence is not in the data — only 7 of
318 reds carry a yellow on the same row, so a second yellow cannot be told from violent
conduct. Guessing three matches would remove a player who is actually available, and that
is the worse direction of error. Yellow accumulation uses the real tiers (5/10/15 within
19/32/38 club matches); the third has never fired in seven seasons.

**Suspensions are a small effect and were measured, not assumed.** On 2025/26 they touch
8.3% of team-matches, and the median change in squad overall is **zero** — usually the
player sent off was not in the expected XI anyway. Mean −0.084 against a column standard
deviation of 3.4. Occasionally the change is positive, because the twelfth-most-used player
can be rated above the eleventh. Do not oversell this feature on the strength of it being
correct.

**AML and AMR are wingers, so they count as attackers.** Understat names positions for a
slot in a formation grid, not for the job — reading its wide attacking codes as midfield
gave sides with four defenders, six midfielders and *no attacker at all*. Man United's
most-used XI rendered as 4-6-0 while genuinely being a 4-3-3 with Mbeumo, Cunha and
Diallo across the front. `AMC` stays a midfielder, since that is a number ten, so prefix
order in `LINE_PREFIXES` has to catch the wide codes first. Across all 5,320 starting XIs
this makes 4-3-3 the most common shape and leaves none with an empty attack. It shifts
`squad_att_overall` and `squad_mid_overall` for every historical match; the backtest moved
by 0.0001 on one model, so re-run `compare --fast` if you touch the mapping again.

**A line that arrives already set is kept, never recomputed.** `starting_ratings` reads
Understat's codes, and an XI built from ratings uses FIFA's (`CB`, `LB`, `ST`) which fall
through to `unknown` — which put every promoted club's entire outfield into midfield on
the pitch view. Anything supplying its own `line` or `fifa_player_name` has it preserved
through the join.

**A promoted club's XI comes from ratings, not appearances, and says so.** It has no
history in this division, so `most_used_eleven` returns nothing and every squad-quality
column lands null — which downstream becomes the training median, describing a newly
promoted side to the model as an average Premier League squad. That is the Phase 9 bug one
division down. `squads.ratings_eleven` picks the best-rated eleven by position instead, and
the fallback also fires when a club's history exists but sits outside the lookup season's
name map (a club relegated and promoted back). Every row carries `xi_source`, and the
report's overlay changes its wording rather than calling a ratings XI a most-used eleven.

**The Fantasy Premier League API is the best free source found for anything live.**
`https://fantasy.premierleague.com/api/bootstrap-static/` gives 564 players with `status`,
`chance_of_playing_next_round` and a `news` string; `/api/fixtures/` gives all 380 fixtures
with kickoff times and *official* gameweek numbers. Free, no authentication, JSON, all 20
clubs. **It is undocumented and can change without notice**, so treat a schema change as
expected rather than exceptional. Club names need mapping like every other source
(`Man Utd`, `Spurs`), and player names match Understat only 57–61% on exact normalisation,
so reaching usable coverage means the Phase 4 cascade again, not a direct join.

**Our score and the market's are always taken over the same fixtures.** `report/results.py`
computes both RPS numbers over matches that were *both played and priced*, never ours over
everything and the bookmaker's over the subset it quoted. Scoring two models on different
match sets produces a flattering number that looks entirely ordinary, and this is the one
page element with a motive to be wrong.

**A card admits both halves of a split result.** An exact scoreline can come with the wrong
call: 1-1 leads the scorelines on a card whose headline verdict is HOME WIN, because the
draw is the most likely single score while the home win is the most likely outcome. When
such a match finishes 1-1 the card says `EXACT SCORE` *and* `WRONG CALL` — showing only the
green half was the first version and it was advertising. Both of gameweek 38's exact hits
are this case, so it is the normal one, not an edge case.

**Small samples are labelled on the page, and a lucky one is contradicted.** Under 30
scored matches the scorecard carries a caveat, and if that sample happens to beat the
closing line the caveat says so explicitly and points at the 2,280-match backtest. Ten
fixtures is a round, not evidence.

**The report states whether it is showing upcoming fixtures or a replay.** A replayed
round otherwise reads as next week's matches, since the cards look identical either way.
Every stored prediction carries a `mode` field for exactly this.

**Four project skills live in `.claude/skills/`**, auto-discovered and committed, so they
work for anyone who clones this, and indexed in `SKILLS.md`: `refresh-squads`,
`audit-squads`, `predict-round`, `check-report`. They encode the *why* and the traps, not
the command list — `COMMANDS.md` is the command list.

**`squad_changes.csv` is gone, and the lesson is why.** It was documented in three places
as the way to move a transferred player, and it never worked: `apply_squad_changes` was
defined, had four passing tests, and was called by nothing, so the file's only effect was a
warning about names it did not recognise. It also keyed on *FIFA* names while the expected
XI is built from *Understat* appearances, so the two could never have met. Departures are
detected from the FPL squad lists instead. **A tested function is not a connected one** —
that is the third time this project has shipped that shape, so check what calls a thing
before trusting that it does anything.

**Squad membership comes from FPL, and only departures are acted on.** Appearances are last
season's, so a departed regular keeps his place forever otherwise — Casemiro was still being
picked for Man United a year after leaving. `predict.transfers` diffs each club's recent
starters against its current FPL squad and removes anyone with no counterpart. **Scope the
check to recent starters, never to a club's whole history**, or it reports Cristiano Ronaldo
as a departure. Matching is deliberately generous in both containment directions, because a
false departure silently deletes a real player: `Amad Diallo Traore` must match `Amad`, and
`Bruno Fernandes` must match `Bruno Borges Fernandes`. **Arrivals are reported and never
selected** — a signing has no appearances to rank against the players who have them, so
picking one would assert a team sheet rather than describe an expectation.

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

**The best-scoring model is not the best report, and the report knows it.**
`predict.gameweek.DEFAULT_MODEL` is `dixon-coles-squad` (RPS 0.2087), not `poisson-glm`
(0.2039). Do not "fix" this without reading the comment there. The GLM hedges towards the
average — which is exactly what RPS rewards — and the cost is that it calls 1-1 in 74% of
matches with nine distinct top scorelines all season. Dixon-Coles estimates each club's
attack and defence directly, commits, and gives 60% and eleven. For Crystal Palace against
Arsenal the GLM predicts 1.27 goals to 1.63 and reports 1-1 where the market has the away
side at 51%; Dixon-Coles predicts 0.92 to 1.68 and reports 0-1, matching the bookmakers.
Lowering the ridge penalty does not help — the fixture stays 1-1 even at 0.25 — so this is
structural. **When the deliverable is a scoreline, a model that says 1-1 three times in
four is failing whatever its score says.**

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
