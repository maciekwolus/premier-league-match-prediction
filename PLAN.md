# Premier League Match Predictor — Project Plan

> **All ten phases are complete.** The pipeline runs from raw downloads to a Streamlit
> report: 2,660 matches, 77,278 player appearances, 98.6% of starters matched to ratings,
> a 99-column feature table, nine models benchmarked against the closing line, and
> predictions for fixtures that have not been played. 336 tests, none needing network or
> data.
>
> The honest headline: **the bookmaker's closing line wins at RPS 0.1965, ahead of the
> best model's 0.2027.** That is the expected result and the project is more trustworthy
> for it — a model beating the market would have deserved suspicion first.
>
> This document is kept as written, including the estimates that turned out wrong, since
> what the plan expected against what happened is the more useful record. `CLAUDE.md`
> describes the code as it now stands.
>
> **Stage two (Phases 11–15) is planned and not started** — preparing for the 2026/27
> season: archived predictions, squads corrected against a ratings edition that will not
> exist, a browsable gameweek history scored against reality, and suspensions. Phase 11
> is first because it is the only one that cannot be done late.

## Goal

Given an upcoming Premier League fixture and its expected lineups, output the most
likely **scorelines with their probabilities** — e.g. `1-1 (12%) · 2-1 (9%) · 1-0 (9%)` —
plus the home-win / draw / away-win summary, displayed in a web report.

## Decisions locked in

| Topic | Decision |
|---|---|
| Language | Python 3.12 |
| Output | Top 3 scorelines + probabilities, plus W/D/L summary |
| Models | Dixon–Coles baseline **and** gradient-boosting (AutoGluon), compared head-to-head |
| Bookmaker odds | Primary role = **benchmark**. Also train a with-odds variant to measure market signal |
| Seasons | 2019/20 → 2025/26 (7 seasons, ~2,660 matches) |
| UI | Streamlit |

### Two expectations to keep calibrated

1. **Exact scores cap around 12%.** That is the ceiling for anyone, bookmakers included.
   A model outputting 80% on a scoreline is a bug, not a breakthrough.
2. **No Hugging Face model applies here.** HF hosts text/vision models. "Don't write a
   neural net" is satisfied by AutoGluon (`.fit()` and it handles the rest) and by
   Dixon–Coles (a well-defined statistical model, ~60 lines).

---

## Architecture

```
football-data.co.uk CSVs ─┐
Understat lineups + xG ───┼─→ player matching ─→ feature table ─→ models ─→ score matrix ─→ Streamlit
FIFA / EA FC ratings ─────┘     (fuzzy)          (1 row/match)              (top-3 + W/D/L)
```

```
src/
├── config.py              season list, paths, team-name maps
├── data/
│   ├── fetch_matches.py   download football-data.co.uk CSVs
│   ├── fetch_lineups.py   Understat scrape
│   ├── load_fifa.py       read Kaggle FIFA/FC CSVs
│   └── clean.py           normalise + unify schemas across seasons
├── matching/
│   ├── team_names.py      Understat ↔ football-data ↔ FIFA club names
│   └── player_names.py    fuzzy player matching + override file
├── features/
│   ├── build.py           the one-row-per-match feature table
│   ├── team_strength.py   FIFA aggregates per XI
│   └── form.py            rolling form, Elo, rest days
├── models/
│   ├── dixon_coles.py     classical Poisson model
│   ├── gbm.py             AutoGluon goal regressors
│   └── score_matrix.py    λ_home, λ_away → 6×6 scoreline probabilities
├── evaluate/
│   ├── backtest.py        walk-forward by season
│   └── benchmark.py       vs bookmaker closing odds
└── predict/
    └── gameweek.py        upcoming fixtures → predictions.json
app.py                     Streamlit report
```

---

## Phase 0 — Setup *(~1 short session)*

- venv, `requirements.txt`, `src/` package skeleton, `config.py`
- `pytest` + `ruff` configured
- Commit: `chore: project skeleton`

**Learn with Claude:** run `/init` at the end of each phase to keep `CLAUDE.md` current.
Add a hook so `ruff` runs automatically on every file edit — ask for `/update-config`.

---

## Phase 1 — Match results *(~1 session)*

Download 7 seasons from `football-data.co.uk/mmz4281/{1920,2021,2122,2223,2324,2425,2526}/E0.csv`
— **verified live, 380 matches each**.

Watch out: column sets differ across seasons (older files carry fewer bookmakers).
Take the intersection of columns we actually need rather than assuming a fixed schema.

Useful columns: `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Referee, HS/AS, HC/AC,
HY/AY, HR/AR`, and odds `B365H/D/A` (opening) plus `B365CH/CD/CA` (**closing** — sharper,
use these for the benchmark).

→ `data/processed/matches.parquet`

---

## Phase 2 — Lineups *(~1–2 sessions — first hard part)*

Pull per-match lineups and xG from Understat via `understatapi`.

The real work is the **join**: Understat and football-data name teams differently
(`Manchester United` vs `Man United`). Build an explicit mapping table in
`team_names.py`, then join on `date + home_team + away_team` and assert 380 matched
rows per season. Fail loudly on any unmatched fixture — silent drops here poison
everything downstream.

Risks and fallbacks: if `understatapi` is stale, use the `soccerdata` library (FBref
+ Understat) or scrape the Understat JSON directly. FBref also gives minutes played.

→ `data/raw/lineups/`, `data/processed/lineups.parquet`
  (match_id, player, team, position, minutes, is_starter)

**Learn with Claude:** run the scrape as a background task (`run_in_background`) so
you keep working while it runs.

---

## Phase 3 — FIFA / EA FC ratings *(~1 session)*

Season mapping — note the mid-series rename:

| Season | Game |
|---|---|
| 2019/20 | FIFA 20 |
| 2020/21 | FIFA 21 |
| 2021/22 | FIFA 22 |
| 2022/23 | FIFA 23 |
| 2023/24 | EA Sports FC 24 |
| 2024/25 | EA Sports FC 25 |
| 2025/26 | EA Sports FC 26 |

Source: Kaggle player-ratings CSVs. Kaggle coverage of the newest editions is the
weak link — if FC 25/26 aren't available as clean CSVs, fall back to scraping SoFIFA,
or carry forward the previous edition's ratings with an age adjustment (and record
that we did, so the backtest can account for it).

Fields we need: `short_name, long_name, club_name, age, overall, potential, value_eur,
player_positions, pace, shooting, passing, dribbling, defending, physic`.

→ `data/raw/fifa/`, `data/processed/fifa_players.parquet`

---

## Phase 4 — Player matching *(~1–2 sessions — the hardest part)*

Join Understat player names to FIFA player names. Understat uses display names
(`Bruno Fernandes`), FIFA has both short and long forms (`B. Fernandes` /
`Bruno Miguel Borges Fernandes`). Accents, hyphens and one-name Brazilians make it messy.

Cascade:
1. Normalise — strip accents, lowercase, collapse punctuation
2. Exact match on `normalised_name + club + season`
3. Fuzzy match with `rapidfuzz`, **scoped within club and season** (this is what makes
   fuzzy safe — you're matching against ~30 candidates, not 18,000)
4. Anything still unmatched → `data/manual/player_name_overrides.csv`

Acceptance gate: **≥95% of starting-XI minutes matched per season**, reported per
season so you can see which one is dragging. Achieved: 97.7–99.4%.

→ `data/processed/player_map.parquet` + a coverage report

**Done.** Reached 98.6% of starting appearances. The lesson worth keeping: seven agents
ran in parallel over the residue *after* the automated cascade, not instead of it — and
what they mostly proved was that the remaining names had no candidate to match, which
pointed at a filtering bug in Phase 3 rather than at any name being hard.

---

## Phase 5 — Feature engineering *(~1–2 sessions)*

One row per match, everything knowable **before kickoff**.

**Correction to our earlier note:** raw shots and cards from the match itself are
*post*-match — using them is data leakage and would produce a model that looks
brilliant in testing and useless on Saturday. Rolling averages of *previous* matches'
shots are fine and valuable.

Feature groups:
- **Squad quality (FIFA):** mean/max overall of the XI; mean overall by line
  (GK / DEF / MID / ATT); mean pace, shooting, passing, defending, physic; squad value; mean age
- **Differentials:** home minus away for each of the above — usually the strongest signals
- **Form:** rolling last-5 goals for/against, points, and **xG** (Understat's xG is far
  more predictive than raw goals)
- **Context:** rest days since last match, matchday number, promoted-team flag
- **Elo:** a rating updated match by match from results alone
- **Odds (variant only):** closing odds → de-overrounded probabilities

### Using the season in progress

Predicting matchday 10 should use matchdays 1–9 of the same season. That is what the
form features are for, and it carries most of the short-term signal — squad ratings say
a team is good, form says whether they are playing well right now.

**Shift before rolling.** The naive version silently includes the match being predicted
inside its own feature, which produces a spectacular backtest and a useless predictor:

```python
df["form"] = df.groupby("team")["goals"].rolling(5).mean()  # WRONG
df["form"] = df.groupby("team")["goals"].shift(1).rolling(5).mean()  # RIGHT
```

Phase 5 needs a test asserting that a match's own result never reaches its own features.
This is the single most common way football models get quietly broken.

**Let the window cross season boundaries.** A team's "last 5" means their last 5 matches
full stop, so August matchday 1 uses the previous May. Summer transfers make this
imperfect, but it beats starting every season blind — and roughly 26% of all matches
fall in the first five matchdays, so the cold-start case is not an edge case.

**Add `season_matches_played`** so the model learns for itself how much to discount form
early in a season, instead of us hard-coding a rule.

**Promoted teams have no history at all** — their previous matches were in the
Championship, which we do not collect. This is precisely where the FIFA ratings earn
their place: in August they are the only signal available for a promoted side.

→ `data/final/features.parquet`

---

## Phase 6 — Models *(~2 sessions)*

Three builds:

- **A. Dixon–Coles** — attack/defence strength per team + home advantage + the
  low-score correction. Produces a scoreline matrix natively. Your baseline and
  sanity check.
- **B. AutoGluon** — two goal models (home goals, away goals) on the full feature
  table, converted to λ values → score matrix.
- **C. B + odds features** — measures how much the market knows that you don't.

**Score matrix:** from λ_home and λ_away, compute P(score) over 0–5 goals per side,
apply the Dixon–Coles correction for 0-0/1-0/0-1/1-1, normalise. Summing the matrix
regions gives home-win / draw / away-win for free — one model, both outputs.

**Validation — walk-forward, never a random split.** Train on seasons 1..n, test on
n+1, roll forward. A random split lets the model see the future and inflates every score.

**Training window.** Season-level walk-forward is the headline benchmark: simple, cheap,
and the number to quote against the bookmakers. It is mildly pessimistic, because the
model never sees any of the season it is predicting. Two refinements:

- **In production, refit on everything completed to date.** When predicting matchday 10
  of the live season, there is no reason to discard matchdays 1–9. Phase 7 should do this.
- **Expanding-window backtest as a variant** — refit each matchday (38 fits per season)
  and see whether it actually beats the season-level number. Expect a small gain,
  concentrated late in the season. Dixon–Coles is cheap to refit and is conventionally
  run this way, so for that model it is the default rather than an upgrade.

Either way the *features* always use the current season; only the training rows differ.

**Metrics:** Ranked Probability Score (the football standard — rewards being close),
log-loss, accuracy, and a calibration plot. Then the honest question: **RPS vs the
bookmaker closing line.** Beating it is genuinely hard; matching it is a real result.

Also report **RPS bucketed by matchday.** Early-season predictions should be measurably
worse, given thin form data and freshly promoted sides. Better to measure that than to
hide it inside a single average.

**Learn with Claude — parallel agents again:** one agent per model family, all training
and reporting simultaneously, then compare.

---

## Phase 7 — Upcoming-fixture predictions *(~1–2 sessions)*

Fetch next gameweek's fixtures, assemble expected XIs, run the model, write
`predictions.json`.

Open problem worth knowing up front: **you don't know the lineups until an hour before
kickoff.** Three options, cheapest first — (a) each team's most-used XI from its last 5
matches, (b) manual entry in the UI, (c) scrape a predicted-lineups site. Start with (a),
and make lineups overridable so you can re-run once teams are announced.

### The squad problem, which is the real work

Everything up to Phase 6 predicted the past, where the squads were a matter of record.
Predicting forwards breaks two assumptions at once.

**A new season has no ratings edition for most of its first two months.** The 2026/27
season starts in August 2026; EA FC 27 is released in late September. For those opening
weeks the newest ratings we can have are FC 26's, published a year earlier. The fix is
carry-forward: point `Season.fifa_edition` at the most recent edition that exists, and
swap it when the new one lands. That is a one-line change in `config.py`, which is the
design working as intended.

**The transfer window is open while the season starts.** A player who signs in July is
listed in FC 26 at their old club — the same September-snapshot problem that broke Phase 4,
except now it is unavoidable rather than historical. Ratings are still correct; only the
club is wrong.

So the mechanism must make squad edits **cheap and incremental**, because they will happen
weekly during a window and occasionally all season. Two committed files under
`data/manual/`, following `player_name_overrides.csv`:

| File | Purpose |
|---|---|
| `squad_changes.csv` | `season, fifa_player_name, team, note` — one row per move. `team` blank means the player has left the league |
| `player_ratings_manual.csv` | `season, fifa_player_name, overall, age, position, note` — for a signing with no FIFA entry at all |

**Record changes, not state.** The default squad for a club is whoever played for it most
recently; the change file only says what is different. Restating 20 full squads to move one
player would guarantee the file goes stale, and a stale squad file is worse than none
because it looks authoritative.

`player_ratings_manual.csv` covers the genuinely new: a signing from a league outside the
dataset, or an academy player with no rating anywhere. An overall rating alone is enough to
compute squad quality; the rest of the columns are optional.

Also needed: `season_clubs()` currently reads the participating clubs from
`matches.parquet`, which has no rows for a season that has not started. Phase 7 needs a
promoted/relegated list for the upcoming season, which is another two-line manual file or a
fixture-list scrape.

**Acceptance test for the mechanism:** adding a single transfer should be one line in one
file, followed by a rebuild — no code change, no threshold to tune, and a clear error if the
player name does not match anything.

---

## Phase 8 — Streamlit report *(~1 session)*

`streamlit run app.py` — one card per fixture:

```
┌────────────────────────────────────────────┐
│  Arsenal  vs  Chelsea      Sat 15:00       │
│                                            │
│  2-1  ████████  11%                        │
│  1-1  ███████   10%                        │
│  2-0  ██████     9%                        │
│                                            │
│  Home 48%  │  Draw 26%  │  Away 26%        │
│  Bookmaker: 45% / 27% / 28%                │
└────────────────────────────────────────────┘
```

Showing the bookmaker line next to yours turns the UI into a live scoreboard for the
model — the most useful thing on the page.

---

## Phase 9 — Wrap up *(~1 session)*

README with results, `CLAUDE.md` updated, tests over the matching and feature code
(the two places where silent bugs hide), clean git history.

**Done, and it found something.** A scan for functions defined but never called turned up
the whole squad-change mechanism — `apply_squad_changes`, `load_manual_ratings` and their
neighbours were written, tested and documented, but nothing in the prediction path called
them. Worse, upcoming fixtures were getting *none* of the 48 squad-quality columns: they
were filled with the training median, so every team looked exactly average to the model.

Both are now wired in, with a regression test asserting that two squads of different
quality produce different numbers. The lesson worth keeping: **passing tests and a green
pipeline do not prove a feature is connected to anything.** Ask what calls it.

---

# Stage two — ready for 2026/27

The first ten phases were Phases 0–9; the numbering resumes at 11 so the two stages stay
visually distinct. Stage one built a pipeline that predicts *a* round. Stage two makes it
survive a live season: correct squads for a season whose ratings edition does not exist,
a browsable history with scores against it, and a model that knows who is unavailable.

**These four phases have a hard ordering constraint, and it is not the usual one.**
Phase 11 is not the most interesting work, but it is the only piece that cannot be done
late. Every other phase can be built whenever. A track record can only be built forwards.

## Phase 11 — Archive every prediction *(~1 session)*

Today `data/final/predictions.json` is a flat list, rewritten from scratch on every run.
Last week's prediction does not exist the moment this week's is made.

Becomes an append-only store, one file per round:

```
data/final/predictions/2026-27/gw01.json
```

**Written once, and refuses to overwrite without an explicit flag.** That refusal is the
feature rather than a safety rail: an archive that can be quietly rewritten is not
evidence of anything, and the whole point of Phase 13 is to show what we said *before*
kickoff.

This phase also introduces the gameweek, which does not currently exist anywhere — no
upstream source gives a round number, only a date. The rule is each team's Nth match.
Verified against 2025/26: **36 of 38 gameweeks come out at exactly 10 fixtures**, the
remaining 2 split by rescheduling. Tests encode that, rather than asserting a clean 10 and
breaking the first time a match is moved.

| Deliverable | Test that it works |
|---|---|
| `gameweek_of(matches)` | A complete season yields gameweeks 1–38; the two split rounds are expected, not a failure |
| Immutable round archive | Re-running a stored gameweek leaves the file byte-identical |
| Migration of the current file | The report reads the archive and renders exactly as before |

## Phase 12 — Squads for 2026/27 *(~1–2 sessions)*

`UPCOMING_SEASON` already points at `EA FC 26`, so ratings carry forward as designed. What
is missing is everything that makes those ratings describe *this* season's clubs.

- **The twenty clubs.** `season_clubs()` reads participants from `matches.parquet`, which
  is empty for a season that has not started. Needs the promoted and relegated three.
- **Transfers.** One line per move in `data/manual/squad_changes.csv` — the file and its
  loader already exist and already run before the matching cascade.
- **Players with no FC 26 entry** — arrivals from other leagues, promoted-club squads —
  go in `data/manual/player_ratings_manual.csv`.

**Acceptance test, taken from CLAUDE.md's own rule: adding one transfer is one line plus a
rebuild.** If it is ever two, the design is wrong.

And per the Phase 9 lesson, the refresh **asserts the ratings actually moved** — a diff of
squad-quality columns before and after, with a named player checked by hand. A squad file
that loads cleanly and changes nothing is exactly the failure this project has already had
once.

Known limitation worth writing down now: by May 2027 these ratings are twenty months old.
A player who improved sharply over 2026/27 is rated as he was in September 2025. That is
a real cost of the carry-forward and it is not fixable without an edition that does not
exist.

## Phase 13 — Gameweek browser and the running scorecard *(~1–2 sessions)*

A selector for any gameweek, played or not. Played rounds show the prediction against the
actual score, and a season-to-date scorecard: our RPS beside the bookmaker's.

That framing matters. The interesting claim is not "we called it" — it is **our hit rate
next to theirs**, which is the same honesty the report already applies to a single fixture
and the reason nothing here has ever claimed to beat the market.

The gameweek selector is page-level, so an ordinary Streamlit widget is fine. The rule that
interactivity inside a *card* must be CSS still holds, and still applies to the expected-XI
overlay.

**Open dependency:** football-data's `fixtures.csv` covers only the next few days, so a
selector spanning a whole season needs the 2026/27 schedule from another source.

## Phase 14 — Availability: suspensions, then injuries *(~1–2 sessions)*

**Suspensions come free from data already on disk.** `lineups.parquet` carries per-player
`yellow_cards` and `red_cards` — 318 reds and 9,575 yellows across seven seasons, about 45
reds a season.

| Rule | Source |
|---|---|
| Red card → out of the next match | Derived from `lineups.parquet` |
| Yellow accumulation at 5 / 10 / 15 → 1 / 2 / 3 matches | Premier League rules, encoded in config |

**The gap to state plainly: we cannot see the offence type.** A three-match violent-conduct
ban will look like a one-match ban. Better to record that limit than to model it wrongly.

Unavailable players drop out of the expected XI and the next-most-used player steps up.
This changes the squad-quality *deviation* — which `dixon-coles-squad` already consumes —
so **no model changes at all**. The feature plugs into a socket that already exists.

Effect size must be measured rather than assumed: removing one starter moves a squad mean
by roughly (player − replacement) ÷ 11. The test asserts the prediction actually moves,
because a correct-looking feature that changes nothing is this project's signature failure.

**Injuries begin with a spike, timeboxed, with a written go/no-go.** Usable means: free,
no authentication, machine-readable, Premier League coverage, updated before gameweeks,
and terms that permit the use. My honest expectation is that it returns nothing, in which
case a manual override file is the answer — and we will have paid a little to know that
rather than having assumed it.

## Phase 15 — Skills *(~1 session)*

`refresh-squads`, `refresh-gameweek`, and predicting the opening round.

**Deliberately last.** A skill encodes a workflow; encoding one before the workflow is
stable just freezes a guess. This also closes the one item from the original learning list
never delivered — custom skills were promised after Phases 4 and 7 and never built.

## Open questions blocking stage two

| Question | Why it cannot be guessed |
|---|---|
| The three promoted and three relegated clubs for 2026/27 | The knowledge cutoff sits at the end of 2025/26. A wrong club list produces a report that looks authoritative and is wrong throughout |
| Where the full 2026/27 fixture list comes from | The existing feed covers days, not seasons — Phase 13 cannot span a season without it |

---

## Learning track with Claude

| When | What | How to trigger |
|---|---|---|
| Every phase | Plan mode before writing code | I'll propose, you approve |
| Every phase | Commit per phase, branch per feature | "commit this" / "open a PR" |
| Phase 2, 3 | Background tasks for long scrapes | happens automatically |
| **Phase 4, 6** | **Parallel subagents** | *"use subagents for this"* |
| ~~After Phase 4~~ | ~~**Custom skill** — `/refresh-data`~~ — not built; moved to Phase 15 | "make this a skill" |
| ~~After Phase 7~~ | ~~**Custom skill** — `/predict-gameweek`~~ — not built; moved to Phase 15 | "make this a skill" |
| **Phase 15** | **Custom skills** — `refresh-squads`, `refresh-gameweek`, predict the opening round | *"build the skills"* |
| Anytime | `/code-review` before merging | you run it |
| Phase 0 | Hooks — auto-lint on edit | "set up a lint hook" |

---

## Order of work

Phases are sequential up to 5, because each feeds the next. Phase 3 (FIFA) is
independent of Phase 2 (lineups), so those two can run in parallel if you want to try
two agents early.

~~**Next action:** Phase 0 + Phase 1 — skeleton and the match-results downloader.
That gets real data on disk within one session.~~ *Done, along with everything through
Phase 9.*

**Next action: Phase 11 — archive every prediction.** Not the most interesting phase, and
that is not the criterion. It is the only one whose cost grows while it waits: each round
predicted without it is a round of track record that no later work can recover. Phases 12
to 14 can be built in any order once it lands, though 12 gates predicting anything in
2026/27 at all.
