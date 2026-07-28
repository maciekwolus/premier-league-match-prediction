# Premier League Match Predictor — Project Plan

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

---

## Learning track with Claude

| When | What | How to trigger |
|---|---|---|
| Every phase | Plan mode before writing code | I'll propose, you approve |
| Every phase | Commit per phase, branch per feature | "commit this" / "open a PR" |
| Phase 2, 3 | Background tasks for long scrapes | happens automatically |
| **Phase 4, 6** | **Parallel subagents** | *"use subagents for this"* |
| After Phase 4 | **Custom skill** — `/refresh-data` re-runs the whole pipeline | "make this a skill" |
| After Phase 7 | **Custom skill** — `/predict-gameweek` | "make this a skill" |
| Anytime | `/code-review` before merging | you run it |
| Phase 0 | Hooks — auto-lint on edit | "set up a lint hook" |

---

## Order of work

Phases are sequential up to 5, because each feeds the next. Phase 3 (FIFA) is
independent of Phase 2 (lineups), so those two can run in parallel if you want to try
two agents early.

**Next action:** Phase 0 + Phase 1 — skeleton and the match-results downloader.
That gets real data on disk within one session.
