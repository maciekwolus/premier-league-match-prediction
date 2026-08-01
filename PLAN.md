# Premier League Match Predictor — Project Plan

> **All ten phases are complete.** The pipeline runs from raw downloads to a Streamlit
> report: 2,660 matches, 77,278 player appearances, 98.6% of starters matched to ratings,
> a 99-column feature table, nine models benchmarked against the closing line, and
> predictions for fixtures that have not been played. 452 tests, none needing network or
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
> **Stage two is complete: Phases 11–15 all done.** **Gameweek 1 of 2026/27 is predicted
> and archived** — the point of the whole stage. Predictions are stored per gameweek and
> never rewritten, the report browses them and scores them against what happened,
> suspensions come out of the cards already on disk, the twenty clubs and 380 fixtures of
> the new season come from the Fantasy Premier League API, and four skills encode the
> recurring workflows. Phase 16 remains as an optional experiment.

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
| `assign_gameweeks(matches)` | A complete season yields gameweeks 1–38; the two split rounds are expected, not a failure |
| Immutable round archive | Re-running a stored gameweek refuses and exits 1 |
| Migration of the current file | The report reads the archive and renders exactly as before |

**Done.** 23 new tests, 359 in total. Two things worth recording.

**The archive nearly landed in an occupied directory.** The plan said
`data/final/predictions/`, which `evaluate.compare --save` already owns for its per-model
backtest tables. Nothing would have broken immediately — the glob is specific enough — but
two unrelated things in one directory is how a glob quietly starts matching the wrong
files later. It went to `data/final/rounds/` instead.

**The gameweek derivation was checked against something it could not have fitted to.**
`--replay` predicts the final round of 2025/26, and the derivation independently called it
gameweek 38. That is the kind of agreement a wrong rule does not produce by accident.

`predictions.json` is gone rather than kept as a convenience mirror. Two copies of one
round invite them to disagree, and the page would show whichever was wrong.

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

**Done.** 22 new tests, 435 in total. **Gameweek 1 of 2026/27 is predicted and archived**,
which was the point of the whole stage.

The blocking questions were answered by the Fantasy Premier League API, found while spiking
for injuries in Phase 14: 380 fixtures with official gameweek numbers, and the club list.
`fetch_fpl` and `clean_fpl` cache and validate it on the same contract as every other
source — exactly 20 clubs, 380 fixtures, 19 home and 19 away each, raising rather than
writing a suspect table.

**And it found the Phase 9 bug again, one division down.** A promoted club has no history
here, so its expected XI was empty, so every squad-quality column was null — and null
becomes the training median downstream, which described Coventry and Hull to the model as
average Premier League squads. Ipswich failed a second way: it *had* an XI from 2024/25,
but none of those players sit in the lookup season's name map, so the join produced nulls
anyway and a global name check would not have noticed.

The fix is `ratings_eleven` — the best-rated eleven by position, from ratings that cover
every club in the world and were simply never mapped for these three. What it changed:

| Fixture | Before | After |
|---|---|---|
| Arsenal v Coventry | 74% home, xG 2.29–0.65 | **85% home**, xG 2.93–0.53 |
| Hull v Man United | 43% away, xG 1.30–1.56 | **60% away**, xG 1.04–2.03 |

**A ratings XI is a different kind of guess and the report says so.** Every row carries
`xi_source`, and the overlay stops calling it a most-used eleven — which would have been a
quiet lie about precisely the club a reader knows least about.

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

**Done.** 30 new tests, 387 in total. Three things worth recording.

**The card was quietly flattering itself, and only the rendered page showed it.** An exact
scoreline can arrive with the wrong call — 1-1 leads the scorelines on a card whose headline
verdict is HOME WIN, since the draw is the likeliest single score while the home win is the
likeliest outcome. The first version showed a green `EXACT SCORE CALLED` and dropped the
fact that the call above it was wrong. Both of gameweek 38's exact hits are this case, so
it is the normal one. Cards now say `EXACT SCORE` *and* `WRONG CALL`.

**The archived round beats the market, which is exactly the trap.** RPS 0.2407 against the
bookmaker's 0.2503 over ten fixtures. That is noise — the 2,280-match backtest has the
closing line winning clearly — so the small-sample caveat detects the case and contradicts
it in as many words rather than letting a good week read as skill.

**Both RPS numbers are computed over the same fixtures**, only those played *and* priced.
Scoring ourselves on everything and the bookmaker on the subset it quoted would produce a
flattering number that looks entirely ordinary, and there is a test whose only job is to
stop that.

The selector stays hidden while one round is stored, since a dropdown with a single choice
is furniture. It appears from the second round on; the multi-round behaviour is covered by
tests rather than waiting for a season to prove it.

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

**Done — and the spike contradicted the expectation above.** 33 new tests, 413 in total.

### Suspensions: built, and the effect measured rather than assumed

Red cards and yellow accumulation come out of `lineups.parquet`, and unavailable players
are dropped from the expected XI *before* the eleven is picked, so the next-most-used
player steps up rather than a hole appearing.

Measured on 2025/26 rather than asserted:

| | |
|---|---|
| Team-matches with at least one suspension | 63 of 760 — **8.3%** |
| Mean change in squad overall | **−0.084** rating points |
| Median change | **0.000** |
| Range | −1.55 to +0.73 |
| Changed by ≥0.5 | 11 of 63 |

**The median is zero**, because more often than not the suspended player was not in the
expected XI to begin with. Against a `squad_overall_mean` standard deviation of 3.4, the
mean effect is around 2.5% of one standard deviation. The feature is correct and it fires,
but it is small, and the occasional *positive* change is real: the twelfth-most-used player
is sometimes rated above the eleventh, so losing a starter can raise the mean.

The offence is not in the data — only 7 of 318 reds carry a yellow on the same row — so
every dismissal is one match. That under-counts violent conduct deliberately: banning an
available player on a guess is the worse error.

### The injury spike: GO, and it found more than injuries

The **official Fantasy Premier League API** meets every criterion: free, no authentication,
JSON, all 20 clubs, updated before each gameweek.

```
https://fantasy.premierleague.com/api/bootstrap-static/
```

564 players with `status`, `chance_of_playing_next_round`, and a `news` string
(*"Groin injury - Expected back 21 Aug"*). 55 were flagged when checked. Caveat worth
keeping: it is undocumented, so it can change without notice.

**The real work is name matching, and it is Phase 4 all over again** — 57% of FPL names
match an Understat name on exact normalisation, 61% counting the short form. Some of that
gap is players who simply never appeared in 2025/26, but a club-scoped fuzzy cascade is the
known answer and it is a piece of work, not a line.

**The same source answers both of Phase 12's blocking questions.** `/api/fixtures/` returns
all **380 fixtures for 2026/27**, each with a kickoff time and an *official* gameweek
number, first deadline 21 August 2026. And the club list gives the turnover directly:

| | |
|---|---|
| Promoted | Coventry City, Hull City, Ipswich Town |
| Relegated | Burnley, West Ham, Wolves |

Not acted on here — that is Phase 12 — but it is no longer blocked. Note the club names
need mapping (`Man Utd`, `Spurs`), exactly as every other source has.

## Phase 15 — Skills *(~1 session)*

`refresh-squads`, `refresh-gameweek`, and predicting the opening round.

**Deliberately last.** A skill encodes a workflow; encoding one before the workflow is
stable just freezes a guess. This also closes the one item from the original learning list
never delivered — custom skills were promised after Phases 4 and 7 and never built.

**Done.** Three skills in `.claude/skills/`, auto-discovered by Claude Code and committed
so they work for anyone who clones the repo:

| Skill | Covers |
|---|---|
| `refresh-squads` | Transfers, signings and injuries — and confirming the edit reached the model |
| `predict-round` | Predicting and archiving a gameweek, and a new season's schedule |
| `check-report` | Verifying the page by measuring the DOM rather than reading the code |

The names ended up as `predict-round` and `check-report` rather than the planned
`refresh-gameweek`, because both do more than refresh: one writes a permanent record, the
other proves a change works.

**What makes them worth more than COMMANDS.md is that they carry the reasoning and the
traps.** A squad edit that loads cleanly may still change nothing — that has happened here.
The archive refusing to overwrite is the feature working, not an error to route around. A
front-end bug in this project is invisible in the source and obvious in the DOM. Each skill
also says what a *correct* result looks like, including that a real squad change often
moves the prediction barely at all, so nobody goes hunting for a bug that is not there.

Every command and snippet in them was run verbatim before committing, and the archive's
refusal message is quoted from the actual output rather than paraphrased.

## Phase 16 — FPL player form as a weekly-updating quality signal *(~1–2 sessions)*

The Fantasy Premier League API is already in this project's vocabulary: Phase 14's injury
spike found it, and Phase 12 takes the club list and all 380 fixtures from it. The same
`bootstrap-static/` payload carries a per-player scoring record — 564 players, each with
`total_points`, `form`, `points_per_game`, `bonus`, `bps`, `influence`, `creativity`,
`threat`, `ict_index`, `now_cost`, `cost_change_start`, `selected_by_percent` and `ep_next`.
This phase asks whether any of it predicts goals. **It is written as an experiment with a
kill criterion, not as a feature to be delivered.**

**The likely answer, stated first: the obvious version is redundant.** FPL points are a
re-encoding of goals, assists, clean sheets, minutes and cards, and `lineups.parquet`
already carries goals, assists, xG, xA, minutes and cards per player per match, with
rolling form features built over them. A rolling mean of a player's FPL points is a rolling
mean of things the feature table already knows, passed through a scoring rulebook that
discards information rather than adding it — a 25-yard screamer and a tap-in are both four
points. If this phase is built as "add rolling FPL points as a feature", the honest
expectation is that it moves nothing, and it should not be built that way.

**The version that could actually pay makes a different claim: a player-quality signal that
updates weekly.** Squad quality currently comes from FIFA ratings — an annual September
snapshot — and Phase 12 already records the cost of that: by May 2027 the ratings are
twenty months old, and a player who improved sharply during 2026/27 is rated as he was
before it started. `form`, `ict_index`, `now_cost` and `selected_by_percent` move every
gameweek. Two of those are not performance measures at all. Price and ownership are crowd
judgement, aggregated over millions of managers, and that is a *different kind* of
information from a rating — closer in character to the odds than to FIFA. It is also the
one thing the FIFA pipeline cannot supply at any refresh rate, which is the whole reason
this phase is worth an experiment rather than a shrug.

Candidates, aggregated over the expected XI and entering the table as `home_`/`away_`/
`diff_` like every other per-team feature:

| Candidate | Source field | What it is |
|---|---|---|
| `squad_fpl_form` | `form` | Mean FPL points per match over the player's recent matches — FPL's own rolling form number |
| `squad_fpl_ict` | `ict_index` | Composite of influence, creativity and threat: FPL's per-player contribution score, not a points total |
| `squad_fpl_price` | `now_cost` | Current price. Moved by transfers in and out — crowd judgement about a player, updated daily |
| `squad_fpl_price_change` | `cost_change_start` | Price movement since the season opened: who the crowd has *re-rated* during this season |
| `squad_fpl_ownership` | `selected_by_percent` | Share of managers owning the player |
| `squad_fpl_ep_next` | `ep_next` | FPL's own expected points for the coming gameweek — an external forecast, and the only field here that is itself somebody's model |

`ict_index`, `now_cost`, `cost_change_start` and `selected_by_percent` are the block worth
testing. The rest are there as the control: if the points-derived fields carry the result
and the judgement-derived fields do not, that is the redundancy hypothesis confirmed, and
the phase ends.

### Leakage is the sharp edge here, sharper than in any previous phase

**Clean-sheet points and goals-conceded points *are* the match result**, re-expressed. So
are goal and assist points. As rolling features over *previous* matches they are entirely
legitimate — that is the same argument that keeps shots and cards in `matches.parquet`. If
the match being predicted falls inside the window, the model is reading the scoreline it is
being asked to predict, and it will backtest beautifully.

The project rule applies unchanged: `.shift(1).rolling(n)`, never `.rolling(n)`. And
`FORBIDDEN` will not catch this one, because `squad_fpl_form` is an innocent-looking name
for a column that can contain Saturday's clean sheet. The check that matters is the one
`tests/test_form.py` already performs — **change a match's score, assert that match's own
features do not move**, with the mirror test that later matches' features *do*. Every
feature in the table above belongs under both, and a feature that cannot be put under them
does not get built.

**A second, quieter leak: the API serves current values only.** There is no as-of query.
`form` fetched today is today's form, so joining it onto a match played in March silently
attaches information from after that match. Historical values must come from a stored
snapshot, and live values must be captured *before* each round and archived, the way
Phase 11 archives predictions. Fetching at prediction time and again at scoring time would
produce two different feature tables for one fixture.

### The blocker is historical coverage, and it is real

`element-summary/{player_id}/` gives `history` — per-gameweek rows — for the **current
season only**. `history_past` gives previous seasons as **season totals**, one row per
season, which cannot be shifted into a rolling pre-match window and is therefore useless
for training. Walk-forward backtesting needs per-gameweek history across all seven seasons,
so the official API alone cannot support this phase at all.

The widely used third-party archive is the `vaastav/Fantasy-Premier-League` GitHub
repository, which is understood to publish per-gameweek CSVs going back to 2016/17. **That
is hearsay until checked** — the first task of this phase is a timeboxed verification: does
it cover 2019/20 onwards, are the per-gameweek rows genuinely per-match, do the field names
survive across seasons the way `load_fifa.COLUMN_ALIASES` had to handle, and does its
licence permit the use. If the archive does not cover the seasons, the phase is dead on
arrival and should be recorded as such rather than half-built against one season.

**Name matching is the other cost, and it is Phase 4 again.** FPL names match Understat
names on exact normalisation **57%** of the time, **61%** counting the short `web_name`
form — the number Phase 14 already measured. That needs the club-and-season-scoped fuzzy
cascade, not a fresh one, plus an overrides file in the established shape. Club names need
mapping too (`Man Utd`, `Spurs`), exactly as every other source has. Budget this as work,
not as a line.

### Acceptance criteria

| Gate | The number |
|---|---|
| Beats the baselines | RPS below `baseline-elo`'s **0.2051**. Plain Elo already beats AutoGluon on 85 features; that is the bar any new signal clears before it has earned its complexity |
| Beats the model it extends | RPS below `dixon-coles-squad`'s **0.2087**, since that is the model that consumes squad quality and the only fair comparison |
| Not leaking | Correlation with goal difference around **0.43**, which is what squad-quality difference and the closing odds both manage. Materially higher is leaking, not clever |
| Leakage tests | Every new column under both `test_form.py` patterns — own match unmoved, later matches moved |
| Coverage | Share of expected-XI minutes carrying an FPL match, reported per season like Phase 4's gate. A feature that is mostly nulls is a feature that does nothing, quietly |
| Actually connected | Per the Phase 9 lesson: assert the predictions *move*. A squad-quality column that loads cleanly and changes no output is this project's signature failure |

**Kill criterion, stated in advance so it cannot be negotiated afterwards.** One honest
walk-forward run against `dixon-coles-squad`. If the full block does not beat 0.2087, and
does not beat it in a majority of the season folds, **the phase is abandoned** — not tuned,
not extended with more fields, not rescued by dropping the redundant control features and
re-running until something clears. A few thousandths of RPS over 2,660 matches is inside
the noise, so "slightly better overall, worse in three folds of six" counts as a failure,
not a marginal pass. The deliverable in that case is a paragraph in this document saying it
was tried and did not work, which is worth more than a feature nobody trusts.

### What would make this fail

- **The archive does not go back far enough**, or its per-gameweek rows are not per-match.
  No training data, no phase.
- **Name matching lands well below Phase 4's 98.6%** and the aggregates are computed over
  half an XI, which makes the feature noise dressed as information.
- **The block is redundant**, which is the base case: FPL points describe goals and assists,
  and the feature table has those already at higher resolution.
- **The API changes without notice.** It is undocumented — this is already recorded in
  Phase 14 — so a field can be renamed or removed between gameweeks. A feature the live
  prediction path depends on must degrade to a null and a warning, not a crash.
- **We talk ourselves past the kill criterion.** The most likely failure of all, and the
  reason the number is written down here before any code exists.

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

~~**Next action: Phase 11 — archive every prediction.** Not the most interesting phase, and
that is not the criterion. It is the only one whose cost grows while it waits: each round
predicted without it is a round of track record that no later work can recover. Phases 12
to 14 can be built in any order once it lands, though 12 gates predicting anything in
2026/27 at all.~~ *Done.*

~~**Next action: Phase 12 — squads for 2026/27.**~~ *Phase 13 was taken first, since 12 is
blocked and 13 was not.*

~~**Next action: Phase 14 — suspensions**, which is unblocked and needs nothing from
outside.~~ *Done.*

~~**Next action: Phase 12 — squads for 2026/27**, now unblocked.~~ *Done.*

~~**Next action: Phase 15 — the skills.**~~ *Done.*

**Nothing is queued.** Stage two is finished and the pipeline runs end to end for a season
that has not started. Phase 16 is on the table as an optional experiment, deliberately
framed with a kill criterion — and its honest expectation is that the obvious version adds
nothing. The other standing job is the seasonal one: as 2026/27 is played, keep predicting
each round *before* it happens, because that record cannot be reconstructed afterwards.
