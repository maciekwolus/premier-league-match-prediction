# Premier League Match Prediction

Predicts the **scoreline** of upcoming Premier League fixtures, with probabilities — and
shows every prediction next to the bookmaker's line, so it is obvious whether the model is
actually adding anything.

```
Crystal Palace vs Arsenal     0-1 (14%) · 0-2 (13%) · 1-1 (11%)
                              Home 15% | Draw 24% | Away 61%
                              Bookmaker: 24% | 26% | 51%
```

## Why this exists

**It is a workout for Claude Code, disguised as a football project.** The prediction is
real and the numbers are honest, but the point was to find out what an AI coding agent can
do when the task is long enough to have a memory problem, a research problem, and a taste
problem all at once.

What that turned into, concretely:

- **A written plan as the contract.** [PLAN.md](PLAN.md) defined ten phases up front, and
  every session started by reading it. An agent with a plan argues with you about scope;
  an agent without one agrees with everything and drifts. It later grew a second stage,
  and the original estimates were left in place where they turned out wrong — what the
  plan expected against what happened is the more useful record.
- **Persistent instructions that outlive the conversation.** [CLAUDE.md](CLAUDE.md) is the
  project's operating memory — the leakage rules, the join contracts, the data quirks that
  cost hours to discover. It exists because the *interesting* failures were never syntax
  errors, they were facts nobody wrote down.
- **Parallel subagents on genuinely parallel work.** Seven agents, one per season, run on
  the *residue* left by automated name-matching rather than instead of it. Their most
  valuable output was negative: by refusing to guess, they exposed an upstream filtering
  bug rather than papering over it.
- **One branch, one PR, one review per phase.** Nothing reached `main` unread.
- **Verifying the UI by measuring the rendered page**, not by reading the source. Every
  front-end bug here was invisible in the code and obvious in the DOM.

The most useful lesson had nothing to do with football: **a feature with passing tests can
be connected to nothing at all.** An entire squad-adjustment mechanism was built, tested
and documented before anyone noticed the prediction path never called it.

## What it does

Seven seasons of results are joined to the **players who actually started each match**, and
those players are matched to their FIFA ratings — so the model knows not just that Arsenal
are playing, but which Arsenal turned up. Rolling form, expected goals, Elo and rest days
go in alongside. A set of goals models turns that into two expected-goals numbers per
fixture, which become a full scoreline probability matrix.

| Input | Source |
|---|---|
| Results, shots, cards, odds | [football-data.co.uk](https://www.football-data.co.uk/englandm.php) |
| Lineups and expected goals | [Understat](https://understat.com) |
| Player ratings | FIFA 20–23 / EA Sports FC 24–26 |

**Trained on 2019/20 through 2025/26 — 2,660 matches, 99 features, 9 models benchmarked.**
It predicts **2026/27**, a season with no results yet: the twenty clubs and all 380
fixtures come from the Fantasy Premier League API, and clubs promoted into the division
get an eleven built from ratings, since there is no history here to read one from.

Every prediction is **archived the moment it is made and never rewritten**, because the
only interesting question about a forecast is what it said *before* the match — and that
cannot be reconstructed afterwards.

## The report

![The report — three fixtures to a row, scoreline bars, and the bookmaker's line beneath each card](docs/screenshots/report.png)

Styled after a teletext results page. Each card gives the most likely scorelines, the
home/draw/away split, and where the model **disagrees with the market by more than ten
points** — the only genuinely interesting thing a model can offer once a market exists.
Club crests are trademarked, so each side gets a pixel kit instead, generated as inline SVG.

Once a round has been played the cards show the final score and whether the call stood up,
with a season-to-date scorecard putting our RPS next to the bookmaker's over the same
fixtures. Under thirty scored matches it says plainly that this is not a sample — and if
that small sample happens to beat the market, it says not to believe it.

Every card opens an **expected XI** — both sides on a pitch, attackers meeting at the
halfway line, with each player's FIFA rating and the team average. This is where the
squad-quality signal stops being a number in a table:

![The expected XI overlay — Burnley and Wolves laid out in formation with FIFA ratings per player](docs/screenshots/expected-xi.png)

## Does it beat the bookmaker?

**No — and that is the correct result.** Ranked Probability Score, walk-forward over 2,280
matches, lower is better:

| | RPS |
|---|---|
| **bookmaker (closing odds)** | **0.1965** |
| gbm-with-odds | 0.2027 |
| poisson-glm | 0.2039 |
| baseline-elo | 0.2051 |
| dixon-coles-squad *(what the report shows)* | 0.2087 |
| baseline-league-average | 0.2346 |

*A selection; [CLAUDE.md](CLAUDE.md) carries all ten.*

A model that beat the closing line should be suspected of data leakage before it is
believed. Two things worth noticing: **plain Elo beats AutoGluon on 85 features**, and the
report deliberately does *not* use the best-scoring model — `poisson-glm` wins on RPS by
hedging, which makes it call 1-1 in 74% of matches. When the deliverable is a scoreline,
that is a failure whatever the score says.

## Running it

**[SETUP.md](SETUP.md) — how to install and run the whole app**, from a fresh clone to the
report on screen.

Already built? Then it is two lines, and the `cd` is the one people skip:

```bash
cd C:\repositories\premier-league-match-prediction
```
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

## Also here

| File | What it is |
|---|---|
| [SETUP.md](SETUP.md) | Install and run the whole app, step by step |
| [COMMANDS.md](COMMANDS.md) | Every command and flag, plus what the error messages mean |
| [PLAN.md](PLAN.md) | The ten-phase build plan and where it landed |
| [CLAUDE.md](CLAUDE.md) | Architecture contracts, leakage rules, and the data quirks worth knowing |
| [.claude/skills/](.claude/skills/) | Three Claude Code skills: refresh squads, predict a round, check the report |

```
src/config.py     seasons, paths, source URLs — the single place seasons are defined
src/data/         acquisition and cleaning, one module per source
src/matching/     name reconciliation between sources
src/features/     squad quality, form, Elo, and the feature table
src/models/       goals models and the scoreline matrix
src/evaluate/     scoring, walk-forward backtesting, the bookmaker benchmark
src/predict/      fixtures, expected XIs, and predictions for unplayed matches
src/report/       shaping predictions for display
app.py            the Streamlit report
tests/            435 tests, no network access and no reading of data/
data/manual/      hand-written overrides: names, squad changes, fixtures (committed)
```

**Status: complete.** Stage one built the pipeline; stage two made it survive a live
season — archived predictions, a scored history, suspensions, and squads for clubs the
data has never seen. Almost no data is committed: it is gitignored and rebuilt from source.
