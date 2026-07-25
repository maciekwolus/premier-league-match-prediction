# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Predicts Premier League scorelines with probabilities. Pipeline: match results +
lineups + FIFA player ratings → per-match feature table → goals models → scoreline
probability matrix → Streamlit report.

`PLAN.md` holds the full ten-phase build plan and is the source of truth for what
comes next. Read it before starting work.

## Commands

Use the venv interpreter directly; there is no need to activate it.

```
.venv/Scripts/python.exe -m pytest        # tests
.venv/Scripts/python.exe -m ruff check .  # lint
.venv/Scripts/python.exe -m ruff format . # format
```

## Architecture

- `src/config.py` — **the only place seasons are defined.** Paths, the `Season`
  dataclass, and the `SEASONS` tuple. Adding a season should mean adding one row here
  and touching nothing else. Season identifiers differ per source, so `Season` carries
  all three: `code` (football-data), `understat`, `fifa_edition`.
- Further packages (`src/data/`, `src/features/`, `src/models/`, …) are created as
  their phase begins rather than as empty stubs.

## Rules that matter

**No data leakage.** Features must only use information available *before* kickoff.
Shots, cards and half-time scores from the match being predicted are post-match facts —
using them produces a model that scores brilliantly in backtests and fails on Saturday.
Rolling averages over *previous* matches are fine.

**Validate walk-forward, never randomly.** Train on seasons 1..n, test on n+1. A random
train/test split lets the model see the future.

**Bookmaker odds are the benchmark, not a free feature.** The headline metric is Ranked
Probability Score compared against closing odds (`B365CH/CD/CA` — closing, not opening).
A model variant that uses odds as features is trained separately, to measure market
signal.

**Fail loudly on data joins.** Every season must yield exactly 380 matches. Team names
differ between sources (`Man United` vs `Manchester United`), so joins go through
explicit mapping tables and assert their row counts. Silently dropped rows here corrupt
everything downstream and are almost impossible to spot later.

## Conventions

- Data files are never committed; `data/manual/` override files are the exception.
- Intermediate data is parquet, written to `data/processed/`; the model-ready table
  goes to `data/final/`.
- Commit at the end of each phase.
