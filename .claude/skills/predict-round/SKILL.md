---
name: predict-round
description: Predict and permanently archive a Premier League gameweek in this project. Use this whenever the user asks to predict the next round, the opening weekend, this week's fixtures, a specific gameweek, or wants to know what the model says about upcoming matches — and also when they ask to refresh what the report is showing, or mention the archive refusing to overwrite a stored round.
---

# Predicting a round

One command predicts a round and writes it to the archive:

```
.venv\Scripts\python.exe -m src.predict.gameweek
```

Between seasons the fixture feed is empty, so `--replay` re-predicts the last round that
was actually played — that is the only way to exercise this path from June to August, and
the report labels the result as a replay so nobody mistakes it for next week's matches.

Useful flags: `--model` picks a different model, `--offline` never downloads fixtures,
`--force` replaces an already-stored round.

## A new season needs its schedule first

`football-data`'s fixture feed only covers the next few days, which is no use in July. The
Fantasy Premier League API carries the whole season:

```
.venv\Scripts\python.exe -m src.data.fetch_fpl
.venv\Scripts\python.exe -m src.data.clean_fpl
```

`clean_fpl` validates before it writes — exactly 20 clubs, 380 fixtures, 19 home and 19
away for each — and raises rather than storing a schedule that is the right size and the
wrong shape. If it complains about 19 clubs, a promoted side is missing from
`FPL_TO_FOOTBALL_DATA` in `src/matching/team_names.py`; add it rather than relaxing the
check, because the count is the only thing standing between a missing club and a silent
gap.

## The archive refuses to overwrite, and that is correct

Re-running a round that is already stored fails:

```
data\final\rounds\2026_27\gw01.json already exists. A stored round is the record of what
was predicted before those matches were played, so it is not rewritten by default.
```

**This is the feature working, not a bug to route around.** The archive answers exactly
one question — what did the model say *before* those matches were played — and a file
that can be quietly replaced after the result is known answers nothing. It also cannot be
reconstructed: re-running a past round predicts it with today's model and today's data,
and that output is indistinguishable from the original.

Use `--force` only when you genuinely mean to replace the record — for instance after
fixing a bug in a round that has not been played yet. If the matches have already been
played, think hard before overwriting, because you are erasing the only evidence of what
was predicted beforehand.

## Check the output before trusting it

Read the printed round. Three things are worth a glance:

- **Do the favourites look right?** A big club at home to a promoted side should be a
  clear favourite. If everything sits near even, something upstream is flat.
- **Is 1-1 on nearly every card?** The default model tops about 60% of matches with 1-1,
  which is the sport rather than a fault — it stays the single most likely score until one
  side is expected to score about 2.4 goals. Literally every card is a different matter,
  and usually means `--model poisson-glm`, which hedges to 74%.
- **Does any club look like an average Premier League squad when it should not?** Squad
  columns that come out null are filled downstream with the training median, which
  describes a newly promoted club to the model as thoroughly average. This project has
  shipped that bug twice.

To check that last one directly:

```python
from src.predict.gameweek import features_for
from src.predict.fixtures import upcoming_fixtures, as_matches

raw, _ = upcoming_fixtures(allow_download=False)
features, problems, _ = features_for(as_matches(raw))
quality = ["home_squad_overall_mean", "away_squad_overall_mean"]
share = ["home_rated_share", "away_rated_share"]
print(features[["match_id", *quality, *share]])
```

Any NaN, or a `rated_share` of 0.0, means a club is reaching the model without squad
quality. That is worth fixing before storing the round, because the archive is permanent.

## Show it

The report opens on the newest stored round, so it picks up a new prediction on reload.
If anything under `src/` changed, restart the server first — Streamlit re-runs `app.py`
but does not reload imported modules, and a stale module raises an `ImportError` for a
function that plainly exists and whose tests pass.
