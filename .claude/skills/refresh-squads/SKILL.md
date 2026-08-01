---
name: refresh-squads
description: Update Premier League squads in this project after a transfer, injury, or suspension, then rebuild so predictions reflect it. Use this whenever the user mentions a player moving club, a new signing, someone being injured or out, a squad being wrong or stale, ratings looking off for a team, or asks to "update the squads" or "add a transfer" — and also when a prediction looks wrong for a club and a squad change might be why.
---

# Refreshing squads

Player ratings are an annual snapshot: EA publishes them in September and they do not
move again. Every transfer after that leaves a player rated correctly and filed at the
wrong club. Four hand-maintained files under `data/manual/` correct that, and this is the
workflow for using them.

**Each file records a *change*, never whole state.** A file restating twenty squads to
move one player goes stale within a week, and a stale file that looks authoritative is
worse than no file at all. One line, then a rebuild.

## Pick the right file

| Situation | File | Columns |
|---|---|---|
| Player transferred; ratings still list the old club | `squad_changes.csv` | `season,fifa_player_name,team,note` |
| Player has no FIFA entry at all — signed from another league or straight out of an academy | `player_ratings_manual.csv` | `season,fifa_player_name,overall,age,position,note` |
| Player is injured or otherwise unavailable | `unavailable.csv` | `season,team,player,from_date,until_date,reason,note` |
| The name matcher missed a pair | `player_name_overrides.csv` | `season,understat_player,fifa_player_name,confidence,reason` |

Three things worth knowing before you edit:

- **A blank `team` in `squad_changes.csv` means the player left the league**, and the row
  drops them rather than moving them.
- **A blank `until_date` in `unavailable.csv` means open-ended**, which is the honest
  default for an injury with no announced return date.
- **Suspensions need no row.** Red cards and yellow accumulation are derived from
  `lineups.parquet` automatically. Adding a suspended player by hand would double-count.

`player_name_overrides.csv` is consulted before the fuzzy matching cascade and always
wins. Reach for a row there rather than loosening a matching threshold — one override
fixes one player, while a looser threshold puts every other match at risk.

## Rebuild

```
.venv\Scripts\python.exe -m src.features.build
.venv\Scripts\python.exe -m src.predict.gameweek
```

Between seasons the fixture feed is empty, so add `--replay` to the second command to
re-predict the last round that was actually played.

## Then verify it actually did something

This is the step that matters and the one that is easy to skip. This project has already
shipped a squad-change mechanism that was fully written, tested and documented while
nothing in the prediction path called it — every test passed and the feature did nothing.
A typo in a name has exactly the same signature: the file loads, the run succeeds, and
the squad is unchanged.

Two checks, both cheap:

**Read the run's output for complaints.** A change naming a player nobody recognises is
reported rather than silently ignored:

```
squad change for an unknown player: 'A. Playr'
```

If you see that, the name does not match what the ratings call the player. Look up the
exact spelling in `fifa_players.parquet` rather than guessing.

**Confirm the number moved.** Compare the squad rating before and after:

```python
import pandas as pd
from src.predict.gameweek import features_for
from src.predict.fixtures import upcoming_fixtures, as_matches

raw, _ = upcoming_fixtures(allow_download=False)
features, problems, _ = features_for(as_matches(raw))
print(features[["match_id", "home_squad_overall_mean", "away_squad_overall_mean"]])
print(problems)
```

A club whose squad you just changed should show a different mean. If it is identical, the
edit did not reach the model, and the run "succeeding" tells you nothing.

Also watch `rated_share`. A value below 1.0 means some of the expected XI could not be
matched to a rating, and a value of 0.0 means none of them could — which usually points
at a club whose players are absent from the lookup season's name map.

## Expect small effects, and say so

Moving one player changes a squad mean by roughly (their rating − their replacement's)
÷ 11. Measured on real suspensions across 2025/26, the median change was **zero**,
because more often than not the missing player was not in the expected XI anyway. Against
a column standard deviation of 3.4, a typical single change is a fraction of that.

So a correct edit that barely moves the prediction is the normal case, not a failure. Do
not go looking for a bug, and do not oversell the update — report what actually changed.
