---
name: audit-squads
description: Check every Premier League club's expected XI against its real current squad and report who has left or arrived, then propose the fixes. Use this whenever the user asks to check or audit the squads, says a lineup contains someone who has transferred or is no longer at a club, mentions the transfer window, asks whether the squads are up to date, or before predicting the first round after a window closes.
---

# Auditing squads against reality

Appearances are last season's and ratings are a September snapshot, so both go stale the
moment a window opens. Casemiro started for Man United all through 2025/26, left, and was
still being picked for the opening round of 2026/27 — a year after leaving.

**FPL is the source that knows.** It publishes every club's current squad, free and
updated weekly, because people pick those players. This skill diffs our expected XIs
against it.

## Run the audit

```
.venv\Scripts\python.exe -m src.predict.gameweek --offline --force
```

Departures are applied automatically and printed as warnings:

```
warning: Man United: left the club - Casemiro
warning: Arsenal: left the club - Leandro Trossard
```

To see the whole picture without writing a round, and to include arrivals:

```python
import pandas as pd
from src.predict.gameweek import lineups_with_dates
from src.predict.squads import recent_starters
from src.predict.transfers import arrivals, departures, fpl_squads, rating_index

squads = fpl_squads()
lineups = lineups_with_dates()
fifa = pd.read_parquet("data/processed/fifa_players.parquet")
when = pd.Timestamp("2026-08-21")

for club in sorted(squads):
    pool = set(recent_starters(lineups, club, when)["player"])
    gone = departures(sorted(pool), club, squads)
    came = arrivals(club, squads, pool, fifa=rating_index(fifa, club, "2025/26"))[:5]
    if gone or came:
        print(club, "| out:", gone, "| in:", [(p["player"], p["overall"]) for p in came])
```

## What to do with the result

**Departures need nothing.** They are already removed from the expected XI and the
next-most-used player steps up. The warning exists so the change is visible, not so you act
on it.

**Arrivals are reported and deliberately not selected.** A signing has never played for the
club, so there is nothing to rank them against the players who have. Putting them straight
into the eleven would be asserting a team sheet rather than describing one, and this project
does not have the evidence for that.

If you know a signing will start, that is a human judgement. Give them a rating in
`data/manual/player_ratings_manual.csv` if FIFA has never heard of them; the `refresh-squads`
skill covers that workflow.

## Read the output carefully

**Scope matters more than it looks.** The check runs over the club's *recent* starters —
the pool the XI is actually drawn from. Run it against a club's full appearance history
instead and it reports Cristiano Ronaldo as a Man United departure, which is true and
useless.

**A long list for one club is usually real.** A relegated-then-promoted side has a recent
history from two seasons ago and a squad that has turned over almost completely; ten
departures there is not a bug.

**A false departure is the error that costs something.** Leaving a departed player in makes
one prediction slightly stale; wrongly removing a current player deletes him from the side
silently. So matching is generous on purpose — `Amad Diallo Traore` matches FPL's `Amad`,
and `Bruno Fernandes` matches `Bruno Borges Fernandes`. If a name is flagged that you know
is still at the club, that is a matching bug worth fixing rather than a transfer.

Check one directly before believing it:

```python
from src.predict.transfers import fpl_squads

squad = fpl_squads()["Man United"]
print([p["web_name"] for p in squad if "casem" in p["full_name"].lower()])
```

An empty list means FPL genuinely does not have them.

## Then re-predict

A stored round is never rewritten, so correcting squads for a round already archived needs
`--force` — and that is only appropriate if those matches have not been played yet. The
`predict-round` skill covers why.
