# Skills

Four skills live in [`.claude/skills/`](.claude/skills/). Claude Code discovers them
automatically, and they are committed, so they work for anyone who clones this repository.

They are not a second copy of [COMMANDS.md](COMMANDS.md). The commands are the easy half
and already written down there. **What a skill carries is the reasoning and the traps** —
what a correct result looks like, which failures are silent, and which alarming-looking
outputs are actually fine.

| Skill | Reach for it when | The trap it exists to prevent |
|---|---|---|
| [`refresh-squads`](.claude/skills/refresh-squads/SKILL.md) | A transfer, a signing, an injury, or a club's ratings look wrong | A squad edit can load cleanly, run successfully and change nothing at all |
| [`audit-squads`](.claude/skills/audit-squads/SKILL.md) | Checking whether squads are still real, or a lineup names someone who has left | Scoping the check to a club's whole history reports Cristiano Ronaldo as a departure |
| [`predict-round`](.claude/skills/predict-round/SKILL.md) | Predicting the next round, the opening weekend, or a specific gameweek | The archive refusing to overwrite is the feature working, not an error to route around |
| [`check-report`](.claude/skills/check-report/SKILL.md) | After any front-end change, or when the page looks wrong | Every front-end bug here was invisible in the source and obvious in the DOM |

## Why these four

They are the workflows that actually recur. Each was written **after** its workflow had
settled rather than before — a skill encodes a way of working, and encoding one too early
just freezes a guess.

Each also says what a *correct* result looks like, which turns out to matter as much as the
steps. A genuine squad change often moves the prediction barely at all — the median effect
of a suspension measured across a season was **zero** — so a skill that only listed commands
would leave you hunting for a bug that is not there.

## Using one

Ask for the task and Claude Code picks the skill up on its own: *"a player transferred, update
the squads"*, *"predict this weekend"*, *"does the report still render?"* You can also name
one directly.

## Adding one

A new skill needs a directory under `.claude/skills/` containing `SKILL.md`, with
frontmatter naming it and describing when it applies:

```markdown
---
name: my-skill
description: What it does, and the situations that should trigger it.
---
```

The `name` must match the directory. The `description` is the whole triggering mechanism,
so it should name the situations in the user's words rather than describe the
implementation.

**Worth writing one when the workflow has a trap in it.** If the steps are self-evident
from `COMMANDS.md`, a skill adds nothing — the value is in what the commands do not tell
you.

## Not in here

The skills Claude Code offers everywhere — code review, document handling, and the rest —
are not listed here. They belong to the user's own setup rather than to this project, they
are already presented to Claude automatically, and any copy of that list kept here would be
wrong the first time a plugin is installed or removed.
