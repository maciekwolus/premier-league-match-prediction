---
name: check-report
description: Verify the Streamlit report in this project actually renders correctly, by measuring the live page rather than reading the source. Use this after any change to app.py, src/report/, the CSS or theme, whenever the user says the page looks wrong, is blank, shows the wrong round, or asks to check or screenshot the report — and before claiming any front-end change works.
---

# Checking the report

**Every front-end bug in this project was invisible in the source and obvious in the DOM.**
A grid that silently stacked because the pane was 393px wide. A pixel font that never
reached the expander label. An icon that would have rendered as the literal text
`keyboard_arrow_right`. A results row that showed a green "EXACT SCORE CALLED" while
quietly dropping the fact that the headline call was wrong.

None of those could be caught by reading the code, and all of them looked authoritative on
screen. So verify by measuring the rendered page, and treat "the code looks right" as
worth nothing here.

## Start it

```
.venv\Scripts\python.exe -m streamlit run app.py
```

Or use the preview tooling, which is already configured — `preview_start` with the
`report` config from `.claude/launch.json`.

**Restart the server after editing anything under `src/`.** Streamlit re-runs `app.py` on
save but does not reload imported modules, so a stale copy raises an `ImportError` for a
function that exists and whose tests pass. This costs a confusing ten minutes every time
it is forgotten.

If the page is blank on first load, it is usually still booting — read it again before
concluding anything is wrong.

## Measure what you changed

`read_page` gives the structure and the text. For anything about layout, size, colour or
computed style, run JavaScript against the page instead, because that is the only way to
see what the browser actually did:

```javascript
const cards = [...document.querySelectorAll('.pl-card')];
JSON.stringify({
  count: cards.length,
  tops: [...new Set(cards.map(c => Math.round(c.getBoundingClientRect().top)))],
  perRow: cards.filter(c => c.getBoundingClientRect().top === cards[0].getBoundingClientRect().top).length
});
```

Things worth checking, depending on what changed:

- **Counts.** Ten cards for a full round. A stat bar's cells all on one row rather than
  wrapping — compare their `top` values, do not eyeball it.
- **Text that must be present.** A caveat, a banner, a label. Assert the string is in the
  DOM rather than trusting that the branch ran.
- **The expected-XI overlay.** It is a pure-CSS checkbox toggle, so it can be opened by
  setting `.pl-modal-toggle` checked, then confirming `.pl-modal` computes to
  `display: flex`.
- **Width.** Before concluding the grid is broken, check the viewport. A narrow pane
  stacks the columns correctly, and that has been mistaken for a layout bug here.

## Be careful reading the numbers

The legend contains live samples built from the same CSS classes as the cards, which is
deliberate — it keeps the guide from drifting out of step with the page. It also means a
naive `querySelectorAll` count includes them. If a count comes out higher than the number
of fixtures, that is usually why, so scope the selector to `.pl-card` rather than adjusting
your expectations.

## Refresh the screenshots when the look changes

The README's images are generated from the running app, so they can be rebuilt rather than
going stale:

```
.venv\Scripts\python.exe tools/screenshots.py docs/screenshots
```

It drives the installed Chrome over the DevTools Protocol, waits for the card count to
*stop changing* rather than for the first card to appear, hides Streamlit's own toolbar as
host chrome, and opens the overlay by checking its toggle.

## Show the evidence

Finish by putting something concrete in front of the user — a screenshot for a visual
change, or the measured values for a structural one. "It renders correctly" is not a
report; "ten cards, five scorecard cells on one row, the caveat text present" is.

And if a server was started only to check something, either leave it running and say so,
or stop it and say so. Silently tearing it down after saying the page works has caused a
"localhost shows nothing" exchange here before.
