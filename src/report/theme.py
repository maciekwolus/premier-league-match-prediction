"""The report's look: a retro league-scoreboard styling.

Kept as one string rather than scattered inline styles so the palette can be changed in
one place, and so ``app.py`` stays layout rather than decoration.

The reference is the printed league table and teletext results page of the late eighties
and nineties - hard edges, no gradients, a monospaced grid, and a palette narrow enough
that everything on the page could plausibly have been printed with three inks.
"""

INK = "#F4F1E8"  # newsprint white
PAPER = "#111418"  # near-black
PANEL = "#1A1F26"
ACCENT = "#E63946"  # club red
ACCENT_2 = "#F1C40F"  # scoreboard amber
MUTED = "#7C8798"
UP = "#4ADE80"
DOWN = "#FB7185"

CSS = f"""
<style>
/* Pixel typography where it is available, with a monospaced fallback that keeps the
   grid honest if the font cannot be fetched. */
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

:root {{
  --ink: {INK};
  --paper: {PAPER};
  --panel: {PANEL};
  --accent: {ACCENT};
  --accent2: {ACCENT_2};
  --muted: {MUTED};
}}

.stApp {{
  background:
    repeating-linear-gradient(
      0deg, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px,
      transparent 1px, transparent 3px
    ),
    {PAPER};
  color: {INK};
}}

/* Streamlit's default max width wastes half the screen on a three-column grid. */
.block-container {{ max-width: 1500px; padding-top: 2.2rem; }}

h1, h2, h3, .pl-pixel {{
  font-family: 'Press Start 2P', ui-monospace, 'Courier New', monospace !important;
  letter-spacing: 0.02em;
}}

h1 {{
  font-size: 1.45rem !important;
  color: {INK} !important;
  text-shadow: 3px 3px 0 {ACCENT};
  margin-bottom: 0.2rem !important;
}}

.pl-sub {{
  font-family: ui-monospace, 'Courier New', monospace;
  color: {MUTED};
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 1.1rem;
}}

/* ------------------------------------------------------------------ stat bar */

.pl-statbar {{
  display: flex; flex-wrap: wrap; gap: 0;
  border: 2px solid {INK};
  margin-bottom: 1.1rem;
}}
.pl-stat {{
  flex: 1 1 25%;
  padding: 0.55rem 0.7rem;
  border-right: 2px solid {INK};
  background: {PANEL};
}}
.pl-stat:last-child {{ border-right: 0; }}
.pl-stat-key {{
  display: block; font-family: ui-monospace, monospace;
  font-size: 0.6rem; letter-spacing: 0.13em; color: {MUTED};
}}
.pl-stat-val {{
  display: block; margin-top: 0.25rem;
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.82rem; color: {ACCENT_2};
}}

/* --------------------------------------------------------------------- card */

.pl-card {{
  background: {PANEL};
  border: 2px solid {INK};
  box-shadow: 5px 5px 0 rgba(0,0,0,0.55);
  padding: 0.7rem 0.75rem 0.8rem;
  margin-bottom: 1.05rem;
  font-family: ui-monospace, 'Courier New', monospace;
}}

.pl-date {{
  display: flex; justify-content: space-between;
  font-size: 0.62rem; letter-spacing: 0.1em;
  color: {MUTED}; border-bottom: 1px dashed rgba(244,241,232,0.25);
  padding-bottom: 0.4rem; margin-bottom: 0.55rem;
}}
.pl-xg {{ color: {MUTED}; }}

.pl-teams {{
  display: grid; grid-template-columns: 1fr auto 1fr;
  align-items: start; gap: 0.35rem; margin-bottom: 0.5rem;
}}
.pl-team {{ display: flex; flex-direction: column; align-items: center; gap: 0.3rem; }}
.pl-badge {{ image-rendering: pixelated; display: block; }}
.pl-name {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.52rem; line-height: 1.5; text-align: center; color: {INK};
}}
.pl-v {{ color: {ACCENT}; font-size: 0.85rem; padding-top: 0.9rem; }}

.pl-verdict {{
  text-align: center; font-size: 0.6rem; letter-spacing: 0.16em;
  color: {PAPER}; background: {ACCENT_2};
  padding: 0.22rem 0; margin-bottom: 0.6rem;
}}

/* ---------------------------------------------------------------- scorelines */

.pl-score-row {{ display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.22rem; }}
.pl-score {{
  width: 2.3rem; font-size: 0.72rem; color: {MUTED};
  font-family: 'Press Start 2P', ui-monospace, monospace;
}}
.pl-score.pl-lead {{ color: {INK}; }}
.pl-bar-track {{ flex: 1; height: 10px; background: rgba(244,241,232,0.10); display: block; }}
.pl-bar {{ display: block; height: 100%; background: {MUTED}; }}
.pl-bar.pl-lead {{ background: {ACCENT}; }}
.pl-pct {{ width: 2.2rem; text-align: right; font-size: 0.68rem; color: {INK}; }}

/* ------------------------------------------------------------------ outcomes */

.pl-outcomes {{
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0.3rem; margin: 0.6rem 0 0.45rem;
}}
.pl-outcome {{
  background: rgba(244,241,232,0.06);
  border: 1px solid rgba(244,241,232,0.16);
  padding: 0.35rem 0.2rem; text-align: center;
}}
.pl-outcome-key {{ font-size: 0.58rem; letter-spacing: 0.1em; color: {MUTED}; }}
.pl-outcome-val {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.72rem; color: {INK}; margin-top: 0.2rem;
}}
.pl-edge {{ display: block; font-size: 0.55rem; margin-top: 0.18rem; }}
.pl-up {{ color: {UP}; }}
.pl-down {{ color: {DOWN}; }}

.pl-market {{
  font-size: 0.58rem; letter-spacing: 0.08em; color: {MUTED};
  border-top: 1px dashed rgba(244,241,232,0.25); padding-top: 0.4rem;
}}
.pl-market-empty {{ color: rgba(124,135,152,0.7); font-style: italic; }}

.pl-flag {{
  margin-top: 0.45rem; padding: 0.3rem 0.4rem;
  background: rgba(230,57,70,0.14); border-left: 3px solid {ACCENT};
  font-size: 0.55rem; letter-spacing: 0.06em; color: {INK};
}}

/* ------------------------------------------------------------------- lineups */

/* The button. It has to read as pressable at a glance - the previous version was a
   line of small grey text and nobody would guess it did anything. */
.pl-xi-button {{
  display: block; margin-top: 0.55rem; cursor: pointer; user-select: none;
  text-align: center;
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.52rem; letter-spacing: 0.08em;
  color: {PAPER}; background: {ACCENT_2};
  border: 2px solid {INK}; box-shadow: 3px 3px 0 rgba(0,0,0,0.5);
  padding: 0.45rem 0.3rem;
  transition: transform 0.04s ease, box-shadow 0.04s ease;
}}
.pl-xi-button:hover {{ background: {INK}; }}
.pl-xi-button:active {{ transform: translate(3px, 3px); box-shadow: none; }}

/* Checkbox toggle rather than a details element or a Streamlit widget: the overlay must
   float above the grid without moving it, and without rerunning the script. */
.pl-modal {{
  display: none;
  position: fixed; inset: 0; z-index: 9999;
  align-items: center; justify-content: center;
  padding: 1.5rem;
}}
.pl-modal-toggle:checked ~ .pl-modal {{ display: flex; }}

.pl-modal-backdrop {{
  position: absolute; inset: 0;
  background: rgba(8,10,12,0.82);
  cursor: pointer;
}}
.pl-modal-box {{
  position: relative;
  width: min(760px, 100%); max-height: 88vh; overflow-y: auto;
  background: {PANEL}; border: 2px solid {INK};
  box-shadow: 8px 8px 0 rgba(0,0,0,0.6);
  padding: 0.9rem 1rem 1rem;
}}
.pl-modal-head {{
  display: flex; justify-content: space-between; align-items: center;
  border-bottom: 2px solid {INK}; padding-bottom: 0.55rem; margin-bottom: 0.8rem;
}}
.pl-modal-title {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.68rem; color: {INK};
}}
.pl-modal-close {{
  cursor: pointer; user-select: none;
  font-family: ui-monospace, monospace; font-size: 0.9rem; line-height: 1;
  color: {PAPER}; background: {ACCENT};
  border: 2px solid {INK}; padding: 0.2rem 0.5rem;
}}
.pl-modal-close:hover {{ background: {INK}; color: {ACCENT}; }}
.pl-modal-note {{
  margin-top: 0.85rem; padding-top: 0.6rem;
  border-top: 1px dashed rgba(244,241,232,0.25);
  font-family: ui-monospace, monospace; font-size: 0.6rem;
  line-height: 1.65; color: {MUTED};
}}
.pl-modal-note b {{ color: {ACCENT_2}; }}

/* ---------------------------------------------------------------------- pitch */

.pl-pitch {{
  background:
    repeating-linear-gradient(
      0deg, rgba(255,255,255,0.022) 0 28px, transparent 28px 56px
    ),
    #16301F;
  border: 2px solid rgba(244,241,232,0.35);
  padding: 0.7rem 0.5rem;
}}
.pl-halfway {{
  border-top: 2px dashed rgba(244,241,232,0.35);
  margin: 0.55rem 0;
}}
.pl-side {{ display: flex; flex-direction: column; gap: 0.45rem; }}
.pl-side-head {{
  display: flex; align-items: baseline; gap: 0.6rem;
  font-family: ui-monospace, monospace; font-size: 0.58rem;
  letter-spacing: 0.08em; color: {INK};
}}
.pl-side-away .pl-side-head {{ order: 99; }}
.pl-side-name {{ font-weight: 700; }}
.pl-side-shape {{ color: {MUTED}; }}
.pl-side-mean {{ margin-left: auto; color: {ACCENT_2}; }}

.pl-pitch-row {{
  display: flex; justify-content: center; flex-wrap: wrap;
  gap: 0.35rem 0.7rem;
}}
.pl-token {{
  display: flex; flex-direction: column; align-items: center; gap: 0.1rem;
  width: 4.6rem; text-align: center;
}}
.pl-token-kit {{ image-rendering: pixelated; display: block; }}
.pl-token-name {{
  font-family: ui-monospace, monospace; font-size: 0.55rem; color: {INK};
  max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.pl-token-rating {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.5rem; color: {ACCENT_2};
}}
.pl-xi-unrated {{ color: {MUTED}; }}
.pl-xi-empty {{
  font-family: ui-monospace, monospace; font-size: 0.6rem;
  color: {MUTED}; font-style: italic; padding: 0.8rem 0; text-align: center;
}}

@media (max-width: 640px) {{
  .pl-token {{ width: 3.9rem; }}
  .pl-modal {{ padding: 0.6rem; }}
}}

/* -------------------------------------------------------------------- legend */

/* Streamlit's expander is the only widget on the page that keeps its own chrome, so
   it is restyled rather than replaced - a details/summary of our own would lose the
   open/closed state on every rerun. */
[data-testid="stExpander"] {{
  border: 2px solid {MUTED} !important;
  border-radius: 0 !important;
  background: {PANEL} !important;
  margin-bottom: 1.1rem;
}}
[data-testid="stExpander"] summary {{ padding: 0.7rem 0.8rem !important; }}

/* The label sits inside a markdown container that sets its own font, so the styling
   has to reach that rather than the summary. The toggle arrow is deliberately left
   alone: it is a Material Symbols ligature, and restyling its font would render the
   word "keyboard_arrow_right" instead of an arrow. */
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {{
  font-family: 'Press Start 2P', ui-monospace, monospace !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.08em;
  color: {ACCENT_2} !important;
  margin: 0 !important;
}}
[data-testid="stExpander"] summary:hover
  [data-testid="stMarkdownContainer"] p {{ color: {INK} !important; }}
[data-testid="stExpander"] [data-testid="stIconMaterial"] {{ color: {ACCENT_2}; }}

.pl-legend {{
  font-family: ui-monospace, 'Courier New', monospace;
  font-size: 0.72rem; line-height: 1.7; color: {INK};
}}
.pl-legend-lead, .pl-legend-foot {{
  color: {MUTED}; padding: 0.2rem 0 0.9rem;
}}
.pl-legend-foot {{
  border-top: 1px dashed rgba(244,241,232,0.25);
  margin-top: 0.7rem; padding-top: 0.8rem; padding-bottom: 0.2rem;
}}
.pl-legend-foot b, .pl-legend-lead b {{ color: {ACCENT_2}; }}

.pl-legend-row {{
  display: grid;
  grid-template-columns: 7rem minmax(180px, 15rem) 1fr;
  gap: 0.9rem; align-items: center;
  padding: 0.7rem 0;
  border-top: 1px dashed rgba(244,241,232,0.18);
}}
.pl-legend-key {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.55rem; color: {ACCENT}; letter-spacing: 0.04em; line-height: 1.6;
}}
.pl-legend-sample {{ min-width: 0; }}
.pl-legend-sample .pl-date {{ margin-bottom: 0; border-bottom: 0; padding-bottom: 0; }}
.pl-legend-sample .pl-outcomes {{ margin: 0; }}
.pl-legend-sample .pl-market {{ border-top: 0; padding-top: 0; }}
.pl-legend-sample .pl-flag {{ margin-top: 0; }}
.pl-legend-text b {{ color: {INK}; }}
.pl-legend-text b.pl-up {{ color: {UP}; }}
.pl-legend-text b.pl-down {{ color: {DOWN}; }}

/* One column on a phone, where three would be unreadable. */
@media (max-width: 640px) {{
  .pl-legend-row {{ grid-template-columns: 1fr; gap: 0.45rem; }}
}}

/* ------------------------------------------------------------- round picker */

/* The archive laid out flat rather than hidden behind a dropdown. A selectbox filtered
   as you typed - baffling with two options - and made "which rounds have we predicted?"
   a question you had to open a menu to answer. These borrow the expected-XI button's
   language, so the page has one idea of what a pressable thing looks like. */

.pl-picker-key {{
  font-family: ui-monospace, monospace;
  font-size: 0.6rem; letter-spacing: 0.13em; color: {MUTED};
  margin: 0 0 0.3rem;
}}

/* Streamlit still emits the widget label when it is collapsed; it is duplicated by the
   key above, so it goes rather than being shown twice. */
[data-testid="stButtonGroup"] [data-testid="stWidgetLabel"] {{ display: none; }}

[data-testid="stButtonGroup"] {{ margin-bottom: 0.7rem; }}
[data-testid="stButtonGroup"] [role="radiogroup"] {{ gap: 0.35rem; flex-wrap: wrap; }}

/* Attribute selectors rather than the emotion class hashes beside them, which change
   whenever Streamlit is upgraded. */
[data-testid="stButtonGroup"] button[data-variant="segmented_control"],
[data-testid="stButtonGroup"] button[data-variant="pills"] {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.52rem; letter-spacing: 0.06em;
  color: {INK}; background: {PANEL};
  border: 2px solid {MUTED}; border-radius: 0;
  padding: 0.5rem 0.7rem;
  transition: none;
}}

[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover,
[data-testid="stButtonGroup"] button[data-variant="pills"]:hover {{
  border-color: {INK}; color: {INK}; background: {PANEL};
}}

/* Selected reads like the verdict bar on a card: amber block, dark text. */
[data-testid="stButtonGroup"] button[data-selected="true"] {{
  color: {PAPER} !important; background: {ACCENT_2} !important;
  border-color: {INK} !important;
  box-shadow: 3px 3px 0 rgba(0,0,0,0.5);
}}

[data-testid="stButtonGroup"] button:focus-visible {{
  outline: 2px solid {ACCENT_2}; outline-offset: 2px;
}}

/* The label markup Streamlit wraps each option in carries its own colour. */
[data-testid="stButtonGroup"] button p {{ color: inherit !important; margin: 0; }}

/* ------------------------------------------------------------------- notices */

.pl-notice {{
  border: 2px solid {ACCENT_2}; background: rgba(241,196,15,0.08);
  padding: 0.7rem 0.85rem; margin-bottom: 1rem;
  font-family: ui-monospace, monospace; font-size: 0.72rem; line-height: 1.65;
}}
.pl-notice b {{ color: {ACCENT_2}; }}
.pl-notice code {{ color: {ACCENT_2}; background: rgba(0,0,0,0.35); padding: 0 0.25rem; }}

/* A caveat, not an alert. Same shape as a notice but muted, so the warning about a
   small sample does not shout louder than the numbers it is qualifying. */
.pl-notice-quiet {{
  border-color: {MUTED}; background: rgba(124,135,152,0.07);
  color: {MUTED};
}}
.pl-notice-quiet b {{ color: {INK}; }}

/* --------------------------------------------------------- results and scorecard */

/* The result of a match that has been played, on its own card. */
.pl-result {{
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0.55rem; margin-bottom: 0.5rem;
  border: 2px solid {MUTED};
  font-family: ui-monospace, monospace; font-size: 0.68rem;
}}
.pl-result-key {{ color: {MUTED}; letter-spacing: 0.1em; }}
.pl-result-score {{
  font-family: 'Press Start 2P', ui-monospace, monospace;
  font-size: 0.8rem; color: {INK};
}}
/* An exact scoreline is rare enough to be worth colouring; the outcome alone is not. */
.pl-hit {{ margin-left: auto; color: {UP}; letter-spacing: 0.08em; }}
.pl-hit-soft {{ color: {MUTED}; }}
.pl-miss {{ margin-left: auto; color: {MUTED}; letter-spacing: 0.08em; }}
/* When both appear together the second must not be pushed to the right as well, or the
   pair splits across the row and reads as two unrelated labels. */
.pl-hit + .pl-miss {{ margin-left: 0.4rem; }}

.pl-scorecard {{ border-color: {ACCENT_2}; }}
/* Five cells, not the four the summary bar carries, so they need a fifth of the width
   each - left at 25% the last one wraps onto a row of its own and reads as an
   afterthought rather than the point. */
.pl-scorecard .pl-stat {{ flex-basis: 20%; border-right-color: {ACCENT_2}; }}
</style>
"""
