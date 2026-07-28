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

/* ------------------------------------------------------------------- notices */

.pl-notice {{
  border: 2px solid {ACCENT_2}; background: rgba(241,196,15,0.08);
  padding: 0.7rem 0.85rem; margin-bottom: 1rem;
  font-family: ui-monospace, monospace; font-size: 0.72rem; line-height: 1.65;
}}
.pl-notice b {{ color: {ACCENT_2}; }}
.pl-notice code {{ color: {ACCENT_2}; background: rgba(0,0,0,0.35); padding: 0 0.25rem; }}
</style>
"""
