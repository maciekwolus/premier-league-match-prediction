"""The permanent record of what was predicted, one file per gameweek.

    data/final/rounds/2026_27/gw01.json

Deliberately not under ``data/final/predictions/`` — ``evaluate.compare --save`` already
owns that for its per-model backtest tables, and two unrelated things sharing a directory
is how a glob quietly starts matching the wrong files.

**A stored round is never rewritten.** ``save_round`` raises rather than overwrite, and
the caller has to pass ``force=True`` to mean it. That refusal is the feature, not a
safety rail around one: the point of keeping predictions is to show what the model said
*before* a match was played, and an archive that can be quietly rewritten after the
result is known is not evidence of anything.

It also cannot be reconstructed later. Re-running a past round predicts it with today's
model and today's data, which answers a different question - and the difference is
invisible in the output, since both produce a perfectly well-formed set of probabilities.
So the record has to be written at the time or not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import FINAL_DIR

ROUNDS_DIR = FINAL_DIR / "rounds"


class RoundAlreadyStored(FileExistsError):
    """Raised when a stored gameweek would be overwritten without an explicit force."""


def round_path(season_slug: str, gameweek: int, root: Path | None = None) -> Path:
    """Where one gameweek's predictions live. Zero-padded so they sort as they read."""
    root = root or ROUNDS_DIR
    return root / season_slug / f"gw{int(gameweek):02d}.json"


def save_round(
    predictions: list[dict],
    season_slug: str,
    gameweek: int,
    root: Path | None = None,
    force: bool = False,
) -> Path:
    """Write one gameweek, refusing to replace one already stored."""
    path = round_path(season_slug, gameweek, root)
    if path.exists() and not force:
        raise RoundAlreadyStored(
            f"{path} already exists. A stored round is the record of what was predicted "
            f"before those matches were played, so it is not rewritten by default. "
            f"Pass force=True (or --force) if you genuinely mean to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return path


def load_round(season_slug: str, gameweek: int, root: Path | None = None) -> list[dict]:
    """One stored gameweek, or an empty list if it was never predicted."""
    path = round_path(season_slug, gameweek, root)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def available_rounds(root: Path | None = None) -> list[tuple[str, int]]:
    """Every stored ``(season_slug, gameweek)``, oldest first.

    Season slugs sort correctly as strings because they are zero-padded years -
    ``2025_26`` before ``2026_27``.
    """
    root = root or ROUNDS_DIR
    if not root.exists():
        return []

    rounds = []
    for path in root.glob("*/gw*.json"):
        try:
            gameweek = int(path.stem.removeprefix("gw"))
        except ValueError:
            continue  # not ours; leave it alone rather than guess
        rounds.append((path.parent.name, gameweek))
    return sorted(rounds)


def group_by_gameweek(predictions: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Split a prediction run into the rounds it actually covers.

    Usually one, but a run is not guaranteed to be: the replay window is a date range,
    and a rescheduled fixture can put two gameweeks inside it. Splitting keeps each
    stored file a real round rather than whatever happened to be predicted together.
    """
    grouped: dict[tuple[str, int], list[dict]] = {}
    for match in predictions:
        key = (match["season_slug"], int(match["gameweek"]))
        grouped.setdefault(key, []).append(match)
    return grouped
