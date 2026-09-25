# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure formatting helpers for the UI layer: no terminal, no state, no I/O.

Everything here is a plain function so the folding matrix and the "never fake a
percentage" guard can be unit-tested without a TTY.
"""

from __future__ import annotations


def fmt_duration(seconds: float) -> str:
    """Human duration, the single formatter behind every elapsed time we print.

    Sub-second reads in ms so a fast step doesn't claim "0.0s"; past a minute we
    switch to `3m 34s`, which is what a long install actually feels like.
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def fmt_clock(seconds: float) -> str:
    """`0:42` / `12:07` — for a counter that ticks up against a deadline."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def fmt_bytes(num: float) -> str:
    """Decimal units, matching what Docker, uv, and curl report."""
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if num >= size:
            return f"{num / size:.1f} {unit}"
    return f"{int(num)} B"


def progress_bar(done: float, total: float, width: int = 14) -> str:
    """Determinate bar, or an empty string when the total is unknown.

    Returning "" for an unknown total is deliberate and load-bearing: the design
    forbids inventing a percentage. Callers show an exact counter instead.
    """
    if total <= 0:
        return ""
    filled = max(0, min(width, round(width * done / total)))
    return "▕" + "█" * filled + "░" * (width - filled) + "▏"


def show_detail(verbose: bool, in_phase: bool) -> bool:
    """The one folding predicate: `verbose or not in_phase`.

    Inside a phase on a normal run the collapsed phase line is the confirmation,
    so routine "done" output is folded; `-v` un-folds it. Outside any phase there
    is nothing to collapse into, so detail shows.

    Gate only routine confirmations on this. Failures, prompts, and actionable
    warnings must never be gated — note that on a phase-less command this returns
    True today, so a wrongly-gated failure looks fine right up until that command
    grows a phase.
    """
    return bool(verbose) or not bool(in_phase)


def tilde(path) -> str:
    """`~/.local/share/...` — shorter, and how people read their own paths."""
    text = str(path)
    try:
        import os

        home = os.path.expanduser("~")
        if home and home != "/" and text.startswith(home):
            return "~" + text[len(home) :]
    except Exception:
        pass
    return text


def elide(text: str, limit: int = 120) -> str:
    """One line of evidence, never a log viewer."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"
