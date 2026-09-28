# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The serve checklist as the user sees it: one row per boot step, on stderr.

A live row is repainted in place while it runs and settled once, permanently,
when it finishes — so what stays on screen after a serve is the list of what
happened, and a piped or `--json` run records the same rows as plain lines
instead of a redraw storm.

Rendering only: the rows are decided by `progress.tracker`. The status channel
is stderr throughout (`output.py`), which keeps `tt --json serve | jq` clean.

It speaks the `ui` design language (docs/cli-output.md): the same glyphs, theme
names, spinner and formatters as a phase body, rows in the body's two-space
gutter, motion only on a real TTY, and no durations when piped. It is its own
Live rather than a `ui.activity()` because a boot needs a counter, a bar and a
detail on the one live line — but it takes the same one-live-display slot.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

from rich.console import RenderableType
from rich.live import Live
from rich.text import Text

from ....output import OutputManager
from ....ui import console as ui_console
from ....ui.format import fmt_bytes, fmt_clock, fmt_duration
from ....ui.theme import (
    ELAPSED_THRESHOLD_S,
    GLYPH_DONE,
    GLYPH_FAIL,
    GLYPH_SKIP,
    SPINNER_FRAMES,
)

_BAR_WIDTH = 8
_EIGHTHS = " ▏▎▍▌▋▊▉"
_GUTTER = "  "
_LIVE_WIDTH = 80
# Fast enough that the spinner and the elapsed seconds read as continuous, and
# cheap: a repaint is a dozen short rows.
_REFRESH_PER_SECOND = 12.5
_SPINNER_FPS = 10  # one full turn a second


@dataclass
class _Row:
    label: str
    mark: str = ""  # "" while active
    style: str = ""
    detail: str = ""
    started: float = field(default_factory=time.monotonic)
    elapsed: float | None = None
    done: float = 0.0
    total: float = 0.0
    is_bytes: bool = False
    #: a row that only holds the spinner honest until something reports; the
    #: next begin() replaces it rather than leaving a ✓ for work never done.
    placeholder: bool = False
    #: an aside between rows, not a step: not counted, never timed.
    is_note: bool = False

    def progress_text(self) -> str:
        if self.total <= 0:
            return ""
        fraction = min(1.0, max(0.0, self.done / self.total))
        full, part = divmod(round(fraction * _BAR_WIDTH * 8), 8)
        bar = ("█" * full + (_EIGHTHS[part] if part else "")).ljust(_BAR_WIDTH)
        if self.is_bytes:
            done, total = fmt_bytes(self.done), fmt_bytes(self.total)
            same_unit = done.split()[-1] == total.split()[-1]
            counts = f"{done.split()[0]}/{total}" if same_unit else f"{done} / {total}"
        else:
            counts = f"{int(self.done)}/{int(self.total)}"
        return f"▕{bar}▏ {counts} · {fraction * 100:.0f}%"


class Checklist:
    """A live list of steps for one long job. Use as a context manager."""

    def __init__(self, output: OutputManager) -> None:
        self._output = output
        self._rows: list[_Row] = []
        self._pending: list[str] = []
        self._closed = False
        self._drawn = 0
        self._started = time.monotonic()
        self._live: Live | None = None
        # Motion on the same terms as every other live row: a real TTY, not
        # --json/--quiet (ui.live), and not -v, whose raw lines would tear it.
        self._animate = output.ui.live and not output.verbose
        self._silent = not output.ui.enabled

    # -- lifecycle --------------------------------------------------------------------
    def __enter__(self) -> "Checklist":
        # One live display at a time: a ui spinner painting while this Live
        # redraws would interleave mid-escape-sequence.
        if self._animate and not ui_console._ACTIVE_LIVE:
            ui_console._ACTIVE_LIVE.append("checklist")
            self._output.ui.note("Ctrl-C stops watching; the server keeps starting.")
            self._live = Live(
                get_renderable=self._render,
                console=self._output.status_console,
                refresh_per_second=_REFRESH_PER_SECOND,
                transient=False,
            )
            self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            self._settle(GLYPH_SKIP, "muted", detail="interrupted")
        elif exc_type is not None:
            self._settle(GLYPH_FAIL, "error")
        else:
            self._settle(GLYPH_DONE, "success")
        self.close()

    def close(self) -> None:
        self._closed = True
        if self._live is not None:
            self._live.refresh()  # settle the last row before the block freezes
            self._live.stop()
            self._live = None
            with contextlib.suppress(ValueError):
                ui_console._ACTIVE_LIVE.remove("checklist")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    # -- rows -------------------------------------------------------------------------
    def plan(self, labels: list[str]) -> None:
        """The steps still to come, drawn dim until they start. One the run
        skips over is dropped when a later one begins."""
        self._pending = list(labels)
        self._refresh()

    def instant(self, label: str, detail: str | None = None) -> None:
        """A step that was already true when we looked (no timing worth showing)."""
        self._settle(GLYPH_DONE, "success")
        self._reach(label)
        self._rows.append(
            _Row(label, mark=GLYPH_DONE, style="success", detail=detail or "", elapsed=0.0)
        )
        self._emit(self._rows[-1])

    def begin(self, label: str, *, placeholder: bool = False) -> None:
        self._settle(GLYPH_DONE, "success")
        self._reach(label)
        self._rows.append(_Row(label, placeholder=placeholder))
        self._refresh()

    def detail(self, text: str) -> None:
        if self._active is not None:
            self._active.detail = text
            self._refresh()

    def progress(self, done: float, total: float, *, is_bytes: bool = False) -> None:
        row = self._active
        if row is None or total <= 0:
            return
        row.done, row.total, row.is_bytes = done, total, is_bytes
        self._refresh()

    def done(self, label: str | None = None, detail: str | None = None) -> None:
        self._settle(GLYPH_DONE, "success", label=label, detail=detail)

    def fail(self, label: str | None = None, detail: str | None = None) -> None:
        self._settle(GLYPH_FAIL, "error", label=label, detail=detail)

    def note(self, text: str) -> None:
        """An aside that belongs between rows rather than on one: a state, as
        `ui.note()` draws one, not a step."""
        self._settle(GLYPH_DONE, "success")
        self._rows.append(_Row(text, mark=GLYPH_SKIP, style="muted", elapsed=0.0, is_note=True))
        self._emit(self._rows[-1])

    # -- internals --------------------------------------------------------------------
    def _reach(self, label: str) -> None:
        if label in self._pending:
            del self._pending[: self._pending.index(label) + 1]

    @property
    def _active(self) -> _Row | None:
        if self._rows and self._rows[-1].elapsed is None:
            return self._rows[-1]
        return None

    def _settle(self, mark: str, style: str, *, label: str | None = None,
                detail: str | None = None) -> None:
        row = self._active
        if row is None:
            return
        # A placeholder only ever vanishes; marking it ✗ or ○ is the honest
        # outcome for a wait that was interrupted or never reported.
        if row.placeholder and mark == GLYPH_DONE and label is None and detail is None:
            self._rows.pop()
            self._refresh()
            return
        if label is not None:
            row.label = label
        if detail is not None:
            row.detail = detail
        if not row.detail and row.is_bytes and row.done:
            # The bar is about to be replaced by the settled row's detail, so
            # keep the one number worth keeping: how much actually came down.
            row.detail = fmt_bytes(row.done)
        row.mark, row.style = mark, style
        row.elapsed = time.monotonic() - row.started
        self._emit(row)
        self._refresh()

    def _emit(self, row: _Row) -> None:
        """Print a settled row once. On a terminal it goes above the live area,
        which only ever holds the active row: a frame taller than the terminal
        cannot be redrawn in place and leaves a copy per refresh."""
        if self._silent:
            return
        if self._live is not None:
            with self._live._lock:
                self._unwrap()
                self._live.console.print(self._line(row, timed=True))
            return
        # Piped (or -v): the same row with no elapsed suffix, as for a ui step —
        # a CI log must not flap around the threshold. soft_wrap, so a long
        # detail is not padded out to the console width.
        self._output.status_console.print(self._line(row, timed=False), soft_wrap=True)

    def _refresh(self) -> None:
        """Repaint now, rather than waiting up to a tick for the refresh thread."""
        if self._live is not None:
            self._live.refresh()

    @staticmethod
    def _line(row: _Row, *, timed: bool) -> Text:
        """A finished row, as a ui step line reads: `✓ label  detail  1.2s`.
        Plain text with no trailing padding, so a narrower terminal has nothing
        to re-wrap."""
        line = Text(_GUTTER, no_wrap=True, overflow="ellipsis")
        line.append(row.mark, style=row.style)
        line.append(f" {row.label}", style="muted" if row.is_note or row.style == "muted" else "")
        if row.detail:
            line.append(f"  {row.detail}", style="muted")
        if timed and row.elapsed and row.elapsed >= ELAPSED_THRESHOLD_S:
            line.append(f"  {fmt_duration(row.elapsed)}", style="muted")
        return line

    def _render(self) -> RenderableType:
        """One short, unpadded line. A terminal made narrower re-wraps what is
        already drawn, and Live then redraws from the wrong row, leaving a copy
        per refresh; one line under _LIVE_WIDTH is rarely re-wrapped at all."""
        self._unwrap()
        if self._closed:
            return Text()
        now = time.monotonic()
        # A snapshot: Live's refresh thread calls this while the main thread is
        # still appending rows and filling them in.
        row = self._active
        counted = sum(1 for r in list(self._rows) if not r.placeholder and not r.is_note)
        total = counted + len(self._pending)
        current = counted if row is not None and not row.placeholder else min(counted + 1, total)
        line = Text(_GUTTER, no_wrap=True, overflow="ellipsis")
        line.append(SPINNER_FRAMES[int(now * _SPINNER_FPS) % len(SPINNER_FRAMES)], style="accent")
        # First, so a long line is clipped in its detail and never here.
        line.append(
            f" [{max(current, 1)}/{max(total, 1)} · {fmt_clock(now - self._started)}]",
            style="muted",
        )
        if row is not None:
            line.append(f"  {row.label}")
            if now - row.started >= 1:
                line.append(f"  {fmt_clock(now - row.started)}", style="muted")
            extra = row.progress_text() or row.detail
            if extra:
                line.append(f"  {extra}", style="muted")
        width = self._output.status_console.width
        line.truncate(min(width, _LIVE_WIDTH), overflow="ellipsis")
        self._drawn = line.cell_len
        return line

    def _unwrap(self) -> None:
        """A terminal made narrower re-wraps the live line onto several rows,
        and Live only clears the last. Clear them all, straight to the terminal
        and before Live redraws: a control inside the live frame would be
        replayed under every row printed above it, erasing that row."""
        console = self._output.status_console
        if self._drawn > console.width:
            rows = -(-self._drawn // console.width) - 1
            console.file.write("\r\x1b[2K" + "\x1b[1A\x1b[2K" * rows)
            console.file.flush()
        self._drawn = min(self._drawn, console.width)
