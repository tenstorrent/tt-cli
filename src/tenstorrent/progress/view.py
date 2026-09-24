# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The serve checklist as the user sees it: one row per boot step, on stderr.

A live row is repainted in place while it runs and settled once, permanently,
when it finishes — so what stays on screen after a serve is the list of what
happened, and a piped or `--json` run records the same rows as plain lines
instead of a redraw storm.

Rendering only: the rows are decided by `progress.tracker`. The status channel
is stderr throughout (`output.py`), which keeps `tt --json serve | jq` clean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..output import OutputManager

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_WIDTH = 8
_EIGHTHS = " ▏▎▍▌▋▊▉"
_LABEL_WIDTH = 32
_LIVE_WIDTH = 80
# Fast enough that the spinner and the elapsed seconds read as continuous, and
# cheap: a repaint is a dozen short rows.
_REFRESH_PER_SECOND = 12.5
_SPINNER_FPS = 10  # one full turn a second


def format_bytes(value: float) -> str:
    """Decimal (1000-based) sizes, matching what Hugging Face and docker report."""
    units = ("B", "kB", "MB", "GB", "TB", "PB")
    index = 0
    while value >= 1000 and index < len(units) - 1:
        value /= 1000
        index += 1
    decimals = 0 if value >= 100 or index == 0 else 1 if value >= 10 else 2
    return f"{value:.{decimals}f} {units[index]}"


def format_duration(seconds: float) -> str:
    """`42s` / `1m 04s` / `1h 07m` — the elapsed column."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


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

    def progress_text(self) -> str:
        if self.total <= 0:
            return ""
        fraction = min(1.0, max(0.0, self.done / self.total))
        full, part = divmod(round(fraction * _BAR_WIDTH * 8), 8)
        bar = ("█" * full + (_EIGHTHS[part] if part else "")).ljust(_BAR_WIDTH)
        if self.is_bytes:
            done, total = format_bytes(self.done), format_bytes(self.total)
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
        console = output.status_console
        self._animate = (
            console.is_terminal
            and not output.quiet
            and not output.json_mode
            and not output.verbose
        )
        self._silent = output.quiet or output.json_mode

    # -- lifecycle --------------------------------------------------------------------
    def __enter__(self) -> "Checklist":
        if self._animate:
            self._output.status("Ctrl-C stops watching; the server keeps starting.", style="dim")
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
            self._settle("○", "dim", detail="interrupted")
        elif exc_type is not None:
            self._settle("✗", "red")
        else:
            self._settle("✓", "green")
        self.close()

    def close(self) -> None:
        self._closed = True
        if self._live is not None:
            self._live.refresh()  # settle the last row before the block freezes
            self._live.stop()
            self._live = None

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
        self._settle("✓", "green")
        self._reach(label)
        self._rows.append(_Row(label, mark="✓", style="green", detail=detail or "", elapsed=0.0))
        self._emit(self._rows[-1])

    def begin(self, label: str, *, placeholder: bool = False) -> None:
        self._settle("✓", "green")
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
        self._settle("✓", "green", label=label, detail=detail)

    def fail(self, label: str | None = None, detail: str | None = None) -> None:
        self._settle("✗", "red", label=label, detail=detail)

    def note(self, text: str) -> None:
        """An aside that belongs between rows rather than on one."""
        self._settle("✓", "green")
        self._rows.append(_Row(text, mark=" ", style="dim", elapsed=0.0))
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
        if row.placeholder and mark == "✓" and label is None and detail is None:
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
            row.detail = format_bytes(row.done)
        row.mark, row.style = mark, style
        row.elapsed = time.monotonic() - row.started
        self._emit(row)
        self._refresh()

    def _emit(self, row: _Row) -> None:
        """Print a settled row once. On a terminal it goes above the live area,
        which only ever holds the active row and the footer: a frame taller than
        the terminal cannot be redrawn in place and leaves a copy per refresh."""
        if self._silent:
            return
        if self._live is not None:
            with self._live._lock:
                self._unwrap()
                self._live.console.print(self._line(row, time.monotonic()))
            return
        if self._animate:
            return
        parts = [f"{row.mark} {row.label}".strip()]
        if row.detail:
            parts.append(row.detail)
        if row.elapsed and row.elapsed >= 1:  # as in _line: sub-second is noise
            parts.append(format_duration(row.elapsed))
        self._output.status("  " + "  ".join(parts), style=row.style)

    def _refresh(self) -> None:
        """Repaint now, rather than waiting up to a tick for the refresh thread."""
        if self._live is not None:
            self._live.refresh()

    def _line(self, row: _Row, now: float) -> Text:
        """A finished row: plain text, no trailing padding, so a narrower
        terminal has nothing to re-wrap."""
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("   ")
        line.append(row.mark, style=row.style)
        line.append(f"  {row.label}".ljust(_LABEL_WIDTH + 2), style="dim")
        if row.detail:
            line.append(f"  {row.detail}", style="dim")
        if row.elapsed and row.elapsed >= 1:
            line.append(f"  {format_duration(row.elapsed)}", style="dim")
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
        counted = sum(1 for r in list(self._rows) if not r.placeholder and r.mark != " ")
        total = counted + len(self._pending)
        current = counted if row is not None and not row.placeholder else min(counted + 1, total)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("   ")
        line.append(_SPINNER[int(now * _SPINNER_FPS) % len(_SPINNER)], style="cyan")
        # First, so a long line is clipped in its detail and never here.
        line.append(
            f"  [{max(current, 1)}/{max(total, 1)} · {format_duration(now - self._started)}]",
            style="dim",
        )
        if row is not None:
            line.append(f"  {row.label}")
            if now - row.started >= 1:
                line.append(f"  {format_duration(now - row.started)}", style="dim")
            extra = row.progress_text() or row.detail
            if extra:
                line.append(f"  {extra}", style="dim")
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


def ready_panel(title: str, rows: list[tuple[str, str]], footer: str) -> Panel:
    """The end-of-serve card: where the server is, and what to do with it next."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(label, value)
    body = Table.grid()
    body.add_column()
    body.add_row(table)
    body.add_row(Text())
    body.add_row(Text(footer, style="dim"))
    return Panel(body, title=f"[bold]{title}[/bold]", title_align="left",
                 border_style="green", padding=(1, 2), expand=False)
