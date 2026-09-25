# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The UI layer: a phase stepper, collapsing step lines, and one live activity row.

Reached as `output.ui`, so every backend that already takes an `OutputManager`
gets it for free. Three rules shape everything here:

1. **Everything renders on stderr**, through `output.status_console`. stdout
   belongs to data and to `--json`; a stepper must never be able to corrupt a
   pipeline. That console re-resolves `sys.stderr` on every access, so CliRunner
   and capsys capture it — do not bind to `sys.__stderr__`.
2. **`Ui` is a no-op under `--json` and `--quiet`.** Handles still work (timing,
   `.detail()`, `.fail()`) so call sites stay branch-free.
3. **Motion needs a real TTY.** Piped output gets one collapsed line per step and
   no escape codes at all — not even an elapsed suffix, which would make CI
   output non-deterministic around the 0.8s threshold.

Capture is deliberately absent: `Runner` owns containing subprocess output, `Ui`
owns presenting it. That removes the reference implementation's redirect_stdout
machinery and the entire "prompt swallowed by a capturing step" hazard.
"""

from __future__ import annotations

import atexit
import contextlib
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

from rich.console import RenderableType
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text

from .format import elide, fmt_duration, show_detail
from .theme import (
    ELAPSED_THRESHOLD_S,
    GLYPH_ACTIVE,
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_FAIL,
    GLYPH_PENDING,
    GLYPH_SKIP,
    SPINNER_FRAMES,
)
from .timings import RunTimings

if TYPE_CHECKING:  # pragma: no cover
    from ..output import OutputManager

# Guards every raw escape write to the terminal. Legitimately module-global:
# it protects a process-wide resource, not per-invocation state.
_TERM_LOCK = threading.RLock()

# Enforces "one live display at a time". A spinner and an activity row both own
# the cursor; two at once interleave mid-escape-sequence and corrupt the row.
_ACTIVE_LIVE: list = []

_SIGNALS_INSTALLED = [False]


def null_ui() -> "Ui":
    """A silent Ui for code constructed without an OutputManager (tests, library
    use). Follows the NULL_SESSION idiom in telemetry/: a real object that does
    nothing, so call sites never branch on None."""
    from ..output import OutputManager

    return OutputManager(quiet=True).ui


def _tty() -> bool:
    """Stricter than Rich's `is_terminal`, which FORCE_COLOR flips true on a pipe.

    Motion is gated on this alone; colour is Rich's business (it honours NO_COLOR).
    """
    try:
        return bool(sys.__stderr__) and sys.__stderr__.isatty()
    except Exception:
        return False


class Step:
    """Handle for one operation rendered as a single collapsing line."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.start = time.monotonic()
        self.failed = False
        self.skipped = False
        self.detail_text = ""
        self.activity_text = ""

    def detail(self, text: str) -> None:
        """A muted suffix on the result line: a version, a count, a size."""
        self.detail_text = text or ""

    def set(self, label: str) -> None:
        """Update the in-progress label without ending the step."""
        self.activity_text = label or ""

    def skip(self, reason: str = "") -> None:
        self.skipped = True
        self.detail_text = reason or self.detail_text

    def fail(self) -> None:
        self.failed = True

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        return "skipped" if self.skipped else "ok"


class Phase:
    """Handle for one phase of a run."""

    def __init__(self, title: str, index: int, total: int) -> None:
        self.title = title
        self.index = index
        self.total = total
        self.start = time.monotonic()
        self.failed = False

    def fail(self) -> None:
        self.failed = True

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


class Activity:
    """One in-place line proving the run is alive while a child is silent.

    Driven by a background ticker so it keeps spinning even when the tool prints
    nothing for minutes — that's what separates "working" from "hung".
    """

    def __init__(self, ui: "Ui", label: str) -> None:
        self._ui = ui
        self.label = label
        self._stop: Any = None
        self._ticker: Any = None
        self._frame = 0

    def _start(self) -> None:
        if not self._ui.live:
            return
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._loop, daemon=True)
        self._ticker.start()

    def running(self) -> bool:
        return self._ticker is not None

    def set(self, label: str) -> None:
        self.label = label or ""

    def milestone(self, text: str) -> None:
        """A real `✓` inside the phase body, printed above the live row."""
        self._ui.milestone(text)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._ui._paint_row(SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)], self.label)
            self._frame += 1
            self._stop.wait(0.1)

    def _end(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._ticker is not None and self._ticker is not threading.current_thread():
            self._ticker.join(timeout=0.5)
        self._ticker = self._stop = None
        self._ui._erase_row()


class Ui:
    """Presentation for one command invocation. Owned by OutputManager."""

    def __init__(self, output: "OutputManager") -> None:
        self.out = output
        self.timings = RunTimings()
        self._phases: list = []
        self._in_phase = False
        self._activity: Any = None
        self._suspended = 0
        self._released = False

    # -- gating ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """False under --json/--quiet: the whole layer goes silent."""
        return not (self.out.json_mode or self.out.quiet)

    @property
    def live(self) -> bool:
        """True only when motion is safe: enabled, a real TTY, not suspended."""
        return self.enabled and _tty() and self._suspended == 0

    def show_detail(self) -> bool:
        return show_detail(self.out.verbose, self._in_phase)

    def in_phase(self) -> bool:
        return self._in_phase

    # -- raw terminal writes (always under the lock) --------------------------
    def _file(self) -> Any:
        return self.out.status_console.file

    def _paint_row(self, glyph: str, label: str) -> None:
        if not self.live:
            return
        text = Text()
        text.append(f"{glyph} ", style="accent")
        text.append(label, style="muted")
        text.no_wrap, text.overflow = True, "crop"
        with _TERM_LOCK:
            try:
                with self.out.status_console.capture() as cap:
                    self.out.status_console.print(text, end="", crop=True)
                self._file().write("\r\033[2K" + cap.get())
                self._file().flush()
            except Exception:
                pass

    def _erase_row(self) -> None:
        if not (self.enabled and _tty()):
            return
        with _TERM_LOCK:
            try:
                self._file().write("\r\033[2K")
                self._file().flush()
            except Exception:
                pass

    def _print(self, renderable: RenderableType) -> None:
        """Print into the body without landing on the live activity row.

        The ticker leaves the cursor mid-row (no trailing newline), so a body
        print must erase that row first; the ticker repaints it next tick. Both
        share _TERM_LOCK, so they cannot interleave mid-escape-sequence.
        """
        if not self.enabled:
            return
        with _TERM_LOCK:
            if self._activity is not None and self._activity.running():
                self._erase_row()
            self.out.status_console.print(renderable)

    # -- body lines -----------------------------------------------------------
    def _body_line(self, text: str, style: str) -> None:
        """A gutter line in the phase body.

        On a TTY it's a padded renderable, so a wrapped line keeps its indent. When
        piped, Padding would expand to the console width and leave trailing spaces
        on every line, which is noise in a CI log and in a substring assertion — so
        indent the string and let the terminal do the wrapping.
        """
        body = Text(text, style=style)
        if self.live:
            self._print(Padding(body, (0, 0, 0, 2)))
            return
        if not self.enabled:
            return
        with _TERM_LOCK:
            self.out.status_console.print(Text("  ") + body, soft_wrap=True)

    def note(self, text: str, *, marker: str = GLYPH_SKIP, style: str = "muted") -> None:
        """A short state line in the phase body: why something was skipped, what
        happens instead. Rendered as Text (not markup) so tool-derived content —
        image refs, paths with brackets — can't trip Rich's parser."""
        prefix = f"{marker} " if marker else "  "
        self._body_line(f"{prefix}{text}", style)

    def milestone(self, text: str, *, marker: str = GLYPH_DONE) -> None:
        self._body_line(f"{marker} {text}", "success")

    def alert(self, text: str) -> None:
        """Actionable — never folded, never gated on show_detail()."""
        self._body_line(f"{GLYPH_ALERT} {text}", "warning")

    def card(self, renderable: RenderableType) -> None:
        """Render a panel. Call it *after* a step collapses, never inside one."""
        self._print(renderable)

    # -- steps ----------------------------------------------------------------
    def _render_step(self, handle: Step) -> str:
        suffix = f"  [muted]{handle.detail_text}[/muted]" if handle.detail_text else ""
        # Elapsed only on a real terminal: piped output must be deterministic.
        if self.live and handle.elapsed >= ELAPSED_THRESHOLD_S and not handle.skipped:
            suffix += f"  [muted]{fmt_duration(handle.elapsed)}[/muted]"
        if handle.failed:
            return f"[error]{GLYPH_FAIL} {handle.label}[/error]{suffix}"
        if handle.skipped:
            return f"[muted]{GLYPH_SKIP} {handle.label}[/muted]{suffix}"
        return f"[success]{GLYPH_DONE}[/success] {handle.label}{suffix}"

    @contextlib.contextmanager
    def step(self, label: str, *, spinner: bool = True) -> Iterator[Step]:
        """One operation as one line: `label…` (spinning) → `✓ label  1.2s`.

        On a non-TTY only the collapsed result is printed — no pre-line, because
        without cursor motion it could not be overwritten and the label would
        appear twice.
        """
        handle = Step(label)
        ticker_stop: Any = None
        ticker: Any = None

        if self.live and spinner and not _ACTIVE_LIVE:
            _ACTIVE_LIVE.append("step")
            self._install_signal_handlers()
            ticker_stop = threading.Event()

            def spin() -> None:
                frame = 0
                while not ticker_stop.is_set():
                    self._paint_row(
                        SPINNER_FRAMES[frame % len(SPINNER_FRAMES)],
                        f"{handle.activity_text or handle.label}…",
                    )
                    frame += 1
                    ticker_stop.wait(0.1)

            ticker = threading.Thread(target=spin, daemon=True)
            ticker.start()

        try:
            yield handle
        except BaseException:
            handle.failed = True
            raise
        finally:
            if ticker is not None:
                ticker_stop.set()
                ticker.join(timeout=0.5)
                with contextlib.suppress(ValueError):
                    _ACTIVE_LIVE.remove("step")
                self._erase_row()
            self.timings.add_step(label, handle.elapsed, handle.status)
            if self.enabled:
                self.out.status_console.print(self._render_step(handle))

    # -- phases ---------------------------------------------------------------
    def register_phases(self, titles: Sequence[str]) -> None:
        """Declare the run's phases once, as a FIXED list.

        A fixed count is what makes `k/N` trustworthy: it must never drift with
        flags. A phase that doesn't apply is *skipped*, not removed.
        """
        self._phases = [{"title": t, "status": "pending"} for t in titles]

    def stepper_line(self) -> Text:
        """`✓ Checks ── ◉ Tools ── ○ System`, colour carrying the progress."""
        parts: list = []
        for i, p in enumerate(self._phases):
            status = p["status"]
            if status == "done":
                parts.append(f"[success]{GLYPH_DONE} {p['title']}[/success]")
            elif status == "active":
                parts.append(f"[accent.bold]{GLYPH_ACTIVE} {p['title']}[/accent.bold]")
            elif status == "failed":
                parts.append(f"[error]{GLYPH_FAIL} {p['title']}[/error]")
            elif status == "skipped":
                parts.append(f"[muted]{GLYPH_SKIP} {p['title']}[/muted]")
            else:
                parts.append(f"[dim]{GLYPH_PENDING} {p['title']}[/dim]")
            if i < len(self._phases) - 1:
                joiner = "[success] ── [/success]" if status == "done" else "[dim] ── [/dim]"
                parts.append(joiner)
        line = Text.from_markup("".join(parts))
        line.no_wrap, line.overflow = True, "crop"
        return line

    def _entry(self, title: str) -> dict:
        for p in self._phases:
            if p["title"] == title:
                return p
        self._phases.append({"title": title, "status": "pending"})
        return self._phases[-1]

    def skip_phase(self, title: str, why: str = "") -> None:
        """Mark a phase skipped, keeping the count intact. Says why, out loud —
        a skipped phase is a decision, not routine output."""
        entry = self._entry(title)
        entry["status"] = "skipped"
        self.timings.add_phase(title, 0.0, "skipped")
        if why:
            self.note(f"{title} skipped — {why}")

    def rename_phase(self, index: int, title: str) -> None:
        """Retitle a phase mid-run (Pull → Build). Count is untouched, by design.

        Use it only when the truth is genuinely unknowable before the phase opens;
        anything decidable up front belongs in register_phases(), and anything that
        just doesn't apply is a skip_phase().
        """
        if 0 <= index < len(self._phases):
            self._phases[index]["title"] = title

    @contextlib.contextmanager
    def phase(self, title: str) -> Iterator[Phase]:
        """Bracket one phase. Its chrome (stepper + rule) is TTY-only; piped runs
        keep the step lines, which carry the same information without motion."""
        entry = self._entry(title)
        entry["status"] = "active"
        index = self._phases.index(entry) + 1
        handle = Phase(title, index, len(self._phases))

        if self.live:
            self.out.status_console.print(self.stepper_line())
            self.out.status_console.print(
                Rule(f"[bold accent]{title}[/bold accent]", align="left", style="muted")
            )

        self._in_phase = True
        try:
            yield handle
        except BaseException:
            entry["status"] = "failed"
            handle.failed = True
            raise
        else:
            entry["status"] = "failed" if handle.failed else "done"
        finally:
            self._in_phase = False
            self.timings.add_phase(title, handle.elapsed, entry["status"])
            if self.live:
                glyph = (
                    f"[error]{GLYPH_FAIL}[/error]"
                    if entry["status"] == "failed"
                    else f"[success]{GLYPH_DONE}[/success]"
                )
                self.out.status_console.print(
                    f"{glyph} [muted]Phase {index}/{handle.total} ·[/muted] "
                    f"[bold accent]{title}[/bold accent]  "
                    f"[muted]{fmt_duration(handle.elapsed)}[/muted]"
                )

    def final_stepper(self) -> None:
        """Leave the completed stepper as a permanent record of the run."""
        if self.live and self._phases:
            self.out.status_console.print(self.stepper_line())

    # -- activity row ---------------------------------------------------------
    @contextlib.contextmanager
    def activity(self, label: str) -> Iterator[Activity]:
        """One live line for a long child process. TTY-only; a no-op when piped."""
        handle = Activity(self, label)
        if self.live and not _ACTIVE_LIVE:
            _ACTIVE_LIVE.append("activity")
            self._install_signal_handlers()
            self._activity = handle
            handle._start()
        try:
            yield handle
        finally:
            if self._activity is handle:
                handle._end()
                self._activity = None
                with contextlib.suppress(ValueError):
                    _ACTIVE_LIVE.remove("activity")

    # -- prompts and hand-off -------------------------------------------------
    @contextlib.contextmanager
    def prompting(self) -> Iterator[None]:
        """Suspend every live row around an input()/getpass/sudo prompt.

        A spinner repainting the row a prompt is sitting on doesn't just look bad,
        it hides the prompt and the CLI appears to hang.
        """
        self._suspended += 1
        activity, self._activity = self._activity, None
        if activity is not None:
            activity._end()
        self._erase_row()
        try:
            yield
        finally:
            self._suspended -= 1
            if activity is not None:
                self._activity = activity
                activity._start()

    def handoff(self) -> None:
        """Give the terminal back before a child takes it over."""
        self.release()

    def release(self) -> None:
        """Idempotent teardown: stop tickers, erase the row, show the cursor.

        Wired into run_before_exec, atexit, and SIGTERM/SIGHUP, because exec_tty
        never returns and a crash must not leave a spinner thread painting.
        """
        if self._released:
            return
        self._released = True
        activity, self._activity = self._activity, None
        if activity is not None:
            with contextlib.suppress(Exception):
                activity._end()
        _ACTIVE_LIVE.clear()
        if _tty():
            with _TERM_LOCK, contextlib.suppress(Exception):
                self._file().write("\r\033[2K\033[?25h")
                self._file().flush()

    def _install_signal_handlers(self) -> None:
        """Restore the terminal on a signal, once per process, main thread only."""
        if _SIGNALS_INSTALLED[0] or threading.current_thread() is not threading.main_thread():
            return
        _SIGNALS_INSTALLED[0] = True
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with contextlib.suppress(Exception):
                previous = signal.getsignal(sig)

                def handler(signum: int, frame: Any, _prev: Any = previous) -> None:
                    self.release()
                    if callable(_prev):
                        _prev(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
        atexit.register(self.release)
