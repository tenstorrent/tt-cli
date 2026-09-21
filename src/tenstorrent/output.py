# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Output management: data on stdout, status on stderr, three modes (human/json/quiet).

The contract that keeps `tt --json ... | jq` clean:
- *Data* (tables, JSON payloads) goes to stdout.
- *Status* (spinners, progress, warnings) goes to stderr.
- Errors are always shown: as a Rich panel on stderr in human/quiet mode, as an
  `{"error": {...}}` object on stdout in JSON mode.
- Long listings page (`less`) when stdout is a terminal and the output would not
  fit on one screen; a pipe, `--json`, `--no-pager` or TT_NO_PAGER=1 never pages.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    from .errors import TTError


def _stdout_isatty() -> bool:
    """Test seam: CliRunner swaps sys.stdout, so tests patch this, not isatty."""
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):  # closed or replaced stream
        return False


def _terminal_lines() -> int:
    return shutil.get_terminal_size(fallback=(80, 24)).lines


def pager_disabled() -> bool:
    """TT_NO_PAGER=1 is the environment form of --no-pager (git's GIT_PAGER=cat)."""
    return os.environ.get("TT_NO_PAGER", "") not in ("", "0")


def maybe_page(text: str, *, disabled: bool = False) -> None:
    """Write `text` to stdout, through the user's pager when it will not fit.

    Pages only when every one of these holds: stdout is a terminal, paging is not
    switched off (`disabled`, TT_NO_PAGER, or a pager of `cat`), and the text is
    taller than the terminal. `less` gets `-FRX` unless LESS is already set — the
    same defaults git uses: quit if it fits, keep colour, leave the screen alone.
    A pager that cannot be started never loses the output: it falls back to a
    plain write.
    """
    if not text:
        return
    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    if (
        disabled
        or pager_disabled()
        or not _stdout_isatty()
        or lines < _terminal_lines()
    ):
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    pager = (os.environ.get("TT_PAGER") or os.environ.get("PAGER") or "less").strip()
    if pager in ("", "cat"):
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    env = dict(os.environ)
    if os.path.basename(pager.split()[0]) == "less":
        env.setdefault("LESS", "-FRX")
    sys.stdout.flush()
    try:
        proc = subprocess.run(pager, shell=True, input=text.encode("utf-8", "replace"), env=env)
    except OSError:
        proc = None
    # 126/127 are the shell saying "cannot run that" — nothing was shown, so show
    # it here. Any other exit is the pager's own business (`q` is 0 in less).
    if proc is None or proc.returncode in (126, 127):
        sys.stdout.write(text)
        sys.stdout.flush()


def to_jsonable(value: Any) -> Any:
    """Convert typed models (dataclasses, enums, paths) into JSON-serializable data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class OutputManager:
    def __init__(
        self,
        *,
        json_mode: bool = False,
        quiet: bool = False,
        verbose: bool = False,
        no_pager: bool = False,
    ) -> None:
        self.json_mode = json_mode
        self.quiet = quiet
        self.verbose = verbose
        self.no_pager = no_pager
        self.data_console = Console(highlight=False)
        self.status_console = Console(stderr=True, highlight=False)

    def apply_flags(
        self,
        *,
        json_mode: bool = False,
        quiet: bool = False,
        verbose: bool = False,
    ) -> None:
        """Leaf-command flags can only turn modes on, never back off a root flag."""
        self.json_mode = self.json_mode or json_mode
        self.quiet = self.quiet or quiet
        self.verbose = self.verbose or verbose

    # -- status channel (stderr) ------------------------------------------------
    def status(self, message: str, *, style: str | None = None) -> None:
        if self.quiet or self.json_mode:
            return
        self.status_console.print(message, style=style)

    def warn(self, message: str) -> None:
        if self.quiet:
            return
        self.status_console.print(f"warning: {message}", style="yellow")

    def debug(self, message: str) -> None:
        if self.verbose and not self.quiet:
            self.status_console.print(message, style="dim")

    # -- data channel (stdout) --------------------------------------------------
    def emit(
        self,
        data: Any,
        renderer: Callable[[Any], RenderableType | None] | None = None,
        *,
        soft_wrap: bool = False,
        page: bool = False,
    ) -> None:
        """Emit a command result. `data` defines the --json schema; `renderer`
        turns it into a Rich renderable for human mode.

        `soft_wrap` disables Rich's word wrapping for renderers whose layout is
        column-aligned: wrapping would break the alignment mid-line and, worse, Rich
        may crop rather than fold. The terminal wraps instead, so nothing is lost.

        `page` sends a listing through the pager when it is taller than the
        terminal (see `maybe_page`). JSON mode is never paged: it is for pipes.
        """
        if self.json_mode:
            # Plain print, not Rich: Rich wraps at terminal width, which would
            # corrupt long JSON lines in pipelines.
            print(json.dumps(to_jsonable(data), indent=2))
            return
        if self.quiet:
            return
        renderable: Any
        if renderer is not None:
            renderable = renderer(data)
            if renderable is None:
                return
        else:
            renderable = to_jsonable(data)
        if not page:
            self.data_console.print(renderable, soft_wrap=soft_wrap)
            return
        # Render to a string first: the pager needs the whole listing, and Rich
        # keeps the colour codes when stdout is a terminal, so `less -R` shows the
        # table exactly as a direct print would.
        with self.data_console.capture() as capture:
            self.data_console.print(renderable, soft_wrap=soft_wrap)
        maybe_page(capture.get(), disabled=self.no_pager)

    # -- errors (always shown) --------------------------------------------------
    def emit_error(self, err: "TTError") -> None:
        if self.json_mode:
            print(json.dumps(to_jsonable(err.to_dict()), indent=2))
            return
        body = Text()
        body.append(err.what, style="bold red")
        if err.why:
            body.append(f"\n{err.why}")
        if err.next_step:
            body.append(f"\n→ {err.next_step}", style="bold cyan")
        log_path = err.details.get("log_path")
        if log_path:
            body.append(f"\nFull output: {log_path}", style="dim")
        self.status_console.print(
            Panel(body, title=f"error [{err.exit_code.name}]", border_style="red", expand=False)
        )
