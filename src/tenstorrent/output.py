# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Output management: data on stdout, status on stderr, three modes (human/json/quiet).

The contract that keeps `tt --json ... | jq` clean:
- *Data* (tables, JSON payloads) goes to stdout.
- *Status* (spinners, progress, warnings) goes to stderr.
- Errors are always shown: as a Rich panel on stderr in human/quiet mode, as an
  `{"error": {...}}` object on stdout in JSON mode.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    from .errors import TTError


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
    ) -> None:
        self.json_mode = json_mode
        self.quiet = quiet
        self.verbose = verbose
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
    def status(
        self, message: str, *, style: str | None = None, soft_wrap: bool = False
    ) -> None:
        """`soft_wrap` as in emit(): for a message carrying a path or command line,
        which Rich would otherwise break mid-token at the terminal width."""
        if self.quiet or self.json_mode:
            return
        self.status_console.print(message, style=style, soft_wrap=soft_wrap)

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
    ) -> None:
        """Emit a command result. `data` defines the --json schema; `renderer`
        turns it into a Rich renderable for human mode.

        `soft_wrap` disables Rich's word wrapping for renderers whose layout is
        column-aligned: wrapping would break the alignment mid-line and, worse, Rich
        may crop rather than fold. The terminal wraps instead, so nothing is lost.
        """
        if self.json_mode:
            # Plain print, not Rich: Rich wraps at terminal width, which would
            # corrupt long JSON lines in pipelines.
            print(json.dumps(to_jsonable(data), indent=2))
            return
        if self.quiet:
            return
        if renderer is not None:
            renderable = renderer(data)
            if renderable is not None:
                self.data_console.print(renderable, soft_wrap=soft_wrap)
        else:
            self.data_console.print(to_jsonable(data))

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
