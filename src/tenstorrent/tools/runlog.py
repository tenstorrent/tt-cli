# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Per-run log files for streamed subprocesses.

Raw tool output is evidence, not UI: we keep all of it on disk and show the user a
sentence we wrote. When something fails, the error carries the path — which is what
finally makes `emit_error`'s "Full output: …" line do something.

The first line of every log is the command itself. `Runner` has always promised
"re-run with --verbose for the full command line"; recording it unconditionally is
better than making the user reproduce the failure to find out what ran.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Keep the log directory bounded: a long-lived install box would otherwise
# accumulate one file per streamed command forever.
KEEP_LOGS = 20


def _slug(text: str) -> str:
    safe = [c if (c.isalnum() or c in "-_.") else "-" for c in text]
    return "".join(safe).strip("-") or "run"


@dataclass
class RunLog:
    """An open log file. Never raises: losing a log must not fail the command."""

    path: Path
    _handle: Any = None

    def write(self, text: str) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(text)
        except Exception:
            self._handle = None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def prune(log_dir: Path, keep: int = KEEP_LOGS) -> None:
    """Drop the oldest logs, newest-first, so the directory can't grow forever."""
    try:
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in logs[keep:]:
            stale.unlink(missing_ok=True)
    except Exception:
        pass


def open_run_log(
    log_dir: Path | None,
    tool: str,
    argv: Sequence[str],
    *,
    keep: int = KEEP_LOGS,
) -> RunLog | None:
    """Open a log for one streamed command, or return None if we can't.

    A missing log is never fatal — the command still runs, the user just doesn't
    get the "Full output" pointer.
    """
    if log_dir is None:
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"{stamp}-{_slug(tool)}.log"
        handle = path.open("w", encoding="utf-8", errors="replace")
    except Exception:
        return None

    log = RunLog(path=path, _handle=handle)
    log.write(f"$ {shlex.join(str(a) for a in argv)}\n")
    log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    prune(log_dir, keep=keep)
    return log
