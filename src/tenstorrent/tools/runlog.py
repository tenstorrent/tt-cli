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

import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Keep the log directory bounded: a long-lived install box would otherwise
# accumulate one file per streamed command forever.
KEEP_LOGS = 20

# Enforce secure POSIX permissions: logs directory is accessible only by the owner (0700),
# and log files are readable/writable only by the owner (0600).
DIR_MODE = 0o700
FILE_MODE = 0o600


def _slug(text: str) -> str:
    safe = [c if (c.isalnum() or c in "-_.") else "-" for c in text]
    return "".join(safe).strip("-") or "run"


def open_secure_file(
    path: Path,
    mode: str = "w",
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    buffering: int = -1,
    dir_mode: int = DIR_MODE,
    file_mode: int = FILE_MODE,
) -> Any:
    """Open a file ensuring its parent directory is 0700 and the file is 0600."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=dir_mode)
    try:
        os.chmod(path.parent, dir_mode)
    except OSError:
        pass

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if "+" in mode:
        flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    elif "a" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND

    fd = os.open(path, flags, file_mode)
    try:
        os.chmod(path, file_mode, follow_symlinks=False)
    except (OSError, NotImplementedError):
        try:
            os.chmod(path, file_mode)
        except OSError:
            pass

    if "b" in mode:
        return open(fd, mode, buffering=buffering)
    return open(fd, mode, buffering=buffering, encoding=encoding, errors=errors)


@dataclass
class RunLog:
    """An open log file. Never raises: losing a log must not fail the command."""

    path: Path
    _handle: Any = None
    _redactor: Any = None

    def write(self, text: str) -> None:
        if self._handle is None:
            return
        try:
            if self._redactor is not None:
                text = self._redactor.text(text)
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

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def prune(log_dir: Path, keep: int = KEEP_LOGS) -> None:
    """Drop the oldest logs, newest-first, so the directory can't grow forever."""
    try:
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in logs[keep:]:
            stale.unlink(missing_ok=True)
    except Exception:
        pass


from ..redaction import Redactor


def open_run_log(
    log_dir: Path | None,
    tool: str,
    argv: Sequence[str],
    *,
    keep: int = KEEP_LOGS,
    redactor: Any = None,
) -> RunLog | None:
    """Open a log for one streamed command, or return None if we can't.

    A missing log is never fatal — the command still runs, the user just doesn't
    get the "Full output" pointer.
    """
    if log_dir is None:
        return None
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"{stamp}-{_slug(tool)}.log"
        handle = open_secure_file(path, "w", encoding="utf-8", errors="replace")
    except Exception:
        return None

    active_redactor = redactor or Redactor()
    clean_argv = active_redactor.argv([str(a) for a in argv])

    log = RunLog(path=path, _handle=handle, _redactor=active_redactor)
    log.write(f"$ {shlex.join(clean_argv)}\n")
    log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    prune(log_dir, keep=keep)
    return log
