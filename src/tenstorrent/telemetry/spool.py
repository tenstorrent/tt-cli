# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The on-disk event spool: append always, hand off rarely.

`tt` must never block on a remote POST (measured: one in-process export against a live
endpoint costs ~300-400 ms, of which only ~11-50 ms is real network). So the live path
writes events here — a JSONL append, measured at 0.01 ms — and a separate detached
process uploads the accumulated batch later (see drain.py).

Each line is one PostHog event exactly as it will appear inside the `batch` array of
the upload (see attributes.build_event), so the drain is a verbatim passthrough: what
you read in the spool is what is sent.

Layout under ``$TT_DATA_DIR/telemetry/``:
    events.jsonl          appended by every command; renamed aside by a drain
    events.sending.jsonl  a drain's in-flight batch; survives a crash and is retried
    events.started        empty marker whose mtime is the oldest entry's age
    spool.lock            flock held by the running drainer

Releases before the switch to PostHog events (tt <= 1.0.1) spooled OpenTelemetry
spans as `spool.jsonl` / `spool.sending.jsonl` / `spool.started`. Those files cannot
be uploaded any more and are deleted, not migrated (`remove_legacy`). The lock keeps
its old name on purpose, so a still-running drainer from the previous version and a
new one stay mutually excluded.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..config.paths import Paths

# Hand off to a drainer once the spool holds this many events...
DRAIN_EVENT_THRESHOLD = 20
# ...or once the oldest event has waited this long, so a light user still reports in.
DRAIN_AGE_SECONDS = 30 * 60

# Hard ceiling on events carried in one upload; oldest are dropped. A permanently
# firewalled machine must not accumulate forever (dotnet bounds each drain at
# MaxBlobsPerDrain = 200 for the same reason).
MAX_EVENTS = 512
# Belt-and-braces byte ceiling checked on the *append* path, from the stat we already
# do. Bounds disk use even if draining never succeeds and never gets to apply MAX_EVENTS.
MAX_SPOOL_BYTES = 4 * 1024 * 1024
# One write() of a line below this size to an O_APPEND descriptor is a single atomic
# syscall (POSIX guarantees it up to PIPE_BUF for pipes; Linux honours it for regular
# files well beyond), which is what makes lock-free appends from concurrent `tt`
# processes safe. A line that would exceed it is dropped rather than risk tearing.
MAX_LINE_BYTES = 8 * 1024

_LEGACY_FILES = ("spool.jsonl", "spool.sending.jsonl", "spool.started")


@dataclass(frozen=True)
class SpoolStats:
    events: int
    bytes: int
    oldest_age_seconds: float | None

    @property
    def ready_to_drain(self) -> bool:
        if self.events <= 0:
            return False
        if self.events >= DRAIN_EVENT_THRESHOLD:
            return True
        age = self.oldest_age_seconds
        return age is not None and age >= DRAIN_AGE_SECONDS


class Spool:
    """Append-only event spool plus the drain hand-off primitives."""

    def __init__(self, paths: Paths) -> None:
        self.dir = paths.telemetry_dir
        self.path = self.dir / "events.jsonl"
        self.sending_path = self.dir / "events.sending.jsonl"
        self.started_path = self.dir / "events.started"
        self.lock_path = self.dir / "spool.lock"

    # -- the live (append) path --------------------------------------------------
    def append(self, event: dict[str, Any]) -> bool:
        """Append one event as a single line. False (never an exception) if it could
        not be recorded — the command must not care."""
        try:
            data = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
            if len(data) > MAX_LINE_BYTES:
                return False
            handle = self._open_for_append()
            if handle is None:
                return False
            with handle:
                handle.write(data)
            return True
        except Exception:
            return False

    def _open_for_append(self):
        """Open the spool for a single append, creating the age marker on first use.

        Unbuffered, so the whole line goes down in one write(). The marker exists
        because we need the age of the *oldest* entry and no stat field gives that:
        mtime is the newest append, and ctime moves with every write.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.path.stat().st_size >= MAX_SPOOL_BYTES:
                return None
        except OSError:
            # No spool yet: this append starts one, so stamp the age marker — and
            # clear out anything a pre-events release left behind.
            self.remove_legacy()
            self.started_path.touch(exist_ok=True)
        return open(self.path, "ab", buffering=0)

    def remove_legacy(self) -> None:
        """Delete span spools from tt <= 1.0.1. They are in a format nothing can
        upload any more; deleting is the honest outcome (see the module docstring)."""
        for name in _LEGACY_FILES:
            _unlink(self.dir / name)

    # -- inspection (cheap enough for every command) -----------------------------
    def stats(self) -> SpoolStats:
        try:
            data = self.path.read_bytes()
        except OSError:
            return SpoolStats(events=0, bytes=0, oldest_age_seconds=None)
        age: float | None = None
        try:
            age = max(0.0, time.time() - self.started_path.stat().st_mtime)
        except OSError:
            pass
        # One line == one event: the CLI writes exactly one event per command.
        return SpoolStats(events=data.count(b"\n"), bytes=len(data), oldest_age_seconds=age)

    # -- the drain path ----------------------------------------------------------
    @contextlib.contextmanager
    def lock(self) -> Iterator[bool]:
        """Hold the drain lock; yields False if another drainer already has it.

        Two commands finishing together will both decide to hand off, so the *child*
        taking this lock is what actually prevents a double upload — the parent's
        `drain_in_progress()` probe is only an optimisation. Degrades to unlocked (yield
        True) where fcntl is unavailable; the worst case is a duplicated batch, which
        the per-event uuid lets the server deduplicate, whereas refusing to drain is not
        survivable.
        """
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield True
            return
        handle = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            handle = open(self.lock_path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    handle.close()

    def drain_in_progress(self) -> bool:
        """Best-effort probe: is a drainer holding the lock right now?"""
        with self.lock() as acquired:
            return not acquired

    def take(self) -> list[str]:
        """Claim the spooled events for upload. Call while holding the lock.

        Renames the spool aside rather than truncating it, so the claim is atomic
        against concurrent appenders and a crashed drain leaves its batch on disk to be
        retried instead of losing it. An `events.sending.jsonl` left by a previous
        failure is picked back up here, oldest first — with the same event uuids, so a
        batch the server already accepted is deduplicated rather than double-counted.
        """
        lines: list[str] = []
        for path in (self.sending_path, self.path):
            lines.extend(_read_lines(path))
        if not lines:
            return []
        # Newest MAX_EVENTS win: on a machine that has never reached the collector, the
        # recent history is the part still worth having.
        dropped = max(0, len(lines) - MAX_EVENTS)
        lines = lines[dropped:]
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.sending_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
            # Only now is it safe to drop the source: the batch is durable under its
            # new name, so a crash here re-reads it rather than losing it.
            _unlink(self.path)
            _unlink(self.started_path)
        except OSError:
            return []
        return lines

    def sent(self) -> None:
        """The claimed batch reached the collector; drop it."""
        _unlink(self.sending_path)

    def discard(self) -> None:
        """Delete every spooled event without sending it (durable opt-out).

        Spooling opens a window where data sits unsent; if the user opts out inside it,
        uploading anyway would be worse than an in-process flush, where opt-out was
        immediate and total. The lock file is left alone — it carries no event data.
        """
        for path in (self.path, self.sending_path, self.started_path):
            _unlink(path)
        self.remove_legacy()


def _read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)
