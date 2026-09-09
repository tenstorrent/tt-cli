# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The on-disk span spool: append always, hand off rarely.

`tt` must never block on a remote POST (measured: one span through OTLPSpanExporter
costs ~300-400 ms against a live endpoint, of which only ~11-50 ms is real network).
So the live path writes spans here — a JSONL append, measured at 0.01 ms — and a
separate detached process uploads the accumulated batch later (see drain.py).

The file format is the spec'd `OTLP File Exporter <https://opentelemetry.io/docs/specs/
otel/protocol/file-exporter/>`_ JSON Lines, produced by the official
`opentelemetry-exporter-otlp-json-file` exporter rather than an invented schema, so the
spool is readable by any OTLP tooling and drain.py can rebuild the exact protobuf
payload the direct exporter would have sent.

Layout under ``$TT_DATA_DIR/telemetry/``:
    spool.jsonl          appended by every command; renamed aside by a drain
    spool.sending.jsonl  a drain's in-flight batch; survives a crash and is retried
    spool.started        empty marker whose mtime is the oldest entry's age
    spool.lock           flock held by the running drainer
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..config.paths import Paths

# Hand off to a drainer once the spool holds this many spans...
DRAIN_SPAN_THRESHOLD = 20
# ...or once the oldest span has waited this long, so a light user still reports in.
DRAIN_AGE_SECONDS = 30 * 60

# Hard ceiling on spans carried in one upload; oldest are dropped. A permanently
# firewalled machine must not accumulate forever (dotnet bounds each drain at
# MaxBlobsPerDrain = 200 for the same reason).
MAX_SPANS = 512
# Belt-and-braces byte ceiling checked on the *append* path, from the stat we already
# do. Bounds disk use even if draining never succeeds and never gets to apply MAX_SPANS.
MAX_SPOOL_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class SpoolStats:
    spans: int
    bytes: int
    oldest_age_seconds: float | None

    @property
    def ready_to_drain(self) -> bool:
        if self.spans <= 0:
            return False
        if self.spans >= DRAIN_SPAN_THRESHOLD:
            return True
        age = self.oldest_age_seconds
        return age is not None and age >= DRAIN_AGE_SECONDS


class Spool:
    """Append-only span spool plus the drain hand-off primitives."""

    def __init__(self, paths: Paths) -> None:
        self.dir = paths.telemetry_dir
        self.path = self.dir / "spool.jsonl"
        self.sending_path = self.dir / "spool.sending.jsonl"
        self.started_path = self.dir / "spool.started"
        self.lock_path = self.dir / "spool.lock"

    # -- the live (append) path --------------------------------------------------
    def exporter(self) -> Any:
        return _AppendingFileSpanExporter(self)

    def _open_for_append(self):
        """Open the spool for a single append, creating the age marker on first use.

        The marker exists because we need the age of the *oldest* entry and no stat
        field gives that: mtime is the newest append, and ctime moves with every write.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.path.stat().st_size >= MAX_SPOOL_BYTES:
                return None
        except OSError:
            # No spool yet: this append starts one, so stamp the age marker.
            self.started_path.touch(exist_ok=True)
        return open(self.path, "a", encoding="utf-8")

    # -- inspection (cheap enough for every command) -----------------------------
    def stats(self) -> SpoolStats:
        try:
            data = self.path.read_bytes()
        except OSError:
            return SpoolStats(spans=0, bytes=0, oldest_age_seconds=None)
        age: float | None = None
        try:
            age = max(0.0, time.time() - self.started_path.stat().st_mtime)
        except OSError:
            pass
        # One line == one span: the CLI writes exactly one span per command, and each
        # export produces exactly one JSONL record.
        return SpoolStats(spans=data.count(b"\n"), bytes=len(data), oldest_age_seconds=age)

    # -- the drain path ----------------------------------------------------------
    @contextlib.contextmanager
    def lock(self) -> Iterator[bool]:
        """Hold the drain lock; yields False if another drainer already has it.

        Two commands finishing together will both decide to hand off, so the *child*
        taking this lock is what actually prevents a double upload — the parent's
        `drain_in_progress()` probe is only an optimisation. Degrades to unlocked (yield
        True) where fcntl is unavailable; the worst case is a duplicated batch, which is
        survivable for analytics, whereas refusing to drain is not.
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
        """Claim the spooled spans for upload. Call while holding the lock.

        Renames the spool aside rather than truncating it, so the claim is atomic
        against concurrent appenders and a crashed drain leaves its batch on disk to be
        retried instead of losing it. A `spool.sending.jsonl` left by a previous failure
        is picked back up here, oldest first.
        """
        lines: list[str] = []
        for path in (self.sending_path, self.path):
            lines.extend(_read_lines(path))
        if not lines:
            return []
        # Newest MAX_SPANS win: on a machine that has never reached the collector, the
        # recent history is the part still worth having.
        dropped = max(0, len(lines) - MAX_SPANS)
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
        """Delete every spooled span without sending it (durable opt-out).

        Spooling opens a window where data sits unsent; if the user opts out inside it,
        uploading anyway would be worse than the old in-process flush, where opt-out was
        immediate and total. The lock file is left alone — it carries no span data.
        """
        for path in (self.path, self.sending_path, self.started_path):
            _unlink(path)


class _AppendingFileSpanExporter:
    """`FileSpanExporter`'s format, but opening the file per export.

    The stock exporter opens its path in ``__init__`` and holds the descriptor for the
    life of the process. That descriptor keeps pointing at the old inode once a drainer
    renames the spool aside, so a long command (`tt update` runs for minutes) would
    append its span into a file the drainer is about to delete. Opening at export time
    narrows that window to the write itself, and means commands that export nothing
    never create the file at all.
    """

    def __init__(self, spool: Spool) -> None:
        self._spool = spool

    def export(self, spans: Any) -> Any:
        from opentelemetry.exporter.otlp.json.file.trace_exporter import FileSpanExporter
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            handle = self._spool._open_for_append()
            if handle is None:
                return SpanExportResult.FAILURE
            with handle:
                # One write of a <8KB line to an O_APPEND descriptor is a single atomic
                # syscall, which is what makes lock-free appends from concurrent `tt`
                # processes safe.
                return FileSpanExporter(stream=handle).export(spans)
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)
