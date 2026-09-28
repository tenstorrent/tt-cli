# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The state machine behind the serve checklist: log lines in, row events out.

Log order is the truth: a later phase announcing itself finishes the current one,
so a row never sticks on "active" because a tool reworded its completion line,
and the scan only ever moves forward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from .phases import CAUSE_RE, IGNORE_RE, Bar, Phase, parse_bars, parse_pull_layer


@dataclass(frozen=True)
class Event:
    """One change to the checklist. `kind` is start, done, progress or detail."""

    kind: str
    label: str = ""
    detail: str | None = None
    done: float = 0.0
    total: float = 0.0
    is_bytes: bool = False


@dataclass
class _Counts:
    done: float = 0.0
    total: float = 0.0
    is_bytes: bool = False


@dataclass
class _Bars:
    """One phase's tqdm bookkeeping.

    Byte bars are kept per description and summed, because a weights download
    draws one bar per file plus an aggregate `Fetching 14 files` *count* — and
    taking whichever repainted last meant the row reported "10/14" while what
    the user wants to know is how many of the gigabytes have landed. Counts are
    not summed: two count bars measure different things (shards, layers, files),
    so the newest is the only meaningful one.
    """

    by_label: dict[str, tuple[float, float]] = field(default_factory=dict)
    count: tuple[float, float] | None = None

    def update(self, bars: Sequence[Bar]) -> None:
        for bar in bars:
            if bar.is_bytes:
                self.by_label[bar.label] = (bar.done, bar.total)
            else:
                self.count = (bar.done, bar.total)

    def counts(self) -> _Counts | None:
        """Bytes once any byte bar has appeared, else the newest count.

        Bytes win permanently: the file count stays on screen beside them and is
        the less useful of the two.
        """
        if self.by_label:
            done = sum(value[0] for value in self.by_label.values())
            # The total grows as the downloader's worker pool picks up further
            # files, so it can rise for the first second or so. Reported as it
            # is: a denominator that corrects itself is honest, and inventing a
            # stable one would mean guessing the size of files not yet started.
            total = sum(value[1] for value in self.by_label.values())
            return _Counts(done, total, True)
        if self.count is not None:
            return _Counts(self.count[0], self.count[1], False)
        return None


class PhaseTracker:
    """Feed it log lines; it says which step the run is on.

    `feed` returns the events one line produced (usually none). `evidence` is
    what a failure card should quote: the cause-naming lines, then the tail.
    """

    def __init__(self, phases: Sequence[Phase], *, tail_lines: int = 40) -> None:
        self.phases = list(phases)
        self.tail: deque[str] = deque(maxlen=tail_lines)
        self.notable: list[str] = []
        self._index = -1  # index of the current (or last) phase
        self._entered: list[str] = []
        self._done = True  # is that phase finished?
        self._progress: dict[str, _Counts] = {}
        self._detail: dict[str, str] = {}
        self._layers: dict[str, dict[str, bool]] = {}
        self._bars: dict[str, _Bars] = {}

    # -- queries ----------------------------------------------------------------------
    @property
    def current(self) -> Phase | None:
        if self._index < 0 or self._done:
            return None
        return self.phases[self._index]

    @property
    def reached(self) -> list[str]:
        """Keys of every phase the log started, in order."""
        return list(self._entered)

    def evidence(self) -> list[str]:
        return [*self.notable, *self.tail]

    def detail_for(self, key: str) -> str | None:
        """The best fact we have for a row: an extracted one, else its final count."""
        if key in self._detail:
            return self._detail[key]
        counts = self._progress.get(key)
        if counts and not counts.is_bytes and counts.done >= counts.total:
            return f"{int(counts.done)}/{int(counts.total)}"
        return None

    # -- feeding ----------------------------------------------------------------------
    def feed(self, line: str) -> list[Event]:
        line = line.rstrip("\r\n")
        if not line.strip():
            return []
        # Dropped before the tail
        if any(rx.search(line) for rx in IGNORE_RE):
            return []
        self.tail.append(line)
        # Bounded: a crashing boot can match hundreds of times and the card
        # quotes one of them.
        if len(self.notable) < 50 and any(rx.search(line) for rx in CAUSE_RE):
            self.notable.append(line)

        # A line that starts the current row continues it rather than jumping ahead.
        active = self.current
        restart = active is not None and any(rx.search(line) for rx in active.start)
        for index in range(self._index + 1, len(self.phases) if not restart else 0):
            phase = self.phases[index]
            if any(rx.search(line) for rx in phase.start):
                events = self.finish()
                self._index, self._done = index, False
                self._entered.append(phase.key)
                events.append(Event("start", phase.label))
                return events + self._extract(phase, line)

        phase = self.current
        if phase is None:
            return []
        events = self._extract(phase, line)
        if any(rx.search(line) for rx in phase.done):
            events += self.finish()
        return events

    def finish(self) -> list[Event]:
        """Close the active row, if any — also how a caller ends the stream."""
        if self._index < 0 or self._done:
            return []
        self._done = True
        phase = self.phases[self._index]
        return [Event("done", phase.done_label, detail=self.detail_for(phase.key))]

    # -- per-line extraction ----------------------------------------------------------
    def _extract(self, phase: Phase, line: str) -> list[Event]:
        events: list[Event] = []
        counts = self._layer_counts(phase, line) if phase.layers else None
        if counts is None:
            bars = parse_bars(line)
            if bars:
                state = self._bars.setdefault(phase.key, _Bars())
                state.update(bars)
                counts = state.counts()
        if counts is not None:
            self._progress[phase.key] = counts
            events.append(
                Event("progress", done=counts.done, total=counts.total,
                      is_bytes=counts.is_bytes)
            )
        if phase.detail is not None:
            text = phase.detail(line)
            if text:
                self._detail[phase.key] = text
                events.append(Event("detail", detail=text))
        return events

    def _layer_counts(self, phase: Phase, line: str) -> _Counts | None:
        """Layers finished / layers seen, for a `docker pull`.

        The denominator grows as the daemon reveals layers, so this can dip; the
        view clamps the percentage rather than the counts, which stay true.
        """
        parsed = parse_pull_layer(line)
        if parsed is None:
            return None
        layer, finished = parsed
        seen = self._layers.setdefault(phase.key, {})
        seen[layer] = seen.get(layer, False) or finished
        return _Counts(float(sum(seen.values())), float(len(seen)))
