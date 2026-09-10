# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""RunTimings: the durations we render for humans, kept for machines too.

Human mode shows `✓ Starting inference server  2.0s` and `Ready in 3m 34s`;
`--json` gets the same numbers as data, which makes install time diffable in CI
instead of something you eyeball in a terminal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Timing:
    label: str
    seconds: float
    status: str = "ok"  # ok | skipped | failed


@dataclass
class RunTimings:
    """Accumulates step and phase durations for one command invocation."""

    started: float = field(default_factory=time.monotonic)
    phases: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    def add_step(self, label: str, seconds: float, status: str = "ok") -> None:
        self.steps.append(Timing(label, round(seconds, 3), status))

    def add_phase(self, label: str, seconds: float, status: str = "ok") -> None:
        self.phases.append(Timing(label, round(seconds, 3), status))

    def total_seconds(self) -> float:
        return time.monotonic() - self.started

    def to_dict(self) -> dict:
        """Shape for an `emit()` payload. Rounded — nobody diffs nanoseconds."""
        return {
            "total_seconds": round(self.total_seconds(), 3),
            "phases": [
                {"title": t.label, "seconds": t.seconds, "status": t.status} for t in self.phases
            ],
            "steps": [
                {"label": t.label, "seconds": t.seconds, "status": t.status} for t in self.steps
            ],
        }

    def as_payload(self) -> dict:
        return {"timings": self.to_dict()}


def merge_timings(payload: Any, timings: RunTimings) -> Any:
    """Attach timings to a command's --json payload without clobbering it."""
    if isinstance(payload, dict):
        merged = dict(payload)
        merged.setdefault("timings", timings.to_dict())
        return merged
    return payload
