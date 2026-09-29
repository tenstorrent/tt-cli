# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Progress out of `git clone --progress`.

Real output (git 2.43, captured 2026-09-10):

    Cloning into 'gitcap'...
    remote: Enumerating objects: 164, done.
    remote: Counting objects:  22% (37/164)
    remote: Compressing objects: 100% (140/164)
    Receiving objects:  57% (94/164), 1.20 MiB | 2.40 MiB/s
    Resolving deltas: 100% (4/4), done.

Git hands us an exact `(done/total)` for every stage, so this is one of the rare
cases where a real percentage is honest rather than invented.

Note git writes progress with carriage returns, not newlines. Python's universal
newline translation splits on `\\r` too, so the reader already sees these as
separate logical lines — verified against a real clone, not assumed.
"""

from __future__ import annotations

import re

from ..format import fmt_bytes, progress_bar
from .uv_pip import parse_size

# "remote: Counting objects:  22% (37/164)" — the `remote: ` prefix is optional
# because Receiving/Resolving are local stages and don't carry it.
_STAGE_RE = re.compile(
    r"^(?:remote:\s*)?(?P<stage>Enumerating objects|Counting objects|Compressing objects|"
    r"Receiving objects|Resolving deltas)\s*:\s*(?P<rest>.+)$"
)
_FRACTION_RE = re.compile(r"\((?P<done>\d+)/(?P<total>\d+)\)")
_BYTES_RE = re.compile(r"(?P<size>[\d.]+\s*[KMG]?i?B)\s*\|")
_CLONING_RE = re.compile(r"^Cloning into '(?P<path>[^']+)'")

# The order git works through them, so a later stage can't look like a regression.
STAGE_ORDER = (
    "Enumerating objects",
    "Counting objects",
    "Compressing objects",
    "Receiving objects",
    "Resolving deltas",
)
_FRIENDLY = {
    "Enumerating objects": "enumerating",
    "Counting objects": "counting",
    "Compressing objects": "compressing",
    "Receiving objects": "receiving",
    "Resolving deltas": "resolving deltas",
}


def parse_git_line(line: str):
    """→ ('cloning', path) | ('stage', name, done, total, bytes) | None."""
    text = (line or "").strip()
    if not text:
        return None
    match = _CLONING_RE.match(text)
    if match:
        return ("cloning", match.group("path"))
    match = _STAGE_RE.match(text)
    if match is None:
        return None
    rest = match.group("rest")
    fraction = _FRACTION_RE.search(rest)
    done, total = (0, 0)
    if fraction:
        done, total = int(fraction.group("done")), int(fraction.group("total"))
    elif rest.split(",")[0].strip().rstrip(".").isdigit():
        # "Enumerating objects: 164, done." — a total with no fraction.
        total = int(rest.split(",")[0].strip().rstrip("."))
    size_match = _BYTES_RE.search(rest)
    size = parse_size(size_match.group("size")) if size_match else None
    return ("stage", match.group("stage"), done, total, size)


class GitCloneProgress:
    """Aggregates a clone into one activity label plus a milestone per stage."""

    def __init__(self, label: str = "Cloning") -> None:
        self.label = label
        self.stage = ""
        self.done = 0
        self.total = 0
        self.bytes = 0.0
        self._announced: set = set()

    def feed(self, line: str) -> str | None:
        event = parse_git_line(line)
        if event is None:
            return None
        if event[0] == "cloning":
            return None
        _, stage, done, total, size = event
        # Never let a stale stage overwrite a newer one (git interleaves remote
        # and local progress).
        if self.stage and stage in STAGE_ORDER and self.stage in STAGE_ORDER:
            if STAGE_ORDER.index(stage) < STAGE_ORDER.index(self.stage):
                return None
        self.stage = stage
        self.done, self.total = done, total
        if size:
            # Each line restates the running total, so keep the max rather than
            # accumulating — otherwise the counter races away.
            self.bytes = max(self.bytes, float(size))
        # One milestone per stage, the first time we see it complete.
        if total and done == total and stage not in self._announced:
            self._announced.add(stage)
            if stage == "Receiving objects":
                suffix = f" · {fmt_bytes(self.bytes)}" if self.bytes else ""
                return f"received {total} objects{suffix}"
        return None

    def activity(self) -> str:
        if not self.stage:
            return f"{self.label} · connecting"
        friendly = _FRIENDLY.get(self.stage, self.stage.lower())
        if not self.total:
            return f"{self.label} · {friendly}"
        text = (
            f"{self.label}  {progress_bar(self.done, self.total)}  "
            f"{self.done}/{self.total} {friendly}"
        )
        if self.stage == "Receiving objects" and self.bytes:
            text += f" · {fmt_bytes(self.bytes)}"
        return text
