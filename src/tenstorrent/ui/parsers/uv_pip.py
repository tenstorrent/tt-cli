# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Progress out of `uv pip install` / `uv venv`.

Real output (uv 0.9, captured 2026-09-10):

    Using CPython 3.10.21
    Creating virtual environment at: capvenv2
    Resolved 9 packages in 71ms
    Downloading numpy (16.0MiB)
     Downloaded numpy
    Prepared 9 packages in 155ms
    Installed 9 packages in 5ms
     + anyio==4.15.1

`Resolved N packages` gives an exact denominator, which is why this can show a
real bar. Note uv reports **binary** units (MiB), not the decimal ones Docker
uses — parse_size converts, so fmt_bytes still gets plain bytes.
"""

from __future__ import annotations

import re

from ..format import fmt_bytes, progress_bar

# "Resolved 9 packages in 71ms"
_RESOLVED_RE = re.compile(r"^Resolved (?P<n>\d+) packages?\b")
# "Prepared 9 packages in 155ms"
_PREPARED_RE = re.compile(r"^Prepared (?P<n>\d+) packages?\b")
# "Installed 9 packages in 5ms"
_INSTALLED_RE = re.compile(r"^Installed (?P<n>\d+) packages?\b")
# "Downloading numpy (16.0MiB)"
_DOWNLOADING_RE = re.compile(r"^Downloading (?P<name>\S+)\s*\((?P<size>[^)]+)\)")
# " Downloaded numpy"
_DOWNLOADED_RE = re.compile(r"^Downloaded (?P<name>\S+)")
# " + anyio==4.15.1"
_ADDED_RE = re.compile(r"^\+ (?P<name>[^=\s]+)==(?P<version>\S+)")
# "Creating virtual environment at: capvenv2"
_VENV_RE = re.compile(r"^Creating virtual environment at:\s*(?P<path>.+)$")
# "Using CPython 3.10.21"
_PYTHON_RE = re.compile(r"^Using (?P<impl>CPython|PyPy) (?P<version>\S+)")

_UNITS = {
    "B": 1,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
}


def parse_size(text: str) -> float | None:
    """`16.0MiB` → bytes. Handles uv's binary units and decimal ones alike."""
    match = re.match(r"^\s*([\d.]+)\s*([KMG]?i?B)\s*$", (text or "").strip(), re.IGNORECASE)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value * _UNITS.get(match.group(2).upper(), 1)


def parse_uv_line(line: str):
    """→ (kind, payload) | None. Pure; the whole parser's vocabulary in one place."""
    text = (line or "").strip()
    if not text:
        return None
    for regex, kind in (
        (_RESOLVED_RE, "resolved"),
        (_PREPARED_RE, "prepared"),
        (_INSTALLED_RE, "installed"),
    ):
        match = regex.match(text)
        if match:
            return (kind, int(match.group("n")))
    match = _DOWNLOADING_RE.match(text)
    if match:
        return ("downloading", (match.group("name"), parse_size(match.group("size"))))
    match = _DOWNLOADED_RE.match(text)
    if match:
        return ("downloaded", match.group("name"))
    match = _ADDED_RE.match(text)
    if match:
        return ("added", (match.group("name"), match.group("version")))
    match = _VENV_RE.match(text)
    if match:
        return ("venv", match.group("path"))
    match = _PYTHON_RE.match(text)
    if match:
        return ("python", f"{match.group('impl')} {match.group('version')}")
    return None


class UvPipProgress:
    """Aggregates a uv run into one activity label plus a few milestones."""

    def __init__(self, label: str = "Installing dependencies") -> None:
        self.label = label
        self.total = 0
        self.downloaded = 0
        self.added = 0
        self.bytes = 0.0
        self.python = ""
        self.stage = "resolving"

    def feed(self, line: str) -> str | None:
        """Returns a milestone string, or None. Never raises on odd input."""
        event = parse_uv_line(line)
        if event is None:
            return None
        kind, payload = event
        if kind == "python":
            self.python = str(payload)
            return None
        if kind == "resolved":
            self.total = int(payload)
            self.stage = "downloading"
            return None
        if kind == "downloading":
            _, size = payload
            if size:
                self.bytes += float(size)
            return None
        if kind == "downloaded":
            self.downloaded += 1
            return None
        if kind == "prepared":
            self.stage = "installing"
            return None
        if kind == "added":
            self.added += 1
            return None
        if kind == "installed":
            count = int(payload)
            self.stage = "done"
            # The one milestone worth a ✓: what actually landed.
            return f"{count} package{'s' if count != 1 else ''} installed"
        return None

    @property
    def done(self) -> int:
        """Whichever sub-count is further along — both are exact."""
        return max(self.downloaded, self.added)

    def activity(self) -> str:
        """One live label, honest about what is actually known yet.

        uv only reports per-package events for downloads and for the final
        `+ name==version` list, which it prints *after* installing — so a bar
        would sit at 0/N for the whole run and then snap to full. While there is
        no real count, name the stage and show the byte counter instead; the bar
        appears once something has actually completed.
        """
        if self.stage == "resolving" or not self.total:
            return f"{self.label} · resolving"
        if self.done:
            text = (
                f"{self.label}  {progress_bar(self.done, self.total)}  "
                f"{self.done}/{self.total} packages"
            )
        else:
            stage = "downloading" if self.stage == "downloading" else "installing"
            text = f"{self.label} · {stage} {self.total} packages"
        if self.bytes:
            text += f" · {fmt_bytes(self.bytes)}"
        return text
