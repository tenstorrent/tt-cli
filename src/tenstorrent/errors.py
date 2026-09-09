# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Structured errors and the documented exit-code contract for the tt CLI."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .output import OutputManager


class ExitCode(IntEnum):
    """Exit codes are a public, documented contract (see docs/DEVELOPERS.md)."""

    OK = 0
    ERROR = 1  # generic / unexpected failure
    USAGE = 2  # bad arguments (Click/Typer convention)
    NO_DEVICES = 3  # no Tenstorrent devices detected
    TOOL_MISSING = 4  # a wrapped tool is not installed
    TOOL_FAILED = 5  # a wrapped tool ran and failed
    NEEDS_SUDO = 6  # privileged operation and sudo is unavailable non-interactively
    UNSUPPORTED = 7  # stub / not implemented on this version or platform
    OFFLINE = 8  # network required but offline (flag or unreachable)
    CONFIG = 9  # invalid configuration or config key


class TTError(Exception):
    """A CLI error that always says what failed, why (if known), and what to do next."""

    def __init__(
        self,
        what: str,
        *,
        why: str | None = None,
        next_step: str | None = None,
        exit_code: ExitCode | int = ExitCode.ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(what)
        self.what = what
        self.why = why
        self.next_step = next_step
        self.exit_code = ExitCode(exit_code)
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "what": self.what,
                "why": self.why,
                "next_step": self.next_step,
                "exit_code": int(self.exit_code),
                "code": self.exit_code.name,
                "details": self.details,
            }
        }


def render_error(err: TTError, output: "OutputManager") -> None:
    """Render a TTError on the appropriate stream for the active output mode."""
    output.emit_error(err)
