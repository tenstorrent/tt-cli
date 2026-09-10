# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The tt CLI's terminal design language.

Reached as `output.ui`. See docs/cli-output.md for the house rules; the short
version is: one line per operation, minimal by default, `-v` reveals everything,
raw tool output is evidence rather than UI, and a failure gets diagnosed instead
of dumped.
"""

from .cards import (
    failure_card,
    interrupted_panel,
    kept_panel,
    notice_panel,
    ready_panel,
)
from .console import Activity, Phase, Step, Ui, null_ui
from .format import elide, fmt_bytes, fmt_clock, fmt_duration, progress_bar, show_detail
from .stream import Parser, SubStepParser, run_streamed, steps_parser
from .theme import PANEL_WIDTH, SPINNER_FRAMES, THEME
from .timings import RunTimings, Timing, merge_timings

__all__ = [
    "Activity",
    "PANEL_WIDTH",
    "Parser",
    "Phase",
    "RunTimings",
    "SPINNER_FRAMES",
    "SubStepParser",
    "Step",
    "THEME",
    "Timing",
    "Ui",
    "elide",
    "failure_card",
    "fmt_bytes",
    "fmt_clock",
    "fmt_duration",
    "interrupted_panel",
    "kept_panel",
    "merge_timings",
    "notice_panel",
    "null_ui",
    "progress_bar",
    "ready_panel",
    "run_streamed",
    "steps_parser",
    "show_detail",
]
