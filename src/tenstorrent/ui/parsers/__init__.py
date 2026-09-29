# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure stream aggregators: lines in, events out.

Each parser turns a tool's own chatter into (a) milestones worth a ✓ and (b) one
activity label. They hold no terminal state and do no I/O, so every one is tested
against real captured output committed as a fixture — the wording a parser has to
survive is the wording the tool actually prints, not the wording we imagined.

The denominator rule: count something you know exactly, and show the rest as a
counter. `9/13 packages · 16.0 MB` beats an invented percentage.
"""

from .git_clone import GitCloneProgress, parse_git_line
from .uv_pip import UvPipProgress, parse_size, parse_uv_line

__all__ = [
    "GitCloneProgress",
    "UvPipProgress",
    "parse_git_line",
    "parse_size",
    "parse_uv_line",
]
