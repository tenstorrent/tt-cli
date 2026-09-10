# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The one place Ui and Runner meet: a streamed child rendered as one live line.

`Runner.stream_parsed` reads the child line by line; a `Parser` turns those lines
into events; this module renders them. Keeping the parser pure — lines in, events
out, no terminal — is what makes it testable against real captured output with no
subprocess, no network, and no TTY.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Parser(Protocol):
    """A pure stream aggregator.

    `feed` returns None to ignore a line, a string to emit a milestone, and
    `activity` renders the current live label. Implementations live in
    ui/parsers/ and are unit-tested against captured fixtures.
    """

    def feed(self, line: str) -> str | None: ...

    def activity(self) -> str: ...


def run_streamed(
    runner: Any,
    ui: Any,
    argv: Sequence[str],
    *,
    label: str,
    parser: Parser | None = None,
    verbose_echo: bool = True,
    **kwargs: Any,
) -> Any:
    """Run `argv`, showing one live activity row instead of the child's output.

    Returns the `StreamResult`. Under `-v` each raw line is also echoed into the
    phase body, dimmed — the escape hatch for when a parser is wrong or the tool's
    wording has drifted.
    """
    echo = verbose_echo and ui.out.verbose

    with ui.activity(label) as row:

        def on_line(line: str) -> None:
            if echo and line.strip():
                ui.note(line.rstrip(), marker="", style="muted")
            if parser is None:
                return
            milestone = parser.feed(line)
            if milestone:
                row.milestone(milestone)
            row.set(parser.activity())

        return runner.stream_parsed(argv, on_line=on_line, **kwargs)


def steps_parser(steps: Sequence[str], label: str) -> "SubStepParser":
    """A parser for a child whose progress is a known, fixed list of sub-steps.

    Honest denominator: we know exactly how many stages we asked for, so the
    counter is exact even though the tool reports no totals of its own.
    """
    return SubStepParser(steps, label)


class SubStepParser:
    """Matches a fixed list of substrings, in order, against the stream."""

    def __init__(self, steps: Sequence[str], label: str) -> None:
        self.steps = list(steps)
        self.label = label
        self.done = 0

    def feed(self, line: str) -> str | None:
        if self.done < len(self.steps) and self.steps[self.done].lower() in line.lower():
            self.done += 1
            return self.steps[self.done - 1]
        return None

    def activity(self) -> str:
        from .format import progress_bar

        if not self.steps:
            return self.label
        bar = progress_bar(self.done, len(self.steps))
        return f"{self.label}  {bar}  {self.done}/{len(self.steps)}"
