# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Cards: grouped state, never anything transient.

Pure builders — they return renderables and touch no terminal — so a fixed-width
Console(file=StringIO) can render them in a test.

The failure card is the shape the whole design turns on: cause in the title, one
line of evidence, the *consequence* (is this fatal, or does the run continue?),
then what to try. Never a log dump: that is a worse version of the log file, with
no cause and no next step, and it scrolls the useful part away.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from rich.box import ROUNDED
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .format import elide, tilde
from .theme import PANEL_WIDTH


def _body(lines: Sequence[Any]) -> Text:
    body = Text()
    for i, line in enumerate(lines):
        if i:
            body.append("\n")
        body.append_text(Text.from_markup(line) if isinstance(line, str) else line)
    return body


def notice_panel(title: str, lines: Sequence[Any], border_style: str = "warning") -> Panel:
    """Content-sized callout for warnings, interrupts, and diagnosis cards."""
    return Panel(
        _body(lines),
        title=title,
        title_align="left",
        box=ROUNDED,
        border_style=border_style,
        padding=(1, 2),
        expand=False,
    )


def ready_panel(title: str, rows: Sequence[Sequence[str]], footer_lines: Sequence[str] = ()) -> Panel:
    """End-of-run summary: what's up, where, and what to do next.

    Produce it from ONE renderer that probes live state, and back any `--info`
    style re-view with the same function — a second copy of the assembly drifts
    within a month.
    """
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="muted", no_wrap=True)
    table.add_column()
    for row in rows:
        status = f"  [muted]{row[2]}[/muted]" if len(row) > 2 else ""
        table.add_row(row[0], f"{row[1]}{status}")

    group = Table.grid()
    group.add_column()
    group.add_row(table)
    if footer_lines:
        group.add_row(Text())
        for line in footer_lines:
            group.add_row(Text.from_markup(line))

    return Panel(
        group,
        title=f"[bold accent]{title}[/bold accent]",
        title_align="left",
        box=ROUNDED,
        border_style="accent",
        padding=(1, 2),
        width=PANEL_WIDTH,
    )


def kept_panel(title: str, rows: Sequence[str], footer_lines: Sequence[str] = ()) -> Panel:
    """Muted "what was preserved" card, so a teardown is honest about data."""
    lines = list(rows)
    if footer_lines:
        lines += [""] + list(footer_lines)
    return notice_panel(f"[bold accent]{title}[/bold accent]", lines, border_style="muted")


def failure_card(
    name: str,
    diagnosis: Mapping[str, Any],
    *,
    log_path: Any = None,
    consequence: str | None = None,
) -> Panel:
    """Render a diagnosis dict — {cause, detail, evidence, actions} — as a card.

    Build the dict in a pure classifier (see ui/diagnose.py) so the wording matrix
    is unit-testable without a terminal, a network, or hardware.
    """
    lines: list = [f"[error]{diagnosis.get('detail', '')}[/error]"]
    evidence = diagnosis.get("evidence")
    if evidence:
        lines.append(f"[muted]Log · {elide(str(evidence))}[/muted]")
    if consequence:
        lines += ["", f"[warning]{consequence}[/warning]"]

    actions = list(diagnosis.get("actions", ()))
    if log_path and not any(str(log_path) in a for a in actions):
        actions.append(f"tail -50 {tilde(log_path)}")
    if actions:
        lines += ["", "[info]Try:[/info]"]
        lines += [f"[muted]  {a}[/muted]" for a in actions]

    cause = diagnosis.get("cause")
    title = f"[error]{name} — {cause}[/error]" if cause else f"[error]{name}[/error]"
    return notice_panel(title, lines, border_style="error")


def interrupted_panel(resume: str, cleanup: str | None = None) -> Panel:
    """Ctrl-C gets a card too: what state the machine is in, how to resume.

    An interrupt should never be a cliff — the user pressed a key, they didn't
    break anything, and they need to know that.
    """
    lines = [f"[muted]Resume · {resume}[/muted]"]
    if cleanup:
        lines.append(f"[muted]Clean up · {cleanup}[/muted]")
    return notice_panel("[warning]Interrupted[/warning]", lines, border_style="warning")
