# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt self` — the CLI itself: upgrade it, inspect its managed tools and telemetry."""

from __future__ import annotations

import dataclasses

import typer
from rich.table import Table

from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..context import get_app_context

self_app = typer.Typer(
    help="Manage the tt CLI itself: upgrade it, inspect its managed tools and telemetry.",
    no_args_is_help=True,
)


@self_app.command("tools")
@handle_tt_errors
def tools_status(
    ctx: typer.Context, json_mode: JsonFlag = False, quiet: QuietFlag = False
) -> None:
    """Show managed tools: golden pin, installed version, resolution source."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    registry = appctx.registry
    rows = registry.status()

    def render(data: dict) -> Table:
        table = Table(title=f"Managed tools ({data['manifest_origin']})")
        for column in ("tool", "kind", "golden", "installed", "source", "path"):
            table.add_column(column)
        for row in data["tools"]:
            table.add_row(
                row["name"],
                row["kind"],
                row["golden_version"],
                row["installed_version"] or "—",
                row["source"],
                row["path"] or "—",
            )
        return table

    appctx.output.emit(
        {
            "manifest_origin": registry.manifest.origin,
            "tools": [dataclasses.asdict(r) for r in rows],
        },
        renderer=render,
    )


# Deliberately NOT wrapped in @handle_tt_errors. The decorator opens a usage span and
# calls session.flush() on the way out, so a decorated drainer would spool a span for
# every upload and could hand off to another drainer — a feedback loop. This command is
# the one leaf in the CLI that must stay invisible to telemetry.
@self_app.command("send-telemetry")
def send_telemetry(ctx: typer.Context, json_mode: JsonFlag = False) -> None:
    """Upload spooled usage spans, then exit. Normally launched detached by `tt` itself.

    Run it by hand to force delivery (or to see why delivery is failing) — the same
    process a command would have spawned, with its result printed instead of discarded.
    """
    from ..telemetry.drain import drain

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode)
    result = drain(appctx.paths, appctx.config)
    appctx.output.emit(
        {"status": result.status, "spans": result.spans, "detail": result.detail},
        renderer=lambda data: (
            f"telemetry: {data['status']}"
            + (f" ({data['spans']} span(s))" if data["spans"] else "")
            + (f" — {data['detail']}" if data["detail"] else "")
        ),
    )
    # Always exit 0: a failed upload is not a CLI failure (the batch stays spooled and
    # the next drain retries it), and the status above already says what happened.


@self_app.command("update")
@handle_tt_errors
def self_update(
    ctx: typer.Context,
    check: bool = typer.Option(
        False, "--check", help="Only report whether a newer tt exists; change nothing."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Upgrade without asking (required when not on a TTY)."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Upgrade tt itself to the newest release. Works where tt owns its environment (a
    `uv tool` or pipx install, or a venv with nothing else in it); in a shared venv it
    prints the exact command instead, so you can review what else would change."""
    from ..selfupdate.update import run_self_update

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    result = run_self_update(appctx, check_only=check, yes=yes)

    def render(data: dict) -> str:
        status = data["status"]
        if status == "up-to-date":
            return f"tt {data['current']} is the newest release."
        if status == "available":
            return (
                f"tt {data['latest']} is available (you have {data['current']}). "
                f"To upgrade: {data['hint']}"
            )
        if status == "declined":
            return "Left as is."
        return f"tt upgraded: {data['from']} → {data['to']}."

    appctx.output.emit(result, renderer=render)


# Deliberately NOT wrapped in @handle_tt_errors, for the same reason as send-telemetry:
# it runs detached in the background, so a span for it would count a check as usage,
# and the decorator's own after-command hook could spawn another check.
@self_app.command("check-update")
def check_update(ctx: typer.Context, json_mode: JsonFlag = False) -> None:
    """Refresh the cached "newest tt release" lookup and print it. Normally launched
    detached by `tt` itself once a day; run it by hand to see what the check found."""
    from ..selfupdate.check import run_check

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode)
    # skip_if_busy: several commands finishing during one slow lookup may each have
    # spawned a check; the flock lets exactly one of them do the work.
    result = run_check(appctx.paths, skip_if_busy=True)
    appctx.output.emit(
        {
            "current": result.current,
            "latest": result.latest,
            "newer": result.newer,
            "source": result.source,
            "error": result.error,
            "busy": result.busy,
        },
        renderer=lambda data: (
            "update check: another check is already running"
            if data["busy"]
            else f"update check: latest {data['latest'] or 'unknown'} (running {data['current']})"
            + (f" — {data['error']}" if data["error"] else "")
        ),
    )
