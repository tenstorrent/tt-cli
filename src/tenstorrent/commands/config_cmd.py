# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt config` — persistent CLI configuration.

Bare `tt config` opens the (comment-rich) TOML file in the user's editor;
list/get/set/path cover scripting.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..config.store import (
    SOURCE_DEFAULT,
    SOURCE_FILE,
    SOURCE_UNRECOGNIZED,
    coerce_value,
)
from ..context import get_app_context
from ..errors import ExitCode, TTError

config_app = typer.Typer(help="Persistent CLI configuration (TOML, comments preserved).")


def _open_in_editor(path: Path) -> None:
    command = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not command:
        command = next((e for e in ("nano", "vim", "vi") if shutil.which(e)), None)
    if not command:
        raise TTError(
            "No editor found.",
            why="$VISUAL and $EDITOR are unset, and nano/vim are not installed.",
            next_step=f"Set $EDITOR, or edit {path} directly.",
            exit_code=ExitCode.CONFIG,
        )
    result = subprocess.run([*shlex.split(command), str(path)])
    if result.returncode != 0:
        raise TTError(
            f"Editor {command!r} exited with status {result.returncode}.",
            next_step=f"Edit {path} directly, or set $EDITOR to a working editor.",
            exit_code=ExitCode.CONFIG,
        )


def _fmt_toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _assignment(key: str, value: object) -> str:
    return f"{key} = {_fmt_toml_scalar(value)}"


@config_app.callback(invoke_without_command=True)
@handle_tt_errors
def config_root(ctx: typer.Context) -> None:
    """Open the config file in your editor ($VISUAL/$EDITOR)."""
    if ctx.invoked_subcommand is not None:
        return
    appctx = get_app_context(ctx)
    appctx.config.ensure_file()
    path = appctx.config.paths.config_file
    # Point out settings that exist but aren't in the file, rather than editing it behind
    # the user's back — not finding a documented key in here is what sends people to
    # append it at the end of the file, where TOML files it under the wrong table.
    missing = appctx.config.missing_keys()
    if missing:
        appctx.output.status(
            f"{len(missing)} newer setting(s) are not in this file yet "
            f"({', '.join(missing[:3])}{', …' if len(missing) > 3 else ''}). "
            "They use their defaults; `tt config sync` adds them with their comments.",
            style="dim",
        )
    appctx.output.status(f"Opening {path} …")
    _open_in_editor(path)


_SOURCE_NOTE = {
    SOURCE_DEFAULT: "(default)",
    SOURCE_FILE: "(set in config.toml)",
    SOURCE_UNRECOGNIZED: "(unrecognized - ignored)",
}


@config_app.command("list")
@handle_tt_errors
def list_config(
    ctx: typer.Context,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    sources: Annotated[
        bool,
        typer.Option("--sources", help="In JSON mode, report where each value comes from."),
    ] = False,
) -> None:
    """Show all effective config values (defaults overlaid with your file)."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    entries = appctx.config.list_entries()

    def render(_: object) -> str:
        # Annotate every line, not just the interesting ones: "no marker means default"
        # would make the absence of information carry the meaning, which is exactly the
        # ambiguity this display exists to remove.
        width = max(len(_assignment(k, v)) for k, (v, _) in entries.items())
        return "\n".join(
            f"{_assignment(key, value).ljust(width)}  {_SOURCE_NOTE[source]}"
            for key, (value, source) in entries.items()
        )

    # Default JSON stays a flat {key: value} mapping — that is a published contract.
    # --sources opts into the richer shape rather than changing it underneath anyone.
    payload: dict = (
        {key: {"value": value, "source": source} for key, (value, source) in entries.items()}
        if sources
        else {key: value for key, (value, _) in entries.items()}
    )
    appctx.output.emit(payload, renderer=render, soft_wrap=True)


def _fmt_bare(value: object) -> str:
    """Script-friendly single-value output: strings unquoted, bools lowercase."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@config_app.command("get", no_args_is_help=True)
@handle_tt_errors
def get_config(
    ctx: typer.Context,
    key: str = typer.Argument(help="Dotted key, e.g. telemetry.enabled"),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Print a single config value."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    value = appctx.config.get(key)
    # Human output stays the bare value — scripts parse it. Provenance is additive on
    # the JSON side only; a stray key in the file is reported via the warning instead.
    appctx.output.emit(
        {"key": key, "value": value, "source": appctx.config.source_of(key)},
        renderer=lambda d: _fmt_bare(d["value"]),
    )


@config_app.command("set", no_args_is_help=True)
@handle_tt_errors
def set_config(
    ctx: typer.Context,
    key: str = typer.Argument(help="Dotted key, e.g. telemetry.enabled"),
    value: str = typer.Argument(help="New value (bool/int/float/string inferred)"),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Set a single config value (comments in the file are preserved)."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    typed = coerce_value(value)
    appctx.config.set(key, typed)
    appctx.output.status(f"{key} = {_fmt_toml_scalar(typed)}")
    if appctx.output.json_mode:
        appctx.output.emit({"key": key, "value": typed})


@config_app.command("sync")
@handle_tt_errors
def sync_config(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List what would be added, change nothing.")
    ] = False,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Add settings introduced by newer tt versions to your config file.

    Additive only: your values, comments and ordering are untouched. Settings you never
    added already work (they fall back to their defaults) — this makes them visible and
    editable, with the explanatory comments that ship in the template.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    added = appctx.config.sync(dry_run=dry_run)

    def render(data: dict) -> str:
        if not data["added"]:
            return "Config is already up to date."
        verb = "Would add" if data["dry_run"] else "Added"
        listing = "\n".join(f"  {key}" for key in data["added"])
        return f"{verb} {len(data['added'])} setting(s) to {data['path']}:\n{listing}"

    appctx.output.emit(
        {
            "path": str(appctx.config.paths.config_file),
            "added": added,
            "dry_run": dry_run,
        },
        renderer=render,
    )


@config_app.command("reset")
@handle_tt_errors
def reset_config(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Replace the config file with a fresh, fully commented default template."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    path = appctx.config.paths.config_file
    if not yes and path.exists():
        # Same contract as `tt device reset`: no silent destruction, and non-interactive
        # callers must say so explicitly rather than hang or guess.
        if appctx.output.json_mode or not sys.stdin.isatty():
            raise TTError(
                "Resetting the config needs confirmation.",
                why=f"This discards every value set in {path}.",
                next_step="Re-run with --yes to confirm (a .bak copy is kept either way).",
                exit_code=ExitCode.USAGE,
            )
        if not typer.confirm(f"Replace {path} with defaults?"):
            raise typer.Exit(int(ExitCode.OK))
    backup = appctx.config.reset()
    appctx.output.emit(
        {"path": str(path), "backup": str(backup) if backup else None},
        renderer=lambda d: f"Wrote defaults to {d['path']}."
        + (f"\nPrevious file saved as {d['backup']}." if d["backup"] else ""),
    )


@config_app.command("path")
@handle_tt_errors
def config_path(
    ctx: typer.Context, json_mode: JsonFlag = False, quiet: QuietFlag = False
) -> None:
    """Print the config file location."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    appctx.output.emit(
        {"path": str(appctx.config.paths.config_file)},
        renderer=lambda d: d["path"],
    )
