# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt launch` — point a client at a model that is already being served.

tt does not start the model's server: it discovers what `tt serve` (or tt-model)
has running, checks the model can do what the client needs, and then either
configures an installed client and hands over the terminal, or — with consent —
starts a client that runs as a container.

One connect command per client, all sharing `_connect` and telling themselves
apart by `ctx.info_name`, so `tt launch --help` enumerates the clients and adding
one stays a change to the registry alone.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import sys

import typer
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .._compat import confirm
from ..backends.serving.inference_server import InferenceServerBackend
from ..backends.serving.model_manager import ModelManagerBackend
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..launchers import LAUNCHERS, Launcher
from ..launchers.base import (
    DEFAULT_WEB_PORT,
    LaunchEnv,
    LaunchOptions,
    Preparation,
    RunningModel,
    resolve_executable,
    serves_chat,
    tool_call_parser,
    tool_calling_models,
)
from ..launchers.discovery import DEFAULT_PORT, base_url_for, discover
from ..modelhub.catalog import ModelCatalog

launch_app = typer.Typer(
    help="Connect a client to a model tt is already serving.", no_args_is_help=True
)
# Reserved by the group itself: a client may not be named any of these.
SUBCOMMANDS = ("list", "stop", "disconnect")


def complete_tool(incomplete: str) -> list[str]:
    return [name for name in LAUNCHERS if name.startswith(incomplete)]


def _stdin_isatty() -> bool:
    """Test seam: CliRunner replaces sys.stdin, so tests patch this, not isatty."""
    return sys.stdin.isatty()


def _launcher(tool: str) -> Launcher:
    launcher = LAUNCHERS.get(tool.lower())
    if launcher is None:
        raise TTError(
            f"Unknown client {tool!r}.",
            why=f"tt launch supports: {', '.join(sorted(LAUNCHERS))}.",
            next_step="Run `tt launch list` to see them.",
            exit_code=ExitCode.USAGE,
        )
    return launcher


def _installed_or_none(launcher: Launcher, config, output) -> str | None:
    """A soft check for a listing/preview: "not installed" is an expected, cheap
    answer here, so it skips the fresh-shell retry `resolve_executable` otherwise
    pays for a miss that would actually block the user."""
    try:
        return resolve_executable(launcher, config, output, retry_path=False)
    except TTError:
        return None


def _local_candidate_ports(appctx) -> list[int]:
    """Ports tt itself is serving on, read off running containers via docker —
    the same lookup `tt model stop` uses to find a model's container. Ordered by
    backend, not meaningfully rankable otherwise."""
    ports: list[int] = []
    inference = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    try:
        ports += [c.port for c in inference.running_containers() if c.port]
    except TTError:
        pass  # no docker/podman, or it failed — model_manager or the plain default may still work
    model_manager = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    try:
        ports += model_manager.running_ports()
    except TTError:
        pass
    return ports


def _discover_local(appctx) -> list[RunningModel]:
    """Every model found without --port/--url: on every container tt itself
    started, or (docker absent, or nothing found there) tt's plain default
    port. Aggregated across every port that answers — with more than one
    container running, --model has to be able to find one regardless of which
    container actually serves it, not just whichever answers first."""
    ports = _local_candidate_ports(appctx) or [DEFAULT_PORT]
    served: list[RunningModel] = []
    last_error: TTError | None = None
    for port in ports:
        try:
            served += discover(f"http://127.0.0.1:{port}/v1")
        except TTError as exc:
            last_error = exc
    if not served:
        raise last_error
    return served


def _select(appctx, served: list[RunningModel], wanted: str | None) -> RunningModel:
    """The model to connect to, with its catalog entry attached when tt knows it."""
    catalog = ModelCatalog()
    # cached_sizes={} skips the HF cache scan: matching only needs names.
    resolved = [
        dataclasses.replace(s, entry=catalog.find(s.served_id, cached_sizes={}))
        for s in served
    ]
    if wanted is None:
        if len(resolved) > 1:
            appctx.output.warn(
                f"the server is serving {len(resolved)} models; using "
                f"{resolved[0].served_id} (choose with --model)."
            )
        return resolved[0]
    target = wanted.lower()
    for candidate in resolved:
        names = {candidate.served_id.lower()}
        if candidate.entry is not None:
            names |= {candidate.entry.name.lower(), candidate.entry.hf_repo.lower()}
        if target in names:
            return candidate
    raise TTError(
        f"{wanted} is not being served at {resolved[0].base_url}.",
        why="The server reports: " + ", ".join(s.served_id for s in resolved) + ".",
        next_step=f"Serve it first (`tt serve {wanted}`), or drop --model.",
        exit_code=ExitCode.USAGE,
    )


def _check_capability(appctx, launcher: Launcher, model: RunningModel, *, force: bool) -> None:
    """Refuse a model the client cannot actually work with."""
    entry = model.entry
    if entry is None:
        if launcher.requires_tool_calling:
            appctx.output.warn(
                f"{model.served_id} is not in the model catalog — tt cannot confirm "
                f"it supports the tool calling {launcher.id} needs."
            )
        return
    chat = serves_chat(entry)
    if chat and (tool_call_parser(entry) or not launcher.requires_tool_calling):
        return
    if not chat:
        what = f"{entry.name} is not a language model."
        why = (
            f"{launcher.id} needs chat completions; tt serves {entry.name} "
            f"({entry.model_type}) through {', '.join(entry.engines) or 'no vLLM engine'}."
        )
    else:
        what = f"{entry.name} cannot do tool calling."
        why = (
            f"{launcher.id} needs a model served with a vLLM tool-call parser, and "
            f"the support list publishes none for {entry.name}."
        )
    if force:
        appctx.output.warn(f"{why} Continuing because --force was given.")
        return
    alternatives = tool_calling_models(ModelCatalog())
    step = "Deploy a model with tool calling instead"
    if alternatives:
        step += f", e.g. `tt serve {alternatives[0]}` ({', '.join(alternatives)})"
    raise TTError(
        what,
        why=why,
        next_step=f"{step}.",
        exit_code=ExitCode.UNSUPPORTED,
        details={"model": entry.name, "tool": launcher.id},
    )


def _ask(appctx, question: str, *, yes: bool) -> None:
    """Ask before touching a file tt does not own, or pulling and starting a
    container. Non-interactive without --yes is a usage error rather than a silent
    yes: these edit your configuration or download gigabytes."""
    if yes:
        return
    if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
        raise TTError(
            "Refusing to make changes without confirmation.",
            why="stdin is not a terminal (or --json/--quiet is in effect), so there "
            "is no way to ask.",
            next_step="Re-run with --yes once you have checked `--dry-run`.",
            exit_code=ExitCode.USAGE,
        )
    if not confirm(f"{question}. Continue?"):
        raise TTError("Nothing was changed.", exit_code=ExitCode.OK)


# -- connect (one command per client) ---------------------------------------------
def _connect(
    ctx: typer.Context,
    model: str = typer.Option(
        None,
        "--model",
        help="Which served model to connect to, when more than one is running.",
    ),
    port: int = typer.Option(
        None, "--port", min=1, max=65535, help="Port the model is served on."
    ),
    url: str = typer.Option(
        None, "--url", help="Full OpenAI-compatible base URL, ending in /v1."
    ),
    web_port: int = typer.Option(
        DEFAULT_WEB_PORT,
        "--web-port",
        min=1,
        max=65535,
        help="Host port, for a client that runs as a web service.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Connect even if the model cannot do tool calling."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    no_exec: bool = typer.Option(
        False, "--no-exec", help="Prepare the client but do not start it."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done, and change nothing."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    # Which client this is comes from the invoked command name, so every client
    # shares this one implementation.
    launcher = _launcher(ctx.info_name)
    if url is None and port is None:
        served = _discover_local(appctx)
    else:
        served = discover(base_url_for(url, port))
    target = _select(appctx, served, model)
    _check_capability(appctx, launcher, target, force=force)
    # A dry run must not require the client to be installed. It still resolves one
    # when it can, so what it reports (an existing container, say) is the truth on
    # this machine rather than a guess.
    executable = (
        _installed_or_none(launcher, appctx.config, appctx.output)
        if dry_run
        else resolve_executable(launcher, appctx.config, appctx.output)
    )
    prep = launcher.plan(
        target, LaunchOptions(web_port=web_port), executable=executable, runner=appctx.runner
    )
    # A terminal client is still configured under --no-exec — that is what the flag
    # is for. For a service, starting it is the only action, so it waits for start.
    start = not dry_run and not no_exec and not (json_mode and launcher.hands_over_terminal)
    act = start or (launcher.hands_over_terminal and not dry_run)
    env = LaunchEnv(executable, appctx.runner, appctx.output)
    if act:
        if prep.consent:
            _ask(appctx, prep.consent, yes=yes)
        launcher.apply(target, prep, env)
    appctx.output.emit(_payload(launcher, target, prep, acted=act), renderer=_renderer)
    if start:
        launcher.handoff(target, prep, env)


for _name, _launcher_obj in LAUNCHERS.items():
    launch_app.command(
        _name,
        help=f"Connect {_name} ({'terminal client' if _launcher_obj.hands_over_terminal else 'web service'}).",
    )(handle_tt_errors(_connect))


# -- list -------------------------------------------------------------------------
@launch_app.command("list")
@handle_tt_errors
def list_apps(
    ctx: typer.Context,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Show every client tt can connect, and whether each is usable now."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    appctx.output.emit(_catalog(appctx), renderer=_catalog_renderer)


def _catalog(appctx) -> list[dict]:
    """What each client is and whether it could run right now."""
    rows = []
    for launcher in LAUNCHERS.values():
        executable = _installed_or_none(launcher, appctx.config, appctx.output)
        # Only container clients have a state to report; asking is one `docker
        # inspect`, so it stays cheap enough for a listing.
        state = getattr(launcher, "container_state", None)
        rows.append(
            {
                "tool": launcher.id,
                "kind": "terminal" if launcher.hands_over_terminal else "web service",
                "requires_tool_calling": launcher.requires_tool_calling,
                "configures": launcher.target(),
                "needs": " or ".join(launcher.binaries),
                "available": executable is not None,
                "container_state": state(executable, appctx.runner) if state else None,
            }
        )
    return rows


def _catalog_renderer(rows: list[dict]) -> Table:
    table = Table(title="tt launch", show_header=True, header_style="bold")
    for column in ("client", "kind", "model", "status", "configures"):
        table.add_column(column, overflow="fold")
    for row in rows:
        if not row["available"]:
            status = f"[dim]needs {row['needs']}[/dim]"
        elif row["container_state"] == "running":
            status = "[green]running[/green]"
        elif row["container_state"] == "stopped":
            status = "stopped"
        else:
            status = "ready"
        table.add_row(
            row["tool"],
            row["kind"],
            "tool calling" if row["requires_tool_calling"] else "any chat model",
            status,
            row["configures"],
        )
    return table


# -- stop / disconnect ------------------------------------------------------------
@launch_app.command("stop", no_args_is_help=True)
@handle_tt_errors
def stop(
    ctx: typer.Context,
    tool: str = typer.Argument(help="Client to stop.", autocompletion=complete_tool),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Stop a client tt runs as a container, keeping its data."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    launcher = _launcher(tool)
    if launcher.hands_over_terminal:
        raise TTError(
            f"{launcher.id} is not something tt runs.",
            why="It is a client on your machine; tt only writes its configuration.",
            next_step=f"Quit {launcher.id} itself, or run `tt launch disconnect "
            f"{launcher.id}` to undo its configuration.",
            exit_code=ExitCode.USAGE,
        )
    executable = resolve_executable(launcher, appctx.config, appctx.output)
    if launcher.container_state(executable, appctx.runner) != "running":
        appctx.output.status(f"{launcher.target()} is not running.")
        appctx.output.emit({"tool": launcher.id, "stopped": False}, renderer=lambda _: None)
        return
    launcher.stop(LaunchEnv(executable, appctx.runner, appctx.output))
    appctx.output.status(f"Stopped {launcher.target()}; its data is kept.")
    appctx.output.emit({"tool": launcher.id, "stopped": True}, renderer=lambda _: None)


@launch_app.command("disconnect", no_args_is_help=True)
@handle_tt_errors
def disconnect(
    ctx: typer.Context,
    tool: str = typer.Argument(
        help="Client to disconnect.", autocompletion=complete_tool
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be undone, and change nothing."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Undo what tt configured for a client, leaving its own data alone."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    launcher = _launcher(tool)
    executable = _installed_or_none(launcher, appctx.config, appctx.output)
    plan = launcher.disconnect_plan(executable, appctx.runner)
    payload = {
        "tool": launcher.id,
        "target": launcher.target(),
        "what": plan,
        "undone": False,
    }
    if plan is None:
        appctx.output.status(f"Nothing to undo for {launcher.id}.")
        appctx.output.emit(payload, renderer=lambda _: None)
        return
    if dry_run:
        appctx.output.status(f"Would undo: {plan}")
        appctx.output.emit(payload, renderer=lambda _: None)
        return
    _ask(appctx, f"Undo: {plan}", yes=yes)
    if executable is None and not launcher.hands_over_terminal:
        raise TTError(
            f"{' or '.join(launcher.binaries)} is not installed.",
            why=f"Removing {launcher.target()} needs a container runtime.",
            next_step=launcher.install_hint,
            exit_code=ExitCode.TOOL_MISSING,
        )
    launcher.disconnect(LaunchEnv(executable or "", appctx.runner, appctx.output))
    appctx.output.status(f"Done: {plan}")
    appctx.output.emit({**payload, "undone": True}, renderer=lambda _: None)


# -- shared rendering -------------------------------------------------------------
def _payload(launcher: Launcher, model: RunningModel, prep: Preparation, *, acted: bool) -> dict:
    return {
        "tool": launcher.id,
        "model": model.served_id,
        "catalog_name": model.entry.name if model.entry else None,
        "base_url": model.base_url,
        "tool_call_parser": tool_call_parser(model.entry) if model.entry else None,
        "config_path": str(prep.config.path) if prep.config else None,
        "config_key": prep.config.key if prep.config else None,
        "config_block": prep.config.block if prep.config else None,
        "url": prep.url,
        "env": prep.env,
        "steps": prep.steps,
        "applied": acted,
        **prep.rows,
    }


def _renderer(payload: dict) -> Group:
    verb = "connected" if payload["applied"] else "would connect"
    table = Table(title=f"tt launch {payload['tool']} — {verb}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")
    model = payload["model"]
    if payload["catalog_name"] and payload["catalog_name"] != model:
        model += f"  [dim]({payload['catalog_name']})[/dim]"
    table.add_row("model", model)
    table.add_row("endpoint", payload["base_url"])
    table.add_row("tool-call parser", payload["tool_call_parser"] or "—")
    for key in ("container", "image", "endpoint_in_container"):
        if payload.get(key):
            table.add_row(key.replace("_", " "), payload[key])
    if payload["config_path"]:
        table.add_row(
            "config", f"{payload['config_path']}  [dim]({payload['config_key']})[/dim]"
        )
    if payload["url"]:
        table.add_row("open", payload["url"])
    for key, value in (payload.get("env") or {}).items():
        table.add_row(key, value)
    body = [table]
    if not payload["applied"]:
        if payload["config_block"]:
            body += [
                Text("\nblock:", style="bold"),
                Text(json.dumps(payload["config_block"], indent=2), style="dim"),
            ]
        if payload["steps"]:
            body.append(Text("\ncommands:", style="bold"))
            body += [
                Text(" ".join(shlex.quote(a) for a in step), style="dim")
                for step in payload["steps"]
            ]
    return Group(*body)
