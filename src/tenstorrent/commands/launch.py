# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt launch` — point a client at a model that is already being served.

tt does not start the model's server: it discovers what `tt serve` (or tt-model)
has running, checks the model can do what the client needs, and then either
configures an installed client and hands over the terminal, or — with consent —
starts a client that runs as a container (or, for deep-research, as an app tt installs).

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

from .._compat import confirm, prompt
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
from ..modelhub.hub import hf_home_dir

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


def _resolve(appctx, launcher: Launcher, *, install: bool = True, retry_path: bool = True) -> str:
    """The client's executable: a tool tt manages from the registry (installed at its
    pin first, with `install`), any other from PATH."""
    tool = getattr(launcher, "managed_tool", None)
    if tool is None:
        return resolve_executable(launcher, appctx.config, appctx.output, retry_path=retry_path)
    return str(appctx.registry.ensure(tool) if install else appctx.registry.resolve(tool))


def _installed_or_none(appctx, launcher: Launcher) -> str | None:
    """A soft check for a listing/preview: "not installed" is an expected, cheap
    answer here, so it skips the fresh-shell retry `resolve_executable` otherwise
    pays for a miss that would actually block the user, and never installs."""
    try:
        return _resolve(appctx, launcher, install=False, retry_path=False)
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
    try:
        ports += _model_manager(appctx).running_ports()
    except TTError:
        pass
    return ports


def _model_manager(appctx) -> ModelManagerBackend:
    return ModelManagerBackend(appctx.registry, appctx.runner, appctx.config, appctx.output)


def _running_bundles(appctx) -> list[str]:
    """tt-model bundles being served, for an app that matches its recipe to them."""
    try:
        return _model_manager(appctx).running_repos()
    except TTError:
        return []


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


def _app_llm(appctx, served: list[RunningModel]) -> str | None:
    """With several models up and no --model: deep-research's LLM, when its recipe's models
    are the ones serving (the others embed and rerank, and tt's catalog may know none)."""
    app = LAUNCHERS["deep-research"]
    bundles = _running_bundles(appctx)
    executable = _installed_or_none(appctx, app) if bundles else None
    if executable is None:
        return None
    llm = app.serving_llm(executable, bundles, appctx.runner)
    if llm not in {s.served_id for s in served}:
        return None
    appctx.output.status(f"Using {llm}, the LLM of the deep-research recipe that is serving.")
    return llm


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
        None,
        "--web-port",
        min=1,
        max=65535,
        help=f"Host port for a web UI (default {DEFAULT_WEB_PORT}, or the next free one).",
    ),
    web: bool = typer.Option(
        False,
        "--web",
        help="Open the client's web UI instead of its terminal UI, where it has one.",
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
    if web and not getattr(launcher, "web_ui", False):
        raise TTError(
            f"{launcher.id} has no web UI to open.",
            why="--web is for clients that ship their own web interface.",
            next_step="Drop --web to use its terminal UI.",
            exit_code=ExitCode.USAGE,
        )
    if url is None and port is None:
        served = _discover_local(appctx)
        if model is None and len(served) > 1:
            model = _app_llm(appctx, served)
    else:
        served = discover(base_url_for(url, port))
    target = _select(appctx, served, model)
    _check_capability(appctx, launcher, target, force=force)
    # A dry run must not require the client to be installed. It still resolves one
    # when it can, so what it reports (an existing container, say) is the truth on
    # this machine rather than a guess.
    executable = (
        _installed_or_none(appctx, launcher) if dry_run else _resolve(appctx, launcher)
    )
    options = LaunchOptions(
        web_port=web_port or DEFAULT_WEB_PORT,
        web_port_pinned=web_port is not None,
        web=web,
    )
    _start(
        appctx, launcher, target, options, executable,
        yes=yes, no_exec=no_exec, dry_run=dry_run, json_mode=json_mode,
    )


def _connect_recipe(
    ctx: typer.Context,
    recipe: str = typer.Option(
        None,
        "--recipe",
        help="Recipe to start, by name (default: the one whose models tt is serving, "
        "else the default).",
    ),
    list_recipes: bool = typer.Option(
        False,
        "--list-recipes",
        help="Show every recipe; on a terminal, pick one to start.",
    ),
    aiq: bool = typer.Option(
        False,
        "--aiq",
        help="Also run AI-Q's own harness beside Deep Agents (its UI on :3001).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done, and change nothing."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """An app that picks its own models, by recipe."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    launcher = _launcher(ctx.info_name)
    # Its recipes come from the app itself, so a real run installs it first.
    executable = _installed_or_none(appctx, launcher) if dry_run else _resolve(appctx, launcher)
    if executable is None:
        appctx.output.status(
            f"{launcher.managed_tool} is not installed yet; `tt launch {launcher.id}` "
            "installs it, then picks a recipe."
        )
        appctx.output.emit({"tool": launcher.id, "installed": False}, renderer=lambda _: None)
        return
    recipes = launcher.recipes(executable, appctx.runner)
    bundles = _running_bundles(appctx)
    interactive = (
        not (yes or dry_run or appctx.output.json_mode or appctx.output.quiet) and _stdin_isatty()
    )
    if list_recipes:
        rows = launcher.listed(recipes)
        if not interactive:
            appctx.output.emit(rows, renderer=_recipes_renderer)
            return
        recipe = _pick_recipe(appctx, rows, optional=True)
        if recipe is None:
            return
        yes = True  # picking one is the yes
    elif recipe is None and interactive:
        rows = launcher.listed(recipes)
        running = {r["name"] for r in launcher.running_recipes(recipes, bundles)}
        if len(running) > 1:  # several fit what is serving: which one
            recipe = _pick_recipe(appctx, [r for r in rows if r["recipe"] in running], optional=False)
            yes = True
        # Nothing running: a choice when one can start; otherwise choose() says what to stop.
        elif not running and launcher.can_start(recipes, bundles):
            recipe = _pick_recipe(appctx, rows, optional=False)
            yes = True
    choice = launcher.choose(recipes, recipe, bundles)
    llm = next(m for m in choice.recipe["models"] if "llm" in m["roles"])
    # When `ttdr up` serves the models, they are not there to ask yet.
    target = (
        RunningModel(served_id=llm["model"], base_url=llm["url"])
        if choice.serves_models
        else launcher.served_llm(choice.recipe)
    )
    options = LaunchOptions(
        home=launcher.home(),
        recipe=choice.recipe["path"],
        serves_models=choice.serves_models,
        hf_home=hf_home_dir(appctx.config),
        aiq=aiq,
    )
    _start(
        appctx, launcher, target, options, executable,
        yes=yes, no_exec=False, dry_run=dry_run, json_mode=json_mode,
        rows={"serve": "ttdr up serves the models"} if choice.serves_models else None,
    )


def _recipes_renderer(rows: list[dict], *, numbered: bool = False) -> Table:
    table = Table(title="deep-research recipes", show_header=True, header_style="bold")
    columns = ("recipe", "serves", "about")
    for column in ("#", *columns) if numbered else columns:
        table.add_column(column, overflow="fold")
    for i, row in enumerate(rows, start=1):
        cells = [row[column] for column in columns]
        table.add_row(*([str(i), *cells] if numbered else cells))
    return table


def _pick_recipe(appctx, rows: list[dict], *, optional: bool) -> str | None:
    """Ask which recipe to start. With `optional`, Enter starts none (None); otherwise
    Enter takes the default recipe."""
    appctx.output.status_console.print(_recipes_renderer(rows, numbered=True))
    names = [row["recipe"] for row in rows]

    def parse(value: str) -> str | None:
        text = value.strip()
        if optional and not text:
            return None
        if text.isdecimal() and 1 <= int(text) <= len(names):
            return names[int(text) - 1]
        raise typer.BadParameter(f"enter a number from 1 to {len(names)}")

    default = "" if optional else str(
        next((i for i, row in enumerate(rows, start=1) if row["default"]), 1)
    )
    question = "Start which recipe (Enter for none)" if optional else "Start which recipe"
    # err=True keeps the prompt on stderr: stdout stays the command's own output.
    return prompt(question, default=default, show_default=not optional, value_proc=parse, err=True)


def _start(
    appctx,
    launcher: Launcher,
    target: RunningModel,
    options: LaunchOptions,
    executable: str | None,
    *,
    yes: bool,
    no_exec: bool,
    dry_run: bool,
    json_mode: bool,
    rows: dict | None = None,
) -> None:
    prep = launcher.plan(target, options, executable=executable, runner=appctx.runner)
    if rows:
        prep = dataclasses.replace(prep, rows={**rows, **prep.rows})
    # A terminal client is still configured under --no-exec — that is what the flag
    # is for. For a service, starting it is the only action, so it waits for start.
    start = not dry_run and not no_exec and not (json_mode and launcher.hands_over_terminal)
    act = start or (launcher.hands_over_terminal and not dry_run)
    env = LaunchEnv(executable, appctx.runner, appctx.output)
    if act:
        if prep.consent:
            _ask(appctx, prep.consent, yes=yes)
        launcher.apply(target, prep, env)
    # An app that reported its own progress has already said where it is; the table would
    # only repeat it. --json still gets the payload.
    quiet = act and getattr(launcher, "reports_itself", False)
    renderer = (lambda _: None) if quiet else _renderer
    appctx.output.emit(_payload(launcher, target, prep, acted=act), renderer=renderer)
    if start:
        launcher.handoff(target, prep, env)


for _name, _launcher_obj in LAUNCHERS.items():
    launch_app.command(
        _name,
        help=f"Connect {_name} ({'terminal client' if _launcher_obj.hands_over_terminal else 'web service'}).",
    )(
        handle_tt_errors(
            _connect_recipe
            if getattr(_launcher_obj, "uses_recipes", False)
            else _connect
        )
    )


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
        executable = _installed_or_none(appctx, launcher)
        # Only container clients have a state to report; asking is one `docker
        # inspect`, so it stays cheap enough for a listing.
        state = getattr(launcher, "container_state", None)
        rows.append(
            {
                "tool": launcher.id,
                "kind": "terminal" if launcher.hands_over_terminal else "web service",
                "requires_tool_calling": launcher.requires_tool_calling,
                # For a client that chooses its own models rather than any served one.
                "model": getattr(launcher, "model_requirement", None),
                "configures": launcher.target(),
                "needs": " or ".join(launcher.binaries),
                # A tool tt manages is installed on first use.
                "available": executable is not None or hasattr(launcher, "managed_tool"),
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
            row["model"]
            or ("tool calling" if row["requires_tool_calling"] else "any chat model"),
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
    keep_models: bool = typer.Option(
        False,
        "--keep-models",
        help="Leave the models it serves running (deep-research).",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Stop a client's web service (a container, Hermes' dashboard, or an app), keeping its data."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    launcher = _launcher(tool)
    # Container clients report container_state; a terminal client with a web mode
    # (Hermes) reports service_state instead.
    state = getattr(launcher, "service_state", None) or getattr(launcher, "container_state", None)
    if state is None:
        raise TTError(
            f"{launcher.id} is not something tt runs.",
            why="It is a client on your machine; tt only writes its configuration.",
            next_step=f"Quit {launcher.id} itself, or run `tt launch disconnect "
            f"{launcher.id}` to undo its configuration.",
            exit_code=ExitCode.USAGE,
        )
    # An app that serves its own models (deep-research) stops them too, unless kept.
    model_bundles = getattr(launcher, "model_bundles", None)
    if keep_models and model_bundles is None:
        raise TTError(
            f"{launcher.id} has no models of its own to keep.",
            why="--keep-models is for an app that serves its own models.",
            next_step="Drop --keep-models.",
            exit_code=ExitCode.USAGE,
        )
    # Never installed just to find out it is not running.
    managed = hasattr(launcher, "managed_tool")
    executable = _installed_or_none(appctx, launcher) if managed else _resolve(appctx, launcher)
    payload: dict = {"tool": launcher.id, "stopped": False}
    running = state(executable, appctx.runner) == "running"
    if model_bundles and not keep_models and not running:  # its models may outlive the app
        running = bool(model_bundles(_running_bundles(appctx), executable, appctx.runner))
    if not running:
        appctx.output.status(f"{launcher.target()} is not running.")
    else:
        env = LaunchEnv(executable or _resolve(appctx, launcher), appctx.runner, appctx.output)
        if model_bundles:
            launcher.stop(env, keep_models=keep_models)
        else:
            launcher.stop(env)
        if not getattr(launcher, "reports_itself", False):  # else it said so already
            appctx.output.status(f"Stopped {launcher.target()}; its data is kept.")
        payload["stopped"] = True
    appctx.output.emit(payload, renderer=lambda _: None)


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
    executable = _installed_or_none(appctx, launcher)
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
    for key in (
        "container", "image", "endpoint_in_container", "serve", "checkout", "recipe", "app"
    ):
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
