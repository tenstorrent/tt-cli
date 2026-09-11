# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt serve` — run inference (or benchmarks/evals) on a model. [beta]

Three serving paths. `--backend auto` (the default) picks by the model name: a
name in tt-inference-server's released spec goes to tt-inference-server; a name
only TT-Studio's catalog knows goes to studio; a Hugging Face bundle id
(`namespace/name`) neither knows falls back to tt-model. An explicit --backend
forces one, and with no model at all opens a picker of what that backend serves."""

from __future__ import annotations

import json
import shlex
import sys
from enum import Enum

import typer
from rich.console import Group
from rich.table import Table
from rich.text import Text

from ..backends.device import get_device_backend
from ..backends.serving.inference_server import (
    InferenceServerBackend,
    infer_device_config,
)
from ..backends.serving.model_manager import (
    ModelManagerBackend,
    looks_like_bundle_id,
)
from ..backends.serving.studio import StudioBackend
from .._compat import IntRange, prompt
from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..models.model import ModelInfo
from ..modelhub import bundles
from ..modelhub.catalog import ModelCatalog, unknown_model_error
from ..modelhub.completions import complete_model


class Workflow(str, Enum):
    server = "server"
    benchmarks = "benchmarks"
    evals = "evals"


class Backend(str, Enum):
    auto = "auto"
    inference_server = "inference-server"
    studio = "studio"
    model_manager = "model-manager"


def _stdin_isatty() -> bool:
    """Test seam: CliRunner replaces sys.stdin, so tests patch this, not isatty."""
    return sys.stdin.isatty()


def _autodetect_device(appctx) -> str | None:
    """Best-effort device config from our own tt-smi snapshot. run.py's built-in
    detection crashes on fresh checkouts (see backends/serving/inference_server.py), so serve
    passes --device explicitly whenever the host is confidently mappable."""
    try:
        snap = get_device_backend(appctx).snapshot()
    except TTError as err:
        appctx.output.warn(
            f"device auto-detect skipped ({err.what}) — "
            "pass --device if the server cannot infer it."
        )
        return None
    device = infer_device_config(snap.devices)
    if device is None:
        seen = ", ".join(sorted({d.board_type or "?" for d in snap.devices})) or "none"
        appctx.output.warn(
            f"could not map detected boards ({seen}) to a device config — "
            "pass --device if the server cannot infer it."
        )
    else:
        appctx.output.status(
            f"Detected device configuration: {device} (override with --device)."
        )
    return device


@handle_tt_errors
def serve(
    ctx: typer.Context,
    model: str = typer.Argument(
        None,
        help="Catalog model name (Llama-3.1-8B-Instruct) or a tt-model bundle id "
        "(namespace/name). Omit it with --backend to pick from a list.",
        autocompletion=complete_model,
    ),
    backend: Backend = typer.Option(
        Backend.auto,
        "--backend",
        help="inference-server, studio, model-manager, or auto: inference-server "
        "when its spec has the model, studio for the models only studio carries, "
        "model-manager for a bundle id.",
    ),
    workflow: Workflow = typer.Option(
        Workflow.server, "--workflow", help="server, benchmarks, or evals."
    ),
    device: str = typer.Option(
        None,
        "--device",
        help="Device config override (e.g. p300x2); default: auto-detect on host.",
    ),
    port: int = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Host port for the OpenAI-compatible API (default: the server's own).",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use only pre-seeded caches; never download."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Serve even on a board tt records the model as failing on.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the resolved configuration and the command, without running it.",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """[beta] Serve a model for inference via tt-inference-server, TT-Studio, or
    tt-model (for a bundle id neither catalog covers).

    For a bundle id, anything tt serve does not recognize is passed to tt-model —
    its own flags and its vLLM passthrough: `tt serve ns/model -- --port 8080 --follow`.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    offline = offline or appctx.offline
    # Unrecognized options are collected rather than rejected (see the command's
    # context_settings) so tt-model's own flags — --port, --follow, --profile — and
    # its vLLM passthrough reach it unchanged.
    extra_args = list(ctx.args)
    catalog = ModelCatalog()
    if model is None:
        model = _pick_model(appctx, catalog, backend)
    entry = catalog.find(model)
    chosen = _resolve_backend(entry, model, backend, catalog_origin=catalog.origin)
    if chosen is Backend.model_manager:
        _serve_with_tt_model_manager(
            appctx, model, catalog_origin=catalog.origin,
            workflow=workflow, device=device, offline=offline,
            port=port, extra_args=extra_args, dry_run=dry_run,
        )
        return
    if extra_args:
        raise TTError(
            f"Unrecognized arguments for a catalog model: {' '.join(extra_args)}",
            why="Extra arguments are passed through to tt-model, which only serves "
            "bundle ids; tt-inference-server takes its options from tt serve itself.",
            next_step="Drop them, or use `tt serve --workflow/--device`.",
            exit_code=ExitCode.USAGE,
        )
    if chosen is Backend.studio:
        _serve_with_studio(
            appctx, entry, workflow=workflow, device=device, offline=offline,
            port=port, dry_run=dry_run,
        )
        return
    server = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    if device is None:
        device = _autodetect_device(appctx)
    # Spec device_type keys are lowercased throughout (`tt model list --hw` does
    # the same), and every per-device lookup is by that key.
    device = device.lower() if device else device
    if dry_run:
        # No preflight and no install: a dry run must not require docker, or fetch
        # a checkout, to describe what it would do.
        plan = server.plan(
            entry, workflow=workflow.value, device=device, port=port, force=force
        )
        appctx.output.emit(plan, renderer=_plan_renderer)
        return
    server.preflight(entry)
    server.serve(
        entry,
        workflow=workflow.value,
        device=device,
        offline=offline,
        port=port,
        force=force,
    )


def _resolve_backend(
    entry: ModelInfo | None, model: str, backend: Backend, *, catalog_origin: str
) -> Backend:
    """Which path serves `model`. auto: the support list first, then studio, then a
    bundle id. An explicit choice the model does not offer is refused rather than
    handed to a tool that will fail later with less context."""
    if entry is None:
        if backend in (Backend.auto, Backend.model_manager):
            # _serve_with_tt_model_manager keeps the bundle-shape guard and its error.
            return Backend.model_manager
        raise TTError(
            f"{model} is not a {backend.value} model.",
            why=f"It is not in the model catalog ({catalog_origin}); only tt-model "
            "bundle ids (namespace/name) serve outside it.",
            next_step=f"Run `tt model list` to see what {backend.value} serves, or drop "
            "--backend.",
            exit_code=ExitCode.UNSUPPORTED,
        )
    if backend is Backend.auto:
        return (
            Backend.inference_server
            if Backend.inference_server.value in entry.backends
            else Backend.studio
        )
    if backend is Backend.model_manager:
        raise TTError(
            f"{entry.name} is a catalog model, not a tt-model bundle.",
            why="model-manager serves bundle ids (namespace/name) only.",
            next_step=f"Drop --backend, or use `tt serve {entry.name} --backend "
            f"{entry.backends[0]}`.",
            exit_code=ExitCode.USAGE,
        )
    if backend.value not in entry.backends:
        alternatives = " or ".join(
            f"`tt serve {entry.name} --backend {b}`" for b in entry.backends
        )
        raise TTError(
            f"{entry.name} is not served through {backend.value}.",
            why=f"tt serves it through {', '.join(entry.backends)}"
            + (
                "; models tt-inference-server serves are not offered through studio."
                if backend is Backend.studio
                else "."
            ),
            next_step=f"Use {alternatives}, or drop --backend.",
            exit_code=ExitCode.UNSUPPORTED,
        )
    return backend


def _pick_model(appctx, catalog: ModelCatalog, backend: Backend) -> str:
    """Interactive fallback for `tt serve --backend X` with no model: a numbered
    list of what that backend serves here, on stderr, answered with a number."""
    if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
        raise TTError(
            "No model given.",
            why="The interactive picker needs a terminal and is disabled with "
            "--json/--quiet.",
            next_step="Pass a model: `tt serve <model>` (see `tt model list`).",
            exit_code=ExitCode.USAGE,
        )
    choices_display = None
    if backend is Backend.model_manager:
        choices = [b.name for b in bundles.local_bundles(config=appctx.config)]
        if not choices:
            raise TTError(
                "No tt-model bundles are pulled on this machine.",
                next_step="`tt model list --community` to browse, "
                "`tt model pull <namespace>/<name>` to fetch one.",
                exit_code=ExitCode.ERROR,
            )
        label = "Pulled tt-model bundles"
    else:
        models = catalog.list()
        if backend is not Backend.auto:
            models = [m for m in models if backend.value in m.backends]
        device = _autodetect_device(appctx)
        if device:
            models = [
                m for m in models if device in m.hardware and m.devices[device].supported
            ]
        if not models:
            raise TTError(
                f"No {backend.value} models for this machine.",
                next_step="`tt model list --all` shows every model on every device.",
                exit_code=ExitCode.ERROR,
            )
        choices = [m.name for m in models]
        label = f"Models for {device}" if device else "Models"
        if backend is Backend.auto:
            width = max(len(name) for name in choices)
            choices_display = [
                f"{m.name:<{width}}  {', '.join(m.backends)}" for m in models
            ]
    appctx.output.status(f"{label} ({backend.value}):")
    for i, text in enumerate(choices_display or choices, start=1):
        appctx.output.status(f"  {i:>3}. {text}")
    # err=True keeps the prompt on stderr: stdout stays the tool's own output.
    choice = prompt("Model", default=1, type=IntRange(1, len(choices)), err=True)
    return choices[choice - 1]


def _serve_with_studio(
    appctx,
    entry: ModelInfo,
    *,
    workflow: Workflow,
    device: str | None,
    offline: bool,
    port: int | None,
    dry_run: bool,
) -> None:
    studio = StudioBackend(appctx.registry, appctx.runner, appctx.config, appctx.output)
    if workflow is not Workflow.server:
        raise studio.unsupported_workflow(workflow.value)
    if dry_run:
        appctx.output.emit(studio.plan(entry, offline=offline), renderer=_plan_renderer)
        return
    studio.preflight(entry)
    studio.serve(entry, offline=offline, device=device, port=port)


def _plan_renderer(plan: dict) -> Group:
    table = Table(title=f"tt serve {plan['model']} — dry run", show_header=False)
    table.add_column("field", style="bold")
    # fold, not the default ellipsis: an image ref or a cache path is only useful
    # in full.
    table.add_column("value", overflow="fold")
    if plan.get("backend") == "tt-model":
        table.add_row("backend", "tt-model (bundle)")
        bundle = plan["bundle"]
        if bundle:
            device = bundle["hardware"] or "—"
            if bundle["arch"] and bundle["device_count"]:
                device += f"  [dim]({bundle['arch']}, {bundle['device_count']} chips)[/dim]"
            table.add_row("device", device)
            table.add_row("engine", bundle["engine"] or "—")
            table.add_row("docker image", bundle["image"] or "—")
            table.add_row("tool-call parser", bundle["tool_call_parser"] or "—")
            table.add_row("reasoning parser", bundle["reasoning_parser"] or "—")
            table.add_row(
                "tt config", json.dumps(bundle["tt_config"]) if bundle["tt_config"] else "—"
            )
            if bundle["max_model_len"]:
                table.add_row("max model len", str(bundle["max_model_len"]))
            table.add_row("weights", bundle["weights_repo"] or "—")
            if bundle["tt_metal_version"]:
                table.add_row("tt-metal", bundle["tt_metal_version"])
            if bundle["profiles"]:
                table.add_row("profiles", ", ".join(bundle["profiles"]))
            table.add_row(
                "port",
                f"{plan['port']}  [dim](--port override)[/dim]"
                if plan["port"]
                else f"{bundle['port'] or '—'}  [dim](from the bundle)[/dim]",
            )
        elif bundle is None:
            table.add_row(
                "configuration",
                "not pulled yet — tt-model resolves it from the bundle at launch",
            )
        else:
            # serve_details() returns {} for both "no manifest on disk" and "the
            # manifest would not parse"; naming only the second reads as damage
            # when the usual cause is simply that it is not there.
            table.add_row(
                "configuration",
                "installed, but no readable manifest — tt-model resolves it at launch",
            )
        if not plan["installed"]:
            table.add_row("tt-model", "not installed — the first serve installs it")
        if plan["extra_args"]:
            table.add_row("passthrough", " ".join(plan["extra_args"]))
    elif plan.get("backend") == "studio":
        table.add_row("backend", "TT-Studio")
        table.add_row("checkout", plan["cwd"] or "not installed — the first serve clones it")
        table.add_row("HF token", _token_cell(plan["hf_token_source"]))
        table.add_row(
            "deploy",
            "[dim]studio picks the chips and the port; run.py reports the endpoint "
            "once the model is healthy[/dim]",
        )
    else:
        table.add_row("backend", "tt-inference-server")
        table.add_row("workflow", plan["workflow"])
        device = plan["device_sent"] or "auto (server decides)"
        if plan["served_through"]:
            device = (
                f"{plan['device_sent']}  "
                f"[dim](asked for {plan['device_requested']}; no spec for it)[/dim]"
            )
        table.add_row("device", device)
        table.add_row("engines", ", ".join(plan["engines"]) or "—")
        table.add_row("status", plan["status"] or "—")
        image = plan["docker_image"] or plan["spec_docker_image"] or "resolved by the server"
        if plan["docker_image"]:
            image += "\n[dim]override — the spec pins "
            image += f"{plan['spec_docker_image'] or 'no image'}[/dim]"
        table.add_row("docker image", image)
        table.add_row("tool-call parser", plan["tool_call_parser"] or "—")
        table.add_row("reasoning parser", plan["reasoning_parser"] or "—")
        tt_config = plan["forced_tt_config"] or plan["spec_tt_config"]
        source = "forced by tt" if plan["forced_tt_config"] else "from the model spec"
        table.add_row(
            "tt config", f"{json.dumps(tt_config)} [dim]({source})[/dim]" if tt_config else "—"
        )
        if plan["host_volume"]:
            table.add_row(
                "weights",
                f"{plan['host_volume']}\n[dim]pre-seeded volume — weights and "
                "tt_metal_cache reused; the HF cache is not consulted[/dim]",
            )
        elif plan["host_hf_cache"]:
            table.add_row("weights cache", plan["host_hf_cache"])
        else:
            table.add_row(
                "weights",
                "[dim]fetched by the container; the host cache is not used[/dim]",
            )
        if not plan["installed"]:
            table.add_row(
                "tt-inference-server", "not installed — serving installs it first"
            )
        table.add_row("HF token", _token_cell(plan["hf_token_source"]))
        table.add_row(
            "port",
            f"{plan['port']}  [dim](--port override)[/dim]"
            if plan["port"]
            else f"{plan['default_port']}  [dim]("
            + ("SERVICE_PORT" if plan["default_port_from_env"] else "server default")
            + ")[/dim]",
        )
    # The command goes below the table, not in a cell: Rich truncates a long cell
    # with an ellipsis, and a command you cannot copy is worse than none.
    command = Text(" ".join(shlex.quote(a) for a in plan["argv"]), style="dim")
    return Group(table, Text("\ncommand:", style="bold"), command)


def _token_cell(source: str | None) -> str:
    """Where HF_TOKEN would come from — never the token itself."""
    if source == "env":
        return "from the shell (HF_TOKEN)"
    if source == "hf-login":
        return "from the Hugging Face login store"
    return "none [dim](gated models need `hf auth login` or HF_TOKEN)[/dim]"


def _serve_with_tt_model_manager(
    appctx,
    model: str,
    *,
    catalog_origin: str,
    workflow: Workflow,
    device: str | None,
    offline: bool,
    port: int | None,
    extra_args: list[str],
    dry_run: bool = False,
) -> None:
    """Fallback path: a name the released spec does not know. Only Hub-style bundle
    ids route here — anything else is a catalog typo and gets the catalog's error."""
    if not looks_like_bundle_id(model):
        raise unknown_model_error(model, catalog_origin)
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    if workflow is not Workflow.server:
        raise backend.unsupported_workflow(workflow.value)
    if device is not None:
        appctx.output.warn(
            "--device is a tt-inference-server option; tt-model detects the machine "
            "itself (override with its own --arch)."
        )
    if dry_run:
        plan = backend.plan(model, offline=offline, port=port, extra_args=extra_args)
        appctx.output.emit(plan, renderer=_plan_renderer)
        return
    appctx.output.status(
        f"{model} is not in the model catalog ({catalog_origin}) — "
        "serving it as a tt-model bundle."
    )
    backend.serve(model, offline=offline, port=port, extra_args=extra_args)
