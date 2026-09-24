# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt serve` — run inference (or benchmarks/evals) on a model. [beta]

Two serving paths, picked by the model name: a name in tt-inference-server's
released spec goes to tt-inference-server; a Hugging Face bundle id
(`namespace/name`) that the spec does not know falls back to tt-model."""

from __future__ import annotations

import json
import shlex
import shutil
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
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..modelhub.catalog import ModelCatalog, unknown_model_error
from ..modelhub.completions import complete_model


class Workflow(str, Enum):
    server = "server"
    benchmarks = "benchmarks"
    evals = "evals"


# Fixed roadmap for the tt-inference-server path. The bundle path hands straight
# off to tt-model, which renders this design itself, so it declares no phases.
PHASES = ["Checks", "Prepare"]


def _autodetect_device(appctx) -> str | None:
    """Best-effort device config from our own tt-smi snapshot. run.py's built-in
    detection crashes on fresh checkouts (see backends/serving/inference_server.py), so serve
    passes --device explicitly whenever the host is confidently mappable."""
    ui = appctx.output.ui
    with ui.step("Detecting device configuration") as step:
        try:
            snap = get_device_backend(appctx).snapshot()
        except TTError as err:
            step.skip("auto-detect unavailable")
            appctx.output.warn(
                f"device auto-detect skipped ({err.what}) — "
                "pass --device if the server cannot infer it."
            )
            return None
        device = infer_device_config(snap.devices)
        if device is None:
            step.skip("no mappable device")
        else:
            step.detail(f"{device} (override with --device)")
    if device is None:
        seen = ", ".join(sorted({d.board_type or "?" for d in snap.devices})) or "none"
        # Actionable, so it is never folded, and it lands after the step collapses.
        appctx.output.warn(
            f"could not map detected boards ({seen}) to a device config — "
            "pass --device if the server cannot infer it."
        )
    return device


@handle_tt_errors
def serve(
    ctx: typer.Context,
    model: str = typer.Argument(
        help="Catalog model name (Llama-3.1-8B-Instruct) or a tt-model bundle id "
        "(namespace/name).",
        autocompletion=complete_model,
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
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """[beta] Serve a model for inference via tt-inference-server, or via tt-model
    when the name is a bundle id the released spec does not cover.

    For a bundle id, anything tt serve does not recognize is passed to tt-model —
    its own flags and its vLLM passthrough: `tt serve ns/model -- --port 8080 --follow`.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    offline = offline or appctx.offline
    # Unrecognized options are collected rather than rejected (see the command's
    # context_settings) so tt-model's own flags — --port, --follow, --profile — and
    # its vLLM passthrough reach it unchanged.
    extra_args = list(ctx.args)
    catalog = ModelCatalog()
    entry = catalog.find(model)
    if entry is None:
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
    backend = InferenceServerBackend(
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
        plan = backend.plan(
            entry, workflow=workflow.value, device=device, port=port, force=force
        )
        appctx.output.emit(plan, renderer=_plan_renderer)
        return
    # Two phases, not three: Checks and Prepare are the work tt owns. Once run.py
    # takes the terminal its lifetime is not our phase to hold open — the stepper
    # completes and the server's output takes over.
    ui = appctx.output.ui
    ui.register_phases(PHASES)
    with ui.phase("Checks"):
        with ui.step("Container runtime") as step:
            backend.preflight(entry)
            step.detail("docker" if shutil.which("docker") else "podman")
    with ui.phase("Prepare"):
        launch = backend.prepare(
            entry,
            workflow=workflow.value,
            device=device,
            offline=offline,
            port=port,
            force=force,
        )
    ui.final_stepper()
    backend.launch(launch)


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
