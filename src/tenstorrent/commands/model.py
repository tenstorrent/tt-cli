# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt model` — browse, inspect, pull, stop and remove models.

Like `tt serve`, the destructive verbs dispatch on the name: a released-spec
model is handled here, a Hub bundle id is passed through to tt-model."""

from __future__ import annotations

import dataclasses
import sys

import typer
from rich.table import Table

from ..backends.device import get_device_backend
from ..backends.serving.inference_server import (
    Artifact,
    InferenceServerBackend,
    infer_device_config,
)
from ..backends.serving.model_manager import (
    ModelManagerBackend,
    looks_like_bundle_id,
)
from .._compat import confirm
from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..models.model import ModelInfo
from ..modelhub.catalog import ModelCatalog, unknown_model_error
from ..modelhub import bundles, hub
from ..modelhub.completions import complete_local_model, complete_model

model_app = typer.Typer(
    help="Model management: browse, pull, and compile models.", no_args_is_help=True
)


def _human_size(size: int | None) -> str:
    if size is None:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def _detect_device(appctx) -> str | None:
    """Detected device config (lowercase of the spec's device_type keys).

    Mirrors serve.py:_autodetect_device — kept as a separate copy on purpose
    while the serve backend is being overhauled; unify afterwards. Never fails
    the command: any detection problem degrades to the unfiltered list."""
    try:
        snap = get_device_backend(appctx).snapshot()
    except TTError as err:
        appctx.output.warn(
            f"device detection skipped ({err.what}) — "
            "showing all models; use --hw <device> or --all."
        )
        return None
    device = infer_device_config(snap.devices)
    if device is None:
        seen = ", ".join(sorted({d.board_type or "?" for d in snap.devices})) or "none"
        appctx.output.warn(
            f"could not map detected boards ({seen}) to a device config — "
            "showing all models; use --hw <device>."
        )
    return device


def _cached_cell(m: dict) -> str:
    return f"✓ {_human_size(m['cache_size_bytes'])}".strip() if m["cached"] else "—"


def _list_table(payload: dict, *, detected: bool = False) -> Table:
    device = payload["device"]
    if device:
        hint = " (detected — `tt model list --all` for every device)" if detected else ""
        table = Table(title=f"Models for {device}{hint}")
        for column in ("name", "type", "engines", "status", "cached"):
            table.add_column(column)
        for m in payload["models"]:
            table.add_row(
                m["name"],
                m["model_type"],
                ", ".join(m["engines"]),
                m["devices"][device]["status"],
                _cached_cell(m),
            )
    else:
        table = Table(title="Model catalog (all devices)")
        for column in ("name", "type", "engines", "hardware", "cached"):
            table.add_column(column)
        for m in payload["models"]:
            table.add_row(
                m["name"],
                m["model_type"],
                ", ".join(m["engines"]),
                ", ".join(m["hardware"]),
                _cached_cell(m),
            )
    return table


_COMMUNITY_CAPTION = (
    "source: `HF` is tt-model's public community catalog on the Hub, `local` is "
    "installed on this machine — a bundle in both is listed twice, once per source. "
    "arch is the architecture family the bundle declares (blackhole, wormhole_b0); "
    "board and mesh tags are left out, as is any tag tt does not recognise. "
    "Every bundle serves with `tt serve <name>`; weights are referenced rather "
    "than shipped. `--json` carries the engine and packaging kind as well."
)


def _community_table(rows: list[dict]) -> Table:
    """Same shape as the catalog table, minus columns the Hub does not publish."""
    table = Table(title="Community model bundles (tt-model)", caption=_COMMUNITY_CAPTION)
    # fold rather than ellipsize: the id is what you paste into `tt serve`
    table.add_column("name", overflow="fold")
    # The table answers "which of these can I run, and is it here already".
    # `serve` would be a constant ✓, `installed` is what source=local says, and
    # `kind`/`engine` are how a bundle is built rather than something you pick one
    # on — all four stay in --json, and `tt serve <id> --dry-run` reports the
    # engine of a bundle that has been pulled.
    for column in ("source", "arch", "weights"):
        table.add_column(column)
    for row in rows:
        # Render the value itself rather than a literal, so the table can never
        # disagree with --json about what a row's source is.
        table.add_row(
            row["name"],
            row.get("source") or "—",
            ", ".join(row.get("arch") or []) or "—",
            _weights_cell(row),
        )
    return table


def _weights_cell(row: dict) -> str:
    """✓ + size when the referenced weights are in the HF cache, — when they are
    not, ? when the bundle is not pulled so the reference is unknown."""
    if not row["installed"] or not row.get("weights_repo"):
        return "?"
    size = row.get("weights_bytes")
    return f"✓ {_human_size(size)}" if size is not None else "—"


@model_app.command("list")
@handle_tt_errors
def list_models(
    ctx: typer.Context,
    cached: bool = typer.Option(False, "--cached", help="Only models present on disk."),
    model_type: str = typer.Option(
        None,
        "--type",
        help="Filter by kind: llm, vlm, cnn, embedding, audio, text_to_speech, "
        "image, video.",
    ),
    hardware: str = typer.Option(
        None,
        "--hw",
        help="Filter to a device config (e.g. p300x2); skips auto-detection.",
    ),
    all_devices: bool = typer.Option(
        False, "--all", help="Every model on every device, not just this machine's."
    ),
    community: bool = typer.Option(
        False,
        "--community",
        help="List community tt-model bundles from the Hub instead of the released "
        "model catalog.",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Browse models that run on this machine (default: detected hardware only)."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    if community:
        _list_community(
            appctx,
            cached=cached,
            model_type=model_type,
            device_flags=[
                flag
                for flag, given in (("--hw", hardware), ("--all", all_devices))
                if given
            ],
        )
        return
    detected = not hardware and not all_devices
    device = hardware.lower() if hardware else (None if all_devices else _detect_device(appctx))
    models = ModelCatalog().list()
    if device:
        # A device the model is known to fail on is not a device it runs on:
        # the whole point of the support list is that `tt model list` never
        # offers something that will not start. `--all` still shows everything.
        models = [
            m for m in models if device in m.hardware and m.devices[device].supported
        ]
    if cached:
        models = [m for m in models if m.cached]
    if model_type:
        models = [m for m in models if m.model_type == model_type.lower()]
    appctx.output.emit(
        {"device": device, "models": [dataclasses.asdict(m) for m in models]},
        renderer=lambda payload: _list_table(payload, detected=detected),
    )


def _list_community(
    appctx, *, cached: bool, model_type: str | None, device_flags: list[str]
) -> None:
    """`tt model list --community`: the Hub-published bundle catalog.

    Separate from the released spec listing rather than merged into it — a bundle
    has no per-device status or max_context, and blurring the two would hide which
    tool serves what. Device filters do not apply (bundles advertise an arch, not a
    tt-inference-server device config), so passing them is a usage error rather
    than a silently ignored flag."""
    if device_flags:
        raise TTError(
            f"{' and '.join(device_flags)} does not apply to community bundles.",
            why="Bundles advertise an arch (blackhole), not a tt-inference-server "
            "device config (p300x2), so there is nothing to filter against.",
            next_step="Drop the flag — every listed bundle targets the arch shown.",
            exit_code=ExitCode.USAGE,
        )
    if model_type:
        raise TTError(
            "--type does not apply to community bundles.",
            why="Model type is not published as a repo tag, so tt cannot filter on it.",
            next_step="Run `tt model info` on a bundle id, or drop --type.",
            exit_code=ExitCode.USAGE,
        )
    # Local installs first: they need no network, and they are the only source for a
    # bundle nobody published — someone shares an id, you pull it, the Hub shows
    # nothing. Catalog rows win on name, since a listed bundle is the richer record.
    local = bundles.local_bundles(config=appctx.config)
    if appctx.offline:
        # The catalog is a Hub index with no bundled copy, but local installs are
        # entirely on disk — show those rather than refusing the whole command.
        appctx.output.warn(
            "--offline: showing only bundles installed on this machine; the "
            "community catalog lives on the Hugging Face Hub."
        )
        listed = []
    else:
        listed = bundles.search_community(config=appctx.config)
        # Refresh the shell-completion cache: tab-time must never touch the Hub,
        # so this listing is where `tt serve <TAB>` learns community bundle ids.
        bundles.save_community_cache([b.name for b in listed])
    # Not merged by name: a bundle that is both published and installed gets one
    # row per source, so the listing shows both facts instead of picking one.
    found = sorted(local + listed, key=lambda b: (b.name.lower(), b.source))
    if cached:  # --cached reads as "what do I have locally" on this listing too
        found = [b for b in found if b.source == "local"]
    appctx.output.emit(
        {"source": "tt-model-catalog", "bundles": [dataclasses.asdict(b) for b in found]},
        renderer=lambda payload: _community_table(payload["bundles"]),
    )


def _info_renderer(payload: dict) -> Table:
    m = payload
    table = Table(title=m["name"], show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("hf_repo", m["hf_repo"])
    table.add_row("type", m["model_type"])
    table.add_row("engines", ", ".join(m["engines"]))
    table.add_row(
        "servable",
        f"yes — `tt serve {m['name']}`"
        if m["tt_model_id"]
        else "no (no tt-inference-server entry)",
    )
    if m["param_count"] is not None:
        table.add_row("parameters", f"{m['param_count']}B")
    if m["min_disk_gb"] is not None:
        table.add_row("min disk", f"{m['min_disk_gb']} GB")
    if m["min_ram_gb"] is not None:
        table.add_row("min ram", f"{m['min_ram_gb']:g} GB")
    table.add_row(
        "cached",
        f"yes ({_human_size(m['cache_size_bytes'])})" if m["cached"] else "no",
    )
    for device, support in m["devices"].items():
        context = f", max context {support['max_context']}" if support["max_context"] else ""
        detail = f"{', '.join(support['engines'])} — {support['status']}{context}"
        if not support["supported"]:
            mark = support["unsupported"]
            detail = (
                f"[red]does not run[/red] — {mark['details']} "
                f"(seen {mark['verified_on']})"
            )
        elif support["serve_as"]:
            detail += f"\n[dim]served as {support['serve_as']}"
            detail += f" — {support['note']}[/dim]" if support["note"] else "[/dim]"
        table.add_row(f"on {device}", detail)
    return table


@model_app.command("info", no_args_is_help=True)
@handle_tt_errors
def model_info(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="Model name (Llama-3.1-8B-Instruct) or a tt-model bundle id (namespace/name).",
        autocompletion=complete_model,
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Show model metadata: engines, per-device support, requirements.

    For a tt-model bundle id: the bundle's manifest and compatibility verdict via
    `tt-model info` when tt-model is installed, otherwise its community-catalog row.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    model, bundle = _dispatch(appctx, name)
    if bundle is not None:
        _bundle_info(appctx, bundle, json_mode=json_mode)
        return
    appctx.output.emit(dataclasses.asdict(model), renderer=_info_renderer)


def _bundle_info(appctx, name: str, *, json_mode: bool) -> None:
    """`tt model info` for a tt-model bundle id — the one model verb that used to
    reject an id `tt model list --community`, `tt model pull` and `tt serve` all accept.

    Two sources, by what is available. With tt-model installed and a human reading,
    delegate to `tt-model info`: it prints the manifest and its compatibility verdict
    against this machine, which tt has no business reimplementing. Otherwise render
    the catalog row tt reads on its own — the same one the community listing shows,
    plus the pulled manifest's launch settings. That covers: tt-model not installed
    (info is inspection and must not clone-and-build a tool to describe a bundle,
    the same rule as stop/rm); --json (tt-model prints a manifest followed by prose,
    not one document); --offline (tt-model info fetches the manifest from the Hub).
    """
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    installed_tool = backend.is_installed()
    if installed_tool and not json_mode and not appctx.offline:
        backend.info(name)
        return
    row = bundles.describe(name, config=appctx.config, offline=appctx.offline)
    # None: not asked (--offline), so neither "listed" nor "unlisted" is honest.
    in_catalog = None if appctx.offline else (row is not None and row.source == "HF")
    if row is None:
        row = _unlisted_bundle(appctx, name)
    appctx.output.emit(
        {
            "source": "tt-model-catalog",
            "bundle": dataclasses.asdict(row),
            "in_catalog": in_catalog,
            "serve": bundles.serve_details(name),
            "tt_model_installed": installed_tool,
            "offline": appctx.offline,
        },
        renderer=_bundle_info_renderer,
    )


def _unlisted_bundle(appctx, name: str) -> bundles.BundleInfo:
    """A bundle-shaped id that is neither installed here nor in the community
    catalog. Offline there is nothing more to ask; online, the Hub says whether the
    repo carries a manifest at all — a plain weights repo is not a bundle, and
    telling the two apart is the difference between "pull it" and "you can't"."""
    if appctx.offline:
        raise TTError(
            f"{name} is not installed on this machine.",
            why="--offline: the community catalog lives on the Hugging Face Hub, "
            "so only an installed bundle can be described from disk.",
            next_step=f"Drop --offline, or install it: `tt model pull {name}`.",
            exit_code=ExitCode.OFFLINE,
        )
    is_bundle = bundles.is_bundle_repo(name)
    if is_bundle is None:
        raise TTError(
            f"Could not check whether {name} is a tt-model bundle.",
            why="It is not in the community catalog or installed here, and the Hub "
            "could not be asked (unreachable, private, or rate-limited).",
            next_step="Check the connection or HF_TOKEN; `tt model list --community` "
            "lists the published bundles.",
            exit_code=ExitCode.ERROR,
        )
    if not is_bundle:
        raise TTError(
            f"{name} is not a tt-model bundle.",
            why="It is not in the model catalog, not in the community bundle catalog, "
            "and its Hub repo carries no bundle manifest.",
            next_step="Run `tt model list --community` for bundle ids; a plain "
            f"HuggingFace repo's weights fetch with `tt model pull {name} --weights-only`.",
            exit_code=ExitCode.USAGE,
        )
    appctx.output.warn(
        f"{name} is a tt-model bundle but not in the community catalog (never "
        "published with `tt-model publish`); only the id is known until it is pulled."
    )
    return bundles.BundleInfo(name=name)


def _bundle_info_renderer(payload: dict) -> Table:
    b, serve = payload["bundle"], payload["serve"]
    name = b["name"]
    table = Table(title=name, show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("kind", "tt-model bundle" + (f" ({b['kind']})" if b["kind"] else ""))
    if payload["in_catalog"] is None:
        catalog = "not checked — --offline skips the Hub"
    elif payload["in_catalog"]:
        catalog = "community (`tt model list --community`)"
    elif b["installed"]:
        catalog = "not in the community catalog — installed here"
    else:
        catalog = "not in the community catalog — published on the Hub"
    table.add_row("catalog", catalog)
    table.add_row("arch", ", ".join(b["arch"]) or "—")
    table.add_row("engine", b["engine"] or "—")
    if b["downloads"] is not None:
        table.add_row("downloads", str(b["downloads"]))
    table.add_row(
        "installed", "yes" if b["installed"] else f"no — `tt model pull {name}`"
    )
    if b["weights_repo"]:
        cached = _weights_cell(b)
        state = "not in the HF cache" if cached == "—" else f"cached {cached}"
        table.add_row("weights", f"{b['weights_repo']} — {state}")
    else:
        table.add_row("weights", "? (known from the manifest once the bundle is pulled)")
    if serve:  # pulled, and its manifest is on disk
        chips = ""
        if serve["arch"] and serve["device_count"]:
            count = serve["device_count"]
            chips = f"{serve['arch']}, {count} chip{'s' if count != 1 else ''}"
        if serve["hardware"]:
            device = serve["hardware"] + (f"  [dim]({chips})[/dim]" if chips else "")
        else:
            device = chips or "—"
        table.add_row("hardware", device)
        table.add_row("docker image", serve["image"] or "—")
        if serve["port"]:
            table.add_row("port", str(serve["port"]))
        if serve["max_model_len"]:
            table.add_row("max model len", str(serve["max_model_len"]))
        if serve["tt_metal_version"]:
            table.add_row("tt-metal", serve["tt_metal_version"])
        if serve["profiles"]:
            table.add_row("profiles", ", ".join(serve["profiles"]))
    table.add_row(
        "servable", f"yes — `tt serve {name}`; preview with `tt serve {name} --dry-run`"
    )
    if payload["offline"] and payload["tt_model_installed"]:
        table.add_row(
            "manifest",
            "[dim]skipped under --offline: `tt-model info` fetches the manifest "
            "from the Hub[/dim]",
        )
    elif not payload["tt_model_installed"]:
        table.add_row(
            "manifest",
            "[dim]tt-model's manifest and compatibility verdict show here once "
            "tt-model is installed (`tt serve` installs it on first use)[/dim]",
        )
    return table


@model_app.command("pull", no_args_is_help=True)
@handle_tt_errors
def pull(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="Catalog model name, a tt-model bundle id, or any HuggingFace repo id.",
        autocompletion=complete_model,
    ),
    bundle: bool = typer.Option(
        False,
        "--bundle",
        help="Treat NAME as a tt-model bundle id, skipping detection (fails rather "
        "than falling back to a weights-only download).",
    ),
    weights_only: bool = typer.Option(
        False,
        "--weights-only",
        help="Treat NAME as a plain HuggingFace repo: fetch only its weights, never "
        "install a bundle. The opposite of --bundle.",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Only use the local cache; never download."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Download a model: a catalog model's weights, or a tt-model bundle."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    offline = offline or appctx.offline
    if bundle and weights_only:
        raise TTError(
            "--bundle and --weights-only are opposites.",
            why="One installs the tt-model bundle, the other deliberately skips it.",
            next_step="Pass at most one, or neither to let tt detect which it is.",
            exit_code=ExitCode.USAGE,
        )
    if bundle:
        # An explicit claim about what NAME is, so it also overrides a release-spec
        # match: `tt model pull <hf repo> --bundle` means "the bundle, not the
        # weights the spec entry would fetch".
        if offline:
            raise TTError(
                "Installing a bundle needs the network.",
                why="`tt-model pull` fetches the bundle (and its image) from the Hub; "
                "unlike serve, it has no local-only mode.",
                next_step="Drop --offline, or `tt serve <name>` if the bundle is "
                "already installed.",
                exit_code=ExitCode.OFFLINE,
            )
        _pull_bundle(appctx, name)
        return
    catalog = ModelCatalog()
    model = catalog.find(name)
    if model is None:
        _pull_unlisted(
            appctx, name, catalog_origin=catalog.origin,
            weights_only=weights_only, offline=offline,
        )
        return
    if not hub.has_hub_weights(model):
        raise TTError(
            f"{model.name} has no weights to download.",
            why=f"Its weights ship inside the container image; {model.hf_repo!r} is "
            "a model label, not a HuggingFace repo.",
            next_step=f"Nothing to pull — `tt serve {model.name}` fetches the image.",
            exit_code=ExitCode.UNSUPPORTED,
        )
    if not hub.uses_host_weight_cache(model):
        appctx.output.warn(
            f"{model.name}'s container downloads its own weights and ignores the "
            "host cache, so this will not speed up `tt serve`. Pulling anyway."
        )
    appctx.output.status(f"Pulling {model.hf_repo} …")
    path = hub.download_weights(
        model, appctx.config, offline=offline, output=appctx.output
    )
    appctx.output.emit(
        {"name": model.name, "hf_repo": model.hf_repo, "path": str(path)},
        renderer=lambda d: f"{d['name']} ready at {d['path']}",
    )


def _pull_bundle(appctx, name: str) -> None:
    """Install a tt-model bundle, with its weights."""
    if not looks_like_bundle_id(name):
        raise TTError(
            f"{name!r} is not a bundle id.",
            why="A tt-model bundle is a Hub repo, addressed namespace/name.",
            next_step="Run `tt model list --community` to see bundle ids.",
            exit_code=ExitCode.USAGE,
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    backend.pull(name)
    appctx.output.emit(
        {"name": name, "kind": "bundle", "pulled_by": "tt-model"},
        renderer=lambda d: f"{d['name']} installed — serve it with `tt serve {d['name']}`.",
    )


def _pull_unlisted(
    appctx, name: str, *, catalog_origin: str, weights_only: bool, offline: bool
) -> None:
    """A name the released spec does not know: a tt-model bundle, or a plain HF repo.

    Both look like `namespace/name`, so the routing asks the Hub whether the repo
    carries a bundle manifest. Weights-only is always available as a fallback, so
    `tt model pull` is never a dead end for something that exists on the Hub."""
    if not looks_like_bundle_id(name):
        raise unknown_model_error(name, catalog_origin)
    is_bundle = None if (offline or weights_only) else bundles.is_bundle_repo(name)
    if is_bundle:
        _pull_bundle(appctx, name)
        return
    # Fetch the weights so they are cached, but never imply more than that. Only the
    # "definitely not a bundle" case can honestly say serving is impossible.
    if weights_only:
        appctx.output.warn(
            f"--weights-only: fetching HuggingFace weights for {name} and nothing "
            "else. If it is a tt-model bundle, drop the flag to install the bundle."
        )
    elif is_bundle is None:
        appctx.output.warn(
            f"could not check whether {name} is a tt-model bundle (Hub unreachable, "
            "private, or --offline); fetching weights only."
        )
    else:
        appctx.output.warn(
            f"{name} is not in the model catalog ({catalog_origin}) and has no "
            "tt-model bundle; downloading weights only. `tt serve` cannot serve it."
        )
    weights = ModelInfo(name=name, hf_repo=name, model_type="")
    path = hub.download_weights(weights, appctx.config, offline=offline, output=appctx.output)
    appctx.output.emit(
        {"name": name, "kind": "weights", "hf_repo": name, "path": str(path)},
        renderer=lambda d: f"{d['name']} weights ready at {d['path']}",
    )


@model_app.command("compile", no_args_is_help=True)
@handle_tt_errors
def compile_model(
    ctx: typer.Context,
    name: str = typer.Argument(help="Model name, e.g. Llama-3.1-8B-Instruct."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """[stub] Pre-compile a model to a TT-optimized format servable by `tt serve`."""
    raise TTError(
        "`tt model compile` is not available yet.",
        why="Ahead-of-time compilation to TT-optimized artifacts is planned; today "
        "`tt serve` compiles on first run.",
        next_step="Track progress at https://github.com/tenstorrent/tt-inference-server.",
        exit_code=ExitCode.UNSUPPORTED,
    )


def _stdin_isatty() -> bool:
    """Test seam: CliRunner replaces sys.stdin, so tests patch this, not isatty."""
    return sys.stdin.isatty()


def _confirm_destructive(appctx, what: str, *, yes: bool) -> None:
    """Ask before deleting. Non-interactive without --yes is a usage error rather
    than a silent yes: these operations reclaim tens of gigabytes."""
    if yes:
        return
    if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
        raise TTError(
            "Refusing to remove anything without confirmation.",
            why="stdin is not a terminal (or --json/--quiet is in effect), so there "
            "is no way to ask.",
            next_step="Re-run with --yes once you have checked `--dry-run`.",
            exit_code=ExitCode.USAGE,
        )
    if not confirm(f"{what} This cannot be undone. Continue?"):
        raise TTError("Nothing was removed.", exit_code=ExitCode.OK)


def _dispatch(appctx, name: str):
    """(ModelInfo, None) for a released-spec model, (None, bundle_id) for a tt-model
    bundle. Same rule as `tt serve`: the spec wins, and only a Hub-shaped name may
    fall through, so a catalog typo stays a catalog error."""
    catalog = ModelCatalog()
    model = catalog.find(name)
    if model is not None:
        return model, None
    if not looks_like_bundle_id(name):
        raise unknown_model_error(name, catalog.origin)
    return None, name


@model_app.command("stop", no_args_is_help=True)
@handle_tt_errors
def stop_model(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="Model name or tt-model bundle id.",
        autocompletion=complete_local_model,
    ),
    profile: str = typer.Option(None, "--profile", help="Stop only this profile."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Stop a running model server."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    model, bundle = _dispatch(appctx, name)
    if bundle is not None:
        backend = ModelManagerBackend(
            appctx.registry, appctx.runner, appctx.config, appctx.output
        )
        backend.stop(bundle, profile=profile)
        return
    _stop_catalog_model(appctx, model)


def _stop_catalog_model(appctx, model) -> None:
    """Find the container serving this model and stop it.

    Containers are named tt-inference-server-<uuid> with no labels, so they are
    identified by their mounts instead (see inference_server.ServerContainer).
    Anything unidentifiable is reported, never guessed at."""
    backend = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    containers = backend.running_containers()
    matched = [c for c in containers if c.matches(model)]
    unknown = [c for c in containers if not c.identified]
    if not matched and unknown:
        # Something IS running that we cannot attribute (e.g. served with
        # --host-weights-dir, whose path need not name the model). Report it rather
        # than stop a server on a guess.
        raise TTError(
            f"Cannot tell which container serves {model.name}.",
            why=f"{len(unknown)} running container(s) could not be identified: "
            + ", ".join(f"{c.name} ({c.id})" for c in unknown),
            next_step="`docker ps` to see what is running, then "
            "`docker stop <container>`.",
            exit_code=ExitCode.ERROR,
        )
    if not matched:
        # Nothing to do is not an error.
        appctx.output.emit(
            {"model": model.name, "stopped": []},
            renderer=lambda d: f"Nothing running for {d['model']}.",
        )
        return
    backend.stop_containers(matched)
    appctx.output.emit(
        {
            "model": model.name,
            "stopped": [{"id": c.id, "name": c.name} for c in matched],
        },
        renderer=lambda d: f"Stopped {len(d['stopped'])} container(s) for {d['model']}.",
    )


@model_app.command("rm", no_args_is_help=True)
@handle_tt_errors
def rm_model(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="Model name or tt-model bundle id.",
        autocompletion=complete_local_model,
    ),
    include_weights: bool = typer.Option(
        False,
        "--include-weights",
        help="Also delete the model's weights from the HuggingFace cache. They are "
        "shared with anything else that uses them and can be tens of gigabytes to "
        "re-download, so this is off by default.",
    ),
    keep_cache: bool = typer.Option(
        False,
        "--keep-cache",
        help="tt-model bundles: keep the JIT kernel cache so a later pull boots fast.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed and exit."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Remove a model's local artifacts, keeping its weights by default."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    model, bundle = _dispatch(appctx, name)
    if bundle is not None:
        _rm_bundle(
            appctx, bundle, include_weights=include_weights,
            keep_cache=keep_cache, dry_run=dry_run, yes=yes,
        )
        return
    _rm_catalog_model(
        appctx, model, include_weights=include_weights, dry_run=dry_run, yes=yes
    )


def _rm_bundle(
    appctx, bundle: str, *, include_weights: bool, keep_cache: bool,
    dry_run: bool, yes: bool,
) -> None:
    """tt-model owns removal for bundles — containers, image, pulled manifest,
    kernel cache, its own snapshot — so this is a passthrough, not a reimplementation."""
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    if dry_run:
        # tt-model has no --dry-run, so report the exact delegation instead of
        # guessing at its plan.
        argv = backend.rm_argv(
            "tt-model", bundle, include_weights=include_weights, keep_cache=keep_cache
        )
        appctx.output.emit(
            {"model": bundle, "dry_run": True, "delegates_to": argv},
            renderer=lambda d: "would run: " + " ".join(d["delegates_to"]),
        )
        return
    weights = "and its weights" if include_weights else "keeping its weights"
    _confirm_destructive(appctx, f"Remove the {bundle} bundle ({weights}).", yes=yes)
    backend.rm(bundle, include_weights=include_weights, keep_cache=keep_cache)
    appctx.output.emit(
        {"model": bundle, "removed_by": "tt-model", "weights_removed": include_weights},
        renderer=lambda d: f"{d['model']} removed.",
    )


def _artifact_rows(
    artifacts: list[Artifact], weights_bytes: int, root=None
) -> list[dict]:
    """--json keeps absolute paths (a published contract); `display` is the short
    form the human renderer prints under a single "in <root>" header."""
    rows = []
    for art in artifacts:
        try:
            display = str(art.path.relative_to(root)) if root else str(art.path)
        except ValueError:
            display = str(art.path)
        rows.append(
            {
                "kind": art.kind,
                "path": str(art.path),
                "display": display,
                "size_bytes": art.size_bytes,
            }
        )
    if weights_bytes:
        rows.append(
            {
                "kind": "weights",
                "path": "HF cache",
                "display": "HF cache",
                "size_bytes": weights_bytes,
            }
        )
    return rows


def _rm_catalog_model(appctx, model, *, include_weights: bool, dry_run: bool, yes: bool) -> None:
    """tt-inference-server has no rm of its own, so tt owns this. Images are never
    touched: they key on (version, tt_metal_commit) and are shared by many models."""
    backend = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    root = backend.checkout_root()
    artifacts = backend.removable_artifacts(model)
    _, weights_bytes = hub.cached_weights(model.hf_repo, appctx.config)
    planned_weights = weights_bytes if include_weights else 0
    rows = _artifact_rows(artifacts, planned_weights, root)
    total = sum(r["size_bytes"] for r in rows)

    def render(d: dict) -> str:
        if not d["artifacts"]:
            return (
                f"Nothing to remove for {d['model']}."
                + ("" if d["weights_kept_bytes"] == 0 else
                   f" Weights kept ({_human_size(d['weights_kept_bytes'])}); "
                   "pass --include-weights to delete them.")
            )
        head = ("Would remove" if d["dry_run"] else "Removed") + f" for {d['model']}"
        lines = [f"{head} (in {d['root']}):" if d.get("root") else f"{head}:"]
        for row in d["artifacts"]:
            lines.append(
                f"  {row['kind']:8} {_human_size(row['size_bytes']):>10}  {row['display']}"
            )
        lines.append(f"  {'total':8} {_human_size(d['total_bytes']):>10}")
        if d["weights_kept_bytes"]:
            lines.append(
                f"Weights kept ({_human_size(d['weights_kept_bytes'])}) — "
                "pass --include-weights to delete them."
            )
        lines.append(
            "Docker images are shared between models and are never removed; "
            "use `docker image prune` if you need the space."
        )
        return "\n".join(lines)

    payload = {
        "model": model.name,
        "root": str(root) if root else None,
        "dry_run": dry_run,
        "artifacts": rows,
        "total_bytes": total,
        "weights_kept_bytes": 0 if include_weights else weights_bytes,
        "images_removed": [],  # never: shared across models
    }
    if dry_run or not rows:
        appctx.output.emit(payload, renderer=render, soft_wrap=True)
        return
    _confirm_destructive(
        appctx, f"Remove {len(rows)} artifact(s), {_human_size(total)}, for {model.name}.",
        yes=yes,
    )
    freed = backend.remove_artifacts(artifacts)
    if include_weights:
        freed += hub.delete_cached_weights(model.hf_repo, appctx.config)
    payload["total_bytes"] = freed
    appctx.output.emit(payload, renderer=render, soft_wrap=True)
