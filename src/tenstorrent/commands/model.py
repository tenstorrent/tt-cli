# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt model` — browse, inspect, pull, stop and remove models.

Like `tt serve`, the destructive verbs dispatch on the name: a released-spec
model is handled here, a Hub bundle id is passed through to tt-model."""

from __future__ import annotations

import collections
import dataclasses
import sys
import time
from pathlib import Path
from typing import Callable

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
from ..backends.serving.studio import StudioBackend
from ..backends.serving.ps import STUDIO, human_duration, list_served
from .._compat import confirm
from ..cli import JsonFlag, PagedHelpGroup, QuietFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..models.model import ModelInfo
from ..modelhub.catalog import ModelCatalog, unknown_model_error
from ..modelhub import bundles, hub
from ..modelhub.studio import studio_only
from ..modelhub.completions import (
    complete_catalog_model,
    complete_local_model,
    complete_model,
)

model_app = typer.Typer(
    help="Model management: browse, pull, and compile models.",
    no_args_is_help=True,
    # `tt model --help` is one of the two help pages long enough to scroll off a
    # small pane; see PagedHelpGroup.
    cls=PagedHelpGroup,
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


def _known_devices() -> set[str]:
    """Every device name a catalog model declares — the release spec's device
    vocabulary (galaxy, t3k, gpu, ...) is not a board/mesh tag, so it has no
    fixed enum to check against other than what the catalog actually uses."""
    return {
        device for model in ModelCatalog().list(cached_sizes={}) for device in model.hardware
    }


def _add_columns(table: Table, columns: tuple[str, ...]) -> None:
    """Columns that fold rather than ellipsize on a narrow terminal.

    Rich's default overflow is `ellipsis`, which trims whatever column happens to
    be widest — on a 40-column tmux pane that was the model name, the one value
    the user needs whole to paste into `tt serve`. Folding wraps a long cell over
    several lines instead, so a narrow terminal costs height, never characters.
    """
    for column in columns:
        table.add_column(column, overflow="fold")


def _validate_hardware(hardware: str) -> str:
    """`hardware`, lowercased, once it is confirmed real — a catalog device or
    a recognised board/mesh tag — so a typo like p250 is refused up front
    instead of quietly returning zero rows."""
    device = hardware.lower()
    if device in _known_devices() or bundles.is_hardware_tag(device):
        return device
    raise TTError(
        f"{hardware!r} is not a recognized hardware target.",
        why="It matches no catalog device and no known board/mesh tag.",
        next_step="Run `tt model list --all` to see catalog devices, or drop "
        "--hw to auto-detect.",
        exit_code=ExitCode.USAGE,
    )


_MODEL_CAPTION = (
    "source: tt-inference-server/tt-studio catalog vs. HuggingFace/local community. "
    "profiles: smallest board/mesh tag per capability. "
    "via: the paths `tt serve` offers — inference-server (its released spec, the "
    "default), studio (TT-Studio's catalog; `--studio` picks it), tt-model for a "
    "bundle. `tt model list --help` for details."
)


def _hardware_cell(row: dict, hardware: str | None) -> str:
    """Comma-separated tags matching `hardware` (see bundles.hardware_satisfies)
    — every satisfying tag is shown, not just the closest one. Depends only on
    the resolved value, never on whether it came from --hw or detection, so
    both read identically."""
    tags = row.get("hardware") or []
    if hardware:
        tags = sorted(t for t in tags if bundles.hardware_satisfies(t, hardware))
    else:
        tags = sorted(tags)
    return ", ".join(tags) or "—"


def _cached_cell(row: dict) -> str:
    """✓ + size when it is on disk (catalog weights, or a pulled bundle's
    referenced weights), — otherwise."""
    return f"✓ {_human_size(row['cache_size_bytes'])}".strip() if row["cached"] else "—"


# Community bundles report their engine tag verbatim (vllm-plugin, vllm); the
# catalog spells the same engine "vLLM". Normalize for display so the merged
# listing shows one name for what is actually the same engine.
_ENGINE_DISPLAY = {"vllm-plugin": "vLLM", "vllm": "vLLM"}


def _engines_cell(row: dict) -> str:
    names = dict.fromkeys(_ENGINE_DISPLAY.get(e.lower(), e) for e in row["engines"])
    return ", ".join(names) or "—"


def _catalog_row(m: dict) -> dict:
    """The table's common columns, plus every catalog-only field verbatim —
    `--json` gains `source`/`type` alongside the raw `model_type` for the
    community side to share. `hardware` keeps only the smallest tag per
    capability (see bundles.drop_superseded_hardware)"""
    supported = {hw: d for hw, d in m["devices"].items() if d["supported"]}
    profiles = {
        hw: (
            d["max_context"],
            d["impl_id"],
            tuple(sorted(d["engines"] or [])),
            d["docker_image"],
            d["override_tt_config"],
        )
        for hw, d in supported.items()
    }
    # A studio-only entry (modelhub/studio.py) is in the catalog, but it is not
    # tt-inference-server's — say where it came from, as the community rows do.
    source = "tt-inference-server" if "inference-server" in m["backends"] else "tt-studio"
    return {
        **m,
        "source": source,
        "type": m["model_type"],
        "hardware": bundles.drop_superseded_hardware(profiles),
    }


def _bundle_row(b: dict) -> dict:
    """Same shape as `_catalog_row`: every BundleInfo field verbatim, plus the
    common `type`/`cached`/`cache_size_bytes` the table renders. `engines` is
    `engine` promoted to a one-item list (or empty), matching the catalog's
    list shape."""
    cached = bool(b["installed"] and b.get("weights_repo") and b.get("weights_bytes") is not None)
    return {
        **b,
        "type": None,
        "engines": [b["engine"]] if b.get("engine") else [],
        "cached": cached,
        "cache_size_bytes": b.get("weights_bytes") if cached else None,
        # Every bundle serves through tt-model; the catalog rows carry theirs.
        "backends": ["tt-model"],
    }


_SCOPE_TITLE = {
    "catalog": "tt catalog",
    "community": "community",
    "all": "tt catalog + community",
}


def _model_table(payload: dict, *, hardware: str | None, detected: bool) -> Table:
    title = f"Models ({_SCOPE_TITLE[payload['scope']]})"
    if hardware:
        title += f" for {hardware}"
        if detected:
            title += " (detected — `tt model list --all` for every device/bundle)"
    table = Table(title=title, caption=_MODEL_CAPTION, caption_justify="left")
    _add_columns(
        table, ("name", "source", "engine", "serving profiles", "via", "weights")
    )
    for row in payload["models"]:
        table.add_row(
            row["name"],
            row["source"],
            _engines_cell(row),
            _hardware_cell(row, hardware),
            ", ".join(row["backends"]),
            _cached_cell(row),
        )
    return table


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
        help="Filter to a device config (e.g. p300x2); skips auto-detection. "
        "For community bundles, matches any whose board/mesh tag needs no more "
        "chips than this, on the same chip family.",
    ),
    all_devices: bool = typer.Option(
        False, "--all", help="Every model on every device, not just this machine's."
    ),
    community: bool = typer.Option(
        False,
        "--community",
        help="Only community bundles (Hub + local installs) — skip the released "
        "catalog. Community bundles are models anyone has packaged with "
        "tt-model-manager and published on the Hugging Face Hub; they are not "
        "tested or maintained by Tenstorrent. The opposite of --catalog.",
    ),
    catalog_only: bool = typer.Option(
        False,
        "--catalog",
        help="Only the released model catalog (tt-inference-server) — skip "
        "community bundles. The opposite of --community.",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Browse models that run on this machine: the released catalog plus
    community tt-model bundles from the Hub (default: detected hardware only).

    source: `tt-inference-server` is the released catalog — models Tenstorrent
    ships and tests, with known per-device support (`tt model info NAME` for
    details); `HuggingFace` is the community catalog on the Hub — bundles
    anyone has packaged with tt-model-manager, not tested or maintained by
    Tenstorrent; `local` is installed here — a bundle on both shows up twice,
    once per source. Pass --catalog or --community to see just one source.
    profiles: the board/mesh target(s) a model supports, collapsed to the
    smallest tag per capability (a bigger board that adds nothing over a
    smaller one is left out). Every entry serves with `tt serve <name>`
    (`tt serve <namespace>/<name>` for a bundle); weights are referenced
    rather than shipped."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    if community and catalog_only:
        raise TTError(
            "--community and --catalog are opposites.",
            why="One shows only community bundles, the other only the released "
            "catalog.",
            next_step="Pass at most one, or neither to see both.",
            exit_code=ExitCode.USAGE,
        )
    detected = not hardware and not all_devices
    device = _validate_hardware(hardware) if hardware else (
        None if all_devices else _detect_device(appctx)
    )
    show_catalog = not community
    show_community = not catalog_only
    models = ModelCatalog().list() if show_catalog else []
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
    rows = [_catalog_row(dataclasses.asdict(m)) for m in models]
    # Community bundles publish no model type, so --type simply drops them —
    # same outcome as any other filter they cannot match, no special-casing.
    if show_community and not model_type:
        rows.extend(_community_rows(appctx, cached=cached, hardware=device))
    rows.sort(key=lambda r: (r["name"].lower(), r["source"]))
    scope = "community" if community else "catalog" if catalog_only else "all"
    appctx.output.emit(
        {"device": device, "scope": scope, "models": rows},
        renderer=lambda payload: _model_table(payload, hardware=device, detected=detected),
        page=True,
    )


def _community_rows(appctx, *, cached: bool, hardware: str | None) -> list[dict]:
    """Community bundles published on the Hub, or installed locally.

    Filtered the same way as the catalog side: detected device by default,
    --hw for an explicit one, --all for everything (see
    bundles.hardware_satisfies for what counts as a match). """
    # Local installs first: they need no network, and they are the only source for a
    # bundle nobody published — someone shares an id, you pull it, the Hub shows
    # nothing. Catalog rows win on name, since a listed bundle is the richer record.
    local = bundles.local_bundles(config=appctx.config)
    if appctx.offline:
        # The catalog is a Hub index with no bundled copy, but local installs are
        # entirely on disk — show those rather than refusing the whole command.
        appctx.output.warn(
            "--offline: showing only community bundles installed on this machine; "
            "the community catalog lives on the Hugging Face Hub."
        )
        listed = []
    else:
        # Community bundles are now just one part of a listing that must still
        # work with no network: a Hub outage degrades to a warning plus the
        # catalog rows, the same way a device-detection failure does above,
        # rather than failing a command that used to need no network at all.
        try:
            listed = bundles.search_community(config=appctx.config)
        except TTError as err:
            appctx.output.warn(
                f"community bundles skipped ({err.what}) — showing the catalog only."
            )
            listed = []
        else:
            # Refresh the shell-completion cache: tab-time must never touch the
            # Hub, so this listing is where `tt serve <TAB>` learns bundle ids.
            bundles.save_community_cache([b.name for b in listed])
    # Not merged by name: a bundle that is both published and installed gets one
    # row per source, so the listing shows both facts instead of picking one.
    found = sorted(local + listed, key=lambda b: (b.name.lower(), b.source))
    if cached:  # --cached reads as "what do I have locally" here too
        found = [b for b in found if b.source == "local"]
    if hardware:
        found = [
            b for b in found
            if any(bundles.hardware_satisfies(hw, hardware) for hw in b.hardware)
        ]
    return [_bundle_row(dataclasses.asdict(b)) for b in found]


def _info_renderer(payload: dict) -> Table:
    m = payload
    table = Table(title=m["name"], show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("hf_repo", m["hf_repo"])
    table.add_row("type", m["model_type"])
    table.add_row("engines", ", ".join(m["engines"]))
    table.add_row("via", ", ".join(m["backends"]))
    if m["tt_model_id"] and "studio" in m["backends"]:
        servable = f"yes — `tt serve {m['name']}` (`--studio` deploys it with TT-Studio)"
    elif m["tt_model_id"]:
        servable = f"yes — `tt serve {m['name']}`"
    elif m["backends"] == ["studio"]:
        servable = f"yes, through TT-Studio — `tt serve {m['name']}`"
    else:
        servable = "no (no tt-inference-server entry)"
    table.add_row("servable", servable)
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
        elif support["note"]:
            detail += f"\n[dim]{support['note']}[/dim]"
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
    in_catalog = None if appctx.offline else (row is not None and row.source != "local")
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
    if not serve:  # pulled bundles show the manifest's target below instead
        table.add_row("hardware", _hardware_cell(b, None))
    table.add_row("engine", b["engine"] or "—")
    if b["downloads"] is not None:
        table.add_row("downloads", str(b["downloads"]))
    table.add_row(
        "installed", "yes" if b["installed"] else f"no — `tt model pull {name}`"
    )
    if b["weights_repo"]:
        cached = _cached_cell(_bundle_row(b))
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
            next_step="Run `tt model list` to see bundle ids.",
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
    if studio_only(model) or _studio_is_serving(appctx, model):
        # Deployed by studio's run.py, so studio stops it (and resets its chips).
        StudioBackend(appctx.registry, appctx.runner, appctx.config, appctx.output).stop(
            model
        )
        return
    _stop_catalog_model(appctx, model)


def _studio_is_serving(appctx, model) -> bool:
    """Whether a running container is studio's deploy of this model. A model both
    paths offer may have come up through `tt serve X --studio`, and studio's
    containers are not tt-inference-server's (`tt model ps` tells them apart the
    same way), so the owner has to be asked to stop it."""
    runtime = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    ).container_runtime()
    return any(
        row.backend == STUDIO and row.name == model.name
        for row in list_served(appctx.runner, runtime, probe=False)
    )


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


# -- logs --------------------------------------------------------------------------
@model_app.command("logs", no_args_is_help=True)
@handle_tt_errors
def logs_model(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="Model name or tt-model bundle id.",
        autocompletion=complete_local_model,
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Keep streaming new output until Ctrl-C."
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="Only output after this docker-style duration or timestamp (10m, 2h, "
        "2026-09-08T12:00:00). Needs a running tt-inference-server container.",
    ),
    tail: int = typer.Option(None, "--tail", help="Only the last N lines."),
    profile: str = typer.Option(
        None, "--profile", help="tt-model bundles: logs for this profile."
    ),
    quiet: QuietFlag = False,
) -> None:
    """Show a served model's output.

    A catalog model prints the newest log tt-inference-server wrote for it under
    the checkout's workflow_logs/ (the running server keeps appending to it); a
    tt-model bundle id passes through to `tt-model logs`.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(quiet=quiet)
    if appctx.output.json_mode:
        raise TTError(
            "`tt model logs` prints plain text, not JSON.",
            next_step="Drop --json; `tt model ps --json` has the structured view.",
            exit_code=ExitCode.USAGE,
        )
    model, bundle = _dispatch(appctx, name)
    try:
        if bundle is not None:
            _logs_bundle(appctx, bundle, follow=follow, since=since, tail=tail, profile=profile)
        else:
            _logs_catalog_model(
                appctx, model, follow=follow, since=since, tail=tail, profile=profile
            )
    except KeyboardInterrupt:
        # Ctrl-C is how --follow ends; nothing went wrong.
        return


def _logs_bundle(appctx, bundle: str, *, follow: bool, since, tail, profile) -> None:
    """Passthrough to `tt-model logs`, which knows only --follow/--profile."""
    if since is not None or tail is not None:
        raise TTError(
            "tt-model logs has no --since or --tail.",
            why="Bundle logs pass straight through to tt-model, which streams the "
            "whole container log or follows it.",
            next_step="`docker logs --since <when> --tail <n> <container>` — "
            "`tt model ps` shows the container name.",
            exit_code=ExitCode.USAGE,
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    if profile is None:
        # tt-model's own default picks the wrong container when profile names
        # nest (p150 vs p150x2); name the one that is actually running instead.
        running = backend.running_profiles(bundle)
        if len(running) == 1:
            profile = running[0]
    rc = backend.logs(bundle, follow=follow, profile=profile)
    if rc not in (0, 130):  # 130 = the user's Ctrl-C on --follow
        raise TTError(
            f"tt-model logs exited with {rc}.",
            why="tt-model could not show the bundle's container log; its message is "
            "above.",
            next_step="`tt model ps` lists the running bundles and their profiles; "
            f"`tt model logs {bundle} --profile <name>` picks one.",
            exit_code=ExitCode.TOOL_FAILED,
        )


def _logs_catalog_model(appctx, model, *, follow: bool, since, tail, profile) -> None:
    if profile is not None:
        raise TTError(
            f"{model.name} is not a tt-model bundle, so it has no profiles.",
            next_step="Drop --profile.",
            exit_code=ExitCode.USAGE,
        )
    backend = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    if since is not None:
        # The log file has no time index, so --since means `docker logs --since`
        # on the live container — the same stream, seekable by time.
        containers = backend.running_containers()
        matched = [c for c in containers if c.matches(model)]
        if not matched:
            raise TTError(
                f"No running tt-inference-server container for {model.name}.",
                why="--since works on a live container's log; the log file on disk "
                "has no time index to seek by.",
                next_step=f"`tt model logs {model.name} --tail 200` for the end of "
                "the last run, or `tt serve` it first.",
                exit_code=ExitCode.USAGE,
            )
        if len(matched) > 1:
            appctx.output.warn(
                f"{len(matched)} containers serve {model.name}; showing "
                f"{matched[0].name} ({matched[0].id})."
            )
        runtime = backend.container_runtime()
        argv = [runtime, "logs", "--since", since]
        if tail is not None:
            argv += ["--tail", str(tail)]
        if follow:
            argv.append("--follow")
        argv.append(matched[0].id)
        appctx.output.status(
            f"showing {Path(runtime).name} logs for {matched[0].name}", soft_wrap=True
        )
        appctx.runner.stream(argv, tool=runtime, check=False)
        return
    path = backend.newest_log_file(model)
    if path is None:
        root = backend.checkout_root()
        where = f"{root / 'workflow_logs'}" if root else "the tt-inference-server checkout"
        raise TTError(
            f"No logs for {model.name}.",
            why=f"Nothing has been served through tt-inference-server on this machine "
            f"(no *_{model.name}_*.log under {where}).",
            next_step=f"`tt serve {model.name}`",
            exit_code=ExitCode.ERROR,
        )
    appctx.output.status(f"showing {path}", soft_wrap=True)
    _print_log_file(path, tail=tail, follow=follow)


def _print_log_file(
    path: Path,
    *,
    tail: int | None,
    follow: bool,
    poll_s: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """cat / tail -n / tail -f for one log file, writing raw bytes to stdout.

    `should_stop` is a test seam for --follow; None means run until Ctrl-C. A
    file that shrinks (rotated or truncated) is re-read from the start rather
    than waiting for it to grow past the old offset."""
    out = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout
    try:
        with open(path, "rb") as fh:
            if tail is not None:
                last = collections.deque(fh, maxlen=max(tail, 0))
                for line in last:
                    out.write(line)
            else:
                out.write(fh.read())
            out.flush()
            if not follow:
                return
            offset = fh.tell()
            while should_stop is None or not should_stop():
                time.sleep(poll_s)
                size = path.stat().st_size
                if size < offset:
                    offset = 0
                fh.seek(offset)
                chunk = fh.read()
                if chunk:
                    out.write(chunk)
                    out.flush()
                    offset = fh.tell()
    except PermissionError as exc:
        raise TTError(
            f"Cannot read {path}.",
            why="It is owned by another user — container-created files are often "
            "owned by root.",
            next_step=f"sudo tail -f {path}",
            exit_code=ExitCode.NEEDS_SUDO,
        ) from exc
    except FileNotFoundError as exc:
        raise TTError(
            f"{path} disappeared while reading it.",
            next_step="Re-run `tt model logs` to pick the newest file.",
            exit_code=ExitCode.ERROR,
        ) from exc


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


# -- what is being served --------------------------------------------------------------
def _ps_table(payload: dict) -> Table | str:
    rows = payload["served"]
    if not rows:
        return (
            "No model servers running. `tt serve <model>` starts one; "
            "`tt model ps --all` includes stopped containers."
        )
    table = Table(
        title="Served models",
        caption=(
            "health: healthy answers GET /v1/models; starting is running but not "
            "answering yet (docker logs -f <container>); stopped appears only with "
            "--all; unknown means not probed or no published port."
        ),
    )
    table.add_column("name", overflow="fold")
    table.add_column("backend")
    table.add_column("container")
    table.add_column("port")
    table.add_column("health")
    table.add_column("uptime")
    for row in rows:
        table.add_row(
            row["name"],
            row["backend"],
            row["container"] or "—",
            str(row["port"]) if row["port"] else "—",
            row["health"],
            human_duration(row["uptime_s"]) or "—",
        )
    return table


@model_app.command("ps")
@handle_tt_errors
def ps_models(
    ctx: typer.Context,
    all_: bool = typer.Option(
        False, "--all", "-a", help="Include stopped model containers, not just running ones."
    ),
    no_probe: bool = typer.Option(
        False,
        "--no-probe",
        help="Skip the HTTP health check (GET /v1/models on each server); "
        "health shows as unknown.",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """List the model servers on this machine: name, backend, container, port, health, uptime.

    Covers tt-inference-server containers (`tt serve`), tt-model bundles and
    TT-Studio model containers. Exits 0 with an empty list when nothing is served.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    runtime = InferenceServerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    ).container_runtime()
    rows = list_served(appctx.runner, runtime, include_stopped=all_, probe=not no_probe)
    appctx.output.emit(
        {"served": [dataclasses.asdict(r) for r in rows], "probed": not no_probe},
        renderer=_ps_table,
    )
