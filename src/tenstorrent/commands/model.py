# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt model` — browse, search, inspect, pull, query, stop and remove models.

Like `tt serve`, the destructive verbs dispatch on the name: a released-spec
model is handled here, a Hub bundle id is passed through to tt-model. The verbs
that only exist for bundles (search, profiles, login, publish, and the authoring
passthroughs) are tt-model's, surfaced here so one CLI covers the whole loop."""

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
from ..backends.serving.ps import human_duration, list_served
from .._compat import confirm
from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..launchers.discovery import DEFAULT_PORT
from ..models.model import ModelInfo
from ..modelhub.catalog import ModelCatalog, unknown_model_error
from ..modelhub import bundles, hub
from ..modelhub.completions import (
    complete_bundle_id,
    complete_catalog_model,
    complete_local_model,
    complete_model,
)

model_app = typer.Typer(
    help="Model management: browse, search, pull, query, stop and remove models.",
    no_args_is_help=True,
)

PANEL_AUTHORING = "Authoring (delegated to tt-model)"


def _human_size(size: int | None) -> str:
    if size is None:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def _applies(flags: list[str]) -> str:
    return "only applies" if len(flags) == 1 else "only apply"


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
    "source: tt-inference-server catalog vs. HuggingFace/local community. "
    "profiles: smallest board/mesh tag per capability. "
    "`tt model list --help` for details."
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
    return {
        **m,
        "source": "tt-inference-server",
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
    # fold rather than ellipsize: the name is what you paste into `tt serve`
    table.add_column("name", overflow="fold")
    for column in ("source", "engine", "serving profiles", "weights"):
        table.add_column(column)
    for row in payload["models"]:
        table.add_row(
            row["name"],
            row["source"],
            _engines_cell(row),
            _hardware_cell(row, hardware),
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
        "catalog. The opposite of --catalog.",
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

    source: `tt-inference-server` is the released catalog; `HuggingFace` is
    the community catalog on the Hub; `local` is installed here — a bundle
    on both shows up twice, once per source. profiles: the board/mesh
    target(s) a model supports, collapsed to the smallest tag per capability
    (a bigger board that adds nothing over a smaller one is left out). Every
    entry serves with `tt serve <name>`; weights are referenced rather than
    shipped."""
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


def _search_table(payload: dict) -> Table:
    scope = "community catalog" if payload["catalog"] else "every published bundle"
    query = f" matching {payload['query']!r}" if payload["query"] else ""
    arch = f", arch {payload['arch']}" if payload["arch"] else ""
    table = Table(
        title=f"tt-model bundles on the Hub — {scope}{query}{arch}",
        caption="Newest first, as `tt-model search` orders them. `installed` is "
        "tt-model's record on this machine. Serve any row with `tt serve <name>`; "
        "`--catalog` narrows to bundles opted into the community catalog.",
    )
    table.add_column("name", overflow="fold")
    for column in ("visibility", "downloads", "updated", "installed"):
        table.add_column(column)
    for row in payload["bundles"]:
        table.add_row(
            row["name"],
            "private" if row["private"] else "public",
            str(row["downloads"]) if row["downloads"] is not None else "—",
            (row["last_modified"] or "")[:10] or "—",
            "✓" if row["installed"] else "—",
        )
    if not payload["bundles"]:
        table.caption = "No matching bundles found."
    return table


@model_app.command("search")
@handle_tt_errors
def search_bundles(
    ctx: typer.Context,
    query: str = typer.Argument(
        "", help="Free-text query over published tt-model bundles (default: all)."
    ),
    catalog: bool = typer.Option(
        False,
        "--catalog",
        help="Only bundles listed in the community catalog (what `tt model list "
        "--community` shows), not every pushed bundle.",
    ),
    arch: str = typer.Option(
        None, "--arch", help="Only bundles tagged for this arch (blackhole, wormhole_b0)."
    ),
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum number of results."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Search the Hugging Face Hub for published tt-model bundles."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    if appctx.offline:
        raise TTError(
            "Searching needs the Hugging Face Hub.",
            why="Published bundles are a Hub index; there is no local copy to search.",
            next_step="Drop --offline, or `tt model list --community --cached` for "
            "the bundles installed on this machine.",
            exit_code=ExitCode.OFFLINE,
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    found = backend.search(query, limit=limit, catalog=catalog, arch=arch)
    installed = bundles.installed_bundles()
    rows = [
        {
            "name": str(row["id"]),
            "private": bool(row.get("private")),
            "downloads": row.get("downloads"),
            "last_modified": row.get("last_modified") or None,
            "installed": str(row["id"]).lower() in installed,
        }
        for row in found
    ]
    # Every id seen here is a valid `tt serve` argument, so teach tab completion.
    bundles.add_to_community_cache([r["name"] for r in rows])
    appctx.output.emit(
        {"query": query, "catalog": catalog, "arch": arch, "bundles": rows},
        renderer=_search_table,
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
        help="Model name, e.g. Llama-3.1-8B-Instruct.",
        autocompletion=complete_catalog_model,
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Show model metadata: engines, per-device support, requirements."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    model = ModelCatalog().get(name)
    appctx.output.emit(dataclasses.asdict(model), renderer=_info_renderer)


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
    force: bool = typer.Option(
        False,
        "--force",
        help="Bundles: reinstall even if already installed, and push past a "
        "compatibility warning (never past an arch mismatch).",
    ),
    no_weights: bool = typer.Option(
        False,
        "--no-weights",
        help="Bundles: install the bundle only; the weights are fetched on the "
        "first serve instead.",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Download a model: a catalog model's weights, or a tt-model bundle."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    offline = offline or appctx.offline
    bundle_flags = [f for f, on in (("--force", force), ("--no-weights", no_weights)) if on]
    if bundle_flags and weights_only:
        raise TTError(
            f"{' and '.join(bundle_flags)} {_applies(bundle_flags)} to a tt-model bundle.",
            why="--weights-only fetches plain HuggingFace weights, which tt-model "
            "never installs.",
            next_step="Drop the flag(s), or drop --weights-only.",
            exit_code=ExitCode.USAGE,
        )
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
        _pull_bundle(appctx, name, force=force, with_weights=not no_weights)
        return
    catalog = ModelCatalog()
    model = catalog.find(name)
    if model is None:
        _pull_unlisted(
            appctx, name, catalog_origin=catalog.origin,
            weights_only=weights_only, offline=offline,
            force=force, with_weights=not no_weights,
        )
        return
    if bundle_flags:
        raise TTError(
            f"{' and '.join(bundle_flags)} {_applies(bundle_flags)} to a tt-model "
            f"bundle; {model.name} is a catalog model.",
            why="A catalog model's weights are one HuggingFace snapshot: there is "
            "nothing to reinstall, and nothing to pull without them.",
            next_step=f"`tt model pull {model.name}` (re-downloads only what is missing).",
            exit_code=ExitCode.USAGE,
        )
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


def _pull_bundle(
    appctx, name: str, *, force: bool = False, with_weights: bool = True
) -> None:
    """Install a tt-model bundle, with its weights unless told otherwise."""
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
    backend.pull(name, with_weights=with_weights, force=force)
    appctx.output.emit(
        {"name": name, "kind": "bundle", "pulled_by": "tt-model",
         "with_weights": with_weights, "force": force},
        renderer=lambda d: f"{d['name']} installed — serve it with `tt serve {d['name']}`.",
    )


def _pull_unlisted(
    appctx, name: str, *, catalog_origin: str, weights_only: bool, offline: bool,
    force: bool = False, with_weights: bool = True,
) -> None:
    """A name the released spec does not know: a tt-model bundle, or a plain HF repo.

    Both look like `namespace/name`, so the routing asks the Hub whether the repo
    carries a bundle manifest. Weights-only is always available as a fallback, so
    `tt model pull` is never a dead end for something that exists on the Hub."""
    if not looks_like_bundle_id(name):
        raise unknown_model_error(name, catalog_origin)
    is_bundle = None if (offline or weights_only) else bundles.is_bundle_repo(name)
    if is_bundle:
        _pull_bundle(appctx, name, force=force, with_weights=with_weights)
        return
    if force or not with_weights:
        raise TTError(
            f"--force / --no-weights only apply to a tt-model bundle, and {name} "
            + ("could not be checked." if is_bundle is None else "is not one."),
            why="Both flags configure `tt-model pull`; a plain weights download has "
            "neither a venv to reinstall nor anything but weights to fetch.",
            next_step=f"`tt model pull {name}` to fetch the weights, or `--bundle` to "
            "insist it is a bundle.",
            exit_code=ExitCode.USAGE,
        )
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
    _confirm_action(
        appctx,
        f"{what} This cannot be undone. Continue?",
        yes=yes,
        refusal="Refusing to remove anything without confirmation.",
        hint="Re-run with --yes once you have checked `--dry-run`.",
        declined="Nothing was removed.",
    )


def _confirm_action(
    appctx, question: str, *, yes: bool, refusal: str, hint: str, declined: str
) -> None:
    """The one confirmation rule for anything irreversible or outward-facing:
    --yes skips; JSON/quiet/non-TTY without --yes is a usage error, never a silent
    yes; declining is a clean exit 0."""
    if yes:
        return
    if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
        raise TTError(
            refusal,
            why="stdin is not a terminal (or --json/--quiet is in effect), so there "
            "is no way to ask.",
            next_step=hint,
            exit_code=ExitCode.USAGE,
        )
    if not confirm(question):
        raise TTError(declined, exit_code=ExitCode.OK)


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


# -- bundle-only verbs (tt-model's own) --------------------------------------------
def _bundle_only(appctx, name: str, verb: str, *, catalog_hint: str) -> str:
    """The bundle id for a verb that only exists for tt-model bundles. A catalog
    model gets told which tt command answers the same question instead."""
    model, bundle = _dispatch(appctx, name)
    if bundle is None:
        raise TTError(
            f"`tt model {verb}` applies to tt-model bundles; {model.name} is a "
            "catalog model served by tt-inference-server.",
            why="Catalog models are configured by the released spec, not by a bundle "
            "manifest.",
            next_step=catalog_hint,
            exit_code=ExitCode.USAGE,
        )
    return bundle


def _profiles_renderer(payload: dict) -> Table:
    table = Table(title=f"{payload['model']} — serve profiles", show_header=True)
    table.add_column("profile")
    table.add_column("default")
    for name in payload["profiles"]:
        table.add_row(name, "✓" if name == payload["default"] else "")
    table.caption = (
        "One image serves every profile; pick one with "
        f"`tt serve {payload['model']} --profile <name>`."
    )
    return table


@model_app.command("profiles", no_args_is_help=True)
@handle_tt_errors
def model_profiles(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="A pulled tt-model bundle id (namespace/name).",
        autocompletion=complete_bundle_id,
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Show a bundle's serve profiles and which one is the default."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    bundle = _bundle_only(
        appctx, name, "profiles",
        catalog_hint=f"`tt model info {name}` shows its per-device support.",
    )
    details = bundles.serve_details(bundle)
    if details is None:
        raise TTError(
            f"{bundle} is not pulled on this machine.",
            why="Profiles are read from the bundle's manifest, which arrives with "
            "`tt model pull`; tt does not fetch one just to list them.",
            next_step=f"`tt model pull {bundle}`, or `tt model info {bundle}`.",
            exit_code=ExitCode.ERROR,
        )
    if appctx.output.json_mode:
        # tt-model prints a checklist, not JSON, so the --json contract is tt's own,
        # read from the same manifest tt-model would.
        appctx.output.emit(
            {
                "model": bundle,
                "profiles": details.get("profiles") or [],
                "default": details.get("default_profile"),
            },
            renderer=_profiles_renderer,
        )
        return
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    backend.profiles(bundle)


def _served_base_url(appctx, *, port: int | None, url: str | None) -> str:
    """Server root for `tt-model curl` (no /v1): --url, --port, or the one server
    tt itself is running — found the same way `tt launch` finds it."""
    if url and port:
        raise TTError(
            "--url and --port cannot both be given.",
            why="They set the same thing.",
            next_step="Pass --port for a local server, --url for anything else.",
            exit_code=ExitCode.USAGE,
        )
    if url:
        return url.rstrip("/").removesuffix("/v1")
    if port:
        return f"http://127.0.0.1:{port}"
    from .launch import _local_candidate_ports

    ports = sorted(set(_local_candidate_ports(appctx)))
    if len(ports) > 1:
        raise TTError(
            f"Several model servers are running (ports {', '.join(map(str, ports))}).",
            why="tt cannot tell which one the prompt is for.",
            next_step="Pass --port <port>; `tt model ps` lists what is serving where.",
            exit_code=ExitCode.USAGE,
        )
    return f"http://127.0.0.1:{ports[0] if ports else DEFAULT_PORT}"


@model_app.command(
    "curl",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@handle_tt_errors
def model_curl(
    ctx: typer.Context,
    prompt: str = typer.Argument(
        None, help="The user message to send (default: tt-model's one-line greeting)."
    ),
    port: int = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Local port the server listens on (default: the one tt is serving on).",
    ),
    url: str = typer.Option(None, "--url", help="Server root, for a non-local server."),
    model: str = typer.Option(
        None, "--model", help="Model id to name in the request (default: ask the server)."
    ),
    print_only: bool = typer.Option(
        False, "--print", help="Print the equivalent curl command instead of sending it."
    ),
    quiet: QuietFlag = False,
) -> None:
    """Send a chat completion to the model being served.

    Anything tt does not recognise goes into the request body, so vLLM's sampling
    options need no flags of their own: `tt model curl "a haiku" --max-tokens 40`.
    Works for catalog models and bundles alike — it is one HTTP request.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(quiet=quiet)
    base_url = _served_base_url(appctx, port=port, url=url)
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    backend.curl(
        prompt, base_url=base_url, model=model, print_only=print_only,
        extra_args=list(ctx.args),
    )


@model_app.command("login")
@handle_tt_errors
def model_login(
    ctx: typer.Context,
    token: str = typer.Option(
        None, "--token", help="Hugging Face token; omit to be prompted for one."
    ),
    quiet: QuietFlag = False,
) -> None:
    """Log in to the Hugging Face Hub for gated or private bundles and weights.

    Uses Hugging Face's own token store, so `hf auth login` and tt agree.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(quiet=quiet)
    if appctx.offline:
        raise TTError(
            "Logging in needs the Hugging Face Hub.",
            why="The token is verified against the Hub before it is stored.",
            next_step="Drop --offline.",
            exit_code=ExitCode.OFFLINE,
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    backend.login(token)


def _catalog_listing(appctx, name: str, *, listed: bool, yes: bool) -> None:
    verb = "publish" if listed else "unpublish"
    if not looks_like_bundle_id(name):
        raise TTError(
            f"{name!r} is not a bundle id.",
            why="Only a pushed tt-model bundle (a Hub repo, namespace/name) can be "
            "listed in the community catalog.",
            next_step="Pass the repo id `tt-model push` printed.",
            exit_code=ExitCode.USAGE,
        )
    if appctx.offline:
        raise TTError(
            f"{verb} needs the Hugging Face Hub.",
            why="The catalog listing is a tag on the Hub repo.",
            next_step="Drop --offline.",
            exit_code=ExitCode.OFFLINE,
        )
    if listed:
        _confirm_action(
            appctx,
            f"Publish {name}: this makes the repo public if it is private, and lists "
            "it in the community catalog for everyone. Continue?",
            yes=yes,
            refusal="Refusing to publish without confirmation.",
            hint="Re-run with --yes.",
            declined="Nothing was published.",
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    backend.set_catalog_listing(name, listed=listed)
    appctx.output.emit(
        {"model": name, "listed": listed},
        renderer=lambda d: (
            f"{d['model']} is listed in the community catalog — it now shows in "
            "`tt model list --community`."
            if d["listed"]
            else f"{d['model']} is no longer listed; the repo itself is untouched."
        ),
    )


@model_app.command("publish", no_args_is_help=True, rich_help_panel=PANEL_AUTHORING)
@handle_tt_errors
def model_publish(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="A pushed tt-model bundle id (namespace/name).",
        autocompletion=complete_bundle_id,
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """List a pushed bundle in the community catalog (makes a private repo public)."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    _catalog_listing(appctx, name, listed=True, yes=yes)


@model_app.command("unpublish", no_args_is_help=True, rich_help_panel=PANEL_AUTHORING)
@handle_tt_errors
def model_unpublish(
    ctx: typer.Context,
    name: str = typer.Argument(
        help="A listed tt-model bundle id (namespace/name).",
        autocompletion=complete_bundle_id,
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Delist a bundle from the community catalog; the repo is untouched."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    _catalog_listing(appctx, name, listed=False, yes=True)


def _passthrough(ctx: typer.Context, subcommand: str) -> None:
    """Forward an authoring command to tt-model untouched, including --help.

    The command's own help option is disabled (help_option_names=[]) so that
    `tt model package --help` reaches tt-model, whose help is the authoritative flag
    list, rather than showing a tt page that would only say "see tt-model"."""
    appctx = get_app_context(ctx)
    if appctx.offline:
        raise TTError(
            f"`tt model {subcommand}` needs the Hugging Face Hub.",
            why="Authoring resolves wheels, images and repos on the Hub.",
            next_step="Drop --offline.",
            exit_code=ExitCode.OFFLINE,
        )
    backend = ModelManagerBackend(
        appctx.registry, appctx.runner, appctx.config, appctx.output
    )
    code = backend.passthrough(subcommand, list(ctx.args))
    if code:
        raise typer.Exit(code)


_PASSTHROUGH_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


@model_app.command("package", context_settings=_PASSTHROUGH_SETTINGS,
                   rich_help_panel=PANEL_AUTHORING)
@handle_tt_errors
def model_package(ctx: typer.Context) -> None:
    """Build a bundle from a bring-up (`tt-model package`; all flags are its own)."""
    _passthrough(ctx, "package")


@model_app.command("package-thin", context_settings=_PASSTHROUGH_SETTINGS,
                   rich_help_panel=PANEL_AUTHORING)
@handle_tt_errors
def model_package_thin(ctx: typer.Context) -> None:
    """Beta: build a v6 thin bundle (`tt-model package-thin`; all flags are its own)."""
    _passthrough(ctx, "package-thin")


@model_app.command("push", context_settings=_PASSTHROUGH_SETTINGS,
                   rich_help_panel=PANEL_AUTHORING)
@handle_tt_errors
def model_push(ctx: typer.Context) -> None:
    """Push a staged container package to the Hub (`tt-model push`; all flags are its own)."""
    _passthrough(ctx, "push")


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
