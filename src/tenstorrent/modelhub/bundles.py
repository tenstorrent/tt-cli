# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Community tt-model bundles, read straight from the Hugging Face Hub.

tt-model-manager publishes bundles as HF model repos and opts them into a
community catalog with a repo tag; `tt-model search --catalog` lists that set.
tt queries the Hub itself rather than shelling out, for two reasons: the tool's
own JSON carries only id/downloads/visibility (the repo *tags* hold the arch and
packaging, which is what a listing wants), and `tt model list` must not have to
install tt-model just to show what exists.

Bundles deliberately do NOT go through ModelCatalog: they share no schema with
the released compat spec (no per-device status, no max_context), and merging them
into the spec listing would blur which tool can serve what.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config.paths import get_paths
from ..config.store import ConfigStore
from ..errors import ExitCode, TTError
from . import hub

# Repo tags written by `tt-model push` (see tt_kernel/__init__.py and the push
# command). Tag vocabulary is upstream's, so treat unknown tags as informational
# rather than an error.
CATALOG_TAG = "tt-model-catalog"  # opted into the community catalog
BUNDLE_TAG = "tt-model-cache"  # any published bundle
_KIND_TAGS = {
    "tt-model-container": "container",
    "self-contained": "self-contained",
    "thin": "thin",
}
_ENGINE_TAGS = ("vllm-plugin", "vllm", "tt-dit-server")
_ENGINE_SUFFIXES = ("-server", "-plugin")


def _is_engine_tag(tag: str) -> bool:
    return tag in _ENGINE_TAGS or tag.endswith(_ENGINE_SUFFIXES)
_SKIP_TAG_PREFIXES = ("region:", "license:", "arxiv:", "dataset:", "base_model:")

# Architecture families only. Recognised rather than inferred by elimination: a
#
# An unrecognised tag is dropped, so a new family belongs here — one line, and
# until then its bundles show an empty arch rather than a wrong one.
_ARCH_TAGS = frozenset({"blackhole", "wormhole_b0", "grayskull"})


@dataclass(frozen=True)
class BundleInfo:
    """One published tt-model bundle. Field order IS the --json contract."""

    name: str  # HF repo id, namespace/name — exactly what `tt serve` takes
    source: str = "HF"
    kind: str | None = None  # container | self-contained | thin
    engine: str | None = None  # vLLM today
    arch: list[str] = field(default_factory=list)  # blackhole, wormhole_b0, 1x4, …
    downloads: int | None = None
    installed: bool = False  # a local install recorded by tt-model
    # Weights live in the shared HF cache, but which repo they come from is only
    # known from the bundle's manifest — which is on disk only once installed. Both
    # stay None for a bundle we have not pulled (no manifest, hence unknowable
    # without a per-repo Hub fetch this listing deliberately avoids).
    weights_repo: str | None = None
    weights_bytes: int | None = None


def _cache_root() -> Path:
    """tt-model's state dir: $XDG_CACHE_HOME (or ~/.cache) / tt-model, preferring an
    existing legacy tt-kernel dir the way tt-model's own compat.data_dir does."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    current, legacy = base / "tt-model", base / "tt-kernel"
    return legacy if not current.exists() and legacy.exists() else current


def installed_bundles() -> dict[str, dict]:
    """tt-model's local install index, keyed by lowercased repo id."""
    path = _cache_root() / "installed.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key).lower(): value if isinstance(value, dict) else {}
        for key, value in data.items()
    }


def installed_bundle_ids() -> set[str]:
    """Repo ids tt-model records as installed locally; empty if it never ran."""
    return set(installed_bundles())


def _manifest_for(repo_id: str, entry: dict) -> dict | None:
    """A pulled bundle's manifest. The index records its path; fall back to the
    conventional location for an older entry that does not."""
    candidates = []
    recorded = entry.get("manifest")
    if recorded:
        candidates.append(Path(str(recorded)))
    candidates.append(
        _cache_root() / "pulled" / repo_id.replace("/", "__") / "tt_kernel_manifest.json"
    )
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def engine_for(repo_id: str, entry: dict) -> str | None:
    """The server a pulled bundle runs, from its manifest's container.kind.

    Bundles are NOT all vLLM — an LLM package is "vllm-plugin", a diffusion one
    "tt-dit-server" — so this is read rather than assumed."""
    manifest = _manifest_for(repo_id, entry) or {}
    container = manifest.get("container")
    if isinstance(container, dict) and container.get("kind"):
        return str(container["kind"])
    return None


def weights_repo_for(repo_id: str, entry: dict) -> str | None:
    """The HF weights repo a pulled bundle points at, from its own manifest.

    A bundle ships no weights, only a reference, so this is the join between a
    bundle and the shared HF cache. The index records the manifest path; fall back
    to the conventional location for an older entry that does not."""
    manifest = _manifest_for(repo_id, entry) or {}
    weights = manifest.get("weights")
    if isinstance(weights, dict):
        # tt-model serializes the field name (repo_id); "repo" is its JSON alias.
        found = weights.get("repo_id") or weights.get("repo")
        if found:
            return str(found)
    return None


def serve_details(repo_id: str) -> dict | None:
    """Launch settings a *pulled* bundle records in its own manifest, or None.

    Local only, by design: a bundle that has never been pulled has no manifest on
    disk, and fetching one from the Hub to describe a serve would turn a preview
    into a download. The keys mirror what tt resolves for the tt-inference-server
    path — image, parsers, tt config, port — so `tt serve --dry-run` can say the
    same things about either backend.

    tt-model owns these values and applies them itself; tt neither passes nor
    overrides them, so this is reporting, not configuration.
    """
    entry = installed_bundles().get(repo_id.lower())
    if entry is None:
        return None
    manifest = _manifest_for(repo_id, entry)
    if manifest is None:
        return {}  # installed, but nothing on disk to read
    container = manifest.get("container") or {}
    serve = container.get("serve") or {}
    capabilities = serve.get("capabilities") or {}
    image = container.get("image") or {}
    profiles = [
        str(p.get("name"))
        for p in (container.get("serve_profiles") or [])
        if isinstance(p, dict) and p.get("name")
    ]
    return {
        "engine": container.get("kind"),
        "image": image.get("tag") or image.get("repository"),
        "arch": manifest.get("arch"),
        "device_count": manifest.get("device_count"),
        "hardware": serve.get("hardware") or serve.get("mesh_device"),
        "tool_call_parser": capabilities.get("tool_parser"),
        "reasoning_parser": capabilities.get("reasoning_parser"),
        "tt_config": (serve.get("additional_config") or {}).get("tt"),
        "port": serve.get("port"),
        "max_model_len": serve.get("max_model_len"),
        "tt_metal_version": manifest.get("tt_metal_version"),
        "weights_repo": weights_repo_for(repo_id, entry),
        "profiles": profiles,
    }


def _classify(tags: list[str]) -> tuple[str | None, str | None, list[str]]:
    kind = engine = None
    arch: list[str] = []
    for tag in tags:
        if tag in _KIND_TAGS:
            kind = kind or _KIND_TAGS[tag]
        elif _is_engine_tag(tag):
            engine = engine or tag  # report upstream's own name for it
        elif tag in (CATALOG_TAG, BUNDLE_TAG) or tag.startswith(_SKIP_TAG_PREFIXES):
            continue
        elif tag in _ARCH_TAGS:
            arch.append(tag)
    return kind, engine, sorted(arch)


MANIFEST_NAME = "tt_kernel_manifest.json"  # tt-model's on-disk contract, unrenamed


def is_bundle_repo(repo_id: str) -> bool | None:
    """Whether a Hub repo is a tt-model bundle rather than a plain weights repo.

    A bundle id and an ordinary HF model id have the same `namespace/name` shape, so
    the only honest discriminator is whether the repo carries a manifest — one small
    existence check. None means "could not tell" (offline, private, rate-limited);
    callers must not treat that as False."""
    if installed_bundles().get(repo_id.lower()) is not None:
        return True  # already pulled here, no need to ask the Hub
    try:
        from huggingface_hub import file_exists

        return bool(file_exists(repo_id, MANIFEST_NAME))
    except Exception:  # noqa: BLE001 — 404/auth/network all mean "unknown", not "no"
        return None


def _weights_state(
    repo_id: str, entry: dict, sizes: dict[str, int] | None
) -> tuple[str | None, int | None]:
    """(weights repo, bytes cached). `sizes` is one cache scan shared by the whole
    listing — see hub.cached_sizes()."""
    weights_repo = weights_repo_for(repo_id, entry)
    if not weights_repo or sizes is None:
        return weights_repo, None
    return weights_repo, sizes.get(weights_repo.lower())


def local_bundles(config: ConfigStore | None = None) -> list[BundleInfo]:
    """Bundles installed on this machine, from tt-model's own index.

    These need no network and are the only way to see a bundle whose repo is
    private, or public but never opted into the community catalog — someone shares
    an id, you pull it, and nothing on the Hub advertises it.

    Unordered: the caller merges this with the catalog listing and sorts once."""
    sizes = hub.cached_sizes(config) if config is not None else None
    found = []
    for key, entry in installed_bundles().items():
        repo_id = str(entry.get("repo_id") or key)  # the index preserves the real case
        weights_repo, weights_bytes = _weights_state(repo_id, entry, sizes)
        arch = str(entry.get("arch") or "")
        found.append(
            BundleInfo(
                name=repo_id,
                source="local",
                kind="container" if entry.get("container") else None,
                engine=engine_for(repo_id, entry),
                arch=[arch] if arch else [],
                installed=True,
                weights_repo=weights_repo,
                weights_bytes=weights_bytes,
            )
        )
    return found


def search_community(
    *,
    limit: int = 100,
    query: str | None = None,
    config: ConfigStore | None = None,
) -> list[BundleInfo]:
    """Bundles opted into the community catalog, in the Hub's own order (newest
    first); the caller sorts the merged listing.

    Network-only by nature: the catalog is a Hub index, so there is nothing local
    to fall back on. `config` enables the weights-cache lookup for installed
    bundles (it resolves the HF cache root)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        found = list(
            HfApi().list_models(filter=CATALOG_TAG, search=query or None, limit=limit)
        )
    except (HfHubHTTPError, OSError) as exc:
        raise TTError(
            "Could not reach the Hugging Face Hub.",
            why=str(exc),
            next_step="Check your connection, or drop --community to list the "
            "released model catalog (which is bundled).",
            exit_code=ExitCode.ERROR,
        ) from exc
    installed = installed_bundles()
    sizes = hub.cached_sizes(config) if config is not None else None
    bundles = []
    for repo in found:
        repo_id = str(getattr(repo, "id", "") or "")
        if not repo_id:
            continue
        kind, engine, arch = _classify(list(getattr(repo, "tags", None) or []))
        entry = installed.get(repo_id.lower())
        weights_repo = weights_bytes = None
        if entry is not None:
            weights_repo, weights_bytes = _weights_state(repo_id, entry, sizes)
            # The manifest is authoritative; the tag is only a hint.
            engine = engine_for(repo_id, entry) or engine
        bundles.append(
            BundleInfo(
                name=repo_id,
                kind=kind,
                engine=engine,
                arch=arch,
                downloads=getattr(repo, "downloads", None),
                installed=entry is not None,
                weights_repo=weights_repo,
                weights_bytes=weights_bytes,
            )
        )
    return bundles


def describe(
    repo_id: str, *, config: ConfigStore | None = None, offline: bool = False
) -> BundleInfo | None:
    """One bundle's catalog row by id (case-insensitive, the Hub's own rule), or
    None when neither the local install index nor the community catalog knows it.

    The community row wins when both exist: it is the richer record (kind, engine,
    downloads) and already folds in the install state and cached weights. The local
    row stands alone for a bundle nobody published, or under `offline`, where the
    Hub is not asked at all — the caller decides whether None then means "not a
    bundle" or merely "not installed here"."""
    wanted = repo_id.lower()
    local = next(
        (b for b in local_bundles(config=config) if b.name.lower() == wanted), None
    )
    if offline:
        return local
    # The Hub's `search` matches on the id, so narrow by the name half; the exact
    # match is decided here, not by the Hub's substring rule.
    listed = next(
        (
            b
            for b in search_community(query=repo_id.rsplit("/", 1)[-1], config=config)
            if b.name.lower() == wanted
        ),
        None,
    )
    return listed or local


def _community_cache_file() -> Path:
    return get_paths().cache_dir / "community-bundles.json"


def save_community_cache(names: list[str]) -> None:
    """Record the community catalog's repo ids after a successful Hub fetch.

    Shell completion reads this so `tt serve <TAB>` can offer bundle ids without a
    Hub request at tab-time (see modelhub/completions.py). Best-effort: a listing
    must never fail because the cache dir is read-only."""
    try:
        path = _community_cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"names": sorted(names)}, indent=2))
    except OSError:
        pass


def cached_community_names() -> list[str]:
    """Repo ids from the last `tt model list --community`; [] before the first run
    (or on a corrupt cache) — fewer completions, never an error."""
    try:
        data = json.loads(_community_cache_file().read_text())
    except (OSError, ValueError):
        return []
    names = data.get("names") if isinstance(data, dict) else None
    return [str(n) for n in names] if isinstance(names, list) else []
