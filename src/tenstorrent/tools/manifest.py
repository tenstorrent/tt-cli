# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Golden-version manifest, backed by tt-sw-manifest's released golden.json.

tt-sw-manifest (github.com/tenstorrent/tt-sw-manifest) CI-validates a pinned
software stack per release. Its `golden.json` asset is a flat, distro-agnostic
component→version map — that is our source of truth for golden versions of the
installer-managed stack (smi, flash, kmd, sfpi, firmware, ...).

Nothing version-shaped is bundled: `supplement.toml` records *which release tag
we pin* (the `[golden]` table, with the asset's sha256) and `tt update` fetches
golden.json at that tag, verifies it, and caches it under TT_DATA_DIR keyed by
the tag. Every other command reads the cache and never touches the network.
Before the first successful `tt update`, golden versions are simply unknown
(empty golden_version), which only affects display — resolving installed tools
never needed them, and the registry refuses to *install* at an unknown pin.
A cache written for a different tag than the supplement records is ignored, so
upgrading tt invalidates stale pins by construction.

`TT_GOLDEN_PATH` overrides the fetch/cache layer with a local golden.json
(tests, air-gapped installs); `TT_MANIFEST_PATH` overrides the supplement.
`supplement.toml` also adds the CLI-only tools golden.json doesn't cover
(tt-installer itself, tt-inference-server).

`ManifestSource` stays a protocol so a remote source can slot in later without
touching consumers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import tomlkit

from ..config.paths import Paths
from ..errors import ExitCode, TTError

SUPPLEMENT_SCHEMA_VERSION = 1
GOLDEN_PATH_ENV = "TT_GOLDEN_PATH"

# Metadata for the tools that arrive via golden.json — the file carries only
# short-key→version, so our tool name, kind/health/sudo live here.
_TT_PYTHON_TOOL_INFO: dict[str, dict[str, Any]] = {
    "tt-smi": {"golden_key": "smi", "health": ["--version"], "needs_sudo": True},  # sudo for -r reset only
    "tt-flash": {"golden_key": "flash", "health": ["--version"]},
}

# golden.json keys that are NOT displayed as part of the system stack:
# smi/flash are the tt-managed uv tools above, firmware is surfaced separately,
# installer is pinned in the supplement instead (deliberately allowed to be newer
# than the release's CI-validated one), test-sha is upstream CI metadata.
_GOLDEN_NON_SYSTEM_KEYS = {"smi", "flash", "firmware", "installer", "test-sha"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str  # "uv-tool" | "script" | "git-venv" | "docker"
    golden_version: str  # PyPI version, release tag, or git ref; "" = not yet known
    package: str | None = None  # uv-tool: PyPI package name
    url: str | None = None  # script: download URL
    url_template: str | None = None  # script: URL with a {version} slot for pinned-off fetches
    sha256: str | None = None  # script: expected digest (None = unpinned)
    repo: str | None = None  # git-venv clone URL; also a uv-tool git source (golden_version = ref)
    entry: str | None = None  # git-venv: entry script relative to repo root
    bin: str | None = None  # executable name (defaults to package/name)
    python: str | None = None  # interpreter constraint for isolated venvs
    deps: tuple[str, ...] = ()  # git-venv: PyPI requirements the entry script needs
    health: tuple[str, ...] = ()  # argv suffix for a health check
    needs_sudo: bool = False
    lazy: bool = False  # Installed on first use by registry.ensure() instead of by `tt update`.

    @property
    def bin_name(self) -> str:
        return self.bin or self.package or self.name


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    tools: dict[str, ToolSpec]
    system: dict[str, str] = field(default_factory=dict)  # golden.json system components
    firmware: str | None = None  # golden.json firmware pin
    origin: str = ""  # human-readable provenance for status/update output
    # The [golden] table from the supplement: the tt-sw-manifest release we pin.
    # golden_tag must match TTIS_GOLDEN_VERSIONS_TAG in the pinned install.sh —
    # `tt update` enforces that before running the golden installer.
    golden_tag: str | None = None
    golden_sha256: str | None = None
    golden_url_template: str | None = None  # a {tag} slot
    golden_data: dict[str, str] | None = None  # raw golden.json map (debugging/provenance)

    def spec(self, name: str) -> ToolSpec:
        try:
            return self.tools[name]
        except KeyError:
            raise TTError(
                f"Unknown tool {name!r}.",
                why="It is not listed in the golden-version manifest.",
                next_step="Run `tt self tools` to see managed tools.",
                exit_code=ExitCode.CONFIG,
            ) from None

    @property
    def golden_url(self) -> str | None:
        if self.golden_tag and self.golden_url_template:
            return self.golden_url_template.format(tag=self.golden_tag)
        return None


class ManifestSource(Protocol):
    def load(self) -> Manifest: ...


def _config_error(origin: str, problem: str, why: str | None = None) -> TTError:
    return TTError(
        f"{problem} ({origin}).",
        why=why,
        next_step="Run `tt update` to refresh goldens, or fix the override env var.",
        exit_code=ExitCode.CONFIG,
    )


def _as_golden_map(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        return value
    return None


def parse_golden(text: str, origin: str) -> dict[str, str]:
    """Parse a released golden.json: a flat JSON object of component→version."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _config_error(origin, "golden.json is not valid JSON", str(exc)) from exc
    golden = _as_golden_map(data)
    if golden is None:
        raise _config_error(
            origin, "golden.json is not a flat JSON object of component→version"
        )
    return golden


def _parse_supplement(text: str, origin: str) -> tuple[dict[str, ToolSpec], dict[str, Any]]:
    try:
        doc = tomlkit.parse(text).unwrap()
    except Exception as exc:
        raise _config_error(origin, "Tool supplement is not valid TOML", str(exc)) from exc
    if int(doc.get("schema_version", 0)) != SUPPLEMENT_SCHEMA_VERSION:
        raise _config_error(
            origin,
            f"Tool supplement has schema_version {doc.get('schema_version')!r}; "
            f"this CLI supports {SUPPLEMENT_SCHEMA_VERSION}",
        )
    tools: dict[str, ToolSpec] = {}
    for name, entry in doc.get("tools", {}).items():
        entry = dict(entry)
        entry["health"] = tuple(entry.get("health", ()))
        entry["deps"] = tuple(entry.get("deps", ()))
        try:
            tools[name] = ToolSpec(name=name, **entry)
        except TypeError as exc:
            raise _config_error(origin, f"Bad supplement entry for tool {name!r}", str(exc)) from exc
    return tools, dict(doc.get("golden", {}))


def build_manifest(
    golden: dict[str, str],
    supplement: dict[str, ToolSpec],
    origin: str,
    golden_meta: dict[str, Any] | None = None,
) -> Manifest:
    tools: dict[str, ToolSpec] = {}
    for name, info in _TT_PYTHON_TOOL_INFO.items():
        tools[name] = ToolSpec(
            name=name,
            kind="uv-tool",
            golden_version=golden.get(info["golden_key"], ""),
            package=name,
            health=tuple(info.get("health", ())),
            needs_sudo=bool(info.get("needs_sudo", False)),
        )
    tools.update(supplement)  # supplement wins on collision (explicit beats derived)
    golden_meta = golden_meta or {}
    return Manifest(
        schema_version=SUPPLEMENT_SCHEMA_VERSION,
        tools=tools,
        system={k: v for k, v in golden.items() if k not in _GOLDEN_NON_SYSTEM_KEYS},
        firmware=golden.get("firmware") or None,
        origin=origin,
        golden_tag=golden_meta.get("tag") or None,
        golden_sha256=golden_meta.get("sha256") or None,
        golden_url_template=golden_meta.get("url_template") or None,
        golden_data=golden or None,
    )


def golden_cache_read(paths: Paths, expected_tag: str | None) -> dict[str, str] | None:
    """The cached golden.json map, or None if absent, unreadable, or written for a
    different tag than the supplement pins (a stale cache from another tt version).
    Tolerant on purpose: a bad cache must degrade to 'unknown versions', never
    break unrelated commands — `tt update` rewrites it."""
    try:
        wrapper = json.loads(paths.golden_file.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(wrapper, dict) or wrapper.get("tag") != expected_tag:
        return None
    return _as_golden_map(wrapper.get("data"))


def golden_cache_write(paths: Paths, tag: str, golden: dict[str, str]) -> None:
    paths.golden_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.golden_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"tag": tag, "data": golden}, indent=2) + "\n")
    os.replace(tmp, paths.golden_file)


class LocalManifestSource:
    """Bundled supplement.toml (TT_MANIFEST_PATH overrides) + the golden.json
    layer: TT_GOLDEN_PATH override, else the tag-keyed cache written by
    `tt update`, else empty (golden versions unknown until the first update)."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    def _read_supplement(self) -> tuple[str, str]:
        override = os.environ.get("TT_MANIFEST_PATH")
        if override:
            path = Path(override)
            if not path.exists():
                raise TTError(
                    f"TT_MANIFEST_PATH points at {path}, which does not exist.",
                    next_step="Fix or unset TT_MANIFEST_PATH to use the bundled copy.",
                    exit_code=ExitCode.CONFIG,
                )
            return path.read_text(), str(path)
        return (
            (resources.files("tenstorrent.tools") / "supplement.toml").read_text(),
            "bundled supplement.toml",
        )

    def _load_golden(self, golden_meta: dict[str, Any]) -> tuple[dict[str, str], str]:
        override = os.environ.get(GOLDEN_PATH_ENV)
        if override:
            path = Path(override)
            if not path.exists():
                raise TTError(
                    f"{GOLDEN_PATH_ENV} points at {path}, which does not exist.",
                    next_step=f"Fix or unset {GOLDEN_PATH_ENV} to use the fetched cache.",
                    exit_code=ExitCode.CONFIG,
                )
            return parse_golden(path.read_text(), str(path)), str(path)
        tag = golden_meta.get("tag")
        cached = golden_cache_read(self.paths, tag)
        if cached is not None:
            return cached, f"golden.json {tag} (cached)"
        return {}, "golden versions not fetched yet (run `tt update`)"

    def load(self) -> Manifest:
        supp_text, supp_origin = self._read_supplement()
        supplement, golden_meta = _parse_supplement(supp_text, supp_origin)
        golden, golden_origin = self._load_golden(golden_meta)
        return build_manifest(
            golden,
            supplement,
            origin=f"{supp_origin} + {golden_origin}",
            golden_meta=golden_meta,
        )
