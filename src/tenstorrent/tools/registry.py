# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""ToolRegistry: where is tool X, and how do I get it at the golden version?

resolve() order (first hit wins):
  1. env TT_TOOL_BIN_<NAME>       (primary test seam, also power users)
  2. config [tools.override]      (persistent per-tool binary override)
  3. installed state              (what we installed via `tt update`)
  4. ~/.tenstorrent-venv/bin/<tool>  (tt-installer's managed venv; smi/flash only)
Missing everywhere → TTError(TOOL_MISSING) pointing at `tt update`.

Installs target the golden version unless config [tools.pin] names another one
(`tt self pin`); target_spec() is that effective spec.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from ..config.store import ConfigStore
from ..config.paths import Paths
from ..errors import ExitCode, TTError
from .installers import (
    DockerInstaller,
    GitVenvInstaller,
    InstallResult,
    Installer,
    ScriptInstaller,
    UvToolInstaller,
    remote_version_exists,
)
from .manifest import (
    INSTALLER_MANAGED_TOOLS,
    LocalManifestSource,
    Manifest,
    ManifestSource,
    ToolSpec,
)
from .runner import Runner
from .state import ToolState


# Kinds whose installers take any upstream version. A script's URL is fixed to the
# golden release (tt-installer is chosen via `tt update <version>` instead).
PINNABLE_KINDS = frozenset({"uv-tool", "git-venv"})


def env_var_for(tool_name: str) -> str:
    return "TT_TOOL_BIN_" + tool_name.upper().replace("-", "_")


def installer_venv_bin(bin_name: str) -> Path:
    """Where tt-installer's managed venv puts a tool's entry point. Computed per call so
    a redirected HOME (tests, containers) is honoured."""
    return Path.home() / ".tenstorrent-venv" / "bin" / bin_name


@dataclass(frozen=True)
class ToolStatus:
    name: str
    kind: str
    golden_version: str
    installed_version: str | None
    path: str | None
    source: str  # "env" | "override" | "installed" | "installer" | "missing"
    pinned_version: str | None = None


class ToolRegistry:
    def __init__(
        self,
        paths: Paths,
        config: ConfigStore,
        *,
        manifest_source: ManifestSource | None = None,
        runner: Runner | None = None,
        installers: dict[str, Installer] | None = None,
        ui: Any | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.state = ToolState(paths)
        self._manifest_source = manifest_source or LocalManifestSource(paths)
        self._manifest: Manifest | None = None
        self._runner = runner or Runner(sudo_command=str(config.get("tools.sudo_command")))
        # Installing a tool is minutes of work, so the installers render progress;
        # without a ui they fall back to a silent one and print nothing.
        self._installers = installers or {
            "uv-tool": UvToolInstaller(paths, self._runner, ui=ui),
            "script": ScriptInstaller(paths),
            "git-venv": GitVenvInstaller(paths, self._runner, ui=ui),
            "docker": DockerInstaller(),
        }

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = self._manifest_source.load()
        return self._manifest

    def reload_manifest(self) -> None:
        """Drop the cached manifest — `tt update` calls this after refreshing the
        golden.json cache so the new pins are visible in the same process."""
        self._manifest = None

    def spec(self, name: str) -> ToolSpec:
        return self.manifest.spec(name)

    # -- version pins ---------------------------------------------------------------
    def pinned_version(self, name: str) -> str | None:
        """The user's pinned version of a pinnable tool, or None to follow golden."""
        if self.spec(name).kind not in PINNABLE_KINDS:
            return None
        value = self.config.get(f"tools.pin.{name}")
        if value is None:
            return None
        return str(value).strip() or None

    def target_spec(self, name: str) -> ToolSpec:
        """The spec to install: the golden version, or the pinned one."""
        spec = self.spec(name)
        pinned = self.pinned_version(name)
        return dataclasses.replace(spec, golden_version=pinned) if pinned else spec

    def pin(self, name: str, version: str, *, offline: bool = False) -> bool:
        """Pin a tool to `version`. Returns whether the version was confirmed to
        exist upstream (False when offline or the lookup was inconclusive)."""
        spec = self.spec(name)
        if spec.kind not in PINNABLE_KINDS:
            hint = "Run `tt update <version>` to use a specific tt-installer."
            raise TTError(
                f"{name} cannot be pinned.",
                why=f"Its {spec.kind!r} release is fixed to the golden version.",
                next_step=hint if spec.kind == "script" else None,
                exit_code=ExitCode.USAGE,
                reason="tool.not_pinnable",
            )
        version = version.strip()
        if not version:
            raise TTError(
                "The version to pin is empty.",
                exit_code=ExitCode.USAGE,
                reason="tool.pin_empty",
            )
        verified = False
        if not offline:
            verified = self._require_published(dataclasses.replace(spec, golden_version=version))
        self.config.set(f"tools.pin.{name}", version)
        return verified

    def unpin(self, name: str) -> None:
        self.spec(name)  # unknown names fail with CONFIG
        self.config.unset(f"tools.pin.{name}")

    def _require_published(self, spec: ToolSpec) -> bool:
        """Raise if `spec.golden_version` is known not to exist upstream; return
        whether it is known to exist."""
        exists = remote_version_exists(spec, self._runner)
        if exists is False:
            if spec.repo:
                where = spec.repo
                listing = f"`git ls-remote --tags {spec.repo}` lists them"
            else:
                where = f"PyPI ({spec.package})"
                listing = f"see https://pypi.org/project/{spec.package}/#history"
            raise TTError(
                f"{spec.name} version {spec.golden_version!r} does not exist.",
                why=f"No release or git ref by that name was found at {where}.",
                next_step=f"Pick a published version ({listing}), then "
                f"`tt self pin {spec.name} <version>`; or `tt self unpin {spec.name}` "
                "to follow the golden version.",
                exit_code=ExitCode.USAGE,
                reason="tool.version_not_found",
                details={"tool": spec.name, "version": spec.golden_version},
            )
        return bool(exists)

    # -- resolution -----------------------------------------------------------------
    def _resolve_or_none(self, name: str) -> tuple[Path, str] | None:
        env_path = os.environ.get(env_var_for(name))
        if env_path:
            return Path(env_path), "env"
        override = self.config.get(f"tools.override.{name}")
        if override:
            return Path(str(override)), "override"
        installed = self.state.get(name)
        if installed and installed.path.exists():
            return installed.path, "installed"
        # Last resort, and only for the tools tt-installer owns: a machine set up by the
        # installer has a working tt-smi before `tt update` has run, and "not installed"
        # was simply wrong there. Anything else keeps failing closed — a stray binary in
        # that venv must never stand in for a pinned tool.
        if name in INSTALLER_MANAGED_TOOLS:
            candidate = installer_venv_bin(self.spec(name).bin_name)
            if candidate.is_file():
                return candidate, "installer"
        return None

    def resolve(self, name: str) -> Path:
        self.spec(name)  # unknown names fail with CONFIG before TOOL_MISSING
        found = self._resolve_or_none(name)
        if found is None:
            raise TTError(
                f"Required tool {name!r} is not installed.",
                next_step="Run `tt update` to install the latest Tenstorrent system software.",
                exit_code=ExitCode.TOOL_MISSING,
                # Manifest names are kebab-case; the slug grammar is snake_case.
                reason=f"tool.missing.{name.replace('-', '_')}",
                details={"tool": name},
            )
        return found[0]

    def ensure(self, name: str, *, offline: bool = False) -> Path:
        """Resolve, installing at the target version (golden, or the user's pin)
        if missing or stale.

        env/override resolutions are the user's business and never touched, but a
        state-installed tool whose recorded version no longer matches the target
        is reinstalled — otherwise a manifest pin bump would never take effect
        for tools installed on demand (script, git-venv). An unknown golden pin
        (golden.json not fetched yet) never triggers a reinstall. The installer-venv
        fallback is a resolve-time convenience only: `tt update` still installs the
        pinned copy, which then wins at step 3."""
        spec = self.target_spec(name)
        found = self._resolve_or_none(name)
        if found is not None:
            path, source = found
            if source in ("env", "override"):
                return path
            if source == "installer":
                return self.install(spec, offline=offline).path
            installed = self.state.get(name)
            if (
                installed is None
                or not spec.golden_version
                or installed.version == spec.golden_version
            ):
                return path
        result = self.install(spec, offline=offline)
        return result.path

    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult:
        if not spec.golden_version:
            raise TTError(
                f"No golden version is known for {spec.name}.",
                why="The pinned golden.json has not been fetched yet.",
                next_step="Run `tt update` once with network access.",
                exit_code=ExitCode.CONFIG,
                reason="tool.golden_unknown",
            )
        if not offline and spec.golden_version == self.pinned_version(spec.name):
            # A pin can come from a hand-edited config, or name a since-deleted release.
            self._require_published(spec)
        installer = self._installers.get(spec.kind)
        if installer is None:
            raise TTError(
                f"Tool {spec.name} has unknown kind {spec.kind!r} in the manifest.",
                exit_code=ExitCode.CONFIG,
                reason="tool.unknown_kind",
            )
        result = installer.install(spec, offline=offline)
        self.state.record(spec.name, version=result.version, path=result.path)
        return result

    # -- reporting ------------------------------------------------------------------
    def status(self) -> list[ToolStatus]:
        rows = []
        installed_state = {n: t for n, t in self.state.all().items() if t}
        for name, spec in sorted(self.manifest.tools.items()):
            found = self._resolve_or_none(name)
            installed = installed_state.get(name)
            rows.append(
                ToolStatus(
                    name=name,
                    kind=spec.kind,
                    golden_version=spec.golden_version or "unknown (run `tt update`)",
                    installed_version=installed.version if installed else None,
                    path=str(found[0]) if found else None,
                    source=found[1] if found else "missing",
                    pinned_version=self.pinned_version(name),
                )
            )
        return rows
