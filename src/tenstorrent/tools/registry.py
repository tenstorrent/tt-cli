# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""ToolRegistry: where is tool X, and how do I get it at the golden version?

resolve() order (first hit wins):
  1. env TT_TOOL_BIN_<NAME>       (primary test seam, also power users)
  2. config [tools.override]      (persistent per-tool binary override)
  3. installed state              (what we installed via `tt update`)
Missing everywhere → TTError(TOOL_MISSING) pointing at `tt update`.
"""

from __future__ import annotations

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
)
from .manifest import LocalManifestSource, Manifest, ManifestSource, ToolSpec
from .runner import Runner
from .state import ToolState


def env_var_for(tool_name: str) -> str:
    return "TT_TOOL_BIN_" + tool_name.upper().replace("-", "_")


@dataclass(frozen=True)
class ToolStatus:
    name: str
    kind: str
    golden_version: str
    installed_version: str | None
    path: str | None
    source: str  # "env" | "override" | "installed" | "missing"


class ToolRegistry:
    def __init__(
        self,
        paths: Paths,
        config: ConfigStore,
        *,
        manifest_source: ManifestSource | None = None,
        runner: Runner | None = None,
        installers: dict[str, Installer] | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.state = ToolState(paths)
        self._manifest_source = manifest_source or LocalManifestSource(paths)
        self._manifest: Manifest | None = None
        self._runner = runner or Runner(sudo_command=str(config.get("tools.sudo_command")))
        self._installers = installers or {
            "uv-tool": UvToolInstaller(paths, self._runner),
            "script": ScriptInstaller(paths),
            "git-venv": GitVenvInstaller(paths, self._runner),
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
        return None

    def resolve(self, name: str) -> Path:
        self.spec(name)  # unknown names fail with CONFIG before TOOL_MISSING
        found = self._resolve_or_none(name)
        if found is None:
            raise TTError(
                f"Required tool {name!r} is not installed.",
                next_step="Run `tt update` to install the latest Tenstorrent system software.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": name},
            )
        return found[0]

    def ensure(self, name: str, *, offline: bool = False) -> Path:
        """Resolve, installing at the golden pin if missing or stale.

        env/override resolutions are the user's business and never touched, but a
        state-installed tool whose recorded version no longer matches the golden
        pin is reinstalled — otherwise a manifest pin bump would never take effect
        for tools installed on demand (script, git-venv). An unknown golden pin
        (golden.json not fetched yet) never triggers a reinstall."""
        spec = self.spec(name)
        found = self._resolve_or_none(name)
        if found is not None:
            path, source = found
            if source != "installed":
                return path
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
            )
        installer = self._installers.get(spec.kind)
        if installer is None:
            raise TTError(
                f"Tool {spec.name} has unknown kind {spec.kind!r} in the manifest.",
                exit_code=ExitCode.CONFIG,
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
                )
            )
        return rows
