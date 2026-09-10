# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Installers, one per ToolSpec kind.

uv-tool installs go into per-tool venvs under our data dir (UV_TOOL_DIR), which
sidesteps cross-tool pin conflicts; tools are invoked by absolute path afterwards
so the user's PATH is never touched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..config.paths import Paths
from ..errors import ExitCode, TTError
from ..ui.console import null_ui
from ..ui.parsers import GitCloneProgress, UvPipProgress
from ..ui.stream import run_streamed
from .manifest import ToolSpec
from .runner import Runner


@dataclass(frozen=True)
class InstallResult:
    name: str
    version: str
    path: Path


class Installer(Protocol):
    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult: ...


def find_uv_bin() -> str:
    """The uv binary: TT_UV_BIN (test seam) → the bundled uv wheel → PATH."""
    override = os.environ.get("TT_UV_BIN")
    if override:
        return override
    try:
        from uv import find_uv_bin as _find

        return _find()
    except (ImportError, FileNotFoundError):
        pass
    found = shutil.which("uv")
    if found:
        return found
    raise TTError(
        "The `uv` binary is missing.",
        why="It normally ships with the tenstorrent package itself.",
        next_step="Reinstall the CLI: pip install --force-reinstall tenstorrent",
        exit_code=ExitCode.TOOL_MISSING,
    )


class UvToolInstaller:
    def __init__(
        self,
        paths: Paths,
        runner: Runner,
        uv_bin: str | None = None,
        ui: Any | None = None,
    ) -> None:
        self.paths = paths
        self.runner = runner
        self._uv_bin = uv_bin
        # Defaults to a silent Ui so direct construction (tests, library use)
        # prints nothing and no call site has to branch on None.
        self._ui = ui if ui is not None else null_ui()

    def _requirement(self, spec: ToolSpec) -> str:
        """What uv installs: a pinned PyPI version, or a PEP 508 direct reference
        when the tool declares a `repo` (not on PyPI yet — golden_version is then
        the git ref). Publishing such a tool later is a supplement edit: drop
        `repo` and set golden_version to the released version."""
        if spec.repo:
            return f"{spec.package} @ git+{spec.repo}@{spec.golden_version}"
        return f"{spec.package}=={spec.golden_version}"

    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult:
        uv = self._uv_bin or find_uv_bin()
        argv = [
            uv,
            "tool",
            "install",
            self._requirement(spec),
            "--force",  # idempotent re-pinning
        ]
        if spec.python:
            argv += ["--python", spec.python]
        if offline:
            argv += ["--offline"]
        env = dict(os.environ)
        env["UV_TOOL_DIR"] = str(self.paths.tools_dir)
        env["UV_TOOL_BIN_DIR"] = str(self.paths.tool_bin_dir)
        self.paths.tools_dir.mkdir(parents=True, exist_ok=True)
        self.paths.tool_bin_dir.mkdir(parents=True, exist_ok=True)
        # Was a silent capture(): a cold install of a big tool spent minutes with
        # nothing on screen. uv's own "Resolved N packages" gives an exact
        # denominator, so the row can show a real bar.
        progress = UvPipProgress(f"Installing {spec.name} {spec.golden_version}")
        run_streamed(
            self.runner,
            self._ui,
            argv,
            label=progress.label,
            parser=progress,
            env_extra={k: v for k, v in env.items() if k.startswith("UV_")},
            tool=f"uv (installing {spec.name})",
        )
        bin_path = self.paths.tool_bin_dir / spec.bin_name
        if not bin_path.exists():
            raise TTError(
                f"uv reported success but {bin_path} does not exist.",
                why="The package may not provide the expected entry point.",
                next_step="Re-run with --verbose and report this with `tt report issue`.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": spec.name},
            )
        return InstallResult(spec.name, spec.golden_version, bin_path)


def fetch_https(url: str) -> bytes:
    import httpx

    try:
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTError(
            f"Failed to download {url}.",
            why=str(exc),
            next_step="Check connectivity, or use --offline with a pre-seeded cache.",
            exit_code=ExitCode.OFFLINE,
        ) from exc
    return response.content


class ScriptInstaller:
    """Downloads a release script (install.sh), verifies sha256 when pinned, and
    stores it executable under our data dir. `fetch_fn` is the network seam."""

    def __init__(
        self,
        paths: Paths,
        fetch_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self.paths = paths
        self._fetch = fetch_fn or self._fetch_https

    @staticmethod
    def _fetch_https(url: str) -> bytes:
        return fetch_https(url)

    def script_path(self, spec: ToolSpec) -> Path:
        return self.paths.data_dir / "installers" / f"{spec.name}-{spec.golden_version}.sh"

    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult:
        target = self.script_path(spec)
        if target.exists():  # already fetched at this exact version
            return InstallResult(spec.name, spec.golden_version, target)
        if offline:
            raise TTError(
                f"{spec.name} {spec.golden_version} is not in the local cache.",
                why="--offline forbids downloading it.",
                next_step=f"Pre-seed it: place the script at {target}",
                exit_code=ExitCode.OFFLINE,
            )
        if not spec.url:
            raise TTError(
                f"Tool {spec.name} has no download URL in the manifest.",
                exit_code=ExitCode.CONFIG,
            )
        blob = self._fetch(spec.url)
        if spec.sha256:
            digest = hashlib.sha256(blob).hexdigest()
            if digest != spec.sha256:
                raise TTError(
                    f"Checksum mismatch for {spec.name} downloaded from {spec.url}.",
                    why=f"expected sha256 {spec.sha256}, got {digest}",
                    next_step="Re-run `tt update`; if it persists, report it — do not run the script.",
                    exit_code=ExitCode.TOOL_FAILED,
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        return InstallResult(spec.name, spec.golden_version, target)


class GitVenvInstaller:
    """Shallow-clones a repo at the golden ref and builds an isolated uv venv for
    it. Used for tools not yet on PyPI (tt-inference-server)."""

    def __init__(
        self,
        paths: Paths,
        runner: Runner,
        uv_bin: str | None = None,
        ui: Any | None = None,
    ) -> None:
        self.paths = paths
        self.runner = runner
        self._uv_bin = uv_bin
        self._ui = ui if ui is not None else null_ui()

    def _tool_dir(self, spec: ToolSpec) -> Path:
        return self.paths.tools_dir / f"{spec.name}-{spec.golden_version}"

    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult:
        tool_dir = self._tool_dir(spec)
        src_dir = tool_dir / "src"
        entry = src_dir / (spec.entry or "run.py")
        if entry.exists():  # already cloned at this exact ref
            return InstallResult(spec.name, spec.golden_version, entry)
        if offline:
            raise TTError(
                f"{spec.name} {spec.golden_version} is not in the local cache.",
                why="--offline forbids cloning it.",
                next_step=f"Pre-seed a checkout at {src_dir}",
                exit_code=ExitCode.OFFLINE,
            )
        if not spec.repo:
            raise TTError(
                f"Tool {spec.name} has no repo URL in the manifest.",
                exit_code=ExitCode.CONFIG,
            )
        tool_dir.mkdir(parents=True, exist_ok=True)
        # These four commands used to be silent captures, so a first `tt serve` on
        # a fresh box sat on a dead terminal for minutes. git and uv both report
        # exact counts, so each gets a live row with a real denominator.
        clone = GitCloneProgress(f"Cloning {spec.name} {spec.golden_version}")
        run_streamed(
            self.runner,
            self._ui,
            [
                "git",
                "clone",
                "--progress",  # git only reports progress when it isn't a tty
                "--depth",
                "1",
                "--branch",
                spec.golden_version,
                spec.repo,
                str(src_dir),
            ],
            label=clone.label,
            parser=clone,
            tool=f"git (cloning {spec.name})",
        )
        uv = self._uv_bin or find_uv_bin()
        venv_dir = tool_dir / "venv"
        argv = [uv, "venv", str(venv_dir)]
        if spec.python:
            argv += ["--python", spec.python]
        with self._ui.step(f"Creating a virtualenv for {spec.name}") as step:
            self.runner.capture(argv, tool=f"uv (venv for {spec.name})")
            if spec.python:
                step.detail(f"python {spec.python}")
        requirements = src_dir / "requirements.txt"
        if requirements.exists():
            deps = UvPipProgress(f"Installing {spec.name} dependencies")
            run_streamed(
                self.runner,
                self._ui,
                [uv, "pip", "install", "--python", str(venv_dir / "bin" / "python"),
                 "-r", str(requirements)],
                label=deps.label,
                parser=deps,
                tool=f"uv (deps for {spec.name})",
            )
        if spec.deps:
            # Bootstrap deps the entry script imports directly, declared in the
            # manifest — for repos with no root requirements.txt (tt-inference-server
            # manages its own heavy per-workflow venvs from inside run.py).
            extra = UvPipProgress(f"Installing {spec.name} bootstrap dependencies")
            run_streamed(
                self.runner,
                self._ui,
                [uv, "pip", "install", "--python", str(venv_dir / "bin" / "python"),
                 *spec.deps],
                label=extra.label,
                parser=extra,
                tool=f"uv (deps for {spec.name})",
            )
        if not entry.exists():
            raise TTError(
                f"Cloned {spec.name} but entry point {entry} does not exist.",
                exit_code=ExitCode.TOOL_FAILED,
            )
        return InstallResult(spec.name, spec.golden_version, entry)


class DockerInstaller:
    """Reserved slot: docker-delivered tools are out of prototype scope."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def install(self, spec: ToolSpec, *, offline: bool = False) -> InstallResult:
        raise TTError(
            f"Tool {spec.name} is docker-delivered, which this prototype does not manage yet.",
            next_step="Install/pull it manually; see the tool's own docs.",
            exit_code=ExitCode.UNSUPPORTED,
        )
