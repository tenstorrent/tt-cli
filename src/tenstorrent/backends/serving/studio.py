# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""TT-Studio serving backend — the third `tt serve` path.

Studio carries a few models neither tt-inference-server's released spec nor a
tt-model bundle covers (modelhub/studio.py lists them). Its run.py owns the whole
stack — docker compose, the deploy, progress and health — so tt only finds the
pinned checkout and streams `run.py run <model>` from it; stopping is
`run.py --stop-model <model>` the same way.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ...config.store import ConfigStore
from ...errors import ExitCode, TTError
from ...models.model import ModelInfo
from ...modelhub.hub import hf_token
from ...output import OutputManager
from ...tools.registry import ToolRegistry
from ...tools.runner import Runner

TOOL = "tt-studio"


class StudioBackend:
    def __init__(
        self,
        registry: ToolRegistry,
        runner: Runner,
        config: ConfigStore,
        output: OutputManager,
    ) -> None:
        self.registry = registry
        self.runner = runner
        self.config = config
        self.output = output

    def preflight(self, model: ModelInfo) -> None:
        if not shutil.which("docker"):
            raise TTError(
                "docker not found.",
                why="TT-Studio runs its services and model containers with docker "
                "compose (podman is not supported).",
                next_step="Install it — https://docs.docker.com/engine/install/ — then re-run.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": "docker"},
            )

    def _env(self) -> dict[str, str]:
        """Runner.stream() replaces the child environment, so os.environ is
        inherited explicitly. Studio adopts HF_TOKEN from the environment on its
        own, so a token from the HF login store is exported for it here."""
        env = dict(os.environ)
        found = hf_token(self.config)
        if found:
            env.setdefault("HF_TOKEN", found[0])
        return env

    def _python_for(self, entry: Path) -> str:
        venv_python = entry.parent.parent / "venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    def _argv(self, model: ModelInfo, *, entry: Path | None) -> list[str]:
        argv = [self._python_for(entry), str(entry)] if entry else ["<run.py>"]
        return argv + ["run", model.name]

    def _refresh_checkout(self, root: Path, *, offline: bool) -> None:
        """Best-effort `git pull` of the branch-pinned checkout.

        The pin is a branch (see supplement.toml), and registry.ensure() only
        reinstalls when the pin string changes, so without this the checkout would
        stay at whatever the branch pointed to the day it was cloned. Failure is a
        warning, not an error: an unreachable GitHub must not stop a deploy that
        needs nothing new."""
        if offline or not (root / ".git").exists():
            return
        result = self.runner.capture(
            ["git", "-C", str(root), "pull", "--ff-only", "--quiet"],
            check=False,
            tool="git (updating tt-studio)",
        )
        if result.returncode != 0:
            self.output.warn(
                "could not update the tt-studio checkout; serving with the version "
                "already on disk."
            )

    def serve(
        self,
        model: ModelInfo,
        *,
        offline: bool = False,
        device: str | None = None,
        port: int | None = None,
    ) -> int:
        if device is not None or port is not None:
            self.output.warn(
                "--device/--port are tt-inference-server options; studio allocates "
                "chips and ports itself (its own `--device-id` picks chips)."
            )
        entry = Path(self.registry.ensure(TOOL, offline=offline))
        found = self.registry._resolve_or_none(TOOL)
        if found is not None and found[1] == "installed":
            self._refresh_checkout(entry.parent, offline=offline)
        self.output.status(
            f"Starting TT-Studio and deploying {model.name} — Ctrl-C stops watching, "
            f"`tt model stop {model.name}` stops the model."
        )
        # run.py resolves the repo root, .env and its compose file from cwd.
        return self.runner.stream(
            self._argv(model, entry=entry),
            env=self._env(),
            cwd=str(entry.parent),
            tool=TOOL,
        )

    def plan(self, model: ModelInfo, *, offline: bool = False) -> dict:
        """What `serve` would run, without installing or touching docker."""
        installed = self.registry._resolve_or_none(TOOL)
        entry = Path(installed[0]) if installed else None
        token = hf_token(self.config)
        return {
            "backend": "studio",
            "model": model.name,
            "offline": offline,
            "installed": entry is not None,
            "cwd": str(entry.parent) if entry else None,
            "hf_token_source": token[1] if token else None,
            "argv": self._argv(model, entry=entry),
        }

    def unsupported_workflow(self, workflow: str) -> TTError:
        return TTError(
            f"TT-Studio cannot run the {workflow!r} workflow.",
            why="It deploys an inference server only; benchmarks and evals are "
            "tt-inference-server workflows.",
            next_step="Serve a model `tt model list` shows for inference-server.",
            exit_code=ExitCode.UNSUPPORTED,
        )

    def _installed_entry(self) -> Path:
        """Resolve the checkout without installing it: teardown must not clone a
        repo to discover there is nothing to tear down."""
        found = self.registry._resolve_or_none(TOOL)
        if found is None:
            raise TTError(
                "TT-Studio is not installed, so there is nothing for tt to stop.",
                why="tt installs it on the first `tt serve` through studio, not for teardown.",
                next_step="`tt serve <model> --backend studio` installs it and deploys.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": TOOL},
            )
        return Path(found[0])

    def stop(self, model: ModelInfo) -> int:
        entry = self._installed_entry()
        argv = [self._python_for(entry), str(entry), "--stop-model", model.name]
        self.output.status(f"Stopping {model.name} via TT-Studio …")
        return self.runner.stream(
            argv, env=self._env(), cwd=str(entry.parent), tool=TOOL
        )
