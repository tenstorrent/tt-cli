# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""TT-Studio serving backend — the third `tt serve` path.

Studio deploys the models in its own catalog (modelhub/studio.py) — most of
them tt-inference-server's, run from the same images, plus a few only studio
carries. Its run.py owns the whole stack — docker compose, the deploy, progress
and health — so tt only finds the pinned checkout and streams `run.py run
<model>` from it. Stopping a model is `run.py --stop-model <model>` (which
resets its chips) followed by `run.py --stop` for studio's own services. A plain
`--stop` leaves model containers running on purpose (they survive a studio
restart), so `tt model stop --studio` stops each model first (see stop_all).
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

# Every container studio starts joins this network — its own cleanup uses it as
# the filter (tt_setup/cleanup/_resource_ops.py).
STUDIO_NETWORK = "tt_studio_network"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


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
        self._docker()

    def _docker(self) -> str:
        found = shutil.which("docker")
        if not found:
            raise TTError(
                "docker not found.",
                why="TT-Studio runs its services and model containers with docker "
                "compose (podman is not supported).",
                next_step="Install it — https://docs.docker.com/engine/install/ — then re-run.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": "docker"},
            )
        return found

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
        self.output.status(
            f"Starting TT-Studio and deploying {model.name} — Ctrl-C stops watching, "
            f"`tt model stop {model.name}` stops the model and studio."
        )
        # Release every live row first: run.py owns the terminal from here on.
        self.output.ui.handoff()
        # run.py resolves the repo root, .env and its compose file from cwd.
        try:
            return self.runner.stream(
                self._argv(model, entry=entry),
                env=self._env(),
                cwd=str(entry.parent),
                tool=TOOL,
            )
        except TTError as exc:
            if exc.exit_code == ExitCode.TOOL_FAILED:
                # A deploy that died part-way leaves studio's stack up with no
                # model behind it; take it down so the error is the only thing
                # left. Only a real failure — Ctrl-C is a KeyboardInterrupt, not
                # a TTError, and "stops watching" as the status line promises.
                self._teardown(entry, after=f"deploying {model.name} failed")
            raise

    def _teardown(self, entry: Path, *, after: str) -> int:
        """`run.py --stop`: stop studio's services (not the models it deployed).
        Best-effort — a teardown that fails must not mask what it was cleaning up
        after — so the status is returned, not raised."""
        self.output.status(f"Stopping TT-Studio ({after}) …")
        rc = self.runner.stream(
            [self._python_for(entry), str(entry), "--stop"],
            env=self._env(),
            cwd=str(entry.parent),
            tool=TOOL,
            check=False,
        )
        if rc != 0:
            self.output.warn(
                f"TT-Studio's `run.py --stop` exited with status {rc}; its containers "
                "may still be running. Check with `docker ps`, or run "
                f"`python {entry} --stop` from {entry.parent} to retry."
            )
        return rc

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
                next_step="`tt serve <model> --studio` installs it and deploys.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": TOOL},
            )
        return Path(found[0])

    def stop(self, model: ModelInfo) -> int:
        """`--stop-model` first, so studio resets the model's chips, then `--stop`
        for the stack itself: with the model gone nothing needs studio's
        containers and services, and leaving them up is what people asked
        `tt model stop` to prevent."""
        entry = self._installed_entry()
        argv = [self._python_for(entry), str(entry), "--stop-model", model.name]
        self.output.status(f"Stopping {model.name} via TT-Studio …")
        # check=False: a `--stop-model` that fails (the model already died, or
        # never finished deploying) must not skip the teardown — the stack is
        # still up either way, and leaving it up is the thing this command
        # exists to prevent. The failure is raised once the stack is down.
        rc = self.runner.stream(
            argv, env=self._env(), cwd=str(entry.parent), tool=TOOL, check=False
        )
        after = f"{model.name} stopped" if rc == 0 else f"stopping {model.name} failed"
        self._teardown(entry, after=after)
        if rc != 0:
            raise TTError(
                f"TT-Studio's `run.py --stop-model {model.name}` exited with status {rc}.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": TOOL, "returncode": rc},
            )
        return rc

    def _containers(self, docker: str) -> tuple[list[str], list[tuple[str, str]]]:
        """Studio's running containers: (service names, [(id, name)] of the models
        it deployed). Read from the same `docker inspect` scan `tt model ps` uses."""
        from .ps import STUDIO_IMAGE_PREFIX, _inspect_all

        services: list[str] = []
        models: list[tuple[str, str]] = []
        for entry in _inspect_all(self.runner, docker, include_stopped=False):
            config = entry.get("Config") or {}
            networks = (entry.get("NetworkSettings") or {}).get("Networks") or {}
            if STUDIO_NETWORK not in networks and not str(config.get("Image") or "").startswith(
                STUDIO_IMAGE_PREFIX
            ):
                continue
            name = str(entry.get("Name") or "").lstrip("/")
            if COMPOSE_PROJECT_LABEL in (config.get("Labels") or {}):
                services.append(name)
            else:
                models.append((str(entry.get("Id") or "")[:12], name))
        return services, models

    def stop_all(self) -> dict:
        """`tt model stop --studio`: every model studio deployed, then its services.

        Each model goes through `--stop-model` (one at a time: studio refuses the
        whole batch if one name does not resolve, as a model still loading does not),
        which resets its chips. Whatever is left — studio's backend is down, or the
        stop failed — gets a plain `docker stop`, because `--stop` alone keeps model
        containers running."""
        docker = self._docker()
        services, models = self._containers(docker)
        if not services and not models:
            return {"running": False, "stopped": [], "services_stopped": False}
        entry = self._installed_entry() if services else None
        if entry is not None:
            for _, name in models:
                self.output.status(f"Stopping {name} via TT-Studio …")
                self.runner.stream(
                    [self._python_for(entry), str(entry), "--stop-model", name],
                    env=self._env(),
                    cwd=str(entry.parent),
                    tool=TOOL,
                    check=False,
                )
            _, left = self._containers(docker)
        else:
            left = models
        for container_id, name in left:
            self.output.status(f"Stopping {name} ({container_id}) with docker …")
            self.runner.capture([docker, "stop", container_id], tool="docker")
        if left:
            self.output.warn(
                f"{', '.join(n for _, n in left)} stopped without TT-Studio's chip reset; "
                "if the next deploy cannot open the device, run `tt-smi -r`."
            )
        if entry is not None and self._teardown(entry, after="its models stopped") != 0:
            raise TTError(
                "TT-Studio's `run.py --stop` failed; its services may still be running.",
                next_step=f"Retry with `python {entry} --stop` from {entry.parent}.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": TOOL},
            )
        return {
            "running": True,
            "stopped": [name for _, name in models],
            "services_stopped": bool(services),
        }
