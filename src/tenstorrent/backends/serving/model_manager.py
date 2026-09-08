# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""tt-model-manager serving backend (its CLI is `tt-model`) — the second `tt serve` path.

tt-inference-server serves the models in its released spec; tt-model serves
self-contained bundles published as Hugging Face repos (`namespace/name`),
through the Tenstorrent vLLM plugin. `tt serve` prefers the spec and falls back
here, so a bundle id nobody has curated still serves with one command.

Kept deliberately thin: bundle resolution, compatibility checks and engine env
are tt-model's own job, and it reports them itself. All we own is finding the
pinned binary (lazily installed at its git ref) and sharing our HF cache.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from ...config.store import ConfigStore
from ...errors import ExitCode, TTError
from ...modelhub import bundles
from ...modelhub.hub import hf_home_dir
from ...output import OutputManager
from ...tools.registry import ToolRegistry
from ...tools.runner import Runner

TOOL = "tt-model"  # the distribution, the command, and the manifest key
# tt-model-manager's own container label (container.py: LABEL); every container
# it starts carries it, and always under docker, never podman.
_LABEL = "org.tenstorrent.tt-model"


def looks_like_bundle_id(name: str) -> bool:
    """Whether `name` could be a tt-model bundle: a Hub repo id, `namespace/name`.

    The routing guard for `tt serve`. Without it a typo'd catalog name ("Llama-3.1-8B-Instrukt")
    would fall through to tt-model and surface a Hub 404 instead of our
    "unknown model, run `tt model list`" error.
    """
    parts = name.split("/")
    return len(parts) == 2 and all(parts) and not any(c.isspace() for c in name)


class ModelManagerBackend:
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

    def _env(self) -> dict[str, str]:
        """Environment for every tt-model call.

        Runner.stream() *replaces* the child environment rather than merging it
        (exec_tty merges), so os.environ is inherited explicitly — tt-model needs
        PATH, HF_TOKEN and the tt-metal/vLLM vars its launch path reads. HF_HOME is
        pinned on **every** subcommand, not just the ones that download: `rm
        --include-weights` deletes from whatever cache its process resolves, so a
        configured paths.hf_model_cache_directory has to reach teardown too, or tt
        would delete from the default cache and orphan the real weights.
        """
        return {**os.environ, "HF_HOME": str(hf_home_dir(self.config))}

    def serve(
        self,
        repo_id: str,
        *,
        offline: bool = False,
        port: int | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Stream `tt-model serve <repo_id>`, which installs the bundle if needed
        and launches its OpenAI-compatible server.

        `extra_args` are appended verbatim after the bundle id — where tt-model
        itself expects them (its serve declares allow_extra_args/ignore_unknown_options
        and forwards what it does not claim to vLLM)."""
        entry = self.registry.ensure(TOOL, offline=offline)
        argv = self._argv(
            repo_id, offline=offline, port=port, extra_args=extra_args, entry=Path(entry)
        )
        self.output.status(f"Serving {repo_id} via tt-model …")
        return self.runner.stream(argv, env=self._env(), tool=TOOL)

    def _argv(
        self,
        repo_id: str,
        *,
        offline: bool,
        port: int | None,
        extra_args: list[str] | None,
        entry: Path | None,
    ) -> list[str]:
        """The tt-model command line. `entry` is None for a dry run, which must not
        install a lazy tool just to describe what it would run."""
        argv = [str(entry)] if entry else ["<tt-model>"]
        argv += ["serve", repo_id]
        if offline:
            # tt-model would otherwise pull the bundle from the Hub.
            argv.append("--local-only")
        if port is not None:
            argv += ["--port", str(port)]
        argv += list(extra_args or ())
        return argv

    def plan(
        self,
        repo_id: str,
        *,
        offline: bool = False,
        port: int | None = None,
        extra_args: list[str] | None = None,
    ) -> dict:
        """What `serve` would run.

        tt-model owns bundle configuration and applies it itself, so tt passes no
        launch flags on this path. For a bundle already pulled, its manifest is on
        disk and records the same things tt resolves for the other backend — image,
        parsers, tt config, port — so report those rather than leaving the preview
        empty. A bundle that has not been pulled has no manifest, and fetching one
        to describe a serve would turn a preview into a download.
        """
        installed = self.registry._resolve_or_none(TOOL)
        return {
            "bundle": bundles.serve_details(repo_id),
            "backend": "tt-model",
            "model": repo_id,
            "offline": offline,
            "port": port,
            "extra_args": list(extra_args or ()),
            "installed": installed is not None,
            "argv": self._argv(
                repo_id,
                offline=offline,
                port=port,
                extra_args=extra_args,
                entry=Path(installed[0]) if installed else None,
            ),
        }

    def running_ports(self) -> list[int]:
        """Host ports tt-model's own containers are published on.

        tt-model publishes host:container on the same port number, so `docker ps`
        alone gives it — no `inspect` call needed, unlike the tt-inference-server
        side where the container's own port is fixed and the host port varies.
        """
        docker = shutil.which("docker")
        if docker is None:
            return []
        listed = self.runner.capture(
            [docker, "ps", "--filter", f"label={_LABEL}", "--format", "{{.Ports}}"],
            tool="docker",
        )
        ports = {
            int(port)
            for line in listed.stdout.splitlines()
            for port in re.findall(r":(\d+)->", line)
        }
        return sorted(ports)

    def unsupported_workflow(self, workflow: str) -> TTError:
        return TTError(
            f"tt-model cannot run the {workflow!r} workflow.",
            why="It serves an OpenAI-compatible server only; benchmarks and evals "
            "are tt-inference-server workflows.",
            next_step="Serve a model from `tt model list` for benchmarks or evals.",
            exit_code=ExitCode.UNSUPPORTED,
        )

    def _installed_entry(self) -> Path:
        """Resolve tt-model *without* installing it.

        stop/rm are teardown: fetching a git repo and building a uv venv just to
        discover there is nothing installed to tear down is backwards. serve() still
        uses ensure(), where installing is the whole point. registry.resolve() is
        avoided here because its next_step says `tt update`, which deliberately skips
        this lazy tool."""
        found = self.registry._resolve_or_none(TOOL)
        if found is None:
            raise TTError(
                "tt-model is not installed, so there is nothing for tt to act on.",
                why="No bundle has been served through the tt-model path on this "
                "machine — tt installs the tool on first use, not for teardown.",
                next_step="`tt serve <namespace>/<name>` installs it and serves a bundle.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": TOOL},
            )
        return Path(found[0])

    def stop(self, repo_id: str, *, profile: str | None = None) -> int:
        """Stop a running container package via `tt-model stop`.

        Straight passthrough on purpose: tt-model SIGTERMs first so the server can
        close the device mesh, and resets the mesh with tt-smi if docker has to
        SIGKILL — behaviour we should delegate to, not reimplement."""
        entry = self._installed_entry()
        argv = [str(entry), "stop", repo_id]
        if profile:
            argv += ["--profile", profile]
        self.output.status(f"Stopping {repo_id} via tt-model …")
        return self.runner.stream(argv, env=self._env(), tool=TOOL)

    def logs(
        self, repo_id: str, *, follow: bool = False, profile: str | None = None
    ) -> int:
        """Show a running container package's output via `tt-model logs`.

        tt-model resolves the running container for the bundle (and profile) and
        runs `docker logs [--follow]` on it. Read-only, so like stop/rm it never
        installs the tool. check=False: Ctrl-C on --follow ends the child with 130,
        which is the user stopping, not the tool failing."""
        entry = self._installed_entry()
        argv = [str(entry), "logs", repo_id]
        if follow:
            argv.append("--follow")
        if profile:
            argv += ["--profile", profile]
        self.output.status(f"Showing {repo_id} logs via tt-model …")
        return self.runner.stream(argv, env=self._env(), tool=TOOL, check=False)

    def rm_argv(
        self, entry: Path, repo_id: str, *, include_weights: bool, keep_cache: bool
    ) -> list[str]:
        """The `tt-model rm` command line (shared with --dry-run, which prints it)."""
        argv = [str(entry), "rm", repo_id]
        if keep_cache:
            argv.append("--keep-cache")
        if include_weights:
            argv.append("--include-weights")
        return argv

    def rm(
        self, repo_id: str, *, include_weights: bool = False, keep_cache: bool = False
    ) -> int:
        """Remove an installed bundle via `tt-model rm`.

        tt-model owns what "remove" means here — containers, image, pulled manifest,
        kernel cache and its own HF snapshot — and keeps the weights unless
        --include-weights, since they are shared with everything else on the host."""
        entry = self._installed_entry()
        argv = self.rm_argv(
            entry, repo_id, include_weights=include_weights, keep_cache=keep_cache
        )
        self.output.status(f"Removing {repo_id} via tt-model …")
        return self.runner.stream(argv, env=self._env(), tool=TOOL)

    def pull(self, repo_id: str, *, with_weights: bool = True) -> int:
        """Install a bundle via `tt-model pull` (its venv/image and, by default, the
        weights too).

        tt-model skips weights by default — the model class fetches them at load —
        but `tt model pull` means "have it ready before serving", so we ask for them.
        Cheap to ask: for a container package (the norm since v5.1) tt-model puts
        them in the host HF cache, so anything already there is skipped and a partial
        download resumes. A legacy venv bundle instead copies them into its own
        directory (`local_dir`), where the shared cache cannot help — pass
        with_weights=False if that copy is not wanted."""
        entry = self.registry.ensure(TOOL)
        argv = [str(Path(entry)), "pull", repo_id]
        if with_weights:
            argv.append("--with-weights")
        self.output.status(f"Pulling {repo_id} via tt-model …")
        return self.runner.stream(argv, env=self._env(), tool=TOOL)
