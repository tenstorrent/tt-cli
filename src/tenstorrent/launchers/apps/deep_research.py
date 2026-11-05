# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""tt-deep-research adapter: its own Compose stack, started from a local checkout.

Neither a binary tt configures nor a container tt runs: `uv run ttdr up` starts
the app from the user's checkout of https://github.com/rnabeelTT/tt-deep-research.
It needs an LLM, an embedder and a reranker; the recipe describing them is picked
from the bundles tt-model reports running. With none of them running and nothing
else on the board, tt serves the default cohost bundle first and waits for it.
`ttdr up` serves any model of its recipe that does not answer, so every endpoint
is checked first: tt never lets it start a model on chips the running ones hold.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from ...errors import ExitCode, TTError
from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel
from ..container import port_is_free
from ..discovery import probe

REPO_URL = "https://github.com/rnabeelTT/tt-deep-research"
CONFIG_KEY = "launch.deep_research_path"
DEFAULT_CHECKOUT = "~/tt-deep-research"
UI_PORT = 3000
# Where every recipe below expects its models.
ENDPOINTS = {"LLM": 8000, "embedder": 8005, "reranker": 8004}
# Each recipe, with the bundles it expects tt to be serving; cohosts serve all
# three models from one bundle. recipe.toml itself also wants Laya on :8002, so
# the separate bundles map to its overlay without it.
RECIPES = {
    "recipes/ornith-35b-cohost.toml": ("raahemnabeel/cohost-ornith-35b-qwen3vl",),
    "recipes/ornith-122b.toml": ("raahemnabeel/cohost-ornith-122b-qwen3vl",),
    "recipes/qwen38-27b.toml": ("raahemnabeel/cohost-qwen38-27b-qwen3vl",),
    "recipes/rerank.toml": (
        "raahemnabeel/ornith-1.5-35b-a3b",
        "raahemnabeel/qwen3-vl-embedding-8b",
        "raahemnabeel/qwen3-vl-reranker-8b",
    ),
}
# Served by tt when nothing is: the LLM, embedder and reranker as one bundle.
DEFAULT_BUNDLE = "raahemnabeel/cohost-ornith-35b-qwen3vl"
DEFAULT_RECIPE = "recipes/ornith-35b-cohost.toml"
DEFAULT_MODEL = "ornith-ai/Ornith-1.5-35B-A3B"
SERVE_FLAGS = ("--local-only", "--port", "8000", "--detach")
RECOMMENDED_SERVE = f"tt-model serve {DEFAULT_BUNDLE} {' '.join(SERVE_FLAGS)}"
# The cohost's own status: each member reports whether it is ready.
STATUS_URL = "http://127.0.0.1:8006"
SERVE_TIMEOUT_S = 30 * 60  # a cold start takes 10-15 minutes
_POLL_S = 5.0
_COMPOSE = "com.docker.compose"
_PROJECT = "ttdr"
# What `ttdr down` stops; its data services (Qdrant, SearXNG, Redis) keep running.
_APP_SERVICES = frozenset({"engine", "aiq", "ui", "deepagents", "ui-deepagents", "ocr"})


class DeepResearch:
    id = "deep-research"
    binaries = ("uv",)
    install_hint = (
        "Install uv (https://docs.astral.sh/uv/getting-started/installation/), then re-run."
    )
    # Its recipe, not tt's catalog, says what the models can do.
    requires_tool_calling = False
    model_requirement = "a bundle its recipes name"
    hands_over_terminal = False
    runs_from_checkout = True
    # What tt serves itself when none of its recipes' bundles is running.
    default_bundle = DEFAULT_BUNDLE
    default_recipe = DEFAULT_RECIPE
    default_model = DEFAULT_MODEL
    serve_flags = SERVE_FLAGS

    def target(self) -> str:
        return "tt-deep-research app"

    def checkout(self, config, path: Path | None = None) -> Path:
        root = self._root(config, path)
        if not (root / "recipe.toml").is_file():
            raise TTError(
                f"No tt-deep-research checkout at {root}.",
                why=f"`tt launch {self.id}` runs the app from your own checkout; "
                "it does not install one.",
                next_step=f"git clone {REPO_URL} {root}, then re-run. Elsewhere: pass "
                f"--path, or `tt config set {CONFIG_KEY} <dir>`.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": self.id, "path": str(root)},
            )
        return root

    def missing(self, config) -> str | None:
        """What `tt launch list` names when the checkout is absent."""
        root = self._root(config)
        return None if (root / "recipe.toml").is_file() else f"a checkout at {root}"

    def recipe_for(
        self, checkout: Path, wanted: str | None, bundles: list[str]
    ) -> str | None:
        """--recipe, else the recipe whose bundles are all running; None if none is."""
        if wanted:
            if not (checkout / wanted).is_file():
                raise TTError(
                    f"No recipe {wanted} in {checkout}.",
                    why="--recipe is a path relative to the checkout.",
                    next_step=f"Pick one of {', '.join(RECIPES)}, or drop --recipe.",
                    exit_code=ExitCode.USAGE,
                    details={"recipe": wanted},
                )
            return wanted
        for recipe, needed in RECIPES.items():
            if set(needed) <= set(bundles):
                return recipe
        return None

    def not_served(self, bundles: list[str], blocker: str | None = None) -> TTError:
        why = (
            "It needs an LLM, an embedder and a reranker from a bundle one of its "
            f"recipes names; tt-model is serving {', '.join(bundles) or 'nothing'}."
        )
        return TTError(
            "tt is not serving the models deep-research needs.",
            why=f"{why} {blocker}" if blocker else why,
            next_step=f"Serve the recommended bundle (all three models): `{RECOMMENDED_SERVE}`, "
            "then re-run. For another bundle, pass --recipe.",
            exit_code=ExitCode.ERROR,
            details={"tool": self.id, "serving": bundles},
        )

    def serve_blocker(self, bundles: list[str]) -> str | None:
        """Why tt must not serve the default bundle here: the board or one of its
        ports is already someone else's."""
        if bundles:
            return "tt will not serve another bundle over those."
        for role, port in ENDPOINTS.items():
            models = probe(f"http://127.0.0.1:{port}/v1")
            if models:
                return (
                    f"Port {port} (its {role}) already serves {models[0].served_id}, "
                    "so tt will not serve over it."
                )
        return None

    def model_bundles(self, bundles: list[str]) -> list[str]:
        """Which of the running bundles are ones its recipes use."""
        known = {bundle for needed in RECIPES.values() for bundle in needed}
        return [bundle for bundle in bundles if bundle in known]

    def ready(self) -> bool:
        """Whether the cohost reports every member ready, or failing that,
        whether all three endpoints answer."""
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=5.0) as response:
                members = (json.load(response) or {}).get("members") or {}
            if members and all(
                isinstance(m, dict) and m.get("ready") is True for m in members.values()
            ):
                return True
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            pass
        return all(probe(f"http://127.0.0.1:{port}/v1") for port in ENDPOINTS.values())

    def wait_until_served(
        self,
        alive: Callable[[], bool],
        on_tick: Callable[[float], None] = lambda elapsed: None,
        timeout_s: float = SERVE_TIMEOUT_S,
    ) -> None:
        """Poll until the default bundle is ready; `alive` says whether its
        container is still running."""
        logs = (
            f"See its boot with `tt model logs {DEFAULT_BUNDLE} --follow` "
            "(tt-model logs), or `docker ps -a` and `docker logs <container>`."
        )
        start = time.monotonic()
        while True:
            if self.ready():
                return
            elapsed = time.monotonic() - start
            if not alive():
                raise TTError(
                    f"{DEFAULT_BUNDLE} stopped before it was ready.",
                    why="Its container is no longer running.",
                    next_step=logs,
                    exit_code=ExitCode.TOOL_FAILED,
                    details={"tool": self.id, "bundle": DEFAULT_BUNDLE},
                )
            if elapsed >= timeout_s:
                raise TTError(
                    f"{DEFAULT_BUNDLE} was not ready after {int(timeout_s // 60)} minutes.",
                    why=f"Its status ({STATUS_URL}) never reported every member ready.",
                    next_step=f"{logs} Once it is, re-run `tt launch {self.id}`.",
                    exit_code=ExitCode.TOOL_FAILED,
                    details={"tool": self.id, "bundle": DEFAULT_BUNDLE},
                )
            on_tick(elapsed)
            time.sleep(_POLL_S)

    def served_llm(self) -> RunningModel:
        """The LLM, once all three endpoints answer."""
        found: dict[str, RunningModel] = {}
        for role, port in ENDPOINTS.items():
            models = probe(f"http://127.0.0.1:{port}/v1")
            if not models:
                raise TTError(
                    f"Nothing answers as deep-research's {role} on port {port}.",
                    why="`ttdr up` would try to serve it itself, on chips the running "
                    "models already hold.",
                    next_step="If you just served it, it is probably still loading — "
                    "watch `tt model logs <bundle> --follow` and retry.",
                    exit_code=ExitCode.ERROR,
                    details={"tool": self.id, "role": role, "port": port},
                )
            found[role] = models[0]
        return found["LLM"]

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        assert options.checkout is not None and options.recipe is not None
        checkout, recipe = options.checkout, options.recipe
        rows = {"checkout": str(checkout), "recipe": recipe}
        url = f"http://localhost:{UI_PORT}"
        running = self.container_state(executable, runner) == "running"
        if running and self._started_with(checkout) == Path(recipe):
            return Preparation(rows={**rows, "app": "already running"}, url=url)
        if not running and not port_is_free(UI_PORT):
            raise TTError(
                f"Host port {UI_PORT} is already in use.",
                why=f"deep-research's web UI always binds {UI_PORT}.",
                next_step="Free it (`tt launch stop openwebui`, say, if a web UI tt "
                "launched holds it), then re-run.",
                exit_code=ExitCode.CONFIG,
                details={"web_port": UI_PORT},
            )
        exe = executable or self.binaries[0]
        return Preparation(
            rows=rows,
            steps=[[exe, "run", "ttdr", "up", "--recipe", recipe]],
            consent=f"Run `ttdr up --recipe {recipe}` in {checkout}, starting its app "
            "containers",
            url=url,
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        for step in prep.steps:
            # Streamed: a first run builds images and waits on every endpoint.
            env.runner.stream(step, cwd=prep.rows["checkout"], tool=self.id)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        # `ttdr up` exits non-zero unless every endpoint answered.
        env.output.status(f"deep-research is ready at {prep.url}")

    def container_state(self, executable: str | None, runner) -> str:
        """running / absent / unknown: running while any container `ttdr down`
        would stop is up. Nothing to ask without docker."""
        services = self._running(runner)
        if services is None:
            return "unknown"
        return "running" if services else "absent"

    def stop(self, env: LaunchEnv) -> None:
        # From the checkout the running app was started in, whatever tt's config says.
        checkout = next(iter((self._running(env.runner) or {}).values()), None)
        env.runner.stream([env.executable, "run", "ttdr", "down"], cwd=checkout, tool=self.id)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists outside the app's own checkout, so there is never
        anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _root(self, config, path: Path | None = None) -> Path:
        return Path(path or config.get(CONFIG_KEY) or DEFAULT_CHECKOUT).expanduser()

    def _running(self, runner) -> dict[str, str] | None:
        """{app service: the checkout Compose started it from}, or None without docker."""
        docker = shutil.which("docker")
        if docker is None:
            return None
        result = runner.capture(
            [
                docker, "ps",
                "--filter", f"label={_COMPOSE}.project={_PROJECT}",
                "--format",
                f'{{{{.Label "{_COMPOSE}.service"}}}}\t'
                f'{{{{.Label "{_COMPOSE}.project.working_dir"}}}}',
            ],
            check=False,
            tool="docker",
        )
        if result.returncode != 0:
            return None
        rows = (line.split("\t") for line in result.stdout.splitlines())
        return {
            row[0]: row[1] for row in rows if len(row) == 2 and row[0] in _APP_SERVICES
        }

    def _started_with(self, checkout: Path) -> Path | None:
        """The recipe the app was last started with: `ttdr up` records it in .env."""
        env = checkout / ".env"
        if not env.is_file():
            return None
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "TTDR_RECIPE":
                return Path(value.strip())
        return None
