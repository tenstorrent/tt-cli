# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""tt-deep-research adapter: its own Compose stack, started by the `ttdr` tt installs.

Neither a binary tt configures nor a container tt runs: tt installs tt-deep-research
at its pin on first launch, and `ttdr up` starts the app from its home folder
($TTDR_HOME, default ~/.tt-deep-research), which ttdr fills and keeps the data in.
It needs an LLM, an embedder and a reranker; the recipe describing them is picked
from the bundles tt-model reports running. With none of them running and nothing
else on the board, the default recipe (`recipe.toml`: the three models on separate
chips) starts and `ttdr up` serves them itself; a cohost bundle is only used when
it is already running or asked for with --recipe. `ttdr up` serves any model of
its recipe that does not answer, so when some bundle is running every endpoint is
checked first: tt never lets it start a model on chips the running ones hold.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ...errors import ExitCode, TTError
from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel
from ..container import port_is_free
from ..discovery import probe

DEFAULT_HOME = "~/.tt-deep-research"
UI_PORT = 3000
# Where every recipe below expects its models.
ENDPOINTS = {"LLM": 8000, "embedder": 8005, "reranker": 8004}
# Each recipe, with the bundles it expects tt-model to be serving. The default,
# recipe.toml, serves the three models on separate chips (rerank.toml is the same
# setup under its old name); the cohost recipes serve all three from one bundle.
SEPARATE_BUNDLES = (
    "raahemnabeel/ornith-1.5-35b-a3b",
    "raahemnabeel/qwen3-vl-embedding-8b",
    "raahemnabeel/qwen3-vl-reranker-8b",
)
RECIPES = {
    "recipe.toml": SEPARATE_BUNDLES,
    "recipes/rerank.toml": SEPARATE_BUNDLES,
    "recipes/ornith-35b-cohost.toml": ("raahemnabeel/cohost-ornith-35b-qwen3vl",),
    "recipes/ornith-122b.toml": ("raahemnabeel/cohost-ornith-122b-qwen3vl",),
    "recipes/qwen38-27b.toml": ("raahemnabeel/cohost-qwen38-27b-qwen3vl",),
}
# Started when nothing is running: `ttdr up` serves its own models, one bundle per chip group.
DEFAULT_RECIPE = "recipe.toml"
DEFAULT_MODEL = "ornith-ai/Ornith-1.5-35B-A3B"
_COMPOSE = "com.docker.compose"
_PROJECT = "ttdr"
# What `ttdr down` stops; its data services (Qdrant, SearXNG, Redis) keep running.
_APP_SERVICES = frozenset({"engine", "aiq", "ui", "deepagents", "ui-deepagents", "ocr"})


class DeepResearch:
    id = "deep-research"
    binaries = ("ttdr",)
    # Installed and pinned by tt, through the tool registry, on first launch.
    managed_tool = "tt-deep-research"
    install_hint = "`tt launch deep-research` installs it."
    # Its recipe, not tt's catalog, says what the models can do.
    requires_tool_calling = False
    model_requirement = "a bundle its recipes name"
    hands_over_terminal = False
    uses_recipes = True
    # What starts when none of its recipes' bundles is running: `ttdr up` serves the models.
    default_recipe = DEFAULT_RECIPE
    default_model = DEFAULT_MODEL

    def target(self) -> str:
        return "tt-deep-research app"

    def home(self) -> Path:
        """Where ttdr keeps the app's files and data."""
        return Path(os.environ.get("TTDR_HOME") or DEFAULT_HOME).expanduser()

    def recipe_for(self, home: Path, wanted: str | None, bundles: list[str]) -> str | None:
        """--recipe, else the recipe whose bundles are all running; None if none is."""
        if wanted:
            if wanted not in RECIPES and not (home / wanted).is_file():
                raise TTError(
                    f"No recipe {wanted} in {home}.",
                    why="--recipe is a path relative to the app's home folder.",
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
            next_step=f"With the board free, `tt launch {self.id}` starts the app and "
            "`ttdr up` serves its models on separate chips. Or serve the bundles one of "
            f"its recipes names ({', '.join(SEPARATE_BUNDLES)}), then re-run; for another "
            "recipe, pass --recipe.",
            exit_code=ExitCode.ERROR,
            details={"tool": self.id, "serving": bundles},
        )

    def serve_blocker(self, bundles: list[str]) -> str | None:
        """Why tt must not start the default recipe here: the board or one of its
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
        assert options.home is not None and options.recipe is not None
        home, recipe = options.home, options.recipe
        rows = {"home": str(home), "recipe": recipe}
        url = f"http://localhost:{UI_PORT}"
        running = self.container_state(executable, runner) == "running"
        if running and self._started_with(home) == Path(recipe):
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
        consent = f"Run `ttdr up --recipe {recipe}`, starting its app containers"
        if options.serves_models:
            consent = (
                f"Run `ttdr up --recipe {recipe}`: it serves its LLM, embedder and reranker "
                "(each on its own chips; 10-15 minutes from cold) and starts its app containers"
            )
        return Preparation(
            rows=rows,
            steps=[[exe, "up", "--recipe", recipe]],
            consent=consent,
            url=url,
            env={"TTDR_HOME": str(home)},
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        for step in prep.steps:
            # Streamed: a first run builds images and waits on every endpoint.
            env.runner.stream(step, env={**os.environ, **prep.env}, tool=self.id)

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
        # From the folder the running app was started in, whatever $TTDR_HOME says now.
        home = next(iter((self._running(env.runner) or {}).values()), None) or self.home()
        env.runner.stream(
            [env.executable, "down"], env={**os.environ, "TTDR_HOME": str(home)}, tool=self.id
        )

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists outside the app's own home folder, so there is never
        anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _running(self, runner) -> dict[str, str] | None:
        """{app service: the folder Compose started it from}, or None without docker."""
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

    def _started_with(self, home: Path) -> Path | None:
        """The recipe the app was last started with: `ttdr up` records it in .env."""
        env = home / ".env"
        if not env.is_file():
            return None
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "TTDR_RECIPE":
                return Path(value.strip())
        return None
