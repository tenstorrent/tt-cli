# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""tt-deep-research adapter: its own Compose stack, started by the `ttdr` tt installs.

Neither a binary tt configures nor a container tt runs: tt installs tt-deep-research
at its pin on first launch, and `ttdr up` starts the app from its home folder
($TTDR_HOME, default ~/.tt-deep-research), which ttdr fills and keeps the data in.
Its recipes say which models it needs; `ttdr recipes --json` lists them with the
tt-model bundles each serves and on which chips. tt starts the recipe whose bundles
are running, else the one asked for (the default unless --recipe names another),
which `ttdr up` then serves itself. It never lets `ttdr up` serve a model on chips or
ports something else holds: it says what to stop, or which recipe fits what runs.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ...errors import ExitCode, TTError
from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel
from ..discovery import probe

DEFAULT_HOME = "~/.tt-deep-research"
UI_PORT = 3000  # ttdr moves it to the next free port when another program holds it
LIST_HINT = "`tt launch deep-research --list-recipes` shows every recipe."
_COMPOSE = "com.docker.compose"
_PROJECT = "ttdr"
# What `ttdr down` stops; its data services (Qdrant, SearXNG, Redis) keep running.
_APP_SERVICES = frozenset({"engine", "aiq", "ui", "deepagents", "ui-deepagents", "ocr"})


@dataclass(frozen=True)
class Choice:
    recipe: dict  # one entry of `ttdr recipes --json`
    serves_models: bool  # `ttdr up` serves its models: the board is free for them


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
    # `ttdr up` prints its own progress and where the app is, so tt adds no summary table.
    reports_itself = True

    def target(self) -> str:
        return "tt-deep-research app"

    def home(self) -> Path:
        """Where ttdr keeps the app's files and data."""
        return Path(os.environ.get("TTDR_HOME") or DEFAULT_HOME).expanduser()

    def recipes(self, executable: str, runner) -> list[dict]:
        """Every recipe the installed ttdr ships (`ttdr recipes --json`)."""
        result = runner.capture(
            [executable, "recipes", "--json"], env=self._env(self.home()), tool=self.id
        )
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise TTError(
                "`ttdr recipes --json` did not print JSON.",
                why=str(exc),
                next_step="Re-run with --verbose and report this with `tt report issue`.",
                exit_code=ExitCode.TOOL_FAILED,
            ) from exc

    def choose(self, recipes: list[dict], wanted: str | None, bundles: list[str]) -> Choice:
        """--recipe, else the recipe whose bundles are running, else the default; served
        by `ttdr up` only when nothing else holds its chips or ports."""
        if wanted:
            recipe = self._named(recipes, wanted)
        else:
            running = self.running_recipes(recipes, bundles)
            if len(running) > 1:
                names = ", ".join(r["name"] for r in running)
                raise TTError(
                    "Several deep-research recipes fit the models tt-model is serving.",
                    why=f"They all use what is running: {names}.",
                    next_step="Pick one: `tt launch deep-research --recipe <name>`.",
                    exit_code=ExitCode.USAGE,
                    details={"tool": self.id, "recipes": [r["name"] for r in running]},
                )
            if running:
                return Choice(running[0], serves_models=False)
            recipe = self._default(recipes)
        if self._ready(recipe, bundles):
            return Choice(recipe, serves_models=False)
        others, port = self._conflicts(recipe, bundles)
        if others or port:
            raise self._blocked(recipe, recipes, bundles, others, port)
        return Choice(recipe, serves_models=True)

    def listed(self, recipes: list[dict]) -> list[dict]:
        """`--list-recipes`: each described recipe and what it serves."""
        return [
            {
                "recipe": recipe["name"],
                "default": recipe["default"],
                "serves": _serves(recipe),
                "about": recipe["description"],
            }
            for recipe in recipes
            if recipe["description"]
        ]

    def can_start(self, recipes: list[dict], bundles: list[str]) -> bool:
        """Some recipe `ttdr up` could serve now: nothing else holds its chips or ports."""
        return any(not any(self._conflicts(recipe, bundles)) for recipe in recipes)

    def model_bundles(self, bundles: list[str], executable: str | None, runner) -> list[str]:
        """Which of the running bundles are ones its recipes use."""
        if executable is None:
            return []
        known = {b for recipe in self.recipes(executable, runner) for b in recipe["bundles"]}
        return [bundle for bundle in bundles if bundle in known]

    def served_llm(self, recipe: dict) -> RunningModel:
        """The LLM, once every model of the recipe answers."""
        llm = None
        for model in filter(_openai, recipe["models"]):
            found = probe(model["url"])
            if not found:
                raise TTError(
                    f"{model['model']} is not answering at {model['url']} yet.",
                    why="Its bundle is running but still loading; starting now would have "
                    "`ttdr up` serve it a second time.",
                    next_step=f"Watch `tt model logs {model['bundle']} --follow`, then re-run.",
                    exit_code=ExitCode.ERROR,
                    details={"tool": self.id, "model": model["model"], "url": model["url"]},
                )
            if "llm" in model["roles"]:
                llm = found[0]
        assert llm is not None
        return llm

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
        url = f"http://localhost:{self._ui_port(home)}"
        # Up already: this home's engine and a UI run (not a checkout's leftovers), its models
        # serve, and it was started with this recipe.
        ours = {s for s, d in (self._running(runner) or {}).items() if Path(d) == home}
        up = "engine" in ours and bool(ours & {"ui", "ui-deepagents"}) and not options.serves_models
        same = up and self._started_with(home) == Path(recipe)
        if same and (not options.aiq or "aiq" in ours):
            return Preparation(rows={**rows, "app": "already running"}, url=url)
        exe = executable or self.binaries[0]
        env = {"TTDR_HOME": str(home)}
        if options.hf_home:  # the model cache `tt serve` mounts at /hf; ttdr stages templates there
            env["HF_HOME"] = str(options.hf_home)
        up = [exe, "up", "--recipe", recipe, *(["--aiq"] if options.aiq else [])]
        shown = " ".join(["ttdr", *up[1:]])
        consent = f"Run `{shown}`, starting its app containers"
        if options.serves_models:
            consent = (
                f"Run `{shown}`: it serves the recipe's models "
                "(10-15 minutes from cold) and starts its app containers"
            )
        return Preparation(
            rows=rows,
            steps=[up],
            consent=consent,
            url=url,
            env=env,
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        for step in prep.steps:
            # Streamed: a first run builds images and waits on every endpoint.
            env.runner.stream(step, env={**os.environ, **prep.env}, tool=self.id)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        # `ttdr up` exits non-zero unless every endpoint answered, and says where the UI is
        # itself; only an app already running needs telling.
        if not prep.steps:
            env.output.status(f"deep-research is already running at {prep.url}")

    def container_state(self, executable: str | None, runner) -> str:
        """running / absent / unknown: running while any container `ttdr down`
        would stop is up. Nothing to ask without docker."""
        services = self._running(runner)
        if services is None:
            return "unknown"
        return "running" if services else "absent"

    def stop(self, env: LaunchEnv, keep_models: bool = False) -> None:
        """`ttdr down`: the app and its models, or with keep_models only the app."""
        # From the folder the running app was started in, whatever $TTDR_HOME says now.
        home = next(iter((self._running(env.runner) or {}).values()), None) or self.home()
        argv = [env.executable, "down", *(["--keep-models"] if keep_models else [])]
        env.runner.stream(argv, env=self._env(home), tool=self.id)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists outside the app's own home folder, so there is never
        anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _env(self, home: Path | str) -> dict[str, str]:
        return {**os.environ, "TTDR_HOME": str(home)}

    def _default(self, recipes: list[dict]) -> dict:
        return next(recipe for recipe in recipes if recipe["default"])

    def _named(self, recipes: list[dict], wanted: str) -> dict:
        if wanted == "default":
            return self._default(recipes)
        for recipe in recipes:
            if wanted in (recipe["name"], recipe["path"]):
                return recipe
        listed = ", ".join(r["name"] for r in recipes if r["description"])
        raise TTError(
            f"deep-research has no recipe {wanted!r}.",
            why=f"Its recipes: {listed}.",
            next_step=LIST_HINT,
            exit_code=ExitCode.USAGE,
            details={"recipe": wanted},
        )

    def serving_llm(self, executable: str, bundles: list[str], runner) -> str | None:
        """The LLM of the recipe whose bundles are serving: of its several models, the one a
        chat client wants (the others embed and rerank)."""
        running = self.running_recipes(self.recipes(executable, runner), bundles)
        return next((m["model"] for r in running for m in r["models"] if "llm" in m["roles"]), None)

    def running_recipes(self, recipes: list[dict], bundles: list[str]) -> list[dict]:
        """Listed recipes whose bundles tt-model is serving now. Where several share them,
        the one the app last started with, else the default, settles it."""
        fits = [
            r for r in recipes
            if r["description"] and r["bundles"] and set(r["bundles"]) <= set(bundles)
        ]
        last = self._started_with(self.home())
        for preferred in (lambda r: Path(r["path"]) == last, lambda r: r["default"]):
            settled = [r for r in fits if preferred(r)]
            if settled:
                return settled[:1]
        return fits

    def _ready(self, recipe: dict, bundles: list[str]) -> bool:
        """Its bundles are running, or its models already answer (served some other way)."""
        if set(recipe["bundles"]) <= set(bundles):
            return True
        return all(self._serving(m) == m["model"] for m in recipe["models"] if _openai(m))

    def _serving(self, model: dict) -> str | None:
        """The model a recipe model's URL answers with, if any."""
        found = probe(model["url"]) if _openai(model) else None
        return found[0].served_id if found else None

    def _conflicts(self, recipe: dict, bundles: list[str]) -> tuple[list[str], str | None]:
        """What keeps `ttdr up` from serving the recipe: other running bundles (they hold
        chips), or another model on the port of a cohost bundle (ttdr moves a bundle of
        its own to the next free port, but not a cohost's models)."""
        others = [b for b in bundles if b not in recipe["bundles"]]
        hosts = {m["cohosted_with"] for m in recipe["models"]}
        for model in recipe["models"]:
            movable = model["bundle"] and model["name"] not in hosts
            if movable or model["cohosted_with"] or model["bundle"] in bundles:
                continue
            serving = self._serving(model)
            if serving and serving != model["model"]:
                return others, f"{model['url']} already serves {serving}"
        return others, None

    def _blocked(
        self,
        recipe: dict,
        recipes: list[dict],
        bundles: list[str],
        others: list[str],
        port: str | None,
    ) -> TTError:
        name = recipe["name"]
        if others:
            why = f"tt-model is serving {', '.join(others)}, on chips the {name} recipe needs."
            steps = [f"Free the board with `tt model stop {' '.join(others)}`, then re-run."]
        else:
            why = f"{port}, not a tt-model bundle."
            steps = ["Stop that server, then re-run."]
        fits = self.running_recipes(recipes, bundles)
        if fits:
            steps.append(
                f"Or use what is running: `tt launch deep-research --recipe {fits[0]['name']}`."
            )
        return TTError(
            f"deep-research can't start its {name} recipe: something else holds the board.",
            why=why,
            next_step=" ".join([*steps, LIST_HINT]),
            exit_code=ExitCode.ERROR,
            details={"tool": self.id, "recipe": name, "serving": bundles},
        )

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

    def _ui_port(self, home: Path) -> int:
        """Where the UI listens: the port `ttdr up` recorded in data/ports.json."""
        try:
            return int(json.loads((home / "data" / "ports.json").read_text())["ui"])
        except (OSError, ValueError, KeyError, TypeError):
            return UI_PORT

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


def _openai(model: dict) -> bool:
    """An OpenAI-compatible endpoint, which tt can ask what it serves."""
    return model["url"].rstrip("/").endswith("/v1")


def _serves(recipe: dict) -> str:
    """What a recipe serves, and on which chips: "one bundle on chips 0-3: A, B, C" or
    "A (chips 0-1), B (chip 2), C (chip 3)"."""
    models = recipe["models"]
    host = next((m for m in models if m["name"] in {x["cohosted_with"] for x in models}), None)
    names = [m["model"].split("/")[-1] for m in models]
    if host:
        return f"one bundle on {_chips(host['chips'])}: " + ", ".join(names)
    return ", ".join(
        f"{name} ({_chips(m['chips'])})" if m["chips"] else name for name, m in zip(names, models)
    )


def _chips(chips: str) -> str:
    return f"chips {chips}" if "-" in chips or "," in chips else f"chip {chips}"
