# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Inference backend: preflight + delegate to tt-inference-server's run.py.

tt-inference-server lives in an isolated git checkout + venv managed by the
registry (GitVenvInstaller). Preflight catches the big environmental gaps
(container runtime, uncached gated models) before a long tool run fails
confusingly halfway through."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ...config.store import ConfigStore
from ...errors import ExitCode, TTError
from ...models.device import DeviceSnapshot
from ...models.model import DeviceSupport, ModelInfo
from ...modelhub.hub import hf_home_dir, uses_host_weight_cache
from ...output import OutputManager
from ...tools.registry import ToolRegistry
from ...tools.runner import Runner

TOOL = "tt-inference-server"
WORKFLOWS = ("server", "benchmarks", "evals")

# (board family, board count) → run.py --device id (v0.18.0 choices). tt-smi
# reports one entry per asic with a revision suffix ("p300c", "n150 L"); dual-asic
# boards (n300, p300) share a board_id across their two entries, so boards are
# counted by unique board_id, not by entry.
_BOARDS_TO_DEVICE = {
    ("e150", 1): "e150",
    ("n150", 1): "n150",
    ("n150", 4): "n150x4",
    ("n300", 1): "n300",
    ("n300", 4): "t3k",
    ("p100", 1): "p100",
    ("p150", 1): "p150",
    ("p150", 4): "p150x4",
    ("p150", 8): "p150x8",
    ("p300", 1): "p300",
    ("p300", 2): "p300x2",
}


def infer_device_config(devices: Sequence[DeviceSnapshot]) -> str | None:
    """Map a tt-smi snapshot to a run.py --device id; None if not confidently
    mappable (mixed board types, unknown family, unmapped count)."""
    boards: dict[str, set[str]] = {}
    for dev in devices:
        if not dev.board_type:
            return None
        family = re.match(r"([a-z]+\d+)", dev.board_type.strip().lower())
        if family is None:
            return None
        boards.setdefault(family.group(1), set()).add(
            dev.board_id or f"entry-{dev.index}"
        )
    if len(boards) != 1:
        return None
    (family_name, ids), = boards.items()
    return _BOARDS_TO_DEVICE.get((family_name, len(ids)))


@dataclass(frozen=True)
class Artifact:
    """One removable on-disk artifact of a served model."""

    kind: str  # "logs" | "volume"
    path: Path
    size_bytes: int


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:  # vanished mid-walk, or unreadable — not worth failing over
            continue
    return total


def _size_of(path: Path) -> int:
    try:
        return _dir_size(path) if path.is_dir() else path.stat().st_size
    except OSError:
        return 0



# tt-inference-server names its containers tt-inference-server-<short_uuid> and sets
# no labels, so the name cannot say which model a container serves. Two mounts can,
# and both are derived from the model spec upstream (v0.18.0):
#   * the readonly weights bind mount, whose source is the HF snapshot dir
#     (…/models--<org>--<name>/snapshots/<rev>) — always present on tt's path, since
#     `tt serve` always passes --host-hf-cache;
#   * otherwise the named volume `volume_id_<impl_id>-<model_name>`
#     (generate_docker_volume_name in workflows/run_docker_server.py).
# Both are implementation details rather than a published contract: if either format
# changes, identification degrades to "unidentified" and the caller must fall back to
# a container id. Track them when the pin is bumped, alongside _BOARDS_TO_DEVICE.
CONTAINER_PREFIX = "tt-inference-server-"
_HF_SNAPSHOT_RE = re.compile(r"models--([^/]+?)--([^/]+?)[/\\]snapshots")
# volume_id_<impl_id>-<model_name> for a docker named volume, and
# volume_id_<impl_id>-<model_name>-v<version> for a --host-volume subdirectory.
# impl_id itself contains hyphens ("tt-transformers", "forge-vllm-plugin"), so
# the name is kept whole and matched against a model rather than split.
_VOLUME_PREFIX = "volume_id_"


@dataclass(frozen=True)
class ServerContainer:
    """A running tt-inference-server container, with whatever identity we could read."""

    id: str
    name: str
    image: str
    hf_repo: str | None = None  # from the weights bind mount
    volume: str | None = None  # volume_id_<impl_id>-<model_name>, kept whole

    @property
    def identified(self) -> bool:
        return bool(self.hf_repo or self.volume)

    def matches(self, model: ModelInfo) -> bool:
        if self.hf_repo and self.hf_repo.lower() == model.hf_repo.lower():
            return True
        # Two real shapes, and impl_id itself contains hyphens so neither can be
        # split reliably: a docker named volume is volume_id_<impl>-<model> (no
        # version — generate_docker_volume_name drops it so image upgrades reuse
        # the volume), while a --host-volume subdirectory is
        # volume_id_<impl>-<model>-v<version>. Match the model name either way.
        if not self.volume:
            return False
        volume, name = self.volume.lower(), model.name.lower()
        return volume.endswith(f"-{name}") or f"-{name}-v" in volume


def _identity_from_inspect(entry: dict) -> tuple[str | None, str | None]:
    """(hf_repo, volume_name) read out of one `docker inspect` record."""
    hf_repo = volume = None
    for mount in entry.get("Mounts") or []:
        match = _HF_SNAPSHOT_RE.search(str(mount.get("Source") or ""))
        if match and not hf_repo:
            hf_repo = f"{match.group(1)}/{match.group(2)}"
        # The same volume arrives two ways: as a docker named volume (Name set)
        # by default, or — when tt passes --host-volume for a pre-seeded one — as
        # a bind whose Source *ends* with the identical volume_id_ directory name.
        source = Path(str(mount.get("Source") or "")).name
        name = str(mount.get("Name") or "")
        candidate = name if name.startswith(_VOLUME_PREFIX) else source
        if candidate.startswith(_VOLUME_PREFIX) and not volume:
            volume = candidate
    return hf_repo, volume



# Conventional location for pre-seeded persistent volumes, overridable with
# paths.preloaded_volume_directory.
DEFAULT_PRELOADED_VOLUME_DIR = Path("data") / "tt-cache"


def preloaded_volume_root(
    model: ModelInfo, support: DeviceSupport | None, config: ConfigStore
) -> Path | None:
    """The volume root to hand the server, when it holds one for this model.

    tt-inference-server takes a *root* and appends
    `volume_id_<impl_id>-<model_name>-v<version>` itself, so the check has to
    reconstruct that exact name. Matching the model alone is not enough: impl and
    version are per device, and Qwen3-32B alone spans four different directory
    names across its boards. A near-miss would pass the flag, find nothing under
    the name the server wants, and quietly build a second copy inside the user's
    directory — worse than not using it at all.

    That exact-directory test is the whole gate. A machine with no such directory
    gets None, which leaves the server on its own docker volume: the default, and
    what every machine without a pre-seeded volume keeps doing.
    """
    if support is None or not (support.impl_id and support.version):
        return None
    configured = str(config.get("paths.preloaded_volume_directory") or "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / DEFAULT_PRELOADED_VOLUME_DIR
    )
    volume = root / f"volume_id_{support.impl_id}-{model.name}-v{support.version}"
    return root if volume.is_dir() else None


def _vllm_override_args(support: DeviceSupport) -> dict[str, object]:
    """vLLM flags the spec publishes but the server does not apply on its own.

    `tool_call_parser_name` and `reasoning_parser_name` are metadata: nothing in
    tt-inference-server reads them, so a container started without them cannot
    answer a `tool_choice: "auto"` request and coding agents get empty replies.
    Both flags are required together — vLLM refuses to start with auto tool choice
    and no parser — so an entry without a parser gets neither.

    Only meaningful on the vLLM branch of run_docker_server; media and forge
    containers never see --vllm-override-args.
    """
    if "vLLM" not in support.engines or not support.tool_call_parser:
        return {}
    overrides: dict[str, object] = {
        "enable-auto-tool-choice": True,
        "tool-call-parser": support.tool_call_parser,
    }
    if support.reasoning_parser:
        overrides["reasoning-parser"] = support.reasoning_parser
    return overrides


class InferenceServerBackend:
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
        if not (shutil.which("docker") or shutil.which("podman")):
            raise TTError(
                "No container runtime found.",
                why="tt-inference-server runs model backends in containers "
                "(docker or podman).",
                next_step="Install one — e.g. https://docs.docker.com/engine/install/ — "
                "then re-run. `tt update` can set this up on supported distros.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": "docker"},
            )
        if not model.cached and uses_host_weight_cache(model):
            self.output.warn(
                f"{model.name} is not in the local model cache; the server will "
                f"download it on startup (`tt model pull {model.name}` avoids the wait)."
            )

    def _env(self, model: ModelInfo) -> dict[str, str]:
        """Environment for run.py. Runner.stream() *replaces* the child
        environment rather than merging, so os.environ is inherited explicitly.

        MODEL_SOURCE is set for two reasons. It stops setup_host prompting "How do
        you want to provide a model? 1) Download from Hugging Face 2) Local
        folder", which it does whenever the variable is unset. And `noaction`
        tells setup_host the server provides its own weights, which is true for
        STT/TTS and forge containers: with `huggingface` the host downloads the
        whole repo (12 GB for distil-large-v3), mounts it readonly, and the
        container ignores it and downloads its own copy anyway.
        """
        env = dict(os.environ)
        source = "huggingface" if uses_host_weight_cache(model) else "noaction"
        env.setdefault("MODEL_SOURCE", source)
        return env

    def _python_for(self, entry: Path) -> str:
        venv_python = entry.parent.parent / "venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    def plan(
        self,
        model: ModelInfo,
        *,
        workflow: str = "server",
        device: str | None = None,
        port: int | None = None,
        force: bool = False,
        entry: Path | None = None,
    ) -> dict:
        """Everything `serve` would run, without running it.

        `entry` is the resolved run.py. When it is not given, the checkout is
        resolved without installing — a dry run must not fetch a git repo just to
        describe what it would do — and falls back to a placeholder.
        """
        self._check_servable(model)
        if entry is None:
            found = self.registry._resolve_or_none(TOOL)
            entry = Path(found[0]) if found else None
        support = self._check_supported(model, device, force=force)
        volume_root = preloaded_volume_root(model, support, self.config)
        settings = {
            "model": model.tt_model_id,
            "workflow": workflow,
            "device_requested": device,
            "device_sent": (support.serve_as or device) if support else device,
            "served_through": support.serve_as if support and support.serve_as else None,
            "engines": list(support.engines) if support else [],
            "status": support.status if support else None,
            "spec_docker_image": support.docker_image if support else None,
            "docker_image": None,
            "tool_call_parser": support.tool_call_parser if support else None,
            "reasoning_parser": support.reasoning_parser if support else None,
            # the spec's own value, applied by the server without a flag
            "spec_tt_config": support.override_tt_config if support else None,
            "forced_tt_config": None,
            # Exactly what _argv will pass: a pre-seeded volume replaces the HF
            # cache rather than layering with it, so reporting both would tell a
            # --json consumer the cache is in play when it is not.
            "host_hf_cache": str(hf_home_dir(self.config))
            if uses_host_weight_cache(model) and volume_root is None
            else None,
            "host_volume": str(volume_root) if volume_root else None,
            "port": port,
            # run.py reads SERVICE_PORT from the environment it inherits, falling
            # back to 8000 — so the effective default is not always 8000.
            "default_port": os.environ.get("SERVICE_PORT", "8000"),
            "default_port_from_env": "SERVICE_PORT" in os.environ,
            "installed": entry is not None,
        }
        forced = (support.serve_overrides or {}) if support else {}
        settings["docker_image"] = forced.get("docker_image")
        settings["forced_tt_config"] = forced.get("override_tt_config")
        settings["argv"] = self._argv(
            model,
            workflow=workflow,
            support=support,
            device=device,
            port=port,
            entry=entry,
        )
        return settings

    def _check_servable(self, model: ModelInfo) -> None:
        if not model.tt_model_id:
            raise TTError(
                f"{model.name} cannot be served by tt-inference-server.",
                why="The catalog entry has no tt-inference-server model id for it.",
                next_step="Run `tt model list` and pick a servable model "
                "(e.g. Llama-3.1-8B-Instruct).",
                exit_code=ExitCode.UNSUPPORTED,
            )
    def _check_supported(
        self, model: ModelInfo, device: str | None, *, force: bool
    ) -> DeviceSupport | None:
        support = model.devices.get(device) if device else None
        if device and support is None:
            # Passing it on means run.py rejects it as an unknown choice, and
            # everything keyed on the device — parsers, image override, a
            # pre-seeded volume — is silently skipped on the way there.
            known = ", ".join(model.devices) or "none"
            raise TTError(
                f"{model.name} has no support entry for {device}.",
                why=f"Devices for this model: {known}.",
                next_step=f"Pick one of those, drop --device to auto-detect, or run "
                f"`tt model info {model.name}`.",
                exit_code=ExitCode.USAGE,
            )
        if support is not None and not support.supported and not force:
            mark = support.unsupported
            raise TTError(
                f"{model.name} does not run on {device}.",
                why=f"{mark.details} (seen {mark.verified_on}"
                + (f", {mark.source}" if mark.source else "")
                + ")",
                next_step="Run `tt model list` for what does run on this board, "
                "or pass --force to try anyway.",
                exit_code=ExitCode.UNSUPPORTED,
            )
        return support

    def _argv(
        self,
        model: ModelInfo,
        *,
        workflow: str,
        support: DeviceSupport | None,
        device: str | None,
        port: int | None,
        entry: Path | None,
    ) -> list[str]:
        """The run.py command line. `entry` is None for a dry run, which has not
        resolved (and must not install) the checkout."""
        # Always prefer an explicit --device: run.py's own auto-detection is
        # broken on fresh checkouts (v0.18.0 ordering bug: parse_arguments()
        # calls infer_default_device(), which needs the bootstrap uv that
        # bootstrap_uv() only creates later in main()). The serve command
        # auto-fills `device` from our own tt-smi snapshot.
        argv = [self._python_for(entry), str(entry)] if entry else ["<run.py>"]
        argv += ["--model", model.tt_model_id, "--workflow", workflow]
        if workflow == "server":
            # the server workflow demands an explicit backend mode; containers
            # are the supported path (preflight already required docker/podman).
            # --local-server (bare-metal tt-metal dev setups) is out of scope
            # for the beta — run run.py directly for that.
            argv += ["--docker-server"]
            # tt serves models for local use, so the server runs unauthenticated.
            argv += ["--no-auth"]
        # A pre-seeded volume and the HF cache are alternatives, not layers:
        # setup_host.check_setup() returns on host_hf_cache before it ever looks
        # at host_model_volume_root, so passing both means the volume is never
        # consulted. When there is a volume for this model, it already holds the
        # weights (and the tt_metal_cache), so it wins.
        volume_root = preloaded_volume_root(model, support, self.config)
        if volume_root:
            argv += ["--host-volume", str(volume_root)]
        elif not uses_host_weight_cache(model):
            # The container fetches its own weights, so the readonly HF mount
            # setup_host would build is never read. Passing the flag would only
            # make it download the repo to the host first (see _env).
            pass
        else:
            # without --host-hf-cache, run.py downloads a fresh copy of the
            # weights into a docker volume and ignores the host HF cache entirely
            # — passing it makes the server reuse (and populate) the same cache
            # tt model pull uses. run.py accepts both <root>/models--* and
            # <root>/hub/models--*.
            argv += ["--host-hf-cache", str(hf_home_dir(self.config))]
        if support is not None:
            # A model reached through a device fallback has no spec under this
            # board's own name; asking for it would 404 on the spec lookup.
            argv += ["--device", support.serve_as or device]
            overrides = _vllm_override_args(support)
            if overrides:
                argv += ["--vllm-override-args", json.dumps(overrides)]
            # Only forced values: the server already folds the spec's own
            # override_tt_config into vllm_args, so passing that back would be a
            # no-op at best (workflows/model_spec.py, DeviceModelSpec.__post_init__).
            forced = support.serve_overrides or {}
            if forced.get("docker_image"):
                argv += ["--override-docker-image", str(forced["docker_image"])]
            if forced.get("override_tt_config"):
                argv += ["--override-tt-config", json.dumps(forced["override_tt_config"])]
        elif device:
            argv += ["--device", device]
        if port is not None:
            # run.py's SERVICE_PORT
            argv += ["--service-port", str(port)]
        return argv

    def serve(
        self,
        model: ModelInfo,
        *,
        workflow: str = "server",
        device: str | None = None,
        offline: bool = False,
        port: int | None = None,
        force: bool = False,
    ) -> int:
        self._check_servable(model)
        support = self._check_supported(model, device, force=force)
        entry = self.registry.ensure(TOOL, offline=offline)
        argv = self._argv(
            model,
            workflow=workflow,
            support=support,
            device=device,
            port=port,
            entry=Path(entry),
        )
        if support is not None and support.serve_as:
            self.output.status(
                f"{model.name} has no {device} spec — serving it through the "
                f"{support.serve_as} one."
            )
        self.output.status(
            f"Starting tt-inference-server ({workflow}) for {model.name} — Ctrl-C to stop."
        )
        # run.py assumes CWD is the repo root (it reads Path("VERSION") etc.);
        # the official installer's wrapper script cd's there too.
        return self.runner.stream(
            argv, env=self._env(model), cwd=str(Path(entry).parent), tool=TOOL
        )

    # -- artifact cleanup (`tt model rm`) --------------------------------------------
    def checkout_root(self) -> Path | None:
        """The managed checkout, or None when the tool was never installed (nothing
        to clean). Deliberately resolve-only: `tt model rm` must never *install*
        tt-inference-server just to find out there is nothing to remove."""
        try:
            found = self.registry._resolve_or_none(TOOL)
        except TTError:
            return None
        if found is None:
            return None
        root = Path(found[0]).parent
        return root if root.is_dir() else None

    def removable_artifacts(self, model: ModelInfo) -> list[Artifact]:
        """Per-model leftovers in the checkout, newest-cost-first.

        Deliberately NOT included: the docker image (derived from
        (version, tt_metal_commit) and shared by up to ~15 catalog models — the
        release spec has 41 images for 67 models), and `.workflow_venvs/`, which
        is per-*workflow* and shared across every model. Removing either to clean
        up one model would slow down or break the others."""
        root = self.checkout_root()
        if root is None:
            return []
        found: list[Artifact] = []
        # Log/spec filenames embed the model name between underscores
        # (vllm_<ts>_<model>_<device>_server.log), so an underscore-delimited glob
        # cannot confuse Llama-3.1-8B with Llama-3.1-8B-Instruct.
        logs_dir = root / "workflow_logs"
        if logs_dir.is_dir():
            for path in sorted(logs_dir.rglob(f"*_{model.name}_*")):
                found.append(Artifact("logs", path, _size_of(path)))
        # persistent_volume/volume_id_<impl>-<model>-v<version>/ — only created when
        # run.py is given --host-volume (tt serve passes --host-hf-cache instead), so
        # this is for checkouts someone also drove by hand.
        volumes = root / "persistent_volume"
        if volumes.is_dir():
            for path in sorted(volumes.glob(f"volume_id_*-{model.name}-v*")):
                found.append(Artifact("volume", path, _size_of(path)))
        return found

    def remove_artifacts(self, artifacts: Sequence[Artifact]) -> int:
        """Delete the given artifacts; returns the bytes reclaimed."""
        import shutil

        freed = 0
        for art in artifacts:
            try:
                if art.path.is_dir():
                    shutil.rmtree(art.path)
                else:
                    art.path.unlink()
            except PermissionError as exc:
                raise TTError(
                    f"Cannot remove {art.path}.",
                    why="It is owned by another user — container-created paths are "
                    "often owned by root.",
                    next_step=f"sudo rm -rf {art.path}",
                    exit_code=ExitCode.NEEDS_SUDO,
                ) from exc
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise TTError(
                    f"Cannot remove {art.path}.", why=str(exc), exit_code=ExitCode.ERROR
                ) from exc
            freed += art.size_bytes
        return freed

    # -- running containers (`tt model stop`) -----------------------------------------
    def container_runtime(self) -> str:
        """docker or podman, whichever is on PATH (preflight accepts either)."""
        found = shutil.which("docker") or shutil.which("podman")
        if not found:
            raise TTError(
                "No container runtime found.",
                why="tt-inference-server runs its model backends in containers, so "
                "stopping one needs docker or podman.",
                next_step="Install one — e.g. https://docs.docker.com/engine/install/",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": "docker"},
            )
        return found

    def running_containers(self) -> list[ServerContainer]:
        """Every running tt-inference-server container, with any identity we can read.

        One `ps` plus one `inspect` for all ids. A container we cannot identify is
        still returned (with identified=False) so the caller can report it instead of
        silently ignoring a running server."""
        runtime = self.container_runtime()
        listed = self.runner.capture(
            [runtime, "ps", "--filter", f"name=^{CONTAINER_PREFIX}", "--format", "{{.ID}}"],
            tool=runtime,
        )
        ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        if not ids:
            return []
        inspected = self.runner.capture(
            [runtime, "inspect", "--format", "{{json .}}", *ids], tool=runtime
        )
        containers: list[ServerContainer] = []
        for line in inspected.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:  # a runtime that formats differently — skip, don't fail
                continue
            hf_repo, volume = _identity_from_inspect(entry)
            containers.append(
                ServerContainer(
                    id=str(entry.get("Id") or "")[:12],
                    name=str(entry.get("Name") or "").lstrip("/"),
                    image=str((entry.get("Config") or {}).get("Image") or ""),
                    hf_repo=hf_repo,
                    volume=volume,
                )
            )
        return containers

    def stop_containers(self, containers: Sequence[ServerContainer]) -> None:
        """`docker stop` each one: SIGTERM plus grace, never kill — the server has to
        close the device mesh on its way out or the devices need a reset."""
        runtime = self.container_runtime()
        for container in containers:
            self.output.status(f"Stopping {container.name} ({container.id}) …")
            self.runner.capture([runtime, "stop", container.id], tool=runtime)
