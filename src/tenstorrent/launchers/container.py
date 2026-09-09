# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Base for clients tt runs as a container.

These are services, not terminal clients: tt pulls the image, starts the
container detached and prints its URL. Starting one *is* the whole action, so it
never happens without consent. Subclasses supply the image, names and
environment; the pull/adopt/drift mechanics are identical and live here.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from contextlib import suppress
from urllib.parse import urlsplit, urlunsplit

from ..errors import ExitCode, TTError
from .base import LaunchEnv, LaunchOptions, Preparation, RunningModel

# Inside a container, a loopback endpoint is the container itself. Docker and
# podman both map this name to the host with --add-host.
_HOST_ALIAS = "host.docker.internal"
_LOOPBACK = frozenset({"127.0.0.1", "0.0.0.0", "localhost", "::1"})


def container_base_url(base_url: str) -> str:
    """`base_url` as seen from inside a container."""
    parts = urlsplit(base_url)
    if parts.hostname not in _LOOPBACK:
        return base_url
    host = _HOST_ALIAS + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def port_is_free(port: int) -> bool:
    """Whether the host port can still be bound.

    Checked before the pull, not after: docker fails the `run` on a taken port,
    but only once several gigabytes have already been downloaded.
    """
    with socket.socket() as probe:
        # Allow a port in TIME_WAIT, which nothing is actually listening on.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def wait_until_ready(url: str, timeout_s: float) -> bool:
    """Poll until the service answers at all. Any reply counts, including a
    redirect to a login page — the question is whether it is up, not what it says."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return response.status < 500
        except urllib.error.HTTPError as err:
            return err.code < 500
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


class ContainerLauncher:
    # -- subclass contract -------------------------------------------------------
    id: str
    image: str
    container: str
    volume: str
    volume_path: str  # where the volume mounts inside the container
    container_port: int
    base_url_env: str  # the env var carrying the model endpoint
    extra_run_args: tuple[str, ...] = ()

    def container_env(self, model: RunningModel, inner_url: str) -> dict:
        """Environment the container needs to reach the model."""
        raise NotImplementedError

    # -- shared for every container client ---------------------------------------
    binaries = ("docker", "podman")
    install_hint = (
        "Install docker or podman — https://docs.docker.com/engine/install/ — "
        "then re-run."
    )
    # These clients only chat, so any model they can list is usable.
    requires_tool_calling = False
    hands_over_terminal = False
    # Generous: a first start unpacks the image and initialises a database.
    ready_timeout_s = 120.0

    def target(self) -> str:
        return f"container {self.container}"

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        inner = container_base_url(model.base_url)
        state, configured_for, published = (
            self._existing(executable, runner) if executable else ("unknown", None, None)
        )
        # An existing container's published port was fixed when it was created, so
        # --web-port cannot move it. Report where it actually answers.
        web_port = published if state in ("running", "stopped") else options.web_port
        url = f"http://localhost:{web_port}"
        if state == "running":
            self._check_endpoint(configured_for, inner)
            return Preparation(
                rows={
                    "container": f"{self.container} (already running)",
                    "endpoint_in_container": inner,
                },
                url=url,
            )
        exe = executable or self.binaries[0]
        if state == "stopped":
            self._check_endpoint(configured_for, inner)
            # Its port may have been taken by something else while it was down.
            self._check_port(web_port, existing=True)
            steps = [[exe, "start", self.container]]
            consent = f"Start the existing {self.container} container"
        else:
            self._check_port(options.web_port)
            steps = [
                [exe, "pull", self.image],
                self._run_argv(exe, model, inner, options.web_port),
            ]
            consent = (
                f"Pull {self.image} (a few GB) and run it as {self.container} "
                f"on port {options.web_port}"
            )
        return Preparation(
            rows={
                # "unknown" only happens on a dry run with no container runtime,
                # where the bare name is the honest answer.
                "container": self.container
                if state == "unknown"
                else f"{self.container} ({state})",
                "image": self.image,
                "endpoint_in_container": inner,
            },
            steps=steps,
            consent=consent,
            url=url,
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        for step in prep.steps:
            # Streamed: a multi-gigabyte pull must show its own progress.
            env.runner.stream(step, tool=self.id)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        assert prep.url is not None
        env.output.status(f"Waiting for {self.container} to answer at {prep.url} …")
        if wait_until_ready(prep.url, self.ready_timeout_s):
            env.output.status(f"{self.container} is ready at {prep.url}")
            return
        # Not an error: the container is up and may simply be slow. Saying so beats
        # printing a URL that answers nothing.
        env.output.warn(
            f"{self.container} did not answer within "
            f"{int(self.ready_timeout_s)}s — it may still be starting. "
            f"Check with: docker logs -f {self.container}"
        )

    def stop(self, env: LaunchEnv) -> None:
        # stop, never kill or remove: the volume holds the user's chats.
        env.runner.stream([env.executable, "stop", self.container], tool=self.id)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        if executable is None or self.container_state(executable, runner) == "absent":
            return None
        return (
            f"remove the {self.container} container "
            f"(its data volume {self.volume} is kept)"
        )

    def disconnect(self, env: LaunchEnv) -> None:
        # Not `rm -f`, and never `volume rm`: the container is tt's, the data is the
        # user's. A stopped container is removable as-is.
        if self.container_state(env.executable, env.runner) == "running":
            self.stop(env)
        env.runner.stream([env.executable, "rm", self.container], tool=self.id)

    def container_state(self, executable: str | None, runner) -> str:
        """One of running / stopped / absent / unknown, for `tt launch list`.

        "unknown" means no container runtime was resolved, so nothing was asked."""
        if executable is None:
            return "unknown"
        return self._existing(executable, runner)[0]

    def _check_port(self, web_port: int | None, *, existing: bool = False) -> None:
        if web_port is None or port_is_free(web_port):
            return
        if existing:
            # --web-port cannot help here: the binding is baked into the container.
            why = f"{self.container} publishes {web_port} and cannot be moved."
            next_step = (
                f"Free the port, or recreate it elsewhere: docker rm {self.container} "
                f"&& tt launch {self.id} --web-port {web_port + 1}"
            )
        else:
            why = (
                f"{self.container} would publish {web_port}, and docker would fail "
                "the run — after pulling the image."
            )
            next_step = (
                f"Free the port, or pick another: tt launch {self.id} "
                f"--web-port {web_port + 1}"
            )
        raise TTError(
            f"Host port {web_port} is already in use.",
            why=why,
            next_step=next_step,
            exit_code=ExitCode.CONFIG,
            details={"web_port": web_port, "container_exists": existing},
        )

    def _existing(self, executable: str, runner) -> tuple[str, str | None, int | None]:
        """("running"|"stopped"|"absent", endpoint it was created for, host port)."""
        result = runner.capture(
            [executable, "inspect", "--format", "{{json .}}", self.container],
            check=False,
        )
        if result.returncode != 0:
            return "absent", None, None
        try:
            doc = json.loads(result.stdout)
        except json.JSONDecodeError:
            return "absent", None, None
        configured_for = None
        for item in (doc.get("Config") or {}).get("Env") or []:
            if item.startswith(f"{self.base_url_env}="):
                configured_for = item.split("=", 1)[1]
        running = bool((doc.get("State") or {}).get("Running"))
        # PortBindings, not NetworkSettings.Ports: the latter is empty while stopped.
        bindings = (doc.get("HostConfig") or {}).get("PortBindings") or {}
        published = None
        for binding in bindings.get(f"{self.container_port}/tcp") or []:
            with suppress(TypeError, ValueError):
                published = int(binding.get("HostPort"))
        return ("running" if running else "stopped"), configured_for, published

    def _check_endpoint(self, configured_for: str | None, wanted: str) -> None:
        """An existing container has its endpoint baked into its env, so reusing one
        built for a different server would quietly serve the wrong model."""
        if configured_for is None or configured_for == wanted:
            return
        raise TTError(
            f"{self.container} is pointed at a different model server.",
            why=f"It was created for {configured_for}, not {wanted}.",
            next_step=f"Remove it and re-run: docker rm -f {self.container}",
            exit_code=ExitCode.CONFIG,
            details={"container": self.container, "configured_for": configured_for},
        )

    def _run_argv(
        self, executable: str, model: RunningModel, inner_url: str, web_port: int
    ) -> list[str]:
        argv = [
            executable, "run", "-d",
            "--name", self.container,
            "-p", f"{web_port}:{self.container_port}",
            "--add-host", f"{_HOST_ALIAS}:host-gateway",
            "-v", f"{self.volume}:{self.volume_path}",
        ]
        for key, value in self.container_env(model, inner_url).items():
            argv += ["-e", f"{key}={value}"]
        argv += list(self.extra_run_args)
        argv.append(self.image)
        return argv
