# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Watching a serve go from the command to ready, whichever backend is serving.

A serve has two halves and they are owned by different things. The backend
prepares — bundle, image, weights — and starts a container; the *engine* then
boots inside it, which is usually ~95% of the wall clock. Only the first half
differs between tt-inference-server and tt-model, so only the first half is
delegated: a `Preparation` (see preparation.py) reports its own rows and names
the container it started, and from that point this module reads the container's
own log and classifies it with one tracker. Both paths therefore render from
the same code over the same bytes, which is what makes them identical rather
than merely similar. See `docs/serve-progress-contract.md`.

Every raw line, from both halves, is teed to one file under tt's own logs
directory.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from ...errors import ExitCode, TTError
from ...launchers import discovery
from ...output import OutputManager
from ...progress import Checklist, PhaseTracker, WeightsProgress, format_bytes, phases_for
from ...tools.runner import LineSplitter, Runner
from .preparation import Preparation

#: A cold boot JIT-compiles kernels and can genuinely take the best part of an
#: hour on a large model; the wait is bounded so a hung device still ends.
READY_TIMEOUT_ENV = "TT_SERVE_READY_TIMEOUT"
DEFAULT_READY_TIMEOUT_S = 60 * 60
_PROBE_INTERVAL_S = 3.0
_PROBE_TIMEOUT_S = 2.0
_LIVENESS_INTERVAL_S = 10.0
_TAIL_POLL_S = 0.25
_CHUNK = 8192
#: Bounds one pass's read of the container log, so a burst of output cannot
#: starve the health probe while still draining far faster than a boot writes.
_MAX_CHUNKS_PER_PASS = 64
#: How often to ask docker whether a named-but-not-yet-created container exists.
_EXISTS_INTERVAL_S = 1.0
#: How often to re-measure the weights cache. It walks a directory, and a
#: download worth watching lasts minutes.
_WEIGH_INTERVAL_S = 1.5
#: After the endpoint answers, how long to let run.py notice that for itself and
#: exit. It polls the same endpoint, so this is a formality — but returning
#: while it is mid-write would break its pipe for no reason.
_EXIT_GRACE_S = 20.0

@dataclass(frozen=True)
class BootResult:
    """The outcome of one watched serve. `ready` is False only when tt could not
    follow the boot at all — a failure raises instead."""

    ready: bool
    endpoint: str
    container: str | None
    raw_log: Path
    elapsed: float


def endpoint_for(prepare: Preparation, port: int) -> str:
    """Where to look for the server: what the container published, if the
    backend chose the port itself, else the one we asked for."""
    return f"http://127.0.0.1:{prepare.endpoint_port or port}/v1"


def ready_timeout_s() -> int:
    """How long to wait for the server, from the environment."""
    raw = os.environ.get(READY_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_READY_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise TTError(
            f"{READY_TIMEOUT_ENV}={raw!r} is not a positive number of seconds.",
            next_step=f"Unset it, or set it to a whole number (default {DEFAULT_READY_TIMEOUT_S}).",
            exit_code=ExitCode.USAGE,
        )
    return value


def raw_log_path(logs_dir: Path, model_name: str) -> Path:
    """Where this serve's full output is teed, one file per run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return logs_dir / f"serve-{model_name}-{stamp}.log"


def watch_serve(
    *,
    runner: Runner,
    output: OutputManager,
    prepare: Preparation,
    argv: Sequence[str],
    env: dict[str, str],
    cwd: str | None,
    tool: str,
    model_name: str,
    engines: Sequence[str],
    port: int,
    raw_log: Path,
    runtime: str,
    weights_cache: Path | None = None,
    hf_token: str | None = None,
) -> BootResult:
    """Run the serve, render it as a checklist, and wait for the server.

    Returns once the endpoint answers. Raises TTError if the boot fails or the
    wait runs out; KeyboardInterrupt is left to the caller, whose container is
    still booting in the background either way.
    """
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + ready_timeout_s()
    # Built at the handover, not here: a backend may only work out which stack
    # it is launching while it prepares (a bundle's manifest arrives with it).
    trackers: dict[str, PhaseTracker] = {}

    with raw_log.open("w", buffering=1) as sink, Checklist(output) as view:
        def keep(line: str) -> None:
            """Every line, from both halves: to the tee'd file always, and to
            the screen under --verbose, where the checklist prints plainly and
            the two interleave in the order things happened."""
            sink.write(line + "\n")
            output.raw(line)

        view.begin(prepare.label, placeholder=True)
        proc = runner.popen_piped(argv, env=env, cwd=cwd, tool=tool)
        outcome = _watch(
            view, prepare, trackers, engines, keep,
            proc=proc, port=port, runner=runner, runtime=runtime, deadline=deadline,
            weights=WeightsProgress(weights_cache, token=hf_token) if weights_cache else None,
        )
        base_url = endpoint_for(prepare, port)
        container = trackers.get("container")
        if outcome == "ready":
            _apply(view, container.finish() if container else prepare.finish())
            # The one row the endpoint proves rather than a log: /v1/models
            # answered, which is what makes `tt launch` work straight after.
            view.instant("endpoint answering")
        elif outcome == "no-container":
            # The backend named no container. Either it started none, or it
            # says so in a way we no longer recognise — so there is nothing to
            # follow and nothing we can honestly claim about readiness. The
            # serve itself is unaffected.
            view.note("no container to follow — see the raw output")
            return BootResult(
                False, endpoint_for(prepare, port), None, raw_log,
                time.monotonic() - started,
            )
        else:
            view.fail()
            raise _boot_error(
                model_name,
                container if container is not None and container.evidence() else prepare,
                raw_log=raw_log,
                exited=outcome != "timeout",
                deadline_s=int(deadline - started),
                verbose=output.verbose,
                diagnosis=getattr(prepare, "diagnosis", None),
            )

    return BootResult(
        True, base_url, prepare.container, raw_log, time.monotonic() - started
    )


# -- the watch loop ---------------------------------------------------------------------
def _watch(
    view: Checklist,
    prepare: Preparation,
    trackers: dict[str, PhaseTracker],
    engines: Sequence[str],
    keep: Callable[[str], None],
    *,
    proc: subprocess.Popen,
    port: int,
    runner: Runner,
    runtime: str,
    deadline: float,
    weights: WeightsProgress | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Drain the backend and the container's log together until one settles it.

    Returns "ready", "timeout", "no-container" or "failed". The endpoint is the
    authority on ready — the logs only decide which row is active — and the
    container going away is the authority on failure.
    """
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    backend = LineSplitter()
    tail = LineSplitter()
    state: dict[str, object] = {
        "logs": None, "gone": False, "next_look": 0.0, "next_weigh": 0.0,
    }

    def pump_backend(chunk: bytes) -> None:
        """The backend's own output. It drives the preparation rows until the
        container log takes over, and after that is kept only for the tail a
        failure quotes — a backend still polling /health is not a boot step."""
        for line in backend.feed(chunk):
            keep(line)
            events = prepare.feed(line)
            if state["logs"] is None:
                _apply(view, events)

    def follow_container() -> None:
        """Hand the checklist over to the container's own log, once there is one.

        Existence is checked first because a backend can name the container
        before it has created one: run.py echoes the whole `docker run` command,
        `--name` and all, and only then executes it. Following too early gets
        "No such container", which reads as a container that has already died.
        """
        # `gone` as well as `logs`: once the stream has ended, following again
        # would replay the whole boot and walk the checklist through it twice.
        if state["logs"] is not None or state["gone"] or prepare.container is None:
            return
        now = time.monotonic()
        if now < state["next_look"]:
            return
        state["next_look"] = now + _EXISTS_INTERVAL_S
        exists = runner.capture(
            [runtime, "inspect", "--format", "{{.Id}}", str(prepare.container)],
            tool=runtime, check=False,
        )
        if exists.returncode != 0:
            return
        if prepare.endpoint_port is None:
            prepare.endpoint_port = min(
                _inspect_ports(runner, runtime, str(prepare.container))[1], default=None
            )
        _apply(view, prepare.finish())
        trackers["container"] = PhaseTracker(phases_for(prepare.engines() or engines))
        view.begin("waiting for the model server", placeholder=True)
        try:
            logs = runner.popen_piped(
                [runtime, "logs", "--follow", prepare.container], tool=runtime
            )
        except TTError:  # no runtime, or it will not start — probe only
            state["gone"] = True
            return
        state["logs"] = logs
        selector.register(logs.stdout, selectors.EVENT_READ)

    def read_logs(stream) -> None:
        """One bounded read of the container's log, only when it has something.

        Never called speculatively: this is a pipe, so `read1` blocks until a
        byte arrives, and a container that has finished booting and gone quiet
        would hang the loop exactly when the health probe matters most.
        """
        for _ in range(_MAX_CHUNKS_PER_PASS):
            chunk = stream.read1(_CHUNK)
            if not chunk:
                # `docker logs --follow` ends when the container does, and
                # `docker run --rm` means a crashed boot is removed rather than
                # left exited — so this, not an inspect, is how we learn.
                state["gone"] = True
                selector.unregister(stream)
                stream.close()
                state["logs"] = None
                return
            for line in tail.feed(chunk):
                keep(line)
                _apply(view, trackers["container"].feed(line))
            if len(chunk) < _CHUNK:
                return  # drained for now; let the loop breathe

    def show_weights() -> None:
        """Measure the weights download, since neither backend reports it.

        Rate-limited because it walks a directory; the row it writes to is
        whichever step the backend said was fetching, so it stops as soon as
        that step does.
        """
        if weights is None:
            return
        weights.track(prepare.weights_repo)
        now = time.monotonic()
        if prepare.weights_repo is None or now < state["next_weigh"]:
            return
        state["next_weigh"] = now + _WEIGH_INTERVAL_S
        sample = weights.sample()
        if sample is None:
            return
        done, total = sample
        if total > 0:
            view.progress(done, total, is_bytes=True)
        elif done:
            # No total from the Hub: say what has landed rather than guess at a
            # percentage of something we do not know.
            view.detail(format_bytes(done))

    def drain() -> None:
        """Read out everything the backend said on its way out, before judging
        it. Deciding first and reading afterwards loses the container it names
        last of all, and truncates the saved log at whatever we had seen."""
        while True:
            chunk = proc.stdout.read1(_CHUNK)
            if not chunk:
                break
            pump_backend(chunk)
        for line in backend.flush():
            keep(line)
            prepare.feed(line)
        if proc.returncode != 0:
            # A backend that failed started nothing of ours.
            return
        if prepare.container is None:
            prepare.resolve_container(runner, runtime)
        follow_container()

    def finish(outcome: str) -> str:
        # Whatever the container has already written and we have not read: the
        # endpoint answering must not cut the saved log short, and a bounded
        # non-blocking sweep cannot hang on a container that is still talking.
        for _ in range(_MAX_CHUNKS_PER_PASS):
            ready = [
                key for key, _ in selector.select(timeout=0)
                if key.fileobj is not proc.stdout
            ]
            if not ready:
                break
            for key in ready:
                read_logs(key.fileobj)
        container = trackers.get("container")
        for line in tail.flush():
            keep(line)
            if container is not None:
                _apply(view, container.feed(line))
        return outcome

    drained = False
    ready_at: float | None = None
    next_probe = 0.0
    try:
        while True:
            # One select for both: the backend's pipe and the container's log.
            # It is also what paces the loop once neither has anything to say.
            for key, _ in selector.select(timeout=_TAIL_POLL_S):
                if key.fileobj is proc.stdout:
                    chunk = proc.stdout.read1(_CHUNK)
                    if not chunk:
                        selector.unregister(proc.stdout)
                        continue
                    pump_backend(chunk)
                else:
                    read_logs(key.fileobj)

            follow_container()
            show_weights()

            if proc.poll() is not None and not drained:
                drain()
                drained = True

            now = time.monotonic()
            # A backend that gave up is a failure, whatever the port says: with
            # another model already serving there, the probe would answer for it.
            if drained and proc.returncode != 0 and ready_at is None:
                return finish("failed")

            if ready_at is None and prepare.container and now >= next_probe:
                next_probe = now + _PROBE_INTERVAL_S
                if discovery.probe(
                    endpoint_for(prepare, port), timeout_s=_PROBE_TIMEOUT_S
                ) and _serves_on(runner, runtime, str(prepare.container), prepare.endpoint_port or port):
                    # Not returned yet unless the backend is done: it may poll
                    # the same endpoint and be about to exit on its own, and
                    # cutting its pipe mid-write would break a serve that has
                    # already succeeded.
                    ready_at = now
            if ready_at is not None and (drained or now - ready_at > _EXIT_GRACE_S):
                return finish("ready")
            if drained and prepare.container is None:
                return finish("no-container")
            if state["gone"] and ready_at is None:
                return finish("failed")
            if now >= deadline:
                return finish("timeout")
    finally:
        selector.close()
        logs = state["logs"]
        if logs is not None:
            _release(logs)
            logs.terminate()
        _release(proc)


def _serves_on(runner: Runner, runtime: str, container: str, port: int) -> bool:
    """Whether `container` is running and is what listens on host `port`.

    What makes an answering endpoint this serve's rather than a neighbour's. A
    container on the host network publishes nothing, so for one of those
    running is all docker can vouch for.
    """
    running, ports = _inspect_ports(runner, runtime, container)
    return running and (not ports or port in ports)


def _inspect_ports(runner: Runner, runtime: str, container: str) -> tuple[bool, set[int]]:
    """(running, host ports it publishes). Not running when docker cannot say."""
    listed = runner.capture(
        [runtime, "inspect", "--format",
         "{{.State.Running}} {{json .NetworkSettings.Ports}}", container],
        tool=runtime, check=False,
    )
    running, _, raw = listed.stdout.strip().partition(" ")
    if listed.returncode != 0 or running != "true":
        return False, set()
    try:
        bindings = json.loads(raw) or {}
    except ValueError:
        bindings = {}
    return True, {
        int(binding["HostPort"])
        for published in bindings.values() if isinstance(published, list)
        for binding in published
        if isinstance(binding, dict) and str(binding.get("HostPort", "")).isdigit()
    }


def port_is_free(port: int) -> bool:
    """Whether this host port can be bound right now, the way docker will bind
    it (the wildcard address). A snapshot: something can still take it before
    the container starts, and then docker's own error stands."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def pick_free_port(preferred: int, *, attempts: int = 100) -> int | None:
    """The first free port at or above `preferred`, as tt-model picks its own."""
    for candidate in range(preferred, min(preferred + attempts, 65536)):
        if port_is_free(candidate):
            return candidate
    return None


def _release(proc: subprocess.Popen) -> None:
    """Stop reading, without stopping the tool.

    run.py outlives a successful watch by a few seconds and the container
    outlives them both, so this never signals anything: it closes our end of the
    pipe and reaps the process if it has already finished.
    """
    if proc.stdout is not None:
        proc.stdout.close()
    proc.poll()


def _apply(view: Checklist, events) -> None:
    for event in events:
        if event.kind == "start":
            view.begin(event.label)
        elif event.kind == "done":
            view.done(event.label, event.detail)
        elif event.kind == "progress":
            view.progress(event.done, event.total, is_bytes=event.is_bytes)
        elif event.kind == "detail" and event.detail:
            view.detail(event.detail)


# -- failure ---------------------------------------------------------------------------
_HELD_RE = re.compile(
    r"Sysmem mapped at unexpected NOC address|CHIP_IN_USE|stale process holding"
    # tt-model refusing up front: every chip is mounted by a running container.
    r"|tt device\(s\) are free"
)
_FLAG_RE = re.compile(r"unrecognized arguments|error: argument|no such option")
_RESET_RE = re.compile(
    r"Try resetting the board|Timed out while waiting for active ethernet core"
)


def _boot_error(
    model_name: str,
    evidence_from,
    *,
    raw_log: Path,
    exited: bool,
    deadline_s: int,
    cause: TTError | None = None,
    verbose: bool = False,
    diagnosis: dict | None = None,
) -> TTError:
    """Say why the serve did not reach ready, quoting the line that explains it.

    Never a dump of the last N lines: the full output is on disk and named in the
    error, and the one line worth reading has usually scrolled past hundreds of
    traceback lines by the time the container exits.
    """
    evidence = list(evidence_from.evidence())
    joined = "\n".join(evidence)
    # No "see the log" in next_step: the error panel prints details["log_path"]
    # under every error, and saying it twice reads as noise.
    if diagnosis:
        # The backend worked out why itself; it knows things we do not.
        return TTError(
            f"{model_name} could not start: "
            f"{_text(diagnosis.get('cause')) or 'the boot failed'}.",
            why=_text(diagnosis.get("detail")) or _text(diagnosis.get("evidence")) or None,
            next_step="  ".join(str(a) for a in diagnosis.get("actions") or ()) or None,
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": str(raw_log)},
        )
    if _HELD_RE.search(joined):
        return TTError(
            f"{model_name} could not start: the Tenstorrent device is already in use.",
            why="Another process has the card open — usually another served model, "
            "or a stale one that did not release it.",
            next_step="`tt model ps` to see what is running, `tt model stop <model>` "
            "to free the card, then re-run.",
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": str(raw_log)},
        )
    if _FLAG_RE.search(joined):
        return TTError(
            f"{model_name} could not start: the engine rejected one of the arguments.",
            why="Everything `tt serve` does not recognise is forwarded to the "
            "engine, and it refused one of them.",
            next_step="Drop it, or check `tt serve --help` for the flag you meant — "
            "tt's own options must come before the model name.",
            exit_code=ExitCode.USAGE,
            details={"log_path": str(raw_log)},
        )
    if _RESET_RE.search(joined):
        return TTError(
            f"{model_name} could not start: the Tenstorrent device needs a reset.",
            why="tt-metal could not bring the mesh up — an ethernet core did not "
            "come back, which is what a board left in a bad state by an earlier "
            "run looks like.",
            next_step="`tt device reset`, then re-run. If it repeats, check "
            "`tt model ps` for another server still holding the card.",
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": str(raw_log)},
        )
    if "Address already in use" in joined:
        return TTError(
            f"{model_name} could not start: its port is already taken.",
            why="Something else was listening when the server tried to bind.",
            next_step=f"Serve it elsewhere: `tt serve {model_name} --port <other>`.",
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": str(raw_log)},
        )
    if exited:
        return TTError(
            f"{model_name} stopped before the server was ready.",
            # The log first: `cause` only ever says "exited with status 1", while
            # the line the tool printed on its way out names the actual reason.
            why=_last_meaningful(evidence) or (cause.what if cause else None),
            next_step="The saved output has the full traceback."
            if verbose
            else "Re-run with `tt --verbose serve` to watch the server's own output "
            "as it happens.",
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": str(raw_log)},
        )
    return TTError(
        f"{model_name} did not report ready within {deadline_s // 60} minutes.",
        why="The container is still running — a cold boot compiles kernels and can "
        "simply be slow.",
        next_step=f"Keep watching with `tt model logs {model_name} --follow`, raise the "
        f"bound with {READY_TIMEOUT_ENV}=<seconds>, or stop it with "
        f"`tt model stop {model_name}`.",
        exit_code=ExitCode.TOOL_FAILED,
        details={"log_path": str(raw_log)},
    )


_NOISE_PREFIXES = ("^", 'File "', "return ", "self.", "raise ")


def _last_meaningful(evidence: Sequence[str]) -> str | None:
    """The last line that reads like a cause, skipping traceback scaffolding."""
    for line in reversed([ln.strip() for ln in evidence if ln.strip()]):
        if not line.startswith(_NOISE_PREFIXES) and "Traceback" not in line:
            return line[:300]
    return None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
