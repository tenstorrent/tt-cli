# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Watching a tt-inference-server model serve.

The boot happens in two places at once, so both are read at once. run.py does
the host work on its own stdout — validation, `docker pull`, `hf download`, the
container launch — while the container's own boot goes to a separate log file
whose path run.py prints as it starts it. One loop therefore drains run.py's
pipe and tails that file together, feeding the host phase template and then the
container one, until `GET /v1/models` answers.

Every raw line, from both stages, is teed to one file under tt's own logs
directory — the checklist replaces the wall of output on screen.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from ...errors import ExitCode, TTError
from ...launchers import discovery
from ...output import OutputManager
from ...progress import HOST_PHASES, Checklist, PhaseTracker, phases_for
from ...tools.runner import LineSplitter, Runner

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
#: After the endpoint answers, how long to let run.py notice that for itself and
#: exit. It polls the same endpoint, so this is a formality — but returning
#: while it is mid-write would break its pipe for no reason.
_EXIT_GRACE_S = 20.0

# run.py's hand-off lines. Without the log path there are no container rows to show — the serve still works, and the wait falls
# back to polling the endpoint.
_DOCKER_LOG_RE = re.compile(r"docker container with log file:\s*(\S+)")
_CONTAINER_ID_RE = re.compile(r"Created Docker container ID:\s*(\S+)")
@dataclass(frozen=True)
class BootResult:
    """The outcome of one watched serve. `ready` is False only when tt could not
    follow the boot at all — a failure raises instead."""

    ready: bool
    endpoint: str
    container: str | None
    raw_log: Path
    elapsed: float


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
    argv: Sequence[str],
    env: dict[str, str],
    cwd: str,
    tool: str,
    model_name: str,
    engines: Sequence[str],
    port: int,
    raw_log: Path,
    runtime: str,
) -> BootResult:
    """Run the serve, render it as a checklist, and wait for the server.

    Returns once the endpoint answers. Raises TTError if the boot fails or the
    wait runs out; KeyboardInterrupt is left to the caller, whose container is
    still booting in the background either way.
    """
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + ready_timeout_s()
    base_url = f"http://127.0.0.1:{port}/v1"
    host = PhaseTracker(HOST_PHASES)
    container = PhaseTracker(phases_for(engines))
    handoff: dict[str, str] = {}

    with raw_log.open("w", buffering=1) as sink, Checklist(output) as view:
        def keep(line: str) -> None:
            """Every line, from both sources: to the tee'd file always, and to
            the screen under --verbose, where the checklist prints plainly and
            the two interleave in the order things happened."""
            sink.write(line + "\n")
            output.raw(line)

        view.begin("starting tt-inference-server", placeholder=True)
        proc = runner.popen_piped(argv, env=env, cwd=cwd, tool=tool)
        outcome = _watch(
            view, host, container, keep,
            proc=proc, handoff=handoff, base_url=base_url,
            runner=runner, runtime=runtime, deadline=deadline,
        )
        if outcome == "ready":
            _apply(view, container.finish() if handoff else host.finish())
            # The one row the endpoint proves rather than the log: /v1/models
            # answered, which is what makes `tt launch` work straight after.
            view.instant("endpoint answering")
        elif outcome == "no-container":
            # run.py named neither the container log nor its id. Either it
            # started nothing, or upstream reworded both lines — so there is
            # nothing to tail and nothing we can honestly claim about
            # readiness. The serve itself is unaffected.
            view.note("no container to follow — see the raw output")
            return BootResult(False, base_url, None, raw_log, time.monotonic() - started)
        else:
            view.fail()
            raise _boot_error(
                model_name,
                container if handoff else host,
                raw_log=raw_log,
                exited=outcome != "timeout",
                deadline_s=int(deadline - started),
                verbose=output.verbose,
            )

    return BootResult(True, base_url, handoff.get("id"), raw_log, time.monotonic() - started)


# -- the watch loop ---------------------------------------------------------------------
def _watch(
    view: Checklist,
    host: PhaseTracker,
    container: PhaseTracker,
    keep: Callable[[str], None],
    *,
    proc: subprocess.Popen,
    handoff: dict[str, str],
    base_url: str,
    runner: Runner,
    runtime: str,
    deadline: float,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Drain run.py and the container log together until one of them settles it.

    Returns "ready", "timeout", "no-container" or "failed". The endpoint is the
    authority on ready — the logs only decide which row is active — and run.py's
    own exit status is the authority on failure, since it is the one that
    retries a boot and tears the container down.
    """
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    liveness = _Liveness(runner, runtime)
    pipe = LineSplitter()
    tail = LineSplitter()
    state: dict[str, object] = {"handle": None}

    def pump_pipe(chunk: bytes) -> None:
        """run.py's own output. It drives the host rows until the container log
        takes over, and after that is kept only for the tail a failure quotes —
        its readiness polling is not a boot step."""
        for line in pipe.feed(chunk):
            keep(line)
            _note_handoff(line, handoff)
            events = host.feed(line)
            if state["handle"] is None:
                _apply(view, events)

    def open_log() -> None:
        """Hand the checklist over to the container's own log, once it exists."""
        if state["handle"] is not None or "log" not in handoff:
            return
        path = Path(handoff["log"])
        if path.is_file():
            _apply(view, host.finish())
            view.begin("waiting for the model server", placeholder=True)
            state["handle"] = path.open("rb")

    def pump_log() -> bool:
        """The container's log, in bounded steps so a burst of it cannot starve
        the health probe. True while there may be more to read."""
        handle = state["handle"]
        if handle is None:
            return False
        for _ in range(_MAX_CHUNKS_PER_PASS):
            chunk = handle.read1(_CHUNK)
            if not chunk:
                return False
            for line in tail.feed(chunk):
                keep(line)
                _apply(view, container.feed(line))
        return True

    def drain() -> None:
        """Read out everything the tool said on its way out, before judging it.

        Only once it has exited, where reading to EOF cannot block. Deciding
        first and reading afterwards loses the container id — run.py prints it
        last of all — and truncates the saved log at whatever we happened to
        have seen.
        """
        while True:
            chunk = proc.stdout.read1(_CHUNK)
            if not chunk:
                break
            pump_pipe(chunk)
        for line in pipe.flush():
            keep(line)
            _note_handoff(line, handoff)
            host.feed(line)
        open_log()
        while pump_log():
            pass

    def finish(outcome: str) -> str:
        for line in tail.flush():
            keep(line)
            _apply(view, container.feed(line))
        return outcome

    drained = False
    ready_at: float | None = None
    next_probe = next_liveness = 0.0
    try:
        while True:
            if not drained:
                for _ in selector.select(timeout=_TAIL_POLL_S):
                    chunk = proc.stdout.read1(_CHUNK)
                    if not chunk:
                        selector.unregister(proc.stdout)
                        break
                    pump_pipe(chunk)
            open_log()
            busy = pump_log()

            if proc.poll() is not None and not drained:
                drain()
                drained = True

            now = time.monotonic()
            # Not before the container is ours to talk about: something already
            # listening on the port would otherwise be reported as this serve,
            # ready in milliseconds, with no container to name.
            if ready_at is None and (handoff or drained) and now >= next_probe:
                next_probe = now + _PROBE_INTERVAL_S
                if discovery.probe(base_url, timeout_s=_PROBE_TIMEOUT_S):
                    # Not returned yet unless run.py is done: it polls the same
                    # endpoint and is about to exit on its own, and cutting its
                    # pipe mid-write would break a serve that has succeeded.
                    ready_at = now
            if ready_at is not None and (drained or now - ready_at > _EXIT_GRACE_S):
                return finish("ready")
            if drained and proc.returncode != 0:
                return finish("failed")
            if drained and not handoff:
                return finish("no-container")
            if state["handle"] is not None and now >= next_liveness:
                next_liveness = now + _LIVENESS_INTERVAL_S
                if not liveness.alive(handoff.get("id")):
                    return finish("failed")
            if now >= deadline:
                return finish("timeout")
            if drained and not busy:
                # Nothing left to block on: the pipe is closed and the log is
                # caught up, so this is the only thing pacing the loop.
                sleep(_TAIL_POLL_S)
    finally:
        selector.close()
        handle = state["handle"]
        if handle is not None:
            handle.close()
        _release(proc)


def _note_handoff(line: str, handoff: dict[str, str]) -> None:
    """Pick the container's log path and id out of run.py's output, once each."""
    for pattern, key in ((_DOCKER_LOG_RE, "log"), (_CONTAINER_ID_RE, "id")):
        match = pattern.search(line)
        if match and key not in handoff:
            handoff[key] = match.group(1)


def _release(proc: subprocess.Popen) -> None:
    """Stop reading, without stopping the tool.

    run.py outlives a successful watch by a few seconds and the container
    outlives them both, so this never signals anything: it closes our end of the
    pipe and reaps the process if it has already finished.
    """
    if proc.stdout is not None:
        proc.stdout.close()
    proc.poll()


class _Liveness:
    """Whether the container we started is still up.

    `docker run --rm` removes a container the instant it exits, so once ours has
    been seen alive, an inspect that can no longer find it *is* the answer — the
    earlier reading of "cannot ask, so assume yes" left a crashed boot being
    waited on for the full hour. Before that first sighting the benefit of the
    doubt still goes to the boot: a docker hiccup must not abort a healthy one.
    """

    def __init__(self, runner: Runner, runtime: str) -> None:
        self._runner = runner
        self._runtime = runtime
        self._seen = False

    def alive(self, container_id: str | None) -> bool:
        if not container_id:
            return True
        result = self._runner.capture(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", container_id],
            tool=self._runtime, check=False,
        )
        if result.returncode != 0:
            return not self._seen
        running = result.stdout.strip() == "true"
        self._seen = self._seen or running
        return running


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
_HELD_RE = re.compile(r"Sysmem mapped at unexpected NOC address|CHIP_IN_USE|stale process holding")
_RESET_RE = re.compile(
    r"Try resetting the board|Timed out while waiting for active ethernet core"
)


def _boot_error(
    model_name: str,
    tracker: PhaseTracker,
    *,
    raw_log: Path,
    exited: bool,
    deadline_s: int,
    cause: TTError | None = None,
    verbose: bool = False,
) -> TTError:
    """Say why the serve did not reach ready, quoting the line that explains it.

    Never a dump of the last N lines: the full output is on disk and named in the
    error, and the one line worth reading has usually scrolled past hundreds of
    traceback lines by the time the container exits.
    """
    evidence = tracker.evidence()
    joined = "\n".join(evidence)
    # No "see the log" in next_step: the error panel prints details["log_path"]
    # under every error, and saying it twice reads as noise.
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
