# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The two-stage boot watch: run.py's own output, then the container's log."""

import io
import subprocess
import textwrap
import time

from rich.console import Console

import pytest

from tenstorrent.backends.serving import boot
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.output import OutputManager
from tenstorrent.tools.runner import Runner

CONTAINER_LOG = """\
INFO [__init__.py:212] Platform plugin tt is activated
2026-09-08 20:35:08.485 | info | Device | Opening user mode device driver (tt_cluster.cpp:228)
INFO worker.py:786] multidevice with 4 devices and grid (1, 4) is created
INFO model_config:591 - Checkpoint directory: /cache_root/weights/m
Loading layers: 100%|##########| 64/64 [06:20<00:00]
INFO kv_cache_utils.py:1308] GPU KV cache size: 263,200 tokens
INFO warmup_utils:120 - Warming up decode for sampling params: None
INFO core.py:286] init engine (profile, create kv cache, warmup model) took 144.83 seconds
INFO api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8010
"""

# Stands in for run.py: prints the host-stage lines tt keys on and writes the
# container log it hands off to, exactly where the real one does.
FAKE_RUN_PY = textwrap.dedent(
    '''
    import pathlib, sys, time
    log, body, rc, linger = (
        pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    )
    print("INFO: TT-Inference version: 0.22.0")
    print("INFO: validating local setup completed")
    print("INFO: Setup already completed for model m.")
    print("INFO: running: docker pull ghcr.io/tenstorrent/vllm:0.22.0-abcdef")
    print("abc12345: Pull complete")
    print("INFO: Docker Image pulled successfully.")
    print("INFO: Docker run command:")
    print("  --name tt-inference-server-abc123 \\\\")
    log.write_text(pathlib.Path(body).read_text())
    print(f"INFO: Running docker container with log file: {log}")
    print("INFO: Created Docker container ID: deadbeef")
    # v0.22.0 blocks here polling /health, for as long as the boot takes.
    print("INFO: Waiting for inference server readiness at http://127.0.0.1:20000/health ...")
    time.sleep(linger)
    sys.exit(rc)
    '''
)


@pytest.fixture
def harness(tmp_path):
    """A watch_serve call wired to a scripted run.py and a scripted health probe."""
    script = tmp_path / "run.py"
    script.write_text(FAKE_RUN_PY)
    body = tmp_path / "container.log"
    body.write_text(CONTAINER_LOG)

    def run(*, container_body=None, rc=0, running=True, probe_answers=(True,),
            linger=0.0, output=None, monkeypatch=None):
        log = tmp_path / "docker.log"
        if container_body is not None:
            body.write_text(container_body)
        answers = list(probe_answers)
        monkeypatch.setattr(
            boot.discovery, "probe",
            lambda url, timeout_s=0: ["served"] if answers and answers.pop(0) else None,
        )
        runner = Runner(
            spawn=lambda *a, **k: subprocess.CompletedProcess(
                a[0], 0, "true\n" if running else "false\n", ""
            )
        )
        return boot.watch_serve(
            runner=runner,
            output=output or OutputManager(),
            argv=["python", str(script), str(log), str(body), str(rc), str(linger)],
            # As the real caller does (InferenceServerBackend._env): a piped
            # Python block-buffers its stdout, and nothing arrives until it exits.
            env={"PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"},
            cwd=str(tmp_path),
            tool="tt-inference-server",
            model_name="Test-Model",
            engines=["vLLM"],
            port=20000,
            raw_log=tmp_path / "logs" / "serve.log",
            runtime="docker",
        )

    run.script, run.body, run.tmp = script, body, tmp_path
    run.log = tmp_path / "docker.log"
    return run


def test_a_watched_serve_reaches_ready_and_tees_everything(harness, tmp_path, monkeypatch):
    result = harness(monkeypatch=monkeypatch)
    assert result.ready
    assert result.endpoint == "http://127.0.0.1:20000/v1"
    assert result.container == "deadbeef"
    # The checklist replaces the wall of output on screen; it must not throw it
    # away, so both stages land in one file.
    teed = result.raw_log.read_text()
    assert "TT-Inference version: 0.22.0" in teed
    assert "GPU KV cache size" in teed


def test_a_run_py_that_names_no_container_degrades_instead_of_waiting(
    harness, tmp_path, monkeypatch
):
    """An upstream rename must not turn a working serve into an hour-long poll."""
    script = tmp_path / "quiet.py"
    script.write_text("print('nothing to see here')\n")
    monkeypatch.setattr(boot.discovery, "probe", lambda url, timeout_s=0: None)
    result = boot.watch_serve(
        runner=Runner(),
        output=OutputManager(),
        argv=["python", str(script)],
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
        tool="tt-inference-server",
        model_name="Test-Model",
        engines=["vLLM"],
        port=20000,
        raw_log=tmp_path / "logs" / "serve.log",
        runtime="docker",
    )
    assert not result.ready


def test_run_py_failing_is_reported_against_the_saved_output(harness, tmp_path, monkeypatch):
    with pytest.raises(TTError) as excinfo:
        harness(rc=1, probe_answers=(False,), monkeypatch=monkeypatch)
    err = excinfo.value
    assert err.exit_code is ExitCode.TOOL_FAILED
    assert err.details["log_path"].endswith("serve.log")


def test_the_reason_comes_from_the_log_not_the_exit_status(tmp_path):
    """"exited with status 1" says nothing. The line the tool printed on its way
    out — `AssertionError: HF_TOKEN validation failed` — is the whole answer."""
    from tenstorrent.progress import HOST_PHASES
    from tenstorrent.progress.tracker import PhaseTracker

    tracker = PhaseTracker(HOST_PHASES)
    for line in ("INFO: TT-Inference version: 0.22.0",
                 '  File "setup_host.py", line 405, in get_hf_env_vars',
                 "AssertionError: HF_TOKEN validation failed."):
        tracker.feed(line)
    err = boot._boot_error(
        "Test-Model", tracker, raw_log=tmp_path / "serve.log", exited=True, deadline_s=0,
        cause=TTError("tt-inference-server exited with status 1."),
    )
    assert err.why == "AssertionError: HF_TOKEN validation failed."


def test_a_container_that_dies_during_boot_stops_the_wait(harness, monkeypatch):
    with pytest.raises(TTError) as excinfo:
        harness(
            container_body="RuntimeError: CHIP_IN_USE: device 0 is held\n",
            running=False, probe_answers=(False,), monkeypatch=monkeypatch,
        )
    assert "already in use" in excinfo.value.what


def test_ready_timeout_env_is_validated_up_front(monkeypatch):
    monkeypatch.setenv(boot.READY_TIMEOUT_ENV, "900")
    assert boot.ready_timeout_s() == 900
    monkeypatch.setenv(boot.READY_TIMEOUT_ENV, "soon")
    with pytest.raises(TTError) as excinfo:
        boot.ready_timeout_s()
    assert excinfo.value.exit_code is ExitCode.USAGE


@pytest.mark.parametrize(
    "evidence, expected",
    [
        (["CHIP_IN_USE: device 0"], "already in use"),
        (["OSError: [Errno 98] Address already in use"], "port is already taken"),
        (["ValueError: bad config"], "stopped before the server was ready"),
    ],
)
def test_a_failed_boot_names_the_cause_rather_than_dumping_the_log(
    evidence, expected, tmp_path
):
    from tenstorrent.progress import VLLM_PHASES
    from tenstorrent.progress.tracker import PhaseTracker

    tracker = PhaseTracker(VLLM_PHASES)
    for line in evidence:
        tracker.feed(line)
    err = boot._boot_error(
        "Test-Model", tracker, raw_log=tmp_path / "serve.log", exited=True, deadline_s=0
    )
    assert expected in err.what


def test_the_container_log_is_read_while_run_py_is_still_running(
    harness, tmp_path, monkeypatch
):
    """The regression that matters: from v0.21.0 run.py blocks polling /health
    until the model is warm, so waiting for it to exit before opening the
    container log showed "starting the container" for the entire boot — the
    whole thing the checklist exists to narrate. Both are read at once.
    """
    monkeypatch.setenv(boot.READY_TIMEOUT_ENV, "2")
    output = OutputManager()
    output.status_console = Console(file=io.StringIO(), width=120)
    with pytest.raises(TTError):  # never ready: we are here for what it showed
        harness(linger=30.0, probe_answers=(False,) * 50, output=output,
                monkeypatch=monkeypatch)
    shown = output.status_console.file.getvalue()
    assert "Tenstorrent device opened" in shown
    assert "KV cache configured" in shown


def test_ready_is_reported_without_waiting_out_run_pys_own_poll(harness, monkeypatch):
    """run.py polls the same endpoint and exits on its own once it answers, so
    tt neither waits out its full gate nor cuts its pipe from under it."""
    started = time.monotonic()
    result = harness(linger=1.0, probe_answers=(True,), monkeypatch=monkeypatch)
    assert result.ready
    assert time.monotonic() - started < boot._EXIT_GRACE_S


def test_a_crashed_container_is_noticed_even_though_docker_rm_removed_it(harness, monkeypatch):
    """run.py starts the container with `--rm`, so a crash removes it and
    `docker inspect` stops finding it. Reading that as "cannot ask, assume it
    is alive" left a dead boot being waited on for the full hour."""
    inspected = {"calls": 0}

    def fake_inspect(*args, **kwargs):
        inspected["calls"] += 1
        if inspected["calls"] == 1:  # alive at the handover
            return subprocess.CompletedProcess(args[0], 0, "true\n", "")
        return subprocess.CompletedProcess(args[0], 1, "", "No such object: deadbeef\n")

    monkeypatch.setattr(boot, "_LIVENESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        boot.discovery, "probe", lambda url, timeout_s=0: None
    )
    runner = Runner(spawn=fake_inspect)
    script = harness.script
    with pytest.raises(TTError) as excinfo:
        boot.watch_serve(
            runner=runner, output=OutputManager(),
            argv=["python", str(script), str(harness.log), str(harness.body), "0", "5"],
            env={"PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"},
            cwd=str(harness.tmp), tool="t", model_name="Test-Model", engines=["vLLM"],
            port=20000, raw_log=harness.tmp / "logs" / "crash.log", runtime="docker",
        )
    assert "stopped before the server was ready" in excinfo.value.what


def test_a_mesh_that_needs_a_reset_says_so(tmp_path):
    """Replayed from a real p300x2 boot: tt-metal's TT_THROW names the cause
    hundreds of traceback lines before vLLM reports the engine dying."""
    from tenstorrent.progress import VLLM_PHASES
    from tenstorrent.progress.tracker import PhaseTracker

    tracker = PhaseTracker(VLLM_PHASES, tail_lines=5)
    tracker.feed("2026-09-23 18:07:03 | info | Device | Opening user mode device driver")
    tracker.feed(
        "2026-09-23 18:07:23.299 | critical | Always | TT_THROW: Device 0: Timed out "
        "while waiting for active ethernet core 29-25 to become active again. "
        "Try resetting the board."
    )
    for index in range(40):  # the traceback that buries it
        tracker.feed(f'(APIServer pid=1)   File "core_client.py", line {index}, in __init__')
    tracker.feed("(APIServer pid=1) RuntimeError: Engine core initialization failed.")
    err = boot._boot_error(
        "Test-Model", tracker, raw_log=tmp_path / "serve.log", exited=True, deadline_s=0
    )
    assert "needs a reset" in err.what
    assert "tt device reset" in err.next_step
