# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The boot watch: a backend's preparation and the container's log, together."""

import io
import re
import subprocess
import sys
import textwrap
from pathlib import Path
import time

from rich.console import Console

import pytest

from tenstorrent.backends.serving import boot
from tenstorrent.backends.serving.preparation import ModelManagerPreparation, RunPyPreparation
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.output import OutputManager
from tenstorrent.backends.serving.progress import Checklist, PhaseTracker, phases_for
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
    import sys, time
    rc, linger = int(sys.argv[1]), float(sys.argv[2])
    print("INFO: TT-Inference version: 0.22.0")
    print("INFO: validating local setup completed")
    print("INFO: Setup already completed for model m.")
    print("INFO: running: docker pull ghcr.io/tenstorrent/vllm:0.22.0-abcdef")
    print("abc12345: Pull complete")
    print("INFO: Docker Image pulled successfully.")
    print("INFO: Docker run command:")
    print("  --name tt-inference-server-abc123 \\\\")
    print("INFO: Created Docker container ID: deadbeef")
    # v0.22.0 blocks here polling /health, for as long as the boot takes.
    print("INFO: Waiting for inference server readiness at http://127.0.0.1:20000/health ...")
    time.sleep(linger)
    sys.exit(rc)
    '''
)


#: Stands in for `docker logs --follow`: replays the container's output, then
#: stays open exactly as the real one does while the container runs.
FAKE_DOCKER_LOGS = textwrap.dedent(
    '''
    import sys, time
    sys.stdout.write(open(sys.argv[1]).read())
    sys.stdout.flush()
    time.sleep(float(sys.argv[2]))
    '''
)


def _running_on(port):
    """A docker that says every container is up and publishing `port`."""
    def spawn(argv, **kwargs):
        out = ""
        if argv[:2] == ["docker", "inspect"] and "{{.State.Running}}" in " ".join(argv):
            out = 'true {"8000/tcp":[{"HostIp":"0.0.0.0","HostPort":"%d"}]}' % port
        return subprocess.CompletedProcess(argv, 0, out, "")
    return spawn


@pytest.fixture
def harness(tmp_path):
    """A watch_serve call wired to a scripted backend and a scripted docker."""
    script = tmp_path / "run.py"
    script.write_text(FAKE_RUN_PY)
    logs_script = tmp_path / "docker-logs.py"
    logs_script.write_text(FAKE_DOCKER_LOGS)
    body = tmp_path / "container.log"
    body.write_text(CONTAINER_LOG)

    def run(*, container_body=None, rc=0, probe_answers=(False, False, True), linger=0.0,
            logs_linger=0.0, prepare=None, output=None, monkeypatch=None,
            capture=None):
        if container_body is not None:
            body.write_text(container_body)
        # A real endpoint answers after the boot, never before it; a probe that
        # succeeds on the first call would let a serve finish before the
        # container had written a line. No interval between them, though — the
        # tests should not pay for the pacing a real boot needs.
        answers = list(probe_answers)
        monkeypatch.setattr(boot, "_PROBE_INTERVAL_S", 0.0)
        monkeypatch.setattr(boot, "_healthy", lambda url: True)
        monkeypatch.setattr(
            boot.discovery, "probe",
            lambda url, timeout_s=0: ["served"] if answers and answers.pop(0) else None,
        )

        def popen(argv, **kwargs):
            # The watcher spawns `docker logs --follow <id>` for the container
            # half; swap in a script that replays a captured boot.
            if argv[:2] == ["docker", "logs"]:
                argv = [sys.executable, str(logs_script), str(body), str(logs_linger)]
            return subprocess.Popen(argv, **kwargs)

        runner = Runner(popen=popen, spawn=capture or _running_on(20000))
        return boot.watch_serve(
            runner=runner,
            output=output or OutputManager(),
            prepare=prepare or RunPyPreparation(),
            argv=[sys.executable, str(script), str(rc), str(linger)],
            # As the real callers do: a piped Python block-buffers its stdout,
            # and nothing arrives until it exits.
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
    return run


def test_a_watched_serve_reaches_ready_and_tees_everything(harness, tmp_path, monkeypatch):
    result = harness(monkeypatch=monkeypatch)
    assert result.ready
    assert result.endpoint == "http://127.0.0.1:20000/v1"
    # The name, not the id: it is what `tt model ps` and `tt model stop` show,
    # and run.py echoes it first.
    assert result.container == "tt-inference-server-abc123"
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
        prepare=RunPyPreparation(),
        argv=[sys.executable, str(script)],
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
    from tenstorrent.backends.serving.progress import HOST_PHASES
    from tenstorrent.backends.serving.progress.tracker import PhaseTracker

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
    from tenstorrent.backends.serving.progress import VLLM_PHASES
    from tenstorrent.backends.serving.progress.tracker import PhaseTracker

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


def test_a_crashed_container_is_noticed_the_moment_its_logs_end(harness, monkeypatch):
    """`docker run --rm` removes a container the instant it exits, so there is
    nothing left to inspect. `docker logs --follow` ending *is* the news, and
    it arrives at once rather than on the next poll."""
    with pytest.raises(TTError) as excinfo:
        harness(
            container_body="RuntimeError: CHIP_IN_USE: device 0 is held\n",
            probe_answers=(False,) * 20, linger=5.0, logs_linger=0.0,
            monkeypatch=monkeypatch,
        )
    assert "already in use" in excinfo.value.what


def test_a_mesh_that_needs_a_reset_says_so(tmp_path):
    """Replayed from a real p300x2 boot: tt-metal's TT_THROW names the cause
    hundreds of traceback lines before vLLM reports the engine dying."""
    from tenstorrent.backends.serving.progress import VLLM_PHASES
    from tenstorrent.backends.serving.progress.tracker import PhaseTracker

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


# -- the tt-model contract (docs/serve-progress-contract.md) ---------------------------
CONTRACT = Path(__file__).parent.parent / "fakes" / "data" / "tt-model-progress.ndjson"


def replay_events(lines):
    """Feed a preparation an event stream; return (rows, prep)."""
    prepare = ModelManagerPreparation("ns/bundle")
    output = OutputManager()
    output.status_console = Console(file=io.StringIO(), width=120)
    with Checklist(output) as view:
        for line in lines:
            boot._apply(view, prepare.feed(line))
        boot._apply(view, prepare.finish())
    return output.status_console.file.getvalue(), prepare


def test_the_published_contract_renders_as_the_shared_checklist():
    """The fixture is the contract: these are the exact bytes tt-model has to
    send, and this is the whole of what tt does with them."""
    shown, prepare = replay_events(CONTRACT.read_text().splitlines())
    # Shared keys borrow the catalog path's own wording, so the two backends
    # cannot drift apart on a rename; an unknown key uses its own label.
    assert "✓ host ready" in shown
    assert "✓ weights ready" in shown
    assert "✓ image ready" in shown
    assert "✓ container started" in shown
    # An unknown key uses its own labels rather than being dropped or shown raw.
    assert "✓ bundle resolved" in shown
    assert "qwen3-coder-30b-a3b @ p300x2" in shown
    assert prepare.container == "tt-model-qwen3-coder-30b-a3b-p300x2"
    assert prepare.endpoint_port == 20000


def test_byte_progress_survives_the_round_trip():
    shown, _ = replay_events([
        '{"event":"step","key":"weights","state":"start"}',
        '{"event":"progress","done":3221225472,"total":16000000000,"unit":"bytes"}',
    ])
    assert "3.2 GB" in shown  # bytes, not a raw count


def test_everything_it_prints_is_kept_as_evidence_whether_or_not_it_is_a_row():
    """Rows are the ones that look like steps; the tail a failure card quotes is
    all of it, including the notes and whatever the tools underneath wrote."""
    _, prepare = replay_events([
        "  ⭻ image tt-model/qwen3:abc is missing from docker (deleted or pruned)",
        "Error response from daemon: manifest unknown",
        "✗ docker pull tt-model/qwen3:abc",
    ])
    assert "manifest unknown" in " ".join(prepare.evidence())
    assert "missing from docker" in " ".join(prepare.evidence())


@pytest.mark.parametrize(
    "line",
    [
        '{"event":"step"}',                                  # no key, no state
        '{"event":"progress","done":1,"total":0}',           # nothing to divide by
        '{"event":"whatever-comes-next","x":1}',             # a later version
        '{"event":"container"}',                             # nothing named
        '{"not":"ours"}',
        '{"event":"step","key":"weights","state":"start"',   # truncated
        "",
    ],
)
def test_a_malformed_or_unknown_event_is_ignored_not_fatal(line):
    """The stream may outgrow this reader; it must never take a serve down."""
    shown, _ = replay_events([line])
    assert "Traceback" not in shown


def test_a_backend_diagnosis_is_preferred_over_our_own_guess(tmp_path):
    """tt-model knows things about bundles tt does not, so when it says why,
    that is what the card shows."""
    prepare = ModelManagerPreparation("ns/bundle")
    prepare.feed(
        '{"event":"error","cause":"the engine rejected a flag",'
        '"detail":"--frobnicate was forwarded to vLLM","actions":["drop it"]}'
    )
    err = boot._boot_error(
        "ns/bundle", prepare, raw_log=tmp_path / "serve.log", exited=True,
        deadline_s=0, diagnosis=prepare.diagnosis,
    )
    assert "the engine rejected a flag" in err.what
    assert "--frobnicate" in err.why
    assert "drop it" in err.next_step


# -- the point of all of it ------------------------------------------------------------
_RUN_PY_OUTPUT = [
    "INFO: TT-Inference version: 0.22.0",
    "INFO: validating local setup completed",
    "INFO: Setup already completed for model m.",
    "INFO: running: docker pull ghcr.io/tenstorrent/vllm:0.22.0-abcdef",
    "INFO: Docker Image pulled successfully.",
    "INFO: Docker run command:",
    "  --name tt-inference-server-abc123 \\",
]


def _rows(prepare, preparation_output):
    """The checklist a serve draws, as row labels, for one backend."""
    output = OutputManager()
    output.status_console = Console(file=io.StringIO(), width=120)
    container = PhaseTracker(phases_for(["vLLM"]))
    with Checklist(output) as view:
        view.begin(prepare.label, placeholder=True)
        for line in preparation_output:
            boot._apply(view, prepare.feed(line))
        boot._apply(view, prepare.finish())
        for line in CONTAINER_LOG.splitlines():
            boot._apply(view, container.feed(line))
        boot._apply(view, container.finish())
        view.instant("endpoint answering")
    return [
        re.sub(r"\s{2,}.*$", "", line.strip().lstrip("✓ "))
        for line in output.status_console.file.getvalue().splitlines()
    ]


def test_both_backends_draw_the_same_checklist():
    """The one assertion that keeps them together.

    Same container log through the same tracker and the same view, so the boot
    half is identical by construction rather than by two implementations
    agreeing. tt-model's preparation adds the one step it genuinely has that
    tt-inference-server does not, and shares the wording for the rest.
    """
    catalog = _rows(RunPyPreparation(), _RUN_PY_OUTPUT)
    bundle = _rows(ModelManagerPreparation("ns/bundle"), CONTRACT.read_text().splitlines())
    assert catalog[0] == "host ready"
    assert bundle == ["bundle resolved"] + catalog


# -- the fallback, which is what today's pinned tt-model actually produces -------------
# Captured from a real `tt-model --verbose serve` re-pulling a pruned image and
# then fetching a 27B model's weights.
_TT_MODEL_VERBOSE = [
    "  ⭻ image tt-model/qwen3.8-27b-p150x4:4233a70b5f90 is missing from docker",
    "  (deleted or pruned); re-pulling it from tt-hous/qwen3.8-27b-p150x4",
    "docker load tt-model/qwen3.8-27b-p150x4:4233a70b5f90…",
    "Loaded image: tt-model/qwen3.8-27b-p150x4:4233a70b5f90",
    "✓ docker load tt-model/qwen3.8-27b-p150x4:4233a70b5f90  57.9s",
    "  ✓ pulled tt-hous/qwen3.8-27b-p150x4",
    "  → next:  tt-model serve tt-hous/qwen3.8-27b-p150x4",
    "weights Qwen/Qwen3.8-27B@1d4bf0f2…",
]


def test_a_tt_model_without_events_still_shows_its_own_steps():
    """Until the contract lands upstream, tt-model's `--verbose` rows are all
    there is — and one motionless row for the ten minutes a 50 GB bundle takes
    is not good enough. Its prose becomes rows; its prose that is not a step
    (notes, the arrow hint, docker's own output) does not."""
    shown, prepare = replay_events(_TT_MODEL_VERBOSE)
    assert "✓ docker load tt-model/qwen3.8-27b-p150x4:4233a70b5f90" in shown
    assert "✓ pulled tt-hous/qwen3.8-27b-p150x4" in shown
    # The last step is still running, so it stays the live row.
    assert "weights Qwen/Qwen3.8-27B@1d4bf0f2" in shown
    assert "next:" not in shown
    assert "Loaded image:" not in shown
    # tt times its own rows; showing tt-model's duration too reads as two clocks.
    assert "57.9s" not in shown


def test_events_win_over_prose_when_both_arrive():
    """A tt-model that speaks the contract may still print its own rows; reading
    both would draw every step twice."""
    shown, _ = replay_events([
        '{"event":"step","key":"image","state":"done"}',
        "✓ docker load tt-model/qwen3:abc  1.2s",
    ])
    assert "image ready" in shown
    assert "docker load" not in shown


# -- finding the container a bundle left behind ---------------------------------------
_LABELS = "org.tenstorrent.tt-model"


def _docker_ps(rows):
    """A Runner whose `docker ps -a` returns these (id, name-label, repo-label)."""
    body = "".join(f"{i}\t{n}\t{r}\n" for i, n, r in rows)
    return Runner(spawn=lambda *a, **k: subprocess.CompletedProcess(a[0], 0, body, ""))


def test_a_bundle_is_found_by_name_not_by_the_namespace_it_was_pulled_from():
    """The repo label records the bundle's canonical home, so one pulled from a
    fork or a personal namespace carries the original — filtering on the
    requested id found nothing and the whole boot went unwatched."""
    runner = _docker_ps([
        ("73b96ef78488", "devstral-small-2-24b-instruct-2512",
         "tenstorrent/devstral-small-2-24b-instruct-2512"),
    ])
    for asked in ("anirud/devstral-small-2-24b-instruct-2512",
                  "tenstorrent/devstral-small-2-24b-instruct-2512"):
        prepare = ModelManagerPreparation(asked)
        assert prepare.resolve_container(runner, "docker") == "73b96ef78488"


def test_another_bundles_container_is_not_mistaken_for_ours():
    prepare = ModelManagerPreparation("ns/qwen3-a3b")
    runner = _docker_ps([("aaa", "devstral-small-2-24b", "tenstorrent/devstral-small-2-24b")])
    assert prepare.resolve_container(runner, "docker") is None


def test_an_event_named_container_beats_asking_docker():
    prepare = ModelManagerPreparation("ns/bundle")
    prepare.feed('{"event":"container","id":"named-by-the-backend"}')
    runner = _docker_ps([("aaa", "bundle", "ns/bundle")])
    assert prepare.resolve_container(runner, "docker") == "named-by-the-backend"


@pytest.mark.parametrize(
    "engine, vllm",
    [
        ("vLLM", True),            # tt-inference-server's spec
        ("vllm-plugin", True),     # a tt-model bundle manifest
        ("vllm-fork", True),       # …and what it used to say
        ("tt-dit-server", False),  # a diffusion bundle
        ("media", False),
        ("forge", False),
    ],
)
def test_every_spelling_of_the_vllm_stack_picks_the_vllm_template(engine, vllm):
    """Both backends boot the same engine and spell it differently; matching on
    equality classified every bundle as media and read its log with the wrong
    phases."""
    from tenstorrent.backends.serving.progress import VLLM_PHASES

    assert (phases_for([engine]) is VLLM_PHASES) is vllm


def test_a_note_is_not_mistaken_for_a_skipped_step():
    """Captured from a real bundle serve. tt-model marks a skipped step with ○
    and writes ordinary notes with the same marker, so a ○ row cannot be told
    from prose — and a note drawn as a step is worse than a step not drawn."""
    shown, prepare = replay_events([
        "  ↻ image tt-model/depth-anything-3-p150:88d067be2a39 is missing from docker",
        "  (deleted or pruned); re-pulling it from changh95/depth-anything-3-p150",
        "  ○ tt-model/depth-anything-3-p150:88d067be2a39 is loaded but is a different",
        "  image than this package records (6bc03e655b0d vs 88d067be2a39) — reloading",
        "docker load tt-model/depth-anything-3-p150:88d067be2a39…",
        "✓ docker load tt-model/depth-anything-3-p150:88d067be2a39  34s",
    ])
    assert [row.strip() for row in shown.splitlines() if row.strip()] == [
        "✓ docker load tt-model/depth-anything-3-p150:88d067be2a39"
    ]
    assert "is loaded but is a different" in " ".join(prepare.evidence())


def _gone(argv, **kwargs):
    """A docker with no such container: the one named was never started."""
    if argv[:2] == ["docker", "inspect"]:
        return subprocess.CompletedProcess(argv, 1, "", "Error: No such object")
    return subprocess.CompletedProcess(argv, 0, "", "")


def test_a_neighbour_answering_on_the_port_is_not_this_serve(harness, monkeypatch):
    """A second serve on a busy port: docker refuses the container, run.py fails,
    and the model already there answers every probe. That is not ready."""
    with pytest.raises(TTError):
        harness(
            rc=1, probe_answers=(True,) * 50, capture=_gone, monkeypatch=monkeypatch
        )


def test_the_endpoint_is_where_the_container_publishes_not_the_default(
    harness, monkeypatch
):
    """tt-model walks up past busy ports and need not say where it landed; the
    default would find whatever model is already there."""
    harness.script.write_text('print(\'{"event":"container","id":"tt-model-m"}\')\n')
    result = harness(
        logs_linger=5.0, capture=_running_on(20001), monkeypatch=monkeypatch,
        prepare=ModelManagerPreparation("ns/m"),
    )
    assert result.ready
    assert result.endpoint == "http://127.0.0.1:20001/v1"


def test_tt_model_refusing_for_want_of_chips_is_a_failure_not_ready(
    harness, monkeypatch
):
    """Captured verbatim from a second `tt serve` beside a p300x2 model that has
    every chip mounted: tt-model refuses before it starts anything."""
    harness.script.write_text(textwrap.dedent(
        '''
        import sys
        print("  \\u2022 port 20000 is in use; serving on 20001 instead")
        print("only 0 of 4 tt device(s) are free (chip(s) 0, 1, 2, 3 in use); "
              "this profile needs 1")
        sys.exit(1)
        '''
    ))
    with pytest.raises(TTError) as excinfo:
        harness(
            probe_answers=(True,) * 50, monkeypatch=monkeypatch,
            prepare=ModelManagerPreparation("mando2222/vibethinker-3b-blackhole-v51"),
        )
    assert "already in use" in excinfo.value.what


def test_a_media_worker_that_cannot_open_its_chip_fails_the_serve(harness, monkeypatch):
    calls = []
    running = _running_on(20000)

    def capture(argv, **kwargs):
        calls.append(argv)
        return running(argv, **kwargs)

    body = (
        "INFO - setup_runner_environment: TT_VISIBLE_DEVICES=0\n"
        "TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 "
        "to become active again. Try resetting the board.\n"
        "ERROR - Worker 0 device init failed: Unexpected device initialization error\n"
    )
    with pytest.raises(TTError) as excinfo:
        harness(
            container_body=body, logs_linger=5.0, linger=5.0, capture=capture,
            probe_answers=(), monkeypatch=monkeypatch,
        )
    assert "needs a reset" in excinfo.value.what
    assert ["docker", "stop", "tt-inference-server-abc123"] in calls


def test_a_vllm_boot_does_not_promise_an_in_container_weights_fetch():
    """--host-hf-cache mounts the host's weights, so the container fetches none."""
    steps = boot._boot_steps(["vLLM"])
    assert "fetching weights into the container" not in steps
    assert "fetching weights" in boot._boot_steps(["media"])


def test_a_device_that_needs_a_reset_fails_a_server_that_stays_up(harness, monkeypatch):
    """Captured from a tt-model image server: it keeps answering /health 500."""
    body = (
        'File "/opt/tt-metal/models/experimental/qwen_image_2_1/common/device.py", line 38\n'
        "RuntimeError: TT_THROW @ /opt/tt-metal/tt_metal/llrt/llrt.cpp:625: tt::exception\n"
        "Device 0: Timed out while waiting for active ethernet core 29-25 to become active "
        "again. Try resetting the board. Minimum tt-firmware version is 18.10.0\n"
    )
    with pytest.raises(TTError) as excinfo:
        harness(container_body=body, logs_linger=5.0, linger=5.0, probe_answers=(),
                monkeypatch=monkeypatch)
    assert "needs a reset" in excinfo.value.what


def test_a_download_inside_the_container_shows_what_has_landed(harness, monkeypatch):
    from tenstorrent.backends.serving.progress import MEDIA_PHASES

    monkeypatch.setattr(boot, "phases_for", lambda engines: MEDIA_PHASES)
    monkeypatch.setattr(boot, "_WEIGH_INTERVAL_S", 0.0)
    monkeypatch.setenv(boot.READY_TIMEOUT_ENV, "2")
    running = _running_on(20000)

    def capture(argv, **kwargs):
        if argv[:2] == ["docker", "exec"]:
            path = "/cache/hub/models--microsoft--speecht5_tts"
            assert argv[-1] == path
            return subprocess.CompletedProcess(argv, 0, f"1171155687\t{path}\n", "")
        if "{{range .Config.Env}}{{println .}}{{end}}" in argv:
            return subprocess.CompletedProcess(argv, 0, "MODEL=x\nHF_HOME=/cache\n", "")
        return running(argv, **kwargs)

    output = OutputManager()
    output.status_console = Console(file=io.StringIO(), width=120)
    body = (
        "INFO - Settings init: MODEL='speecht5_tts'\n"
        "INFO - Device -1: Loading HuggingFace model: microsoft/speecht5_tts\n"
    )
    with pytest.raises(TTError):
        harness(container_body=body, logs_linger=5.0, linger=5.0, capture=capture,
                probe_answers=(), output=output, monkeypatch=monkeypatch)
    assert "1.2 GB" in output.status_console.file.getvalue()
