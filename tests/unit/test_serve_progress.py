# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Phase classification: captured log lines in, checklist rows out."""

import pytest

from tenstorrent.backends.serving.progress import HOST_PHASES, MEDIA_PHASES, VLLM_PHASES, phases_for
from tenstorrent.backends.serving.progress.phases import parse_bars, parse_pull_layer
from tenstorrent.backends.serving.progress.tracker import PhaseTracker

# Trimmed from a real run.py server run (v0.22.0) — the lines the host template
# keys on, in the order run.py prints them.
HOST_LOG = """\
2026-08-25 03:14:43,304 - run.py:1022 - INFO: TT-Inference version: 0.20.0
2026-08-25 03:14:43,304 - validate_setup.py:313 - INFO: Starting local setup validation
2026-08-25 03:14:43,671 - validate_setup.py:349 - INFO: ✅ validating local setup completed
2026-08-25 03:14:43,672 - setup_host.py:298 - INFO: ✅ Setup already completed for model Qwen3-Coder-30B-A3B-Instruct.
2026-08-25 03:14:43,673 - run_docker_server.py:160 - INFO: running: docker pull ghcr.io/tenstorrent/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-a41684988-bd150c7e9
a1b2c3d4e5f6: Pulling fs layer
9f8e7d6c5b4a: Pulling fs layer
a1b2c3d4e5f6: Pull complete
9f8e7d6c5b4a: Pull complete
2026-08-25 03:14:44,028 - run_docker_server.py:183 - INFO: ✅ Docker Image pulled successfully.
2026-08-25 03:14:44,028 - run_docker_server.py:613 - INFO: Docker run command:
  --name tt-inference-server-f0b78572 \\
2026-08-25 03:14:44,028 - run_docker_server.py:516 - INFO: Running docker container with log file: /repo/workflow_logs/docker_server/vllm_x_server.log
2026-08-25 03:14:44,545 - run_docker_server.py:560 - INFO: Created Docker container ID: e183b242a4e4
"""

# Trimmed from a real Qwen3.6-27B p300x2 boot, in log order.
VLLM_LOG = """\
INFO 09-08 20:34:43 [__init__.py:212] Platform plugin tt is activated
2026-09-08 20:34:45,604 - __main__ - INFO - Downloading weights from Qwen/Qwen3.6-27B to /cache_root/weights/Qwen3.6-27B
Fetching 29 files:  55%|#####     | 16/29 [00:00<00:00, 4151.08it/s]
(EngineCore_DP0 pid=127) INFO 09-08 20:35:07 core.py:98] Initializing a V1 LLM engine (v0.1.dev14179) with config: model='Qwen/Qwen3.6-27B'
2026-09-08 20:35:08.485 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:228)
2026-09-08 20:35:12.704 | info     |           Metal | Fabric initialized on 4 devices (fabric_firmware_initializer.cpp:444)
(EngineCore_DP0 pid=127) INFO 09-08 20:35:14 worker.py:786] multidevice with 4 devices and grid (1, 4) is created
(EngineCore_DP0 pid=127) 2026-09-08 20:35:14.930 | INFO | model_config:591 - Checkpoint directory: /cache_root/weights/Qwen3.6-27B
(EngineCore_DP0 pid=127) Loading layers:  25%|##5       | 16/64 [00:06<07:20,  6.98s/it]
(EngineCore_DP0 pid=127) INFO 09-08 20:41:43 kv_cache_utils.py:1308] GPU KV cache size: 263,200 tokens
(EngineCore_DP0 pid=127) 2026-09-08 20:41:45.702 | INFO | warmup_utils:120 - Warming up decode for sampling params: None
(EngineCore_DP0 pid=127) INFO 09-08 20:44:08 core.py:286] init engine (profile, create kv cache, warmup model) took 144.83 seconds
(APIServer pid=1) INFO 09-08 20:44:17 api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8010
"""


def drive(tracker, log):
    """Feed a log and return (rows_reached, {key: detail})."""
    for line in log.splitlines():
        tracker.feed(line)
    tracker.finish()
    return tracker.reached, {key: tracker.detail_for(key) for key in tracker.reached}


def test_host_template_walks_run_py_in_order():
    reached, details = drive(PhaseTracker(HOST_PHASES), HOST_LOG)
    assert reached == ["host", "weights", "image", "container"]
    assert details["host"] == "tt-inference-server 0.20.0"
    assert details["weights"] == "already in the cache"
    # The tag's tt-metal commit is trimmed off; the version stays.
    assert details["image"] == "vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0"
    assert details["container"] == "tt-inference-server-f0b78572"


def test_docker_pull_layers_are_the_image_rows_progress():
    """Piped `docker pull` reports layer statuses and no bytes, so layers are
    the only honest denominator."""
    tracker = PhaseTracker(HOST_PHASES)
    events = [e for line in HOST_LOG.splitlines() for e in tracker.feed(line)]
    progress = [e for e in events if e.kind == "progress"]
    assert (progress[-1].done, progress[-1].total) == (2.0, 2.0)
    assert not progress[-1].is_bytes


def test_vllm_template_walks_a_real_boot():
    reached, details = drive(PhaseTracker(VLLM_PHASES), VLLM_LOG)
    assert reached == ["engine", "fetch", "device", "weights", "kv", "warmup", "server"]
    assert details["device"] == "4 chips · mesh (1, 4)"
    assert details["kv"] == "263,200 tokens"
    assert details["warmup"] == "145s of warmup"
    assert details["weights"] is None  # a count that never finished is not a result


def test_a_repeated_early_marker_cannot_drag_the_checklist_backwards():
    """tt-metal reopens the device driver once per worker, long after the model
    is loading; the scan only moves forward, so the row must not reopen."""
    tracker = PhaseTracker(VLLM_PHASES)
    drive(tracker, VLLM_LOG)
    before = tracker.reached
    tracker.feed("2026-09-08 20:46:01.1 | info | Device | Opening user mode device driver")
    assert tracker.reached == before


def test_polling_chatter_never_becomes_a_row_or_crowds_out_the_tail():
    tracker = PhaseTracker(VLLM_PHASES, tail_lines=3)
    tracker.feed("INFO 09-08 [__init__.py:212] Platform plugin tt is activated")
    for _ in range(10):
        tracker.feed('INFO:     172.18.0.3:33278 - "GET /health HTTP/1.1" 200 OK')
    assert tracker.reached == ["engine"]
    assert "Platform plugin" in tracker.evidence()[-1]


def test_cause_lines_survive_a_traceback_that_outruns_the_tail():
    tracker = PhaseTracker(VLLM_PHASES, tail_lines=2)
    tracker.feed("RuntimeError: CHIP_IN_USE: device 0 is held by pid 4211")
    for index in range(50):
        tracker.feed(f'  File "engine.py", line {index}, in run')
    assert "CHIP_IN_USE" in tracker.evidence()[0]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("INFO worker.py:786] multidevice with 4 devices and grid (1, 4) is created",
         "4 chips · mesh (1, 4)"),
        ("INFO worker.py:786] multidevice with 1 devices and grid (1, 1) is created",
         "1 chip · mesh (1, 1)"),
        ("| Metal | Fabric initialized on 1 devices", "1 chip"),
    ],
)
def test_one_chip_is_not_one_chips(line, expected):
    tracker = PhaseTracker(VLLM_PHASES)
    tracker.feed("2026-09-08 | info | Device | Opening user mode device driver")
    tracker.feed(line)
    assert tracker.detail_for("device") == expected


def test_media_models_get_the_media_template():
    """Under tt-media-server, uvicorn is up seconds into the boot — reading its
    banner as "the API server started" would report ready minutes early."""
    assert phases_for(["media"]) is MEDIA_PHASES
    assert phases_for(["forge"]) is MEDIA_PHASES
    assert phases_for(["vLLM"]) is VLLM_PHASES

    tracker = PhaseTracker(MEDIA_PHASES)
    tracker.feed("2026-09-01 21:23:05,584 - INFO - Settings init: MODEL='Falcon3-7B-Instruct'")
    tracker.feed("INFO:     Started server process [19]")
    tracker.feed("INFO:     Uvicorn running on http://0.0.0.0:8000")
    assert tracker.reached == ["service"]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Loading layers:  25%|##5    | 16/64 [00:06<07:20]",
         [("Loading layers", 16.0, 64.0, False)]),
        ("model.safetensors:  34%|###  | 1.68G/4.98G [00:12<00:24]",
         [("model.safetensors", 1.68e9, 4.98e9, True)]),
        ("Loading layers:   0%|       | 0/0 [00:00<?, ?it/s]", []),
        ("nothing to see here", []),
    ],
)
def test_tqdm_bars_are_read_as_counts_or_bytes(line, expected):
    """A unit suffix is what tells bytes from a plain count: hf prints file
    counts raw and byte totals scaled."""
    assert [(b.label, b.done, b.total, b.is_bytes) for b in parse_bars(line)] == expected


def test_repaints_sharing_a_line_keep_one_identity_per_bar():
    """Carriage-return repaints usually arrive as separate lines, but when two
    share one they must not read as two different files — that would count the
    same download twice."""
    line = "Loading layers: 5%|# | 3/64 [00:01<00:20]Loading layers: 20%|## | 13/64 [00:04<00:20]"
    bars = parse_bars(line)
    assert [b.done for b in bars] == [3.0, 13.0]
    assert {b.label for b in bars} == {"Loading layers"}


def test_a_weights_download_reports_bytes_not_the_file_count():
    """`hf download` draws one byte bar per file plus an aggregate `Fetching N
    files` count. Taking whichever repainted last showed "10/14" — the count —
    when what the user wants is how many of the gigabytes have landed."""
    tracker = PhaseTracker(HOST_PHASES)
    tracker.feed("INFO: Downloading model to host HF cache: meta-llama/Llama-3.1-8B-Instruct")
    events = []
    for shard in (1, 2):
        events += tracker.feed(
            f"model-0000{shard}-of-00002.safetensors: 50%|##| 2.00G/4.00G [00:12<00:12]"
        )
    # The aggregate count arrives last and must not clobber the byte reading.
    events += tracker.feed("Fetching 14 files:  71%|####| 10/14 [00:30<00:10,  1.2it/s]")
    last = [e for e in events if e.kind == "progress"][-1]
    assert (last.done, last.total, last.is_bytes) == (4.0e9, 8.0e9, True)


def test_the_download_total_grows_as_the_worker_pool_picks_up_more_files():
    """Only the files already started have a known size, so the denominator
    rises for the first moments. Reported as it is rather than guessed at."""
    tracker = PhaseTracker(HOST_PHASES)
    tracker.feed("INFO: Downloading model to host HF cache: org/model")
    tracker.feed("a.safetensors: 50%|##| 2.00G/4.00G [00:12<00:12]")
    events = tracker.feed("b.safetensors:  0%|  | 0.00/4.00G [00:00<?, ?B/s]")
    last = [e for e in events if e.kind == "progress"][-1]
    assert (last.done, last.total) == (2.0e9, 8.0e9)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("a1b2c3d4e5f6: Pull complete", ("a1b2c3d4e5f6", True)),
        ("a1b2c3d4e5f6: Already exists", ("a1b2c3d4e5f6", True)),
        ("a1b2c3d4e5f6: Downloading", ("a1b2c3d4e5f6", False)),
        ("Status: Downloaded newer image for ghcr.io/x:1", None),
    ],
)
def test_docker_pull_layer_lines(line, expected):
    assert parse_pull_layer(line) == expected


def test_a_fetch_after_the_device_opens_gets_its_own_row():
    log = """\
INFO 08-21 23:47:00 core.py:98] Initializing a V1 LLM engine (v0.1) with config
(EngineCore pid=99) INFO 08-21 23:47:18 tt/worker.py:739] Attempting to open mesh device with grid shape (1, 4)
(EngineCore pid=99) INFO 08-21 23:47:25 tt/worker.py:752] multidevice with 4 devices and grid (1, 4) is created
Fetching 28 files:   7%|▋         | 2/28 [00:00<00:01, 18.83it/s]
(EngineCore pid=99) INFO 08-21 23:57:40 kv_cache_utils.py:2146] GPU KV cache size: 264,192 tokens
(EngineCore pid=99) 2026-08-21 23:58:21.128 | INFO | generator_vllm:warmup_model_prefill:512 - Prefill warmup done
"""
    reached, _ = drive(PhaseTracker(VLLM_PHASES), log)
    assert reached == ["engine", "device", "fetch_late", "kv", "warmup"]


def test_an_early_fetch_restating_itself_does_not_skip_the_device():
    log = """\
INFO 08-21 00:25:50 core.py:98] Initializing a V1 LLM engine (v0.1) with config
INFO 08-21 00:25:51 Downloading weights from Qwen/Qwen3-Coder-30B-A3B-Instruct to /cache
Fetching 28 files:   0%|          | 0/28 [00:00<?, ?it/s]
2026-08-21 00:31:07.921 | info | Device | Opening user mode device driver (tt_cluster.cpp:228)
"""
    reached, _ = drive(PhaseTracker(VLLM_PHASES), log)
    assert reached == ["engine", "fetch", "device"]


def test_warmup_names_the_prefill_length_while_it_runs():
    tracker = PhaseTracker(VLLM_PHASES)
    events = tracker.feed(
        "2026-07-18 05:13:18.574 | INFO | generator:warmup_model_prefill:192 - "
        "Warming up prefill for sequence length: 2048"
    )
    assert any(e.kind == "detail" and e.detail == "prefill 2048" for e in events)
