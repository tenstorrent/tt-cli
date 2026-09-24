# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Which step of a serve a log line belongs to — text in, named phases out.

Pure: no I/O, no clock, no terminal, so the whole matrix is testable against
captured logs. Serving a model prints thousands of lines over 1-10 minutes and
the user needs about eight of them: host checked, image pulled, weights ready,
container up, device open, weights loaded, KV cache sized, warmed up.

Three templates. `HOST_PHASES` reads tt-inference-server's own run.py output
(host setup, `docker pull`, container launch); `VLLM_PHASES` and `MEDIA_PHASES`
read the container's log, and which one applies is decided by the model's engine
(`phases_for`) rather than guessed — the media server boots a FastAPI service
with workers long before the engine exists, so its early uvicorn lines mean the
opposite of what they mean in a bare vLLM boot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

Pattern = re.Pattern


def _rx(*patterns: str) -> tuple[Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


@dataclass(frozen=True)
class Phase:
    """One row of the checklist."""

    key: str
    label: str  # while it runs: "opening the Tenstorrent device"
    done_label: str  # once finished: "Tenstorrent device opened"
    start: tuple[Pattern, ...]
    done: tuple[Pattern, ...] = ()
    #: line -> a short fact for the row ("4 chips · mesh (1, 4)"), or None
    detail: Optional[Callable[[str], Optional[str]]] = None
    #: count `docker pull`'s per-layer lines as this phase's progress. Piped
    #: output gets layer statuses and no byte counts (docker only draws the
    #: byte bars on a terminal), so layers are the honest denominator.
    layers: bool = False


# -- shared line parsing ---------------------------------------------------------------

#: tqdm's bar: ` 38%|███▊      | 323/851 [00:00<00:00, 3190it/s]`, and its
#: unit-scaled form ` 34%|███▎      | 1.68G/4.98G [00:12<00:24, 136MB/s]`. Only
#: the counts are trusted; the percentage is re-derived so a rounding quirk
#: cannot show 100% at 850/851.
TQDM_RE = re.compile(r"\d+%\|[^|]*\|\s*([\d.]+)([kKMGTP]?)/([\d.]+)([kKMGTP]?)")
_SCALE = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}


@dataclass(frozen=True)
class Bar:
    """One tqdm bar: what it is called, where it is, and in what units.

    `label` is tqdm's description — a filename for a weights download — and is
    what lets several concurrent bars be told apart and added up.
    """

    label: str
    done: float
    total: float
    is_bytes: bool

#: `docker pull` on a pipe: `a1b2c3d4e5f6: Pull complete`.
_LAYER_RE = re.compile(r"^([0-9a-f]{8,}):\s+(\S.*?)\s*$")
_LAYER_DONE = ("Pull complete", "Already exists")


def _bar_label(before: str) -> str:
    """tqdm's description, from the text in front of a bar.

    A carriage return makes each repaint its own line, so that text is normally
    just `<desc>: `. When two repaints do share a line the leading part is the
    previous bar's `[00:12<00:24, 136MB/s]` tail, which is dropped — otherwise
    the same file would be counted twice under two different names.
    """
    return before.rsplit("]", 1)[-1].strip().rstrip(":").strip()


def parse_bars(line: str) -> list[Bar]:
    """Every tqdm bar on the line, in order.

    A unit suffix only ever marks a byte bar in these logs — file and shard
    counts are printed raw — so the suffix doubles as the "these are bytes" flag.
    """
    bars: list[Bar] = []
    previous_end = 0
    for match in TQDM_RE.finditer(line):
        done_unit, total_unit = match.group(2), match.group(4)
        try:
            done = float(match.group(1)) * _SCALE[done_unit]
            total = float(match.group(3)) * _SCALE[total_unit]
        except (KeyError, ValueError):
            done = total = 0.0
        if total > 0:
            bars.append(
                Bar(
                    _bar_label(line[previous_end : match.start()]),
                    done,
                    total,
                    bool(done_unit or total_unit),
                )
            )
        previous_end = match.end()
    return bars


def parse_pull_layer(line: str) -> Optional[tuple[str, bool]]:
    """`(layer_id, finished)` for a `docker pull` status line, else None."""
    match = _LAYER_RE.match(line)
    if match is None:
        return None
    return match.group(1), match.group(2).startswith(_LAYER_DONE)


# -- host side: tt-inference-server's run.py -------------------------------------------


def _detail_host(line: str) -> Optional[str]:
    match = re.search(r"TT-Inference version:\s*(\S+)", line)
    return f"tt-inference-server {match.group(1)}" if match else None


def _detail_host_weights(line: str) -> Optional[str]:
    if "Setup already completed for model" in line:
        return "already in the cache"
    match = re.search(r"Downloading model to host (?:HF cache|volume):\s*(\S+)", line)
    return match.group(1) if match else None


def _detail_image(line: str) -> Optional[str]:
    if "available locally" in line:
        return "already on this host"
    match = re.search(r"running: docker pull \S*?([^/\s]+:[^\s]+)", line)
    # Tags embed a tt-metal commit and run past 100 characters; the repo name and
    # the version in front of it are what identifies the image to a human.
    return re.sub(r"(:[^-\s]+)-.*", r"\1", match.group(1)) if match else None


def _detail_container(line: str) -> Optional[str]:
    # The name, from run.py's echo of the docker run command.
    match = re.search(r"--name\s+(\S+)", line)
    return match.group(1) if match else None


HOST_PHASES: tuple[Phase, ...] = (
    Phase(
        "host", "checking the host", "host ready",
        start=_rx(r"TT-Inference version:", r"Starting local setup validation"),
        done=_rx(r"validating local setup completed"),
        detail=_detail_host,
    ),
    Phase(
        "weights", "downloading weights", "weights ready",
        start=_rx(r"Downloading model to host (?:HF cache|volume):",
                  r"Setup already completed for model"),
        done=_rx(r"Using weights directory:", r"done setup_weights",
                 r"Setup already completed for model"),
        detail=_detail_host_weights,
    ),
    Phase(
        "image", "pulling the container image", "image ready",
        start=_rx(r"running: docker pull "),
        done=_rx(r"Docker Image pulled successfully", r"Docker Image available locally"),
        detail=_detail_image, layers=True,
    ),
    Phase(
        "container", "starting the container", "container started",
        # "Docker run command:" is where run.py echoes the command it is about to
        # launch.
        start=_rx(r"Docker run command:", r"Running docker container with log file:"),
        done=_rx(r"Created Docker container ID:"),
        detail=_detail_container,
    ),
)


# -- container side: a vLLM boot on tt-metal -------------------------------------------


def _detail_engine(line: str) -> Optional[str]:
    # tt-metal's UMD logs a firmware bundle version in the same window; only a
    # vLLM-shaped line may name the engine version.
    if "| UMD |" in line or "firmware" in line:
        return None
    match = re.search(
        r"(?:vLLM (?:API )?server version|LLM engine \(v)\s*v?(\d+\.\d+[\w.+-]*)", line
    )
    return f"vLLM {match.group(1)}" if match else None


def _chips(count: str) -> str:
    return f"{count} chip" + ("s" if count != "1" else "")


def _detail_device(line: str) -> Optional[str]:
    match = re.search(r"multidevice with (\d+) devices? and grid \(([^)]*)\)", line)
    if match:
        return f"{_chips(match.group(1))} · mesh ({match.group(2)})"
    match = re.search(r"Fabric initialized on (\d+) devices", line)
    if match:
        return _chips(match.group(1))
    return None


def _detail_kv(line: str) -> Optional[str]:
    match = re.search(r"KV cache size:\s*([\d,]+) tokens", line)
    return f"{match.group(1)} tokens" if match else None


def _detail_warmup(line: str) -> Optional[str]:
    match = re.search(r"init engine .* took ([\d.]+) s(?:econds)?\b", line)
    return f"{float(match.group(1)):.0f}s of warmup" if match else None


VLLM_PHASES: tuple[Phase, ...] = (
    Phase(
        "engine", "starting the engine", "engine started",
        start=_rx(r"Available plugins for group vllm\.platform_plugins",
                  r"Platform plugin tt is activated", r"vLLM (?:API )?server version",
                  r"Initializing a V\d LLM engine"),
        detail=_detail_engine,
    ),
    # Only when the container fetches its own weights. tt serve passes
    # --host-hf-cache, so on that path the host has them already and this row
    # never appears — a missing row is honest, an empty one is noise.
    Phase(
        "fetch", "fetching weights into the container", "weights fetched",
        start=_rx(r"Downloading weights from \S+ to", r"Fetching \d+ files"),
    ),
    Phase(
        "device", "opening the Tenstorrent device", "Tenstorrent device opened",
        start=_rx(r"Opening user mode device driver", r"Attempting to open mesh device",
                  r"Starting devices in cluster"),
        done=_rx(r"multidevice with \d+ devices? and grid .* is created"),
        detail=_detail_device,
    ),
    Phase(
        "weights", "loading weights", "weights loaded",
        start=_rx(r"Checkpoint directory:", r"Loading checkpoint shards",
                  r"Loading safetensors", r"Loading weights:", r"Loading layers:"),
    ),
    Phase(
        "kv", "configuring the KV cache", "KV cache configured",
        start=_rx(r"KV cache size", r"Allocating TT kv caches", r"num_gpu_blocks"),
        detail=_detail_kv,
    ),
    # Deliberately anchored so that a bare "warming up" does not also match the server
    # phase's "Warming up chat template processing"
    Phase(
        "warmup", "warming up the model", "model warmed up",
        start=_rx(r"Warming up prefill", r"Warming up decode", r"Starting decode warmup",
                  r"Done Compiling Model", r"Capturing .*[Tt]race"),
        done=_rx(r"init engine .* took [\d.]+ s(?:econds)?\b"),
        detail=_detail_warmup,
    ),
    Phase(
        "server", "starting the API server", "API server started",
        start=_rx(r"Starting vLLM API server", r"Warming up chat template"),
    ),
)


# -- container side: the media server (media and forge engines) ------------------------

MEDIA_PHASES: tuple[Phase, ...] = (
    # One row for all of the early bookkeeping — settings, Prometheus, uvicorn,
    # the worker pool. It all happens inside the first few seconds.
    Phase(
        "service", "starting the server", "server started",
        start=_rx(r"Settings init:", r"Config lookup:", r"Settings resolved:",
                  r"Setting up Prometheus metrics", r"Started server process"),
        done=_rx(r"All workers started in sequence", r"Application startup complete"),
    ),
    # Only lines that mean the container is actually moving bytes.
    Phase(
        "fetch", "fetching weights", "weights fetched",
        start=_rx(r"Downloading weights for model:", r"Loading HuggingFace model:"),
    ),
    Phase(
        "device", "opening the Tenstorrent device", "Tenstorrent device opened",
        start=_rx(r"setup_runner_environment", r"TT_VISIBLE_DEVICES",
                  r"Device \d+: Loading", r"Opening user mode device driver",
                  r"Creating TopologyDiscovery"),
        done=_rx(r"Created mesh device", r"Fabric initialized on \d+ devices"),
        detail=_detail_device,
    ),
    Phase(
        "load", "loading the model", "model loaded",
        start=_rx(r"Loading checkpoint shards", r"Loading weights:", r"Loading layers:",
                  r"Initializing TTNN", r"Creating inference pipeline",
                  r"Creating TTNN (?:encoder|decoder|postnet)",
                  r"Loading (?:Whisper|SpeechT5|VAE|UNet|transformer|pipeline)"),
        done=_rx(r"Model loaded and pipeline ready", r"Successfully created inference pipeline",
                 r"Model initialization completed"),
    ),
    Phase(
        "warmup", "warming up the model", "model warmed up",
        start=_rx(r"\[warmup\]", r"Starting model warmup", r"[Ww]arm-?up done",
                  r"Warming up (?:prefill|decode)", r"Done Compiling Model",
                  r"Capturing .*[Tt]race"),
        done=_rx(r"All devices are warmed up", r"Model warmup completed"),
    ),
)


#: Lines that name a boot failure. Kept aside from the rolling tail.
CAUSE_RE = _rx(
    r"Sysmem mapped at unexpected NOC address", r"CHIP_IN_USE", r"stale process holding",
    r"Address already in use", r"Out of memory|OutOfMemory|OOM",
    r"unrecognized arguments", r"error: argument", r"pull access denied",
    # tt-metal's own fatal, and the two ways vLLM reports the engine dying with it.
    r"TT_THROW", r"Engine core initialization failed", r"EngineCore failed to start",
)

#: Per-request access logs and polling chatter.
IGNORE_RE = _rx(
    r'"(?:GET|POST) /\S* HTTP/[\d.]+"', r"utils\.prompt_client", r"utils\.cache_monitor",
    r"Avg prompt throughput:",
)


def phases_for(engines: Sequence[str]) -> tuple[Phase, ...]:
    """The container template for a model, chosen by its spec engines.

    vLLM containers boot vLLM directly; `media` and `forge` models run inside
    tt-media-server, which wraps the engine in a worker pool.
    """
    # Substring, not equality: the same stack is called "vLLM" by
    # tt-inference-server's spec and "vllm-plugin" (once "vllm-fork") by a
    # tt-model bundle manifest.
    return VLLM_PHASES if any("vllm" in e.lower() for e in engines) else MEDIA_PHASES
