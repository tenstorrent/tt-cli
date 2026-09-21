# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""What is being served on this machine — the engine behind `tt model ps`.

One `docker ps` (running containers, or every container with --all) plus one
`docker inspect` for all of them, then each record is classified in Python. Three
backends start model containers here and each is recognised differently, none by a
shared label, so the listing cannot be a docker filter:

* tt-inference-server — containers named `tt-inference-server-<uuid>`, no labels;
  the model comes from the weights mount (`_identity_from_inspect`, shared with
  `tt model stop`).
* tt-model — every container carries the `org.tenstorrent.tt-model` label family
  (`.repo`, `.profile`, …); the port is whatever it published, never a profile
  default (a live bundle published 20000).
* TT-Studio — model containers are named after the model and built from the
  `studio_images` image family; studio's own services (backend on 8000, frontend,
  agent, litellm, chroma) use other images and are excluded by that rule.

Every rule is pinned to upstream naming and belongs on the re-check list when a
pin is bumped, alongside `_BOARDS_TO_DEVICE`. A rename upstream makes rows vanish;
it never makes the listing fail.

Health is an HTTP probe — GET /v1/models on each published port, in parallel with
a short timeout — plus the default endpoint `tt launch` would try, so a server tt
did not start (or whose container we cannot recognise) still shows up, as
backend "unknown".
"""

from __future__ import annotations

import dataclasses
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
from urllib.parse import urlparse

from ...launchers import discovery
from ...models.served import ServedModel
from ...tools.runner import Runner
from .inference_server import CONTAINER_PREFIX, ServerContainer, _identity_from_inspect

UTC = timezone.utc

INFERENCE_SERVER = "inference-server"
MODEL_MANAGER = "model-manager"
STUDIO = "studio"
UNKNOWN = "unknown"

# tt-model-manager labels every container it starts (src/tt_kernel/container.py):
# org.tenstorrent.tt-model=<name>, plus .repo=<namespace/name>, .profile, .kind,
# .arch, .weights. Verified against a live container 2026-09-08.
TT_MODEL_LABEL = "org.tenstorrent.tt-model"
# TT-Studio model containers carry no tt labels and are named after the model
# (`Qwen3.5-9B`); the image family is the only thing that ties them to studio.
STUDIO_IMAGE_PREFIX = "ghcr.io/tenstorrent/tt-studio/studio_images"
# Short on purpose: a server that is up answers /v1/models in milliseconds, and
# this runs once per row on every `tt model ps`.
PROBE_TIMEOUT_S = 1.5
_MAX_PROBES = 8

_FRACTION_RE = re.compile(r"\.\d+")


def list_served(
    runner: Runner,
    runtime: str,
    *,
    include_stopped: bool = False,
    probe: bool = True,
    now: datetime | None = None,
) -> list[ServedModel]:
    """Every model server on this machine, running rows first."""
    now = now or datetime.now(UTC)
    rows: list[ServedModel] = []
    catalog = _Catalog()
    for entry in _inspect_all(runner, runtime, include_stopped):
        row = _classify(entry, catalog, now)
        if row is not None:
            rows.append(row)
    if probe:
        rows = _probe_rows(rows, catalog)
    rows.sort(key=lambda r: (r.status != "running", r.port or 0, r.name.lower()))
    return rows


def human_duration(seconds: int | None) -> str:
    """42s / 18m / 3h 12m / 1d 2h — the table's uptime cell."""
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


# -- docker ---------------------------------------------------------------------------
def _inspect_all(runner: Runner, runtime: str, include_stopped: bool) -> list[dict]:
    argv = [runtime, "ps"]
    if include_stopped:
        argv.append("--all")
    listed = runner.capture([*argv, "--format", "{{.ID}}"], tool=runtime)
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not ids:
        return []
    inspected = runner.capture(
        [runtime, "inspect", "--format", "{{json .}}", *ids], tool=runtime
    )
    entries: list[dict] = []
    for line in inspected.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict):
            entries.append(doc)
    if not entries and inspected.stdout.strip():
        # podman pretty-prints `inspect` output as one multi-line JSON array.
        with suppress(ValueError):
            doc = json.loads(inspected.stdout)
            if isinstance(doc, list):
                entries = [d for d in doc if isinstance(d, dict)]
    return entries


# -- classification -------------------------------------------------------------------
class _Catalog:
    """The model catalog, built once and only if a row needs it (it reads a JSON
    file — cheap, but `tt model ps` on a box with only studio rows never should)."""

    def __init__(self) -> None:
        self._models = None

    def models(self):
        if self._models is None:
            from ...modelhub.catalog import ModelCatalog

            # cached_sizes={} skips the HF cache scan: matching only needs names.
            self._models = ModelCatalog().list(cached_sizes={})
        return self._models

    def name_for(self, served_id: str) -> str:
        wanted = served_id.lower()
        for model in self.models():
            if wanted in (model.name.lower(), model.hf_repo.lower()):
                return model.name
        return served_id


def _classify(entry: dict, catalog: _Catalog, now: datetime) -> ServedModel | None:
    """A ServedModel for a container one of the backends started, else None."""
    config = entry.get("Config") or {}
    name = str(entry.get("Name") or "").lstrip("/")
    image = str(config.get("Image") or "")
    labels = config.get("Labels") or {}
    profile = None

    if name.startswith(CONTAINER_PREFIX):
        backend = INFERENCE_SERVER
        hf_repo, volume = _identity_from_inspect(entry)
        container = ServerContainer(id="", name=name, image=image, hf_repo=hf_repo, volume=volume)
        matched = next((m for m in catalog.models() if container.matches(m)), None)
        model_name = matched.name if matched else (hf_repo or volume or name)
    elif TT_MODEL_LABEL in labels:
        backend = MODEL_MANAGER
        model_name = str(labels.get(f"{TT_MODEL_LABEL}.repo") or labels[TT_MODEL_LABEL])
        profile = labels.get(f"{TT_MODEL_LABEL}.profile") or None
    elif image.startswith(STUDIO_IMAGE_PREFIX):
        backend = STUDIO
        model_name = name
    else:
        return None

    state = entry.get("State") or {}
    status = str(state.get("Status") or ("running" if state.get("Running") else "exited"))
    running = status == "running"
    started = _started_at(entry)
    port = _published_port(entry)
    return ServedModel(
        name=model_name,
        backend=backend,
        container=name,
        container_id=str(entry.get("Id") or "")[:12] or None,
        image=image or None,
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1" if port else None,
        status=status,
        health="unknown" if running else "stopped",
        started_at=started.strftime("%Y-%m-%dT%H:%M:%SZ") if started else None,
        uptime_s=int((now - started).total_seconds()) if running and started else None,
        profile=profile,
        served_id=None,
    )


def _published_port(entry: dict) -> int | None:
    """The lowest host port the container publishes, or None.

    HostConfig.PortBindings, not NetworkSettings.Ports: the latter is empty while
    the container is stopped. IPv4 and IPv6 bindings of one port collapse; a
    container publishing several ports reports the lowest, deterministically."""
    bindings = (entry.get("HostConfig") or {}).get("PortBindings") or {}
    ports: set[int] = set()
    for published in bindings.values():
        for binding in published or []:
            with suppress(TypeError, ValueError, AttributeError):
                ports.add(int(binding.get("HostPort")))
    return min(ports) if ports else None


def _started_at(entry: dict) -> datetime | None:
    """State.StartedAt as an aware UTC datetime, or None for a never-started
    container (docker writes the zero time 0001-01-01T00:00:00Z)."""
    raw = str((entry.get("State") or {}).get("StartedAt") or "")
    if not raw or raw.startswith("0001-"):
        return None
    # docker prints nanoseconds (8-9 digits) and a trailing "Z"; fromisoformat
    # rejects the former everywhere and the latter before Python 3.11.
    trimmed = _FRACTION_RE.sub("", raw, count=1)
    if trimmed.endswith("Z"):
        trimmed = trimmed[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# -- health -----------------------------------------------------------------------------
def _probe_rows(rows: list[ServedModel], catalog: _Catalog) -> list[ServedModel]:
    """Mark running rows healthy/starting from GET /v1/models, and add a row for a
    server at the default endpoint that no container explains."""
    default_url = discovery.DEFAULT_BASE_URL
    default_port = urlparse(default_url).port
    urls: dict[str, None] = {}
    for row in rows:
        if row.status == "running" and row.base_url:
            urls[row.base_url] = None
    probe_default = not any(r.port == default_port and r.status == "running" for r in rows)
    if probe_default:
        urls[default_url] = None
    if not urls:
        return rows

    def one(url: str):
        return discovery.probe(url, timeout_s=PROBE_TIMEOUT_S)

    with ThreadPoolExecutor(max_workers=min(_MAX_PROBES, len(urls))) as pool:
        results = dict(zip(urls, pool.map(one, urls)))

    probed: list[ServedModel] = []
    for row in rows:
        if row.status != "running" or not row.base_url:
            probed.append(row)
            continue
        served = results.get(row.base_url)
        if served:
            probed.append(
                dataclasses.replace(row, health="healthy", served_id=served[0].served_id)
            )
        else:
            probed.append(dataclasses.replace(row, health="starting"))

    # The default endpoint is probed on its own only when no running row owns that
    # port, so anything answering there is a server no container explains — tt did
    # not start it, or cannot recognise it.
    for model in (results.get(default_url) or []) if probe_default else []:
        probed.append(
            ServedModel(
                name=catalog.name_for(model.served_id),
                backend=UNKNOWN,
                container=None,
                container_id=None,
                image=None,
                port=default_port,
                base_url=default_url,
                status="running",
                health="healthy",
                started_at=None,
                uptime_s=None,
                profile=None,
                served_id=model.served_id,
            )
        )
    return probed
