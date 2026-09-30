# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Which Tenstorrent chips running containers hold, read the way tt-model reads it
(tt_kernel/container.py: _claimed_from_container), so tt, tt-model and tt-studio
deployments all see each other."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ...tools.runner import Runner

TT_DEVICE = "/dev/tenstorrent"
ROOT_ENV = "TT_DEVICE_ROOT"
_NODE_RE = re.compile(r"^/dev/tenstorrent/(\d+)$")
_WHOLE_BOARD = {"/", "/dev", "/dev/tenstorrent"}
_DEVICES_LABEL = "org.tenstorrent.tt-model.devices"


def all_ids() -> list[int]:
    try:
        root = Path(os.environ.get(ROOT_ENV) or TT_DEVICE)
        return sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
    except OSError:
        return []


def claimed(runner: Runner, runtime: str, ids: list[int]) -> set[int] | None:
    ps = runner.capture([runtime, "ps", "-q"], tool=runtime, check=False)
    if ps.returncode != 0:
        return None
    running = ps.stdout.split()
    if not running:
        return set()
    inspected = runner.capture([runtime, "inspect", *running], tool=runtime, check=False)
    if inspected.returncode != 0:
        return None
    try:
        infos = json.loads(inspected.stdout)
    except ValueError:
        return None
    taken: set[int] = set()
    for info in infos if isinstance(infos, list) else []:
        if isinstance(info, dict):
            taken |= _claims(info, ids)
    return taken


def _claims(info: dict, ids: list[int]) -> set[int]:
    labels = (info.get("Config") or {}).get("Labels") or {}
    taken = {int(x) for x in str(labels.get(_DEVICES_LABEL) or "").split(",") if x.strip().isdigit()}
    host = info.get("HostConfig") or {}
    ipc = str(host.get("IpcMode") or "")
    # Only a container in the host ipc namespace contends for the UMD lock; one
    # that mounts the devices privately (tt-studio's backend) holds nothing.
    if ipc != "host" and not ipc.startswith("container:"):
        return taken
    if host.get("Privileged"):
        return set(ids)
    paths = [d.get("PathOnHost") or "" for d in host.get("Devices") or []]
    paths += [m.get("Source") or "" for m in info.get("Mounts") or []]
    for path in filter(None, paths):
        path = path.rstrip("/") or "/"
        if path in _WHOLE_BOARD:
            return set(ids)
        match = _NODE_RE.match(path)
        if match:
            taken.add(int(match.group(1)))
    return taken
