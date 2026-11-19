# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Which Tenstorrent chips are held: by running containers, read the way tt-model reads
it (tt_kernel/container.py: _claimed_from_container), and by tt-bio runs, read from
tt-bio's own per-card leases (tt_bio/device_lease.py) — so tt, tt-model, tt-studio
deployments and tt-bio all see each other."""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
from pathlib import Path

from ...tools.runner import Runner

TT_DEVICE = "/dev/tenstorrent"
ROOT_ENV = "TT_DEVICE_ROOT"
_NODE_RE = re.compile(r"^/dev/tenstorrent/(\d+)$")
_WHOLE_BOARD = {"/", "/dev", "/dev/tenstorrent"}
_DEVICES_LABEL = "org.tenstorrent.tt-model.devices"
LEASE_DIR_ENV = "TT_BIO_LEASE_DIR"
LEASE_HOST_ENV = "TT_BIO_LEASE_HOST"
_LEASE_FALLBACK_DIR = Path("/tmp/tt-bio-device-leases")


def all_ids() -> list[int]:
    try:
        root = Path(os.environ.get(ROOT_ENV) or TT_DEVICE)
        return sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
    except OSError:
        return []


def claimed(runner: Runner, runtime: str, ids: list[int]) -> set[int] | None:
    """The chips a running container or a live tt-bio run holds, or None when the
    container side cannot be read (a chip tt could not check is never reported free)."""
    taken = _container_claims(runner, runtime, ids)
    if taken is None:
        return None
    return taken | leased(ids)


def _container_claims(runner: Runner, runtime: str, ids: list[int]) -> set[int] | None:
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


def lease_dir() -> Path:
    """tt-bio's lease directory, resolved exactly as tt_bio.device_lease.lease_dir does."""
    override = os.environ.get(LEASE_DIR_ENV)
    if override:
        return Path(override)
    coworker = Path.home() / ".coworker"
    if coworker.is_dir():
        return coworker / "state" / "leases"
    return _LEASE_FALLBACK_DIR


def leased(ids: list[int]) -> set[int]:
    """The chips a live tt-bio run holds a lease on.

    tt-bio takes an exclusive flock on `<host>-card<N>.json` for as long as it has
    the card open, and the kernel drops it when the process dies, however it dies.
    So the file existing proves nothing — a crashed run leaves one behind — and only
    a held flock is a claim. Leases for other hosts (the directory can be shared
    by a fleet) are ignored.
    """
    host = os.environ.get(LEASE_HOST_ENV) or socket.gethostname()
    directory = lease_dir()
    return {i for i in ids if _flock_held(directory / f"{host}-card{i}.json")}


def _flock_held(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        # A lease file we cannot open is one we cannot check: report it held.
        return True
    try:
        # Non-blocking probe: getting the lock means nobody holds it, so let it go
        # at once rather than sit on a card tt-bio may be about to open.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # EWOULDBLOCK is a live holder; anything else means we could not check.
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def free(runner: Runner, runtime: str, ids: list[int] | None = None) -> list[int] | None:
    """The chips no running container or tt-bio run holds, or None when that cannot be determined.

    `claimed` already answers "which are taken"; this is the complement, which is
    what a placement decision actually wants. None (runtime unreachable, inspect
    unparseable) means *unknown*, never "all of them" — a caller must not place
    work on a chip it failed to check.
    """
    ids = all_ids() if ids is None else ids
    if not ids:
        return None
    taken = claimed(runner, runtime, ids)
    if taken is None:
        return None
    return [i for i in ids if i not in taken]
