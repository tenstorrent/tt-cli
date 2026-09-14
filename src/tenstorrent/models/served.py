# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The typed row behind `tt model ps` and its --json.

One ServedModel per model server on this machine, whichever backend started it.
TT-Studio and `tt report issue` read this shape, so field order IS the --json
contract and the `backend` / `health` vocabularies are closed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServedModel:
    """One model server on this machine.

    health:
      healthy  — GET http://127.0.0.1:<port>/v1/models answered with a model list
      starting — the container is running but nothing OpenAI-compatible answered
                 yet (refused, timeout, a foreign server); loading is the usual
                 cause, `docker logs -f <container>` the next step
      stopped  — the container is not running (only listed with --all)
      unknown  — not probed (--no-probe), or running with no published port
    """

    name: str  # catalog name when known, else the best identity we have
    backend: str  # inference-server | model-manager | studio | unknown
    container: str | None  # None for a server found only by probing
    container_id: str | None  # 12 chars
    image: str | None
    port: int | None  # published host port; None if the container publishes none
    base_url: str | None  # http://127.0.0.1:<port>/v1 when the port is known
    status: str  # docker State.Status verbatim: running | exited | created | …
    health: str  # healthy | starting | stopped | unknown
    started_at: str | None  # ISO 8601 UTC, whole seconds; None if never started
    uptime_s: int | None  # running rows only
    profile: str | None  # tt-model profile; None for other backends
    served_id: str | None  # the id GET /v1/models reports, once healthy
