# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Ask a running server what it is serving.

Models served by tt are unauthenticated, so one GET /v1/models answers both "is
anything there" and "what model id does it expect" — for tt-inference-server and
tt-model containers alike, without involving docker.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..errors import ExitCode, TTError
from .base import RunningModel

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_TIMEOUT_S = 5.0


def base_url_for(url: str | None, port: int | None) -> str:
    """The endpoint to probe, from --url, --port, or the server's own default."""
    if url and port:
        raise TTError(
            "--url and --port cannot both be given.",
            why="They set the same thing.",
            next_step="Pass --port for a local server, --url for anything else.",
            exit_code=ExitCode.USAGE,
        )
    if url:
        return url.rstrip("/")
    if port:
        return f"http://127.0.0.1:{port}/v1"
    return DEFAULT_BASE_URL


def discover(base_url: str) -> list[RunningModel]:
    """Every model the server at `base_url` reports, in its own order."""
    url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            doc = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise TTError(
            f"No OpenAI-compatible server answered at {base_url}.",
            why=f"GET {url} failed ({exc}).",
            # `tt serve` returns once the container is listed, which is minutes
            # before a large model finishes loading — so "not up yet" is the more
            # likely cause than "not started".
            next_step="If you just started it, the model is probably still loading — "
            "watch `docker logs -f <container>` and retry. Otherwise serve one "
            "(`tt serve <model>`), or pass --port/--url if it is listening elsewhere.",
            exit_code=ExitCode.ERROR,
            details={"base_url": base_url},
        ) from exc
    served = [
        RunningModel(
            served_id=str(item["id"]),
            base_url=base_url,
            max_context=item.get("max_model_len"),
        )
        for item in (doc.get("data") or [])
        if item.get("id")
    ]
    if not served:
        raise TTError(
            f"The server at {base_url} reports no models.",
            why="GET /v1/models returned an empty list.",
            next_step="Check the server's own log; it may still be starting up.",
            exit_code=ExitCode.ERROR,
            details={"base_url": base_url},
        )
    return served
