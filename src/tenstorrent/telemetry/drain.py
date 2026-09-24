# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Upload the spooled events. Runs detached, or synchronously via `tt self send-telemetry`.

This is the only place in `tt` that talks to PostHog, and it is never on a command's
critical path (see spool.py for why). The upload is PostHog's batch capture API — one
JSON document, ``{"api_key", "batch": [event, ...], "sent_at"}`` POSTed to
``.../batch/`` — sent with httpx directly. No PostHog SDK: its background consumer
thread and atexit flush are exactly the block-on-exit behaviour the spool exists to
avoid, and the wire format is a plain JSON object we can produce verbatim from the
spool lines.

A 2xx from PostHog proves the request was *accepted*, not that every event was
ingested: it answers 200 and silently drops an event with no `event` name or no
`distinct_id`. So `build_batch` refuses such lines itself, and the tests assert on
what a loopback collector actually received, never on the status code alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config.paths import Paths
from ..config.store import ConfigStore
from .spool import Spool

# One attempt, generously bounded. No retry loop: a failed batch stays in
# events.sending.jsonl and the next drain retries it, which gets us at-least-once
# delivery without a backoff sequence that could keep a detached process alive for
# minutes on a firewalled machine. Per-event uuids make the retry idempotent.
_POST_TIMEOUT_S = 10.0
# How much of an error response to keep. PostHog's failures are short JSON documents
# that say what went wrong ("Project API key invalid").
_DETAIL_CHARS = 200

# A spooled line must carry these to be worth sending; PostHog drops the event
# otherwise (with a 200), and `uuid` is what makes a retried batch safe to resend.
_REQUIRED_FIELDS = ("event", "distinct_id", "uuid")


@dataclass(frozen=True)
class DrainResult:
    """Outcome of one drain. `status` is one of: sent, empty, discarded, busy, disabled,
    unconfigured, failed, error — reported by `tt self send-telemetry`, ignored by the
    detached uploader (nobody is listening, and a failed batch is retried anyway)."""

    status: str
    events: int = 0
    detail: str = ""


def drain(paths: Paths, config: ConfigStore) -> DrainResult:
    """Upload (or discard) everything in the spool. Never raises."""
    try:
        return _drain(paths, config)
    except Exception as exc:  # a best-effort uploader must not crash the process
        return DrainResult("error", detail=f"{type(exc).__name__}: {exc}")


def _drain(paths: Paths, config: ConfigStore) -> DrainResult:
    from .session import opted_out, resolve_endpoint

    spool = Spool(paths)
    with spool.lock() as acquired:
        if not acquired:
            # Another drainer is already uploading this spool.
            return DrainResult("busy")

        # Span spools from tt <= 1.0.1 cannot be uploaded any more (see spool.py).
        spool.remove_legacy()

        if opted_out(config):
            # Re-checked here, not just at collection time: the spool is a window in
            # which the user can revoke their opt-in before the data has left the
            # machine, and honouring that late opt-out is the whole point of deleting
            # rather than uploading.
            spool.discard()
            return DrainResult("discarded")

        # A per-run kill switch (or CI) means "do nothing now", NOT "opt out" — so the
        # batch is kept for a later run rather than deleted.
        if os.environ.get("TT_TELEMETRY_DISABLED"):
            return DrainResult("disabled")

        endpoint, token = resolve_endpoint(config)
        if not endpoint or not token:
            # Unconfigured: keep the batch (the user may be mid-edit) and let
            # Spool.MAX_EVENTS bound it.
            return DrainResult("unconfigured")

        lines = spool.take()
        if not lines:
            return DrainResult("empty")

        batch = build_batch(lines)
        if not batch:
            # Nothing salvageable (e.g. a torn write); don't retry it forever.
            spool.sent()
            return DrainResult("empty")

        status, detail = post_batch(endpoint, token, batch)
        if status is not None and 200 <= status < 300:
            spool.sent()
            return DrainResult("sent", events=len(batch))
        # Left in events.sending.jsonl for the next drain to retry.
        return DrainResult("failed", events=len(batch), detail=detail or f"HTTP {status}")


def build_batch(lines: list[str]) -> list[dict[str, Any]]:
    """The spooled lines as the `batch` array, verbatim.

    Unparseable lines are skipped rather than fatal: concurrent appenders make a torn
    line possible in principle, and one bad record must not strand the whole batch.
    Records missing a required field are skipped too — PostHog would accept and then
    silently discard them.
    """
    batch: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if not all(record.get(field) for field in _REQUIRED_FIELDS):
            continue
        batch.append(record)
    return batch


def build_payload(token: str, batch: list[dict[str, Any]]) -> bytes:
    """The request body. `sent_at` is the upload time: PostHog compares it with its own
    receive time to correct the client's clock skew, and one offset per batch is exactly
    right for events that were all stamped by the same clock."""
    return json.dumps(
        {
            "api_key": token,
            "batch": batch,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def post_batch(endpoint: str, token: str, batch: list[dict[str, Any]]) -> tuple[int | None, str]:
    """POST one batch. Returns (HTTP status or None, detail). Never raises."""
    import httpx

    try:
        response = httpx.post(
            endpoint,
            content=build_payload(token, batch),
            headers={"Content-Type": "application/json"},
            timeout=_POST_TIMEOUT_S,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if 200 <= response.status_code < 300:
        return response.status_code, ""
    body = " ".join(response.text.split())[:_DETAIL_CHARS]
    return response.status_code, f"HTTP {response.status_code}" + (f": {body}" if body else "")
