# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Upload the spooled spans. Runs detached, or synchronously via `tt self send-telemetry`.

This is the only place in `tt` that talks to the collector, and it is never on a
command's critical path (see spool.py for why). It rebuilds the OTLP protobuf payload
from the spooled JSON Lines and POSTs it directly with httpx rather than going back
through `OTLPSpanExporter`, for three reasons:

- the exporter's `force_flush` ignores its timeout and its retry sequence has no knob
  (`_MAX_RETRYS = 6`, exponential backoff), so the only way to bound it is externally;
- ~350 ms of the ~400 ms per-export cost is unattributed overhead inside the exporter,
  while a raw POST of the same bytes takes 11-50 ms;
- going back through the exporter would mean reconstructing SDK `ReadableSpan` objects
  from JSON, which is lossier and more fragile than mapping JSON to the wire format.

The payload is byte-identical to what the direct exporter sends — pinned by
`test_spooled_payload_matches_the_direct_exporter_bytes`, so this shortcut cannot
silently drift from the path that was verified against the real endpoint.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from ..config.paths import Paths
from ..config.store import ConfigStore
from .spool import Spool

# One attempt, generously bounded. No retry loop: a failed batch stays in
# spool.sending.jsonl and the next drain retries it, which gets us at-least-once
# delivery without a backoff sequence that could keep a detached process alive for
# minutes on a firewalled machine.
_POST_TIMEOUT_S = 10.0

# OTLP/JSON encodes span and trace ids as hex strings, but protobuf's canonical JSON
# mapping for `bytes` fields is base64. Feeding the spooled hex straight to ParseDict
# does NOT fail loudly — it base64-decodes the hex text into 24 bytes of garbage and
# reports success, producing spans with invalid, unrelated trace ids. Since PostHog
# answers 200 for payloads it cannot use, that corruption would have been invisible.
_ID_FIELDS = frozenset({"traceId", "spanId", "parentSpanId"})


@dataclass(frozen=True)
class DrainResult:
    """Outcome of one drain. `status` is one of: sent, empty, discarded, busy, disabled,
    unconfigured, failed, error — reported by `tt self send-telemetry`, ignored by the
    detached uploader (nobody is listening, and a failed batch is retried anyway)."""

    status: str
    spans: int = 0
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
            # Spool.MAX_SPANS bound it.
            return DrainResult("unconfigured")

        lines = spool.take()
        if not lines:
            return DrainResult("empty")

        payload, spans = build_payload(lines)
        if spans == 0:
            # Nothing salvageable (e.g. a torn write); don't retry it forever.
            spool.sent()
            return DrainResult("empty")

        status, detail = _post(endpoint, token, payload)
        if status is not None and 200 <= status < 300:
            spool.sent()
            return DrainResult("sent", spans=spans)
        # Left in spool.sending.jsonl for the next drain to retry.
        return DrainResult("failed", spans=spans, detail=detail or f"HTTP {status}")


def build_payload(lines: list[str]) -> tuple[bytes, int]:
    """Rebuild the OTLP protobuf request from spooled JSONL. Returns (bytes, span count).

    Unparseable lines are skipped rather than fatal: concurrent appenders make a torn
    line possible in principle, and one bad record must not strand the whole batch.
    """
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    merged = ExportTraceServiceRequest()
    # Spans from different commands carry the same resource and scope, so they are
    # regrouped into one block each instead of one block per line. That is what makes
    # the batch byte-identical to a single direct export of the same spans.
    resource_blocks: dict[bytes, Any] = {}
    count = 0
    for line in lines:
        one = _parse_line(line, ExportTraceServiceRequest)
        if one is None:
            continue
        for resource_spans in one.resource_spans:
            key = resource_spans.resource.SerializeToString()
            block = resource_blocks.get(key)
            if block is None:
                block = merged.resource_spans.add()
                block.resource.CopyFrom(resource_spans.resource)
                block.schema_url = resource_spans.schema_url
                resource_blocks[key] = block
            for scope_spans in resource_spans.scope_spans:
                count += len(scope_spans.spans)
                _merge_scope(block, scope_spans)
    return merged.SerializeToString(), count


def _parse_line(line: str, message_type: Any) -> Any:
    from google.protobuf.json_format import ParseDict

    try:
        return ParseDict(_hex_ids_to_base64(json.loads(line)), message_type())
    except Exception:
        return None


def _merge_scope(block: Any, scope_spans: Any) -> None:
    key = scope_spans.scope.SerializeToString()
    for existing in block.scope_spans:
        if existing.scope.SerializeToString() == key:
            existing.spans.extend(scope_spans.spans)
            return
    block.scope_spans.append(scope_spans)


def _hex_ids_to_base64(node: Any) -> Any:
    """Re-encode OTLP/JSON's hex trace/span ids as the base64 protobuf JSON expects."""
    if isinstance(node, dict):
        return {
            key: base64.b64encode(bytes.fromhex(value)).decode("ascii")
            if key in _ID_FIELDS and isinstance(value, str)
            else _hex_ids_to_base64(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_hex_ids_to_base64(item) for item in node]
    return node


def _post(endpoint: str, token: str, payload: bytes) -> tuple[int | None, str]:
    import httpx

    try:
        response = httpx.post(
            endpoint,
            content=payload,
            headers={
                "Content-Type": "application/x-protobuf",
                "Authorization": f"Bearer {token}",
            },
            timeout=_POST_TIMEOUT_S,
        )
        return response.status_code, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
