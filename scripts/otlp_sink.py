# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Local OTLP/HTTP trace receiver, for eyeballing what `tt` telemetry actually sends.

Stands in for PostHog: decodes the protobuf spans and prints them, so you can see the
exact wire payload — resource attributes, span name, allowlisted argument values, exit
code — without a PostHog project or a network round trip.

Run it inside the project venv (it needs `opentelemetry.proto`, which arrives with the
OTLP exporter dependency):

    uv run scripts/otlp_sink.py                 # loopback only
    uv run scripts/otlp_sink.py --host 0.0.0.0  # reachable from another machine

It prints the env block to paste into the shell where you run `tt`. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
except ModuleNotFoundError:  # pragma: no cover - operator error, not a code path
    sys.exit(
        "opentelemetry.proto is missing — run this inside the project venv:\n"
        "    uv run scripts/otlp_sink.py"
    )

TRACES_PATH = "/i/v1/traces"  # PostHog's path; tt passes the endpoint through verbatim
# Attributes every span carries, so the per-span line can highlight just the interesting
# ones (tt.model, tt.hardware, tt.config_key, …) without repeating the boilerplate.
_ALWAYS = ("tt.command", "tt.options_set", "tt.exit_code", "tt.exit_code_name")

_count = 0


def _unwrap(value):
    """OTLP AnyValue -> plain Python."""
    field = value.WhichOneof("value")
    if field == "array_value":
        return [_unwrap(v) for v in value.array_value.values]
    if field == "kvlist_value":
        return {kv.key: _unwrap(kv.value) for kv in value.kvlist_value.values}
    return getattr(value, field) if field else None


def _primary_ip() -> str | None:
    """Best-effort LAN address, for pointing another box at this sink. Sends nothing."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    compact = False
    log_file = None
    stdout_broken = False

    # -- plumbing ---------------------------------------------------------------
    def _respond(self, status: int, body: bytes = b"", content_type: str = "application/x-protobuf"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args):  # silence the default access log
        pass

    def _emit(self, text: str) -> None:
        # A broken stdout (piped into `head`/`less` and closed) must not take the
        # receiver down with it — otherwise the next POST gets no response and the
        # sender reports a confusing RemoteDisconnected instead of "sink is gone".
        if not Handler.stdout_broken:
            try:
                print(text, flush=True)
            except BrokenPipeError:
                Handler.stdout_broken = True
                try:
                    sys.stderr.write("stdout closed; still receiving (use --log to keep output)\n")
                except OSError:
                    pass
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    # -- handlers ---------------------------------------------------------------
    def do_GET(self):
        # Liveness, so you can `curl http://host:port/` from the other machine.
        self._respond(200, b"tt OTLP sink alive\n", "text/plain")

    def do_POST(self):
        global _count
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)

        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        peer = self.client_address[0]

        request = ExportTraceServiceRequest()
        try:
            request.ParseFromString(body)
        except Exception as exc:  # a sink must never die on bad input
            self._emit(f"\n[{stamp}] {peer} UNPARSEABLE ({len(body)} bytes): {exc}")
            self._respond(200)
            return

        if self.path != TRACES_PATH:
            # Not fatal: tt sends wherever TT_TELEMETRY_ENDPOINT points. Worth flagging,
            # since a wrong path is silent against real PostHog.
            self._emit(f"\n[{stamp}] note: POST to {self.path!r}, expected {TRACES_PATH!r}")

        payload = {
            "received_at": stamp,
            "from": peer,
            "http": {
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "content_encoding": self.headers.get("Content-Encoding"),
                "authorization": self.headers.get("Authorization"),
                "user_agent": self.headers.get("User-Agent"),
            },
            "resource_spans": [],
        }

        for resource_spans in request.resource_spans:
            resource = {a.key: _unwrap(a.value) for a in resource_spans.resource.attributes}
            spans = []
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    attrs = {a.key: _unwrap(a.value) for a in span.attributes}
                    duration_ms = round(
                        (span.end_time_unix_nano - span.start_time_unix_nano) / 1e6, 1
                    )
                    _count += 1
                    extra = {k: v for k, v in attrs.items() if k not in _ALWAYS}
                    self._emit(
                        f"\n[{stamp}] {peer}  {span.name!r}  "
                        f"exit={attrs.get('tt.exit_code_name', '?')}  {duration_ms}ms  "
                        f"opts={list(attrs.get('tt.options_set') or [])}"
                        + (f"  {extra}" if extra else "")
                    )
                    spans.append(
                        {
                            "name": span.name,
                            "scope": scope_spans.scope.name,
                            "trace_id": span.trace_id.hex(),
                            "span_id": span.span_id.hex(),
                            "duration_ms": duration_ms,
                            "status_code": span.status.code,
                            "attributes": attrs,
                        }
                    )
            payload["resource_spans"].append({"resource": resource, "spans": spans})

        if not self.compact:
            self._emit(json.dumps(payload, indent=2))
        self._respond(200)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local OTLP/HTTP trace receiver for tt telemetry.",
        epilog="Point tt at it with TT_TELEMETRY_ENDPOINT + TT_TELEMETRY_POSTHOG_KEY.",
    )
    parser.add_argument("--port", type=int, default=4318, help="listen port (default 4318)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; use 0.0.0.0 to accept spans from another machine "
        "(default 127.0.0.1, loopback only)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="one line per span, no full JSON payload"
    )
    parser.add_argument("--log", metavar="FILE", help="also append output to FILE")
    parser.add_argument("--show-info", help="Print info output")
    args = parser.parse_args()

    Handler.compact = args.compact
    Handler.log_file = open(args.log, "a") if args.log else None

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"cannot bind {args.host}:{args.port} — {exc}", file=sys.stderr)
        return 1

    # The endpoint tt should use: 0.0.0.0 isn't dialable, so suggest the LAN address.
    advertised = args.host
    if args.host in ("0.0.0.0", "::"):
        advertised = _primary_ip() or "<this-machine-ip>"
    endpoint = f"http://{advertised}:{args.port}{TRACES_PATH}"

    if not args.show_info:
        print(f"OTLP sink listening on http://{args.host}:{args.port}{TRACES_PATH}")
        if args.host == "127.0.0.1":
            print("  loopback only — pass --host 0.0.0.0 to accept spans from another machine")
        print("\nIn the shell where you run tt:\n")
        print(f"  export TT_TELEMETRY_ENDPOINT={endpoint}")
        print("  export TT_TELEMETRY_POSTHOG_KEY=local-sink-key   # any non-empty value")
        print("  export TT_TELEMETRY_FLUSH_MODE=sync              # see spans immediately")
        print("  unset TT_TELEMETRY_DISABLED")
        print(
            "\nReminders: an empty key makes the session inert (nothing sent, no notice), "
            "and\n`--offline` disables telemetry by design."
            "\n\nWithout FLUSH_MODE=sync, tt spools spans to "
            "$TT_DATA_DIR/telemetry/spool.jsonl and\nonly uploads once ~20 have accumulated "
            "(or the oldest is ~30 min old), from a\ndetached process — so nothing shows up "
            "here for the first several commands. That is\nthe shipping default; "
            "`tt self send-telemetry` forces a drain if you want to watch one."
        )
        if args.host in ("0.0.0.0", "::"):
            print(
                f"\nFrom the other machine, check reachability first:\n"
                f"  curl http://{advertised}:{args.port}/\n"
                f"If that hangs or says 'No route to host', a firewall is rejecting it."
            )
        print("\nCtrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped after {_count} span(s)")
    finally:
        server.server_close()
        if Handler.log_file:
            Handler.log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
