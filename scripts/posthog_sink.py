# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Local stand-in for PostHog's batch capture endpoint, for eyeballing what `tt` sends.

Accepts the same `POST /batch/` request `tt` makes to PostHog and prints every event in
it — properties, distinct id, capture timestamp — so you can check the exact wire
payload without a PostHog project or a network round trip. Standard library only.

    uv run scripts/posthog_sink.py                 # loopback only
    uv run scripts/posthog_sink.py --host 0.0.0.0  # reachable from another machine
    uv run scripts/posthog_sink.py --compact       # one line per event

It prints the env block to paste into the shell where you run `tt`. Export
TT_TELEMETRY_FLUSH_MODE=sync there as well, or nothing arrives until the spool crosses
its hand-off threshold (~20 commands) — `tt self send-telemetry` forces a drain if you
want to watch the async path instead. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BATCH_PATH = "/batch/"  # PostHog's path; tt passes the endpoint through verbatim
# Stamped on every event; the per-event line highlights only what varies between them.
BORING = {"tt_version", "os_type", "os_arch", "python_version", "$lib", "$lib_version", "$geoip_disable", "$set", "$set_once"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


class Handler(BaseHTTPRequestHandler):
    compact = False
    log = None

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        try:
            body = json.loads(raw)
        except ValueError:
            self._reply(400, {"status": 0, "error": "body is not JSON"})
            self._out(f"[{_now()}] {self.path}: {length} bytes of non-JSON")
            return
        if self.path != BATCH_PATH:
            self._out(f"[{_now()}] WARNING: POST to {self.path}, PostHog expects {BATCH_PATH}")
        batch = body.get("batch") if isinstance(body, dict) else None
        if not isinstance(batch, list):
            self._reply(400, {"status": 0, "error": "no batch array"})
            self._out(f"[{_now()}] {self.path}: JSON without a `batch` array: {raw[:200]!r}")
            return
        key = body.get("api_key") or ""
        self._out(
            f"[{_now()}] {len(batch)} event(s)  api_key={key[:8]}…  sent_at={body.get('sent_at')}"
        )
        for event in batch:
            self._print_event(event)
        self._reply(200, {"status": 1})

    def _print_event(self, event: dict) -> None:
        props = dict(event.get("properties") or {})
        if self.compact:
            interesting = {k: v for k, v in props.items() if k not in BORING}
            self._out(
                f"  {event.get('event')}  {event.get('timestamp')}  "
                f"{event.get('distinct_id', '')[:8]}…  {json.dumps(interesting, sort_keys=True)}"
            )
            return
        self._out(json.dumps(event, indent=2, sort_keys=True))

    def _reply(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _out(self, line: str) -> None:
        print(line, flush=True)
        if self.log is not None:
            self.log.write(line + "\n")
            self.log.flush()

    def log_message(self, *args) -> None:  # silence the default access log
        pass


def _advertised_host(bind_host: str) -> str:
    if bind_host not in ("0.0.0.0", ""):
        return bind_host
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "<this-host>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default loopback)")
    parser.add_argument("--port", type=int, default=0, help="port (default: any free one)")
    parser.add_argument("--compact", action="store_true", help="one line per event")
    parser.add_argument("--log", metavar="FILE", help="also append everything printed to FILE")
    args = parser.parse_args()

    Handler.compact = args.compact
    Handler.log = open(args.log, "a", encoding="utf-8") if args.log else None
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    port = server.server_port
    print(
        f"posthog_sink listening on {args.host}:{port}\n"
        "Paste into the shell running tt:\n\n"
        f"    export TT_TELEMETRY_ENDPOINT=http://{_advertised_host(args.host)}:{port}{BATCH_PATH}\n"
        "    export TT_TELEMETRY_POSTHOG_KEY=phc_local_sink\n"
        "    export TT_TELEMETRY_FLUSH_MODE=sync\n"
        "    export NO_PROXY=127.0.0.1,localhost\n",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if Handler.log is not None:
            Handler.log.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
