# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Real-hardware smoke tests.

These run only under ``pytest --hardware`` on a machine with a working Tenstorrent
stack installed (they are skipped otherwise — see the ``hardware`` marker gating in
``tests/conftest.py``). Unlike the fake-tool tests, they can't assert exact board
types or telemetry values (those vary by machine), so they check the structural
contract instead: the command succeeds, real devices are reported, and the public
``--json`` shape holds against the real tt-smi output.

Config/data/cache/HF are still redirected to a temp dir by the autouse ``isolated_dirs``
fixture, so a run never touches the user's real ``tt`` state — only the hardware,
through the real tools, is exercised.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode
from tenstorrent.telemetry import spool as spool_module
from tenstorrent.tools.registry import env_var_for

pytestmark = pytest.mark.hardware

# The public --json device contract (must match tests/cli/test_device_cmd.py).
DEVICE_JSON_KEYS = {
    "index", "board_type", "board_id", "bus_id", "coords", "dram_status",
    "pcie_speed", "pcie_width", "voltage_v", "current_a", "power_w",
    "aiclk_mhz", "temperature_c", "firmware",
}


def _json(result):
    """Parse tt's JSON document out of a CliRunner result.

    Warnings go to stderr, which CliRunner merges into `output`, so the document
    need not start at char 0 — on a real machine `tt update` may warn about stale
    tt-installer clones in ~/.local/lib."""
    return json.loads(result.output[result.output.index("{"):])


def test_device_status_reports_real_devices(runner, smi_bin):
    result = runner.invoke(app, ["device", "status", "--json"])
    assert result.exit_code == 0, result.output
    data = _json(result)
    assert set(data) == {"host", "devices", "warnings"}
    assert data["devices"], "no Tenstorrent devices detected by the real tt-smi"
    for dev in data["devices"]:
        assert set(dev) == DEVICE_JSON_KEYS
        assert dev["board_type"], "a real board should report a board_type"


def test_device_status_human_renders(runner, smi_bin):
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == 0, result.output
    assert "Tenstorrent devices" in result.output


def test_device_status_raw_is_tt_smi_shape(runner, smi_bin):
    result = runner.invoke(app, ["device", "status", "--raw"])
    assert result.exit_code == 0, result.output
    assert "device_info" in _json(result)  # tt-smi's own shape


def test_device_info_first_device(runner, smi_bin):
    result = runner.invoke(app, ["device", "info", "0", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["devices"][0]["index"] == 0
    assert payload["devices"][0]["firmware"], "real firmware versions expected"


def test_device_info_out_of_range_is_usage_error(runner, smi_bin):
    # 999 is safely beyond any real machine's device count.
    result = runner.invoke(app, ["device", "info", "999"])
    assert result.exit_code == ExitCode.USAGE
    assert "No such device index" in result.output


def test_self_tools_resolves_real_installs(runner, real_tools, monkeypatch):
    """Point tt at the real binaries we could find and confirm `self tools`
    resolves them (source "env") rather than reporting them missing."""
    resolvable = {n: p for n, p in real_tools.items() if p and n != "uv"}
    if not resolvable:
        pytest.skip("no real tt tools found on PATH or in the installer venv")
    for name, path in resolvable.items():
        monkeypatch.setenv(env_var_for(name), str(path))
    result = runner.invoke(app, ["self", "tools", "--json"])
    assert result.exit_code == 0, result.output
    rows = {t["name"]: t for t in _json(result)["tools"]}
    for name in resolvable:
        assert rows[name]["source"] == "env"
        assert rows[name]["path"]


def test_update_dry_run_plans_against_golden(runner):
    # Plan only — no tool is invoked, so this is safe on a real machine. The isolated
    # data dir means nothing is recorded as installed, so everything plans as install.
    result = runner.invoke(app, ["update", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    plan = _json(result)
    assert plan["dry_run"] is True
    assert plan["items"], "the golden manifest should yield a non-empty plan"


@pytest.mark.destructive
def test_device_reset_all(runner, smi_bin):
    """A real `tt device reset` of every device. Interrupts running workloads, so it
    only runs with `--hardware --run-destructive`. Needs privileges to reset (run as
    root or with passwordless sudo); we drive tt-smi directly without the sudo wrapper.
    """
    assert (
        runner.invoke(app, ["config", "set", "tools.sudo_command", ""]).exit_code == 0
    )
    result = runner.invoke(app, ["device", "reset", "--yes"])
    assert result.exit_code == ExitCode.OK, result.output


def _why(spool, child_env: dict) -> str:
    """Explain a delivery failure from the spool's own state.

    Without this the assertion reads `assert []`, which says only that nothing arrived —
    not whether the spans were spooled, whether an uploader ran, or whether the POST
    itself failed. Each of those has a distinct fingerprint on disk, and reading it is
    the difference between a five-minute diagnosis and an afternoon.
    """
    from tenstorrent.config.paths import get_paths
    from tenstorrent.config.store import ConfigStore
    from tenstorrent.telemetry.drain import drain

    lines = ["  spool state:"]
    try:
        contents = sorted(p.name for p in spool.dir.iterdir())
    except OSError:
        contents = []
    if not contents:
        lines.append(
            f"    {spool.dir} is empty or missing -> nothing was ever spooled, so "
            "telemetry was inert (no project key, not opted in, or sync mode via a "
            "CI marker)."
        )
    else:
        lines.append(f"    {spool.dir}: {', '.join(contents)}")
        if spool.sending_path.exists():
            lines.append(
                "    spool.sending.jsonl exists -> an uploader DID run and claimed the "
                "batch; the POST is what failed (proxy, firewall, or a rejecting "
                "collector). NB: httpx and the OTLP exporter both honour HTTP_PROXY / "
                "ALL_PROXY with no implicit localhost bypass."
            )
        elif spool.path.exists():
            lines.append(
                f"    spool.jsonl still holds {spool.stats().spans} span(s) with no "
                "in-flight batch -> the hand-off never happened (threshold not reached, "
                "or the uploader failed to launch)."
            )
    # Attempt a synchronous drain, which reports the actual transport error instead of
    # leaving us with a timeout. It has to borrow the *child's* telemetry environment:
    # this process has TT_TELEMETRY_DISABLED=1 from isolated_dirs, under which the
    # drainer correctly declines to send and would report a useless "disabled".
    borrowed = {
        key: child_env[key]
        for key in ("TT_TELEMETRY_ENDPOINT", "TT_TELEMETRY_POSTHOG_KEY")
        if key in child_env
    }
    try:
        with mock.patch.dict(os.environ, borrowed):
            os.environ.pop("TT_TELEMETRY_DISABLED", None)
            result = drain(get_paths(), ConfigStore(get_paths()))
        lines.append(f"  synchronous drain says: {result.status} {result.detail}".rstrip())
    except Exception as exc:  # never mask the original assertion
        lines.append(f"  synchronous drain raised: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def test_detached_drainer_delivers_through_a_real_process(isolated_dirs, monkeypatch):
    """The one part of telemetry the fake-mode tests cannot cover: an actual detached
    `tt self send-telemetry` reaching an actual socket.

    In fake mode the drain runs in-process, so a broken spawn (wrong interpreter, wrong
    module path, an inherited descriptor that wedges the parent) would go unnoticed. This
    is hardware-marked only because it needs a real installed `tt` on PATH, not because
    it touches a device — it uses a loopback collector and never leaves the machine.
    """
    import subprocess
    import sys
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from tenstorrent.telemetry.spool import Spool
    from tenstorrent.config.paths import get_paths

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "TT_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{server.server_port}/i/v1/traces",
        "TT_TELEMETRY_POSTHOG_KEY": "phc_hardware_smoke",
    }
    env.pop("TT_TELEMETRY_DISABLED", None)
    try:
        # Telemetry is opt-in, and the isolated TT_CONFIG_DIR starts with no consent
        # recorded — opt in the way a user would, through the real CLI.
        done = subprocess.run(
            [sys.executable, "-m", "tenstorrent", "config", "set", "telemetry.enabled", "true"],
            env=env, capture_output=True, timeout=60,
        )
        assert done.returncode == 0, done.stderr
        # Enough commands to cross the hand-off threshold, each a real subprocess.
        for _ in range(spool_module.DRAIN_SPAN_THRESHOLD):
            done = subprocess.run(
                [sys.executable, "-m", "tenstorrent", "config", "path"],
                env=env, capture_output=True, timeout=60,
            )
            assert done.returncode == 0, done.stderr

        spool = Spool(get_paths())
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not received:
            time.sleep(0.2)

        assert received, f"the detached drainer never reached the collector.\n{_why(spool, env)}"
        # The collector records the body *before* it replies, and the drainer only drops
        # spool.sending.jsonl once it has seen the 2xx — so give it the rest of the
        # deadline to finish rather than asserting inside that window.
        while time.monotonic() < deadline and spool.sending_path.exists():
            time.sleep(0.2)
        assert spool.stats().spans == 0, "spool not cleaned up after a successful upload"
        assert not spool.sending_path.exists(), "in-flight batch not dropped after a successful upload"
    finally:
        server.shutdown()
        server.server_close()


def test_launch_list_reports_the_real_clients(runner):
    """`tt launch list` is read-only: it resolves clients on PATH and asks docker
    about our containers, but starts nothing and writes nothing.

    Structural only — which clients are installed varies by machine.
    """
    result = runner.invoke(app, ["launch", "list", "--json"])
    assert result.exit_code == ExitCode.OK, result.output
    rows = json.loads(result.output[result.output.index("[") :])
    assert rows, "no clients registered"
    for row in rows:
        assert set(row) == {
            "tool", "kind", "requires_tool_calling", "configures",
            "needs", "available", "container_state",
        }
        assert row["kind"] in ("terminal", "web service")
        assert isinstance(row["available"], bool)
        # Only container clients have a state, and it is one of a closed set.
        assert row["container_state"] in (None, "running", "stopped", "absent", "unknown")


def test_launch_disconnect_is_clean_when_nothing_is_configured(runner):
    """Redirected HOME/XDG mean no client is configured, so disconnect has nothing
    to undo — and must say so rather than fail."""
    result = runner.invoke(app, ["launch", "disconnect", "aider"])
    assert result.exit_code == ExitCode.OK, result.output
    assert "Nothing to undo" in result.output
