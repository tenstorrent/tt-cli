# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report issue`: URL building, repo selection, browser behavior.

The autouse `browser` fixture replaces webbrowser.open for every test in this
file — no test may ever open a real browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tenstorrent import __version__
from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


@pytest.fixture(autouse=True)
def no_container_runtime(request, monkeypatch):
    """The served-models collector asks docker; without this every test here would
    run the real `docker ps` (and probe port 8000) on a developer's box. A test
    that requests `fake_docker` gets the fake instead."""
    if "fake_docker" in request.fixturenames:
        return
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )


@pytest.fixture(autouse=True)
def browser(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "tenstorrent.commands.report.webbrowser.open",
        lambda url: calls.append(url) or True,
    )
    return calls


def _issue_url(output: str) -> dict:
    """Parse the printed URL into path + single-valued query params."""
    url = next(line for line in output.splitlines() if line.startswith("https://"))
    parsed = urlparse(url)
    return {"url": url, "path": parsed.path, **{k: v[0] for k, v in parse_qs(parsed.query).items()}}


def test_issue_explicit_repo_url(runner):
    result = runner.invoke(app, ["report", "issue", "tt-metal", "--no-browser"])
    assert result.exit_code == 0
    q = _issue_url(result.output)
    assert q["path"] == "/tenstorrent/tt-metal/issues/new"
    assert q["title"] == "[tt cli report] <short description>"
    assert q["labels"] == "bug"
    assert f"tt CLI: {__version__}" in q["body"]
    import platform

    assert platform.python_version() in q["body"]
    # isolated_dirs strips TT_TOOL_BIN_*, so the device section degrades gracefully
    assert "devices: unavailable (TOOL_MISSING)" in q["body"]
    assert "served models: unavailable (TOOL_MISSING)" in q["body"]
    # nothing that identifies the machine or user leaves the box
    assert str(Path.home()) not in q["body"]


def test_issue_unknown_repo_is_usage_error(runner):
    result = runner.invoke(app, ["report", "issue", "bogus"])
    assert result.exit_code == ExitCode.USAGE
    assert "tt-cli" in result.output
    assert "tt-metal" in result.output


def test_issue_missing_repo_noninteractive_is_usage_error(runner):
    # CliRunner stdin is not a TTY, so the picker is refused
    result = runner.invoke(app, ["report", "issue"])
    assert result.exit_code == ExitCode.USAGE
    assert "tt-metal" in result.output


def test_issue_missing_repo_json_is_usage_error(runner):
    result = runner.invoke(app, ["--json", "report", "issue"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.output)["error"]["code"] == "USAGE"


def test_issue_interactive_picker(runner, monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: True)
    result = runner.invoke(app, ["report", "issue", "--no-browser"], input="2\n")
    assert result.exit_code == 0
    assert "/tenstorrent/tt-metal/issues/new" in result.output

    result = runner.invoke(app, ["report", "issue", "--no-browser"], input="\n")
    assert result.exit_code == 0  # bare enter takes the default (1 = tt-cli)
    assert "/tenstorrent/tt-cli/issues/new" in result.output


def test_issue_opens_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue", "tt-cli"])
    assert result.exit_code == 0
    assert browser == [_issue_url(result.output)["url"]]


def test_issue_no_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--no-browser"])
    assert result.exit_code == 0
    assert browser == []
    assert "/tenstorrent/tt-cli/issues/new" in result.output


def test_issue_json_mode_never_opens_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["repo"] == "tt-cli"
    assert payload["github"] == "tenstorrent/tt-cli"
    assert payload["labels"] == ["bug"]
    assert "tenstorrent/tt-cli/issues/new" in payload["url"]
    assert browser == []


def test_issue_browser_failure_still_exits_zero(runner, monkeypatch):
    def boom(url):
        raise RuntimeError("no display")

    monkeypatch.setattr("tenstorrent.commands.report.webbrowser.open", boom)
    result = runner.invoke(app, ["report", "issue", "tt-cli"])
    assert result.exit_code == 0
    assert "warning:" in result.output
    assert "copy the URL above" in result.output
    assert "/tenstorrent/tt-cli/issues/new" in result.output


@pytest.mark.fakes_only
def test_issue_body_includes_device_snapshot(runner, smi_bin):
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--no-browser"])
    assert result.exit_code == 0
    body = _issue_url(result.output)["body"]
    # devices render as a markdown table, one row per device
    assert "| # | board |" in body
    assert "| 0 | p300c |" in body
    # host_info carries a Hostname; it must never reach a public issue body
    assert "tt-quietbox" not in body
    # tt_flash_version is always N/A (a fw field tt-flash doesn't write) — dropped
    assert "tt_flash_version" not in body


def test_issue_title_option(runner):
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--no-browser", "-t", "boom"])
    assert result.exit_code == 0
    assert _issue_url(result.output)["title"] == "boom"


def _record(cid, name, *, image, port, mounts=()):
    return {
        "Id": cid,
        "Name": f"/{name}",
        "Config": {"Image": image, "Labels": {}},
        "Mounts": list(mounts),
        "State": {"Status": "running", "Running": True, "StartedAt": "2026-09-08T10:00:00Z"},
        "HostConfig": {"PortBindings": {f"{port}/tcp": [{"HostIp": "", "HostPort": str(port)}]}},
    }


@pytest.mark.fakes_only
def test_issue_body_lists_served_models(runner, fake_docker, monkeypatch):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = sock.getsockname()[1]
    monkeypatch.setattr(
        "tenstorrent.launchers.discovery.DEFAULT_BASE_URL", f"http://127.0.0.1:{dead}/v1"
    )
    other = dead - 1 if dead > 1024 else dead + 1
    set_containers, _ = fake_docker
    set_containers([
        _record("1" * 64, "Qwen3.5-9B",
                image="ghcr.io/tenstorrent/tt-studio/studio_images:qwen35", port=dead),
        _record("2" * 64, "tt-inference-server-abcd", image="vllm:1", port=other,
                mounts=[{
                    "Type": "volume",
                    "Name": "volume_id_tt-transformers-Qwen3-32B",
                    "Source": str(Path.home() / "volumes" / "volume_id_tt-transformers-Qwen3-32B"),
                }]),
    ])
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--no-browser"])
    assert result.exit_code == 0
    body = _issue_url(result.output)["body"]
    assert f"- served: Qwen3.5-9B (studio, port {dead}, starting, up " in body
    assert "- served: Qwen3-32B (inference-server, port " in body
    # the collector prints identity only — never mount sources or volume names
    assert "volume_id_" not in body
    assert str(Path.home()) not in body


@pytest.mark.fakes_only
def test_issue_body_says_none_when_nothing_is_served(runner, fake_docker, monkeypatch):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = sock.getsockname()[1]
    monkeypatch.setattr(
        "tenstorrent.launchers.discovery.DEFAULT_BASE_URL", f"http://127.0.0.1:{dead}/v1"
    )
    set_containers, _ = fake_docker
    set_containers([])
    result = runner.invoke(app, ["report", "issue", "tt-cli", "--no-browser"])
    assert "- served models: none" in _issue_url(result.output)["body"]
