# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report issue`: URL building, target repo, browser behavior.

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


def test_issue_url_targets_tt_cli(runner):
    result = runner.invoke(app, ["report", "issue", "--no-browser"])
    assert result.exit_code == 0
    q = _issue_url(result.output)
    assert q["path"] == "/tenstorrent/tt-cli/issues/new"
    assert q["title"] == "[tt cli report] <short description>"
    assert q["labels"] == "bug"
    assert f"tt CLI: {__version__}" in q["body"]
    import platform

    assert platform.python_version() in q["body"]
    # isolated_dirs strips TT_TOOL_BIN_*, so the device section degrades gracefully
    assert "devices: unavailable (TOOL_MISSING)" in q["body"]
    # nothing that identifies the machine or user leaves the box
    assert str(Path.home()) not in q["body"]


def test_issue_rejects_repo_argument(runner):
    # There is no repo to pick: every issue goes to tt-cli.
    result = runner.invoke(app, ["report", "issue", "tt-metal", "--no-browser"])
    assert result.exit_code != 0
    assert "/issues/new" not in result.output


def test_issue_opens_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue"])
    assert result.exit_code == 0
    assert browser == [_issue_url(result.output)["url"]]


def test_issue_no_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue", "--no-browser"])
    assert result.exit_code == 0
    assert browser == []
    assert "/tenstorrent/tt-cli/issues/new" in result.output


def test_issue_json_mode_never_opens_browser(runner, browser):
    result = runner.invoke(app, ["report", "issue", "--json"])
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
    result = runner.invoke(app, ["report", "issue"])
    assert result.exit_code == 0
    assert "warning:" in result.output
    assert "copy the URL above" in result.output
    assert "/tenstorrent/tt-cli/issues/new" in result.output


@pytest.mark.fakes_only
def test_issue_body_includes_device_snapshot(runner, smi_bin):
    result = runner.invoke(app, ["report", "issue", "--no-browser"])
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
    result = runner.invoke(app, ["report", "issue", "--no-browser", "-t", "boom"])
    assert result.exit_code == 0
    assert _issue_url(result.output)["title"] == "boom"
