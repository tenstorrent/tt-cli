# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


def test_self_tools_json(runner):
    result = runner.invoke(app, ["self", "tools", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    names = {t["name"] for t in data["tools"]}
    assert {"tt-smi", "tt-flash", "tt-installer", "tt-inference-server"} <= names
    smi = next(t for t in data["tools"] if t["name"] == "tt-smi")
    assert smi["source"] == "missing"
    assert smi["golden_version"]


def test_self_tools_human_table(runner):
    result = runner.invoke(app, ["self", "tools"])
    assert result.exit_code == 0
    assert "tt-smi" in result.output


def test_self_is_listed_in_help(runner):
    result = runner.invoke(app, ["--help"])
    assert "self" in result.output
    assert "update" in runner.invoke(app, ["self", "--help"]).output


@pytest.fixture
def upstream_has(monkeypatch):
    """Set what the upstream version lookup answers (default: the version exists)."""
    answer = {"exists": True}
    monkeypatch.setattr(
        "tenstorrent.tools.registry.remote_version_exists",
        lambda spec, runner: answer["exists"],
    )
    return answer


def test_self_pin_and_unpin(runner, upstream_has):
    result = runner.invoke(app, ["self", "pin", "tt-inference-server", "v0.25.0", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["pinned_version"] == "v0.25.0"
    assert data["verified"] is True

    tools = json.loads(runner.invoke(app, ["self", "tools", "--json"]).output)["tools"]
    server = next(t for t in tools if t["name"] == "tt-inference-server")
    assert server["pinned_version"] == "v0.25.0"

    result = runner.invoke(app, ["self", "unpin", "tt-inference-server"])
    assert result.exit_code == 0
    assert "follows its golden version again" in result.output
    assert "was not pinned" in runner.invoke(app, ["self", "unpin", "tt-inference-server"]).output


def test_self_pin_missing_version_is_a_clear_error(runner, upstream_has):
    upstream_has["exists"] = False
    result = runner.invoke(app, ["self", "pin", "tt-inference-server", "v9.9.9"])
    assert result.exit_code == ExitCode.USAGE
    assert "'v9.9.9' does not exist" in result.output
    tools = json.loads(runner.invoke(app, ["self", "tools", "--json"]).output)["tools"]
    assert all(t["pinned_version"] is None for t in tools)


def test_self_pin_warns_when_the_version_cannot_be_checked(runner, upstream_has):
    upstream_has["exists"] = None
    result = runner.invoke(app, ["self", "pin", "tt-smi", "6.5.0"])
    assert result.exit_code == 0
    assert "Could not confirm" in result.output


def test_self_pin_unknown_tool(runner, upstream_has):
    result = runner.invoke(app, ["self", "pin", "not-a-tool", "1.0"])
    assert result.exit_code == ExitCode.CONFIG
