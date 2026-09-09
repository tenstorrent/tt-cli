# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json

from tenstorrent.cli import app


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
