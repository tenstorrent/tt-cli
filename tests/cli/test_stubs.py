# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


@pytest.mark.parametrize(
    "argv",
    [
        ["report", "feedback"],
        ["train"],
        ["compile"],
    ],
)
def test_stubs_exit_unsupported_with_guidance(runner, argv):
    result = runner.invoke(app, argv)
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "not available yet" in result.output
    assert "→" in result.output  # every stub tells the user what to do instead


@pytest.mark.parametrize("name", ["train", "compile"])
def test_unready_stubs_are_hidden_from_help(runner, name):
    """train/compile still answer with exit 7, but `tt --help` must not offer
    them while the workflows behind them are unimplemented."""
    help_out = runner.invoke(app, ["--help"]).output
    listed = {
        line.strip("| \u2502").split()[0]
        for line in help_out.splitlines()
        if line.strip("| \u2502").split()
    }
    assert name not in listed


def test_stub_json_error_payload(runner):
    import json

    result = runner.invoke(app, ["--json", "train"])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert json.loads(result.output)["error"]["code"] == "UNSUPPORTED"


def test_smi_alias_is_hidden_and_execs_tui(runner, monkeypatch, fake_bin):
    monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(fake_bin / "tt-smi"))
    recorded = {}

    def fake_exec(file, argv, env):
        recorded["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_exec)
    result = runner.invoke(app, ["smi"])
    assert result.exit_code == 0
    assert recorded["argv"] == [str(fake_bin / "tt-smi")]
    help_out = runner.invoke(app, ["--help"]).output
    assert " smi " not in help_out
