# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


@pytest.fixture(autouse=True)
def fake_smi(smi_bin):
    """Wire tt-smi for every device test: the fake in default mode, the real
    tt-smi under --hardware. Returns the recorded-argv log path (fake mode only)."""
    return smi_bin


@pytest.fixture
def no_sudo(runner):
    result = runner.invoke(app, ["config", "set", "tools.sudo_command", ""])
    assert result.exit_code == 0


@pytest.mark.fakes_only
def test_status_human_table(runner):
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == 0
    assert "p300c" in result.output
    assert "0000:01:00.0" in result.output


@pytest.mark.fakes_only
def test_status_json_schema(runner):
    result = runner.invoke(app, ["device", "status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data) == {"host", "devices", "warnings"}
    dev = data["devices"][0]
    # this key set is the public --json contract; a native backend must keep it
    assert set(dev) == {
        "index", "board_type", "board_id", "bus_id", "coords", "dram_status",
        "pcie_speed", "pcie_width", "voltage_v", "current_a", "power_w",
        "aiclk_mhz", "temperature_c", "firmware",
    }
    assert dev["board_type"] == "p300c"
    assert data["host"]["Driver"] == "TT-KMD 2.9.0"


@pytest.mark.fakes_only
def test_status_multi(runner, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "multi")
    result = runner.invoke(app, ["device", "status", "--json"])
    assert [d["board_type"] for d in json.loads(result.output)["devices"]] == [
        "p300c"
    ] * 4


@pytest.mark.fakes_only
def test_status_no_devices_exits_3(runner, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "empty")
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == ExitCode.NO_DEVICES
    assert "No Tenstorrent devices detected" in result.output


@pytest.mark.fakes_only
def test_status_smi_error_exits_5(runner, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "error")
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "No Tenstorrent devices found!" in result.output  # stderr tail surfaced


@pytest.mark.fakes_only
def test_status_malformed_warns_but_renders(runner, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "malformed")
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == 0
    assert "warning" in result.output


def test_status_raw_dumps_tt_smi_json(runner):
    result = runner.invoke(app, ["device", "status", "--raw"])
    assert result.exit_code == 0
    raw = json.loads(result.output)
    assert "device_info" in raw  # tt-smi's shape, not ours


def test_status_missing_smi_exits_4(runner, monkeypatch):
    monkeypatch.delenv("TT_TOOL_BIN_TT_SMI")
    result = runner.invoke(app, ["device", "status"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "tt update" in result.output


@pytest.mark.fakes_only
def test_info_shows_metadata(runner):
    result = runner.invoke(app, ["device", "info"])
    assert result.exit_code == 0
    assert "0000046131924027" in result.output
    assert "fw.cm_fw" in result.output


def test_info_bad_index_is_usage_error(runner):
    result = runner.invoke(app, ["device", "info", "7"])
    assert result.exit_code == ExitCode.USAGE
    assert "No such device index" in result.output


@pytest.mark.fakes_only
def test_reset_requires_confirmation_noninteractive(runner, no_sudo, fake_smi):
    result = runner.invoke(app, ["device", "reset", "0"])
    assert result.exit_code == ExitCode.USAGE
    assert "--yes" in result.output
    assert not fake_smi.exists()  # no reset was attempted


@pytest.mark.fakes_only
def test_reset_with_yes_streams_tt_smi_r(runner, no_sudo, fake_smi):
    result = runner.invoke(app, ["device", "reset", "0", "--yes"])
    assert result.exit_code == 0
    argv = json.loads(fake_smi.read_text().splitlines()[-1])
    assert argv == ["-r", "0"]


@pytest.mark.fakes_only
def test_reset_all_devices(runner, no_sudo, fake_smi):
    result = runner.invoke(app, ["device", "reset", "--yes"])
    assert result.exit_code == 0
    argv = json.loads(fake_smi.read_text().splitlines()[-1])
    assert argv == ["-r"]


@pytest.mark.fakes_only
@pytest.mark.destructive
def test_reset_json_result(runner, no_sudo):
    result = runner.invoke(app, ["device", "reset", "1", "--yes", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "ok": True,
        "devices": [1],
        "message": "reset complete",
    }


@pytest.mark.fakes_only
def test_top_execs_tui(runner, monkeypatch, fake_bin):
    recorded = {}

    def fake_exec(file, argv, env):
        recorded["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_exec)
    result = runner.invoke(app, ["device", "top"])
    assert result.exit_code == 0
    assert recorded["argv"] == [str(fake_bin / "tt-smi")]
