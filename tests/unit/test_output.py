# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from dataclasses import dataclass
from pathlib import Path

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.output import OutputManager, to_jsonable


@dataclass
class Point:
    x: int
    path: Path


def test_to_jsonable_handles_dataclasses_and_paths():
    assert to_jsonable(Point(1, Path("/a"))) == {"x": 1, "path": "/a"}
    assert to_jsonable({"k": [Point(2, Path("b"))]}) == {"k": [{"x": 2, "path": "b"}]}


def test_json_mode_prints_json_to_stdout_only(capsys):
    out = OutputManager(json_mode=True)
    out.status("spinner-ish message")  # suppressed in json mode
    out.emit({"a": 1})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"a": 1}
    assert captured.err == ""


def test_human_mode_status_goes_to_stderr(capsys):
    out = OutputManager()
    out.status("working…")
    out.emit({"a": 1}, renderer=lambda d: f"a is {d['a']}")
    captured = capsys.readouterr()
    assert "working…" in captured.err
    assert "a is 1" in captured.out


def test_quiet_suppresses_data_and_status_but_not_errors(capsys):
    out = OutputManager(quiet=True)
    out.status("nope")
    out.emit({"a": 1}, renderer=lambda d: "nope")
    out.emit_error(TTError("it broke", exit_code=ExitCode.TOOL_FAILED))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "it broke" in captured.err


def test_error_panel_includes_what_why_next(capsys):
    out = OutputManager()
    out.emit_error(
        TTError("Flash failed.", why="power loss", next_step="run `tt update` again")
    )
    err = capsys.readouterr().err
    assert "Flash failed." in err
    assert "power loss" in err
    assert "tt update" in err


def test_json_mode_errors_are_json_on_stdout(capsys):
    out = OutputManager(json_mode=True)
    out.emit_error(TTError("bad", exit_code=ExitCode.CONFIG))
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "CONFIG"
    assert payload["error"]["exit_code"] == 9
