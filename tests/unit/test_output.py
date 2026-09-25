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


# -- paging ------------------------------------------------------------------------
def _fake_pager(tmp_path: Path) -> tuple[str, Path]:
    """A pager that records what it was fed: `TT_PAGER` is run through the shell,
    so this is a python one-liner rather than a script needing an exec bit."""
    import sys

    sink = tmp_path / "paged.txt"
    cmd = (
        f'"{sys.executable}" -c "import sys, pathlib; '
        f"pathlib.Path(r'{sink}').write_bytes(sys.stdin.buffer.read())\""
    )
    return cmd, sink


def _tall_tty(monkeypatch, *, rows: int = 5) -> None:
    """Pretend stdout is a terminal `rows` lines high; CliRunner/capsys never are."""
    monkeypatch.setattr("tenstorrent.output._stdout_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.output._terminal_lines", lambda: rows)
    monkeypatch.delenv("TT_NO_PAGER", raising=False)


TALL = "".join(f"line {i}\n" for i in range(20))


def test_maybe_page_writes_straight_through_when_not_a_terminal(capsys, monkeypatch, tmp_path):
    from tenstorrent.output import maybe_page

    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    monkeypatch.setattr("tenstorrent.output._stdout_isatty", lambda: False)
    maybe_page(TALL)
    assert capsys.readouterr().out == TALL
    assert not sink.exists()


def test_maybe_page_does_not_page_what_fits_on_one_screen(capsys, monkeypatch, tmp_path):
    from tenstorrent.output import maybe_page

    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch, rows=50)
    maybe_page(TALL)
    assert capsys.readouterr().out == TALL
    assert not sink.exists()


def test_maybe_page_pages_a_tall_listing_on_a_terminal(capsys, monkeypatch, tmp_path):
    from tenstorrent.output import maybe_page

    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch)
    maybe_page(TALL)
    assert sink.read_text() == TALL
    assert capsys.readouterr().out == ""  # the pager owns the screen


def test_maybe_page_respects_every_opt_out(capsys, monkeypatch, tmp_path):
    from tenstorrent.output import maybe_page

    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch)
    maybe_page(TALL, disabled=True)  # --no-pager
    monkeypatch.setenv("TT_NO_PAGER", "1")
    maybe_page(TALL)
    monkeypatch.delenv("TT_NO_PAGER")
    monkeypatch.setenv("TT_PAGER", "cat")  # git's spelling of "no pager"
    maybe_page(TALL)
    assert capsys.readouterr().out == TALL * 3
    assert not sink.exists()


def test_maybe_page_gives_less_git_style_defaults_only_when_unset(monkeypatch):
    from tenstorrent import output

    seen: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen.append({"cmd": cmd, "LESS": kwargs["env"].get("LESS")})
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(output.subprocess, "run", fake_run)
    _tall_tty(monkeypatch)
    monkeypatch.delenv("TT_PAGER", raising=False)
    monkeypatch.delenv("PAGER", raising=False)
    monkeypatch.delenv("LESS", raising=False)
    output.maybe_page(TALL)  # default pager
    monkeypatch.setenv("LESS", "-S")
    output.maybe_page(TALL)  # the user's own LESS wins
    monkeypatch.delenv("LESS")
    monkeypatch.setenv("PAGER", "more")
    output.maybe_page(TALL)  # not less: nothing injected
    assert [s["cmd"] for s in seen] == ["less", "less", "more"]
    assert [s["LESS"] for s in seen] == ["-FRX", "-S", None]


def test_maybe_page_falls_back_to_plain_output_when_the_pager_cannot_run(
    capsys, monkeypatch, tmp_path
):
    from tenstorrent.output import maybe_page

    _tall_tty(monkeypatch)
    monkeypatch.setenv("TT_PAGER", str(tmp_path / "no-such-pager"))  # shell: 127
    maybe_page(TALL)
    assert capsys.readouterr().out == TALL


def test_emit_page_routes_the_rendered_table_through_the_pager(monkeypatch, tmp_path, capsys):
    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch)
    out = OutputManager()
    out.emit({"n": 20}, renderer=lambda d: "\n".join(f"row {i}" for i in range(d["n"])), page=True)
    assert "row 19" in sink.read_text()
    assert capsys.readouterr().out == ""


def test_emit_page_is_inert_for_json_and_no_pager(monkeypatch, tmp_path, capsys):
    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch)
    OutputManager(json_mode=True).emit({"n": 1}, renderer=lambda d: "x" * 99, page=True)
    OutputManager(no_pager=True).emit({"n": 20}, renderer=lambda d: TALL, page=True)
    captured = capsys.readouterr().out
    json_text, _, rest = captured.partition("}\n")
    assert json.loads(json_text + "}") == {"n": 1}  # JSON went straight to stdout
    assert "line 19" in rest  # and so did the --no-pager listing
    assert not sink.exists()
