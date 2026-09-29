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


def test_error_panel_shows_the_log_path_when_details_carry_one(capsys):
    """`details["log_path"]` has always been rendered but never populated; the
    streaming Runner mode sets it, so pin the branch."""
    out = OutputManager()
    out.emit_error(
        TTError(
            "tt-installer exited with status 1.",
            why="ERROR: sha256 mismatch",
            next_step="tt update --refresh",
            exit_code=ExitCode.TOOL_FAILED,
            details={"log_path": "/tmp/logs/tt-installer.log"},
        )
    )
    err = capsys.readouterr().err
    assert "Full output" in err
    assert "tt-installer.log" in err


def test_error_panel_omits_the_log_line_when_there_is_no_log(capsys):
    out = OutputManager()
    out.emit_error(TTError("something broke"))
    assert "Full output" not in capsys.readouterr().err


def test_ui_layer_never_writes_to_stdout(capsys):
    """The stdout/stderr contract, enforced against the presentation layer."""
    out = OutputManager()
    out.ui.register_phases(["One"])
    with out.ui.phase("One"):
        with out.ui.step("A step") as step:
            step.detail("d")
        out.ui.note("a note")
    out.ui.final_stepper()
    assert capsys.readouterr().out == ""

def test_no_color_disables_styling_on_both_consoles():
    out = OutputManager(no_color=True)
    assert out.no_color is True
    assert out.data_console.no_color is True
    assert out.status_console.no_color is True


def test_no_color_env_var_is_honoured_without_a_flag(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    out = OutputManager()
    assert out.no_color is True


def test_apply_flags_retints_the_existing_consoles():
    """The consoles are built in __init__, so a leaf flag has to reach them."""
    out = OutputManager()
    assert out.status_console.no_color is False
    out.apply_flags(no_color=True)
    assert out.no_color is True
    assert out.status_console.no_color is True


def test_apply_flags_cannot_turn_a_root_flag_back_off():
    out = OutputManager(verbose=True, no_color=True)
    out.apply_flags(verbose=False, no_color=False)
    assert out.verbose is True
    assert out.no_color is True


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


class _FakePager:
    """Stands in for subprocess.Popen: records what tt does to the pager process.
    `wait_interrupts` Ctrl+Cs arrive while tt waits; `write_interrupts` makes the
    Ctrl+C land while tt is still feeding the pager."""

    def __init__(self, *, wait_interrupts: int = 0, write_interrupts: bool = False):
        self.wait_interrupts = wait_interrupts
        self.write_interrupts = write_interrupts
        self.calls: list[str] = []
        self.launches: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.launches.append({"cmd": cmd, "LESS": kwargs["env"].get("LESS")})
        fake = self

        class _Stdin:
            def write(self, data):
                fake.calls.append("write")
                if fake.write_interrupts:
                    raise KeyboardInterrupt

            def close(self):
                fake.calls.append("close")

        class _Proc:
            stdin = _Stdin()

            def wait(self):
                fake.calls.append("wait")
                if fake.wait_interrupts:
                    fake.wait_interrupts -= 1
                    raise KeyboardInterrupt
                return 0

            def kill(self):
                fake.calls.append("kill")

            def terminate(self):
                fake.calls.append("terminate")

        return _Proc()


def _default_pager(monkeypatch, fake: _FakePager) -> None:
    from tenstorrent import output

    monkeypatch.setattr(output.subprocess, "Popen", fake)
    _tall_tty(monkeypatch)
    monkeypatch.delenv("TT_PAGER", raising=False)
    monkeypatch.delenv("PAGER", raising=False)
    monkeypatch.delenv("LESS", raising=False)


def test_maybe_page_gives_less_its_defaults_only_when_unset(monkeypatch):
    from tenstorrent import output

    fake = _FakePager()
    _default_pager(monkeypatch, fake)
    output.maybe_page(TALL)  # default pager
    monkeypatch.setenv("LESS", "-S")
    output.maybe_page(TALL)  # the user's own LESS wins
    monkeypatch.delenv("LESS")
    monkeypatch.setenv("PAGER", "more")
    output.maybe_page(TALL)  # not less: nothing injected
    assert [s["cmd"] for s in fake.launches] == ["less", "less", "more"]
    assert [s["LESS"] for s in fake.launches] == [output.less_defaults(20), "-S", None]
    # git's FRX, K so Ctrl+C quits less cleanly, and a prompt saying where you are and
    # how to leave; the -P prompt runs to the end of the string, so it has to be last.
    defaults = output.less_defaults(20)
    assert defaults.startswith("-FRXK ")
    assert defaults.split(" -")[-1].startswith("Ps")
    assert "of 20 " in defaults  # tt knows the total; less would not until the end
    assert "Enter/Space for more" in defaults
    assert defaults.endswith("q to quit")


def test_ctrl_c_at_the_pager_waits_for_it_instead_of_killing_it(monkeypatch, capsys):
    """Killing less on Ctrl+C (what subprocess.run does) leaves the terminal with no
    echo. tt keeps waiting for the pager to exit and treats Ctrl+C as done reading."""
    from tenstorrent import output

    fake = _FakePager(wait_interrupts=2)
    _default_pager(monkeypatch, fake)
    output.maybe_page(TALL)  # returns normally: no KeyboardInterrupt escapes
    assert fake.calls == ["write", "close", "wait", "wait", "wait"]
    assert capsys.readouterr().out == ""  # the pager showed it; nothing re-printed


def test_ctrl_c_while_feeding_the_pager_still_closes_and_waits(monkeypatch):
    from tenstorrent import output

    fake = _FakePager(write_interrupts=True)
    _default_pager(monkeypatch, fake)
    output.maybe_page(TALL)
    assert fake.calls == ["write", "close", "wait"]


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


def test_emit_page_honours_the_output_pager_setting(monkeypatch, tmp_path, capsys):
    from tenstorrent.context import AppContext

    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    _tall_tty(monkeypatch)
    appctx = AppContext.create()
    assert appctx.output.pager_enabled() is True  # default: page long listings
    appctx.config.set("output.pager", False)
    assert appctx.output.pager_enabled() is False
    appctx.output.emit({"n": 20}, renderer=lambda d: TALL, page=True)
    assert "line 19" in capsys.readouterr().out
    assert not sink.exists()
