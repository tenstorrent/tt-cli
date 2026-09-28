# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Runner.stream_parsed: real `sh -c` children, so the wording and the ordering
are the shell's rather than a mock's."""

from __future__ import annotations

import sys

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.output import OutputManager
from tenstorrent.tools.runner import Runner
from tenstorrent.ui.stream import run_streamed, steps_parser


@pytest.fixture
def runner(tmp_path):
    return Runner(sudo_command="", log_dir=tmp_path)


def test_lines_arrive_in_order_with_stderr_merged(runner):
    seen = []
    result = runner.stream_parsed(
        ["sh", "-c", "echo one; echo two >&2; echo three"],
        on_line=seen.append,
        tool="demo",
    )
    assert result.returncode == 0
    assert seen == ["one", "two", "three"]


def test_output_is_returned_and_teed_to_a_log(runner):
    result = runner.stream_parsed(["sh", "-c", "echo hello"], tool="demo")
    assert "hello" in result.output
    assert result.log_path is not None
    text = result.log_path.read_text()
    assert "hello" in text
    # The command echo the old error copy only ever promised.
    assert text.startswith("$ sh -c 'echo hello'")


def test_log_can_be_switched_off(runner):
    result = runner.stream_parsed(["sh", "-c", "echo hi"], tool="demo", log=False)
    assert result.log_path is None


def test_no_log_dir_means_no_log_but_still_runs(tmp_path):
    bare = Runner(sudo_command="")
    result = bare.stream_parsed(["sh", "-c", "echo hi"], tool="demo")
    assert result.returncode == 0
    assert result.log_path is None


def test_only_the_tail_is_kept_in_memory_and_truncation_is_flagged(runner):
    result = runner.stream_parsed(
        ["sh", "-c", "for i in $(seq 1 50); do echo line$i; done"],
        tool="demo",
        keep_lines=10,
    )
    assert result.truncated is True
    assert len(result.output.strip().splitlines()) == 10
    assert "line50" in result.output
    assert "line1\n" not in result.output
    # The full stream still reaches the log.
    assert "line1\n" in result.log_path.read_text()


def test_failure_raises_tterror_carrying_the_log_path(runner):
    with pytest.raises(TTError) as excinfo:
        runner.stream_parsed(["sh", "-c", "echo boom >&2; exit 3"], tool="demo")
    err = excinfo.value
    assert err.exit_code == ExitCode.TOOL_FAILED
    assert "status 3" in err.what
    assert err.why is not None and "boom" in err.why
    assert "log_path" in err.details
    assert "Full output" in (err.next_step or "")
    # And the panel renders that path — the branch that was dead until now.
    assert err.details["returncode"] == 3


def test_failure_why_is_a_short_tail_not_the_whole_stream(runner):
    with pytest.raises(TTError) as excinfo:
        runner.stream_parsed(
            ["sh", "-c", "for i in $(seq 1 40); do echo noise$i; done; exit 1"],
            tool="demo",
        )
    assert len((excinfo.value.why or "").splitlines()) <= 8


def test_check_false_returns_the_failure_instead_of_raising(runner):
    result = runner.stream_parsed(["sh", "-c", "exit 7"], tool="demo", check=False)
    assert result.returncode == 7


def test_missing_executable_maps_to_tool_missing(runner):
    with pytest.raises(TTError) as excinfo:
        runner.stream_parsed(["definitely-not-a-real-binary-xyz"], tool="ghost")
    assert excinfo.value.exit_code == ExitCode.TOOL_MISSING


def test_env_extra_merges_over_the_environment(runner):
    seen = []
    runner.stream_parsed(
        ["sh", "-c", "echo $TT_TEST_MARKER"],
        on_line=seen.append,
        env_extra={"TT_TEST_MARKER": "present"},
        tool="demo",
    )
    assert seen == ["present"]
    # PATH survived, i.e. the environment was merged and not replaced.
    seen.clear()
    runner.stream_parsed(["sh", "-c", "echo ${PATH:+haspath}"], on_line=seen.append, tool="demo")
    assert seen == ["haspath"]


def test_cwd_is_honoured(runner, tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    seen = []
    runner.stream_parsed(["sh", "-c", "ls"], on_line=seen.append, cwd=str(tmp_path), tool="demo")
    assert "marker.txt" in seen


def test_a_parser_exception_never_takes_down_the_command(runner):
    def broken(line):
        raise ValueError("parser bug")

    result = runner.stream_parsed(["sh", "-c", "echo hi"], on_line=broken, tool="demo")
    assert result.returncode == 0


def test_stdin_is_closed_by_default_so_a_child_cannot_block_on_a_prompt(runner):
    """A piped child inheriting stdin is how a password prompt ends up invisible
    underneath a repainting spinner row."""
    seen = []
    result = runner.stream_parsed(
        ["sh", "-c", "read x && echo got:$x || echo no-stdin"],
        on_line=seen.append,
        tool="demo",
        check=False,
    )
    assert result.returncode is not None
    assert seen == ["no-stdin"]


def test_keyboard_interrupt_reaps_the_child_and_propagates(runner):
    class Boom:
        stdout = iter(["one\n"])
        returncode = None
        terminated = False

        def __iter__(self):
            return self

        def wait(self, timeout=None):
            raise KeyboardInterrupt

        def terminate(self):
            Boom.terminated = True

        def kill(self):
            pass

        def poll(self):
            return 1 if Boom.terminated else None

    def fake_popen(*args, **kwargs):
        return Boom()

    r = Runner(sudo_command="", popen=fake_popen)
    with pytest.raises(KeyboardInterrupt):
        r.stream_parsed(["sh", "-c", "true"], tool="demo")
    assert Boom.terminated is True


# -- the Ui bridge --------------------------------------------------------------
def test_run_streamed_emits_milestones_and_tracks_an_exact_denominator(runner, capsys):
    out = OutputManager()
    parser = steps_parser(["Cloning", "Creating virtualenv", "Installing"], "Installing tool")
    result = run_streamed(
        runner,
        out.ui,
        ["sh", "-c", "echo Cloning x; echo Creating virtualenv; echo Installing deps"],
        label="Installing tool",
        parser=parser,
        tool="demo",
    )
    assert result.returncode == 0
    assert parser.done == 3
    assert "3/3" in parser.activity()
    err = capsys.readouterr().err
    assert "✓ Cloning" in err
    assert "✓ Installing" in err


def test_run_streamed_stays_silent_under_json(runner, capsys):
    out = OutputManager(json_mode=True)
    run_streamed(runner, out.ui, ["sh", "-c", "echo Cloning x"], label="x", tool="demo")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_verbose_echoes_the_raw_lines_as_the_escape_hatch(runner, capsys):
    out = OutputManager(verbose=True)
    run_streamed(
        runner, out.ui, ["sh", "-c", "echo some-raw-tool-line"], label="x", tool="demo"
    )
    assert "some-raw-tool-line" in capsys.readouterr().err


def test_normal_run_hides_the_raw_lines(runner, capsys):
    out = OutputManager()
    run_streamed(
        runner, out.ui, ["sh", "-c", "echo some-raw-tool-line"], label="x", tool="demo"
    )
    assert "some-raw-tool-line" not in capsys.readouterr().err
