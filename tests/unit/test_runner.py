# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import contextlib
import os
import subprocess
import time

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.runner import LineSplitter, Runner


def test_capture_returns_stdout():
    runner = Runner(sudo_command="")
    result = runner.capture(["echo", "hello"])
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_capture_missing_binary_is_tool_missing():
    runner = Runner(sudo_command="")
    with pytest.raises(TTError) as exc:
        runner.capture(["definitely-not-a-real-binary-xyz"])
    assert exc.value.exit_code == ExitCode.TOOL_MISSING
    assert "tt update" in exc.value.next_step


def test_capture_failure_is_tool_failed_with_stderr_tail():
    runner = Runner(sudo_command="")
    with pytest.raises(TTError) as exc:
        runner.capture(["sh", "-c", "echo broken >&2; exit 3"], tool="fake-tool")
    err = exc.value
    assert err.exit_code == ExitCode.TOOL_FAILED
    assert "fake-tool" in err.what
    assert "broken" in (err.why or "")
    assert err.details["returncode"] == 3


def test_capture_check_false_returns_result():
    runner = Runner(sudo_command="")
    result = runner.capture(["sh", "-c", "exit 5"], check=False)
    assert result.returncode == 5


def test_stream_success_and_failure():
    runner = Runner(sudo_command="")
    assert runner.stream(["true"]) == 0
    with pytest.raises(TTError) as exc:
        runner.stream(["false"], tool="streamer")
    assert exc.value.exit_code == ExitCode.TOOL_FAILED


def test_wrap_sudo_disabled_by_empty_command():
    runner = Runner(sudo_command="")
    assert runner.wrap_sudo(["reset-things"]) == ["reset-things"]


def test_wrap_sudo_noop_as_root(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    runner = Runner(sudo_command="sudo")
    assert runner.wrap_sudo(["x"]) == ["x"]


def _spawn_recorder(probe_rc):
    calls = []

    def spawn(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, probe_rc)

    return spawn, calls


def test_wrap_sudo_probes_and_prefixes_when_cached(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    spawn, calls = _spawn_recorder(probe_rc=0)  # sudo -n succeeds (cached creds)
    runner = Runner(sudo_command="sudo", spawn=spawn, isatty_fn=lambda: False)
    assert runner.wrap_sudo(["tt-smi", "-r"]) == ["sudo", "tt-smi", "-r"]
    assert calls[0] == ["sudo", "-n", "true"]


def test_wrap_sudo_fails_fast_when_noninteractive(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    spawn, _ = _spawn_recorder(probe_rc=1)  # sudo needs a password
    runner = Runner(sudo_command="sudo", spawn=spawn, isatty_fn=lambda: False)
    with pytest.raises(TTError) as exc:
        runner.wrap_sudo(["tt-smi", "-r"])
    err = exc.value
    assert err.exit_code == ExitCode.NEEDS_SUDO
    assert "sudo tt-smi -r" in err.next_step  # exact rerun command


def test_wrap_sudo_allows_prompt_on_tty(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    spawn, _ = _spawn_recorder(probe_rc=1)
    runner = Runner(sudo_command="sudo", spawn=spawn, isatty_fn=lambda: True)
    assert runner.wrap_sudo(["x"], allow_prompt=True) == ["sudo", "x"]


def test_exec_tty_runs_the_before_exec_hook_first():
    """Anything that needs the command to finish — the telemetry span above all —
    has to happen before the process is replaced."""
    order: list[str] = []
    runner = Runner(
        exec_fn=lambda f, a, e: order.append("exec"),
        before_exec=lambda: order.append("hook"),
    )
    with contextlib.suppress(AssertionError):  # exec_fn returned
        runner.exec_tty(["tt-smi"])
    assert order == ["hook", "exec"]


def test_exec_tty_uses_exec_fn_seam():
    recorded = {}

    def fake_exec(file, argv, env):
        recorded["file"] = file
        recorded["argv"] = argv
        raise SystemExit(0)  # simulate process replacement

    runner = Runner(sudo_command="", exec_fn=fake_exec)
    with pytest.raises(SystemExit):
        runner.exec_tty(["tt-smi"])
    assert recorded["file"] == "tt-smi"
    assert recorded["argv"] == ["tt-smi"]


# -- stream_lines ---------------------------------------------------------------------
def test_stream_lines_hands_over_every_line_from_both_streams():
    runner = Runner(sudo_command="")
    seen: list[str] = []
    code = runner.stream_lines(
        ["python3", "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        on_line=seen.append,
    )
    assert code == 0
    assert set(seen) == {"out", "err"}


def test_stream_lines_reports_a_failing_tool_with_its_status():
    runner = Runner(sudo_command="")
    with pytest.raises(TTError) as excinfo:
        runner.stream_lines(["python3", "-c", "raise SystemExit(3)"], on_line=lambda _: None)
    assert excinfo.value.exit_code is ExitCode.TOOL_FAILED
    assert excinfo.value.details["returncode"] == 3


def test_stream_lines_missing_binary_is_tool_missing():
    runner = Runner(sudo_command="")
    with pytest.raises(TTError) as excinfo:
        runner.stream_lines(["definitely-not-a-binary"], on_line=lambda _: None)
    assert excinfo.value.exit_code is ExitCode.TOOL_MISSING


def test_a_carriage_return_bar_is_a_line_as_soon_as_it_repaints():
    """tqdm repaints one "line" for minutes; splitting on LF alone would hold
    every update back until the download finished."""
    splitter = LineSplitter()
    assert splitter.feed(b"12%|# | 1/8\r34%|### | 3/8\r") == ["12%|# | 1/8", "34%|### | 3/8"]
    assert splitter.feed(b"done\n") == ["done"]
    assert splitter.flush() == []


def test_a_multi_byte_character_split_across_chunks_survives():
    splitter = LineSplitter()
    assert splitter.feed("✅ setup".encode()[:2]) == []
    assert splitter.feed("✅ setup".encode()[2:]) == []
    assert splitter.flush() == ["✅ setup"]


def test_stream_lines_sees_a_pythons_output_before_it_exits_only_when_unbuffered():
    """Why every piped tool gets PYTHONUNBUFFERED: Python block-buffers stdout
    at 8 KB when it is a pipe, so a long-running tool's early lines arrive only
    when it finally exits — which is exactly when progress stops being useful."""
    runner = Runner(sudo_command="")
    script = "import os, sys, time; print('early'); time.sleep(1.5)"
    for env, expect_early in (({"PYTHONUNBUFFERED": "1"}, True), ({}, False)):
        seen: list[tuple[str, float]] = []
        start = time.monotonic()
        runner.stream_lines(
            ["python3", "-c", script],
            on_line=lambda line: seen.append((line, time.monotonic() - start)),
            env={"PATH": os.environ.get("PATH", ""), **env},
        )
        assert [line for line, _ in seen] == ["early"]
        assert (seen[0][1] < 1.0) is expect_early
