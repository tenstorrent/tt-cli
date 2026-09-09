# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import contextlib
import subprocess

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.runner import Runner


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
