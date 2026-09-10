# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Unexpected failures and Ctrl-C become cards, never raw tracebacks.

CliRunner never reaches main(), so these test the renderers as units and the real
entry point through a subprocess.
"""

from __future__ import annotations

import subprocess
import sys

from tenstorrent.cli import _argv_verbose, _report_unexpected, _resume_command, app


def test_pretty_exceptions_are_disabled():
    """Typer's pretty traceback renders local variables — which here can include
    environment values and tokens. A traceback is not an error message."""
    assert app.pretty_exceptions_enable is False


def test_resume_command_reconstructs_the_invocation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tt", "update", "--offline"])
    assert _resume_command() == "tt update --offline"
    # -v is dropped: it is about output, not about what to resume.
    monkeypatch.setattr(sys, "argv", ["tt", "update", "-v"])
    assert _resume_command() == "tt update"
    monkeypatch.setattr(sys, "argv", ["tt"])
    assert _resume_command() == "tt"


def test_argv_verbose_reads_the_flag_without_an_app_context(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tt", "update"])
    assert _argv_verbose() is False
    monkeypatch.setattr(sys, "argv", ["tt", "update", "-v"])
    assert _argv_verbose() is True
    monkeypatch.setattr(sys, "argv", ["tt", "--verbose", "update"])
    assert _argv_verbose() is True


def test_unexpected_error_renders_a_card_and_keeps_the_traceback_off_screen(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["tt", "device", "status"])
    try:
        raise ValueError("a wild exception")
    except ValueError as exc:
        _report_unexpected(exc)
    err = capsys.readouterr().err
    assert "tt hit an unexpected error" in err
    assert "ValueError" in err
    assert "a wild exception" in err
    # The consequence is stated, and the next actions are offered.
    assert "stopped here" in err
    assert "tt report issue" in err
    # But the traceback itself stays out of the terminal.
    assert "Traceback (most recent call last)" not in err


def test_verbose_brings_the_traceback_back(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tt", "device", "status", "-v"])
    try:
        raise ValueError("a wild exception")
    except ValueError as exc:
        _report_unexpected(exc)
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_the_traceback_is_written_to_a_log_the_card_points_at(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tt", "device", "status"])
    try:
        raise ValueError("a wild exception")
    except ValueError as exc:
        _report_unexpected(exc)
    assert "crash.log" in capsys.readouterr().err

    from tenstorrent.config.paths import get_paths

    logs = sorted(get_paths().logs_dir.glob("*-crash.log"))
    assert logs, "no crash log was written"
    text = logs[-1].read_text()
    assert "Traceback (most recent call last)" in text
    assert "a wild exception" in text


def test_a_crash_in_the_real_entry_point_prints_a_card_not_a_traceback(tmp_path):
    """End to end through main(), which CliRunner never reaches."""
    program = (
        "import sys;"
        "import tenstorrent.cli as c;"
        "c.app = None;"  # any attribute error inside main()'s try block
        "c.main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "tt hit an unexpected error" in combined
    assert "Traceback (most recent call last)" not in combined


def test_a_deliberate_exit_code_survives_the_crash_handler(tmp_path):
    """Regression: `raise typer.Exit(ExitCode.TOOL_FAILED)` is how a command
    reports a partial failure. typer.Exit subclasses RuntimeError, so main()'s
    `except Exception` caught it and reported a crash with exit 1 — silently
    breaking the documented exit-code contract.

    CliRunner handles Exit itself, so this can only be seen through main().
    """
    program = (
        "import sys, typer;"
        "import tenstorrent.cli as c;"
        "c.app = lambda **kw: (_ for _ in ()).throw(typer.Exit(5));"
        "c.main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 5, result.stdout + result.stderr
    # And it is not misreported as a crash.
    assert "unexpected error" not in (result.stdout + result.stderr)


def test_exit_zero_still_means_success(tmp_path):
    program = (
        "import typer;"
        "import tenstorrent.cli as c;"
        "c.app = lambda **kw: (_ for _ in ()).throw(typer.Exit(0));"
        "c.main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0
