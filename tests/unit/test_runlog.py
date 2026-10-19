# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Tests for secure run logging and log file permissions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tenstorrent.redaction import Redactor
from tenstorrent.tools.runlog import DIR_MODE, FILE_MODE, RunLog, open_run_log, open_secure_file


def test_open_secure_file_sets_0700_dir_and_0600_file(tmp_path: Path):
    target = tmp_path / "logs_subdir" / "test.log"
    with open_secure_file(target, "w") as handle:
        handle.write("hello secure world\n")

    dir_stat = target.parent.stat()
    file_stat = target.stat()

    assert (dir_stat.st_mode & 0o777) == DIR_MODE, f"Expected dir mode 0700, got {oct(dir_stat.st_mode)}"
    assert (file_stat.st_mode & 0o777) == FILE_MODE, f"Expected file mode 0600, got {oct(file_stat.st_mode)}"
    assert target.read_text() == "hello secure world\n"


def test_open_secure_file_tightens_preexisting_loose_permissions(tmp_path: Path):
    loose_dir = tmp_path / "loose_dir"
    loose_dir.mkdir(mode=0o755)
    os.chmod(loose_dir, 0o755)

    loose_file = loose_dir / "loose.log"
    loose_file.write_text("old text")
    os.chmod(loose_file, 0o644)

    # Verify they were actually loose
    assert (loose_dir.stat().st_mode & 0o777) == 0o755
    assert (loose_file.stat().st_mode & 0o777) == 0o644

    with open_secure_file(loose_file, "w") as handle:
        handle.write("overwritten securely")

    assert (loose_dir.stat().st_mode & 0o777) == DIR_MODE
    assert (loose_file.stat().st_mode & 0o777) == FILE_MODE
    assert loose_file.read_text() == "overwritten securely"


def test_runlog_context_manager(tmp_path: Path):
    log_path = tmp_path / "test.log"
    with open_secure_file(log_path, "w") as handle:
        run_log = RunLog(path=log_path, _handle=handle)
        with run_log as log:
            log.write("inside context manager\n")
        # Handle should now be closed
        assert run_log._handle is None


def test_runlog_with_redactor(tmp_path: Path):
    log_path = tmp_path / "redacted.log"
    with open_secure_file(log_path, "w") as handle:
        redactor = Redactor()
        run_log = RunLog(path=log_path, _handle=handle, _redactor=redactor)
        run_log.write("Setting HF_TOKEN=hf_abcdef1234567890abcdef12 in environment\n")
        run_log.close()

    content = log_path.read_text()
    assert "hf_abcdef" not in content
    assert "<redacted>" in content


def test_open_run_log_creates_secure_permissions(tmp_path: Path):
    log_dir = tmp_path / "run_logs"
    log = open_run_log(log_dir, "my_tool", ["echo", "test"])
    assert log is not None
    try:
        log.write("some output\n")
    finally:
        log.close()

    assert (log_dir.stat().st_mode & 0o777) == DIR_MODE
    assert (log.path.stat().st_mode & 0o777) == FILE_MODE


def test_open_run_log_redacts_argv_secrets(tmp_path: Path):
    log_dir = tmp_path / "secret_logs"
    secret = "hf_secretToken1234567890abcdef12"
    log = open_run_log(
        log_dir,
        "serve",
        ["tt", "serve", "--token", secret, f"HF_AUTH={secret}"],
    )
    assert log is not None
    log.close()

    content = log.path.read_text()
    assert secret not in content
    assert "$ tt serve --token '<redacted>' 'HF_AUTH=<redacted>'" in content


