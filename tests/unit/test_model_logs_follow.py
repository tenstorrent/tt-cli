# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt model logs` file reader: cat, tail -n and the --follow loop."""

import pytest

import tenstorrent.cli  # noqa: F401  -- commands.model imports from cli; load cli first
from tenstorrent.commands.model import _print_log_file
from tenstorrent.errors import ExitCode, TTError


def test_follow_prints_lines_appended_after_the_first_read(tmp_path, capfdbinary):
    path = tmp_path / "server.log"
    path.write_text("first\n")
    polls = []

    def should_stop() -> bool:
        polls.append(1)
        if len(polls) == 1:
            with path.open("a") as fh:
                fh.write("second\n")
            return False
        return len(polls) > 2

    _print_log_file(path, tail=None, follow=True, poll_s=0.01, should_stop=should_stop)
    assert capfdbinary.readouterr().out == b"first\nsecond\n"


def test_follow_restarts_from_the_top_when_the_file_shrinks(tmp_path, capfdbinary):
    """A rotated/truncated file must not leave the reader waiting past the old offset."""
    path = tmp_path / "server.log"
    path.write_text("a long first line\n")
    polls = []

    def should_stop() -> bool:
        polls.append(1)
        if len(polls) == 1:
            path.write_text("new\n")  # shorter than what was already printed
            return False
        return len(polls) > 2

    _print_log_file(path, tail=None, follow=True, poll_s=0.01, should_stop=should_stop)
    assert capfdbinary.readouterr().out == b"a long first line\nnew\n"


def test_tail_keeps_the_last_n_lines_only(tmp_path, capfdbinary):
    path = tmp_path / "server.log"
    path.write_text("1\n2\n3\n4\n")
    _print_log_file(path, tail=2, follow=False)
    assert capfdbinary.readouterr().out == b"3\n4\n"


def test_tail_larger_than_the_file_prints_everything(tmp_path, capfdbinary):
    path = tmp_path / "server.log"
    path.write_text("1\n2\n")
    _print_log_file(path, tail=10, follow=False)
    assert capfdbinary.readouterr().out == b"1\n2\n"


def test_unreadable_file_is_needs_sudo(tmp_path, monkeypatch):
    path = tmp_path / "server.log"
    path.write_text("x\n")
    import builtins

    real_open = builtins.open

    def denied(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)
    with pytest.raises(TTError) as info:
        _print_log_file(path, tail=None, follow=False)
    assert info.value.exit_code == ExitCode.NEEDS_SUDO
    assert f"sudo tail -f {path}" in info.value.next_step
