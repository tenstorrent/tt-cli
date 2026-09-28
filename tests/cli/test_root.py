# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from tenstorrent import __version__
from tenstorrent.cli import app


def test_help_exits_zero(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Tenstorrent" in result.output


def test_bare_tt_shows_help_and_exits_zero(runner):
    # bare `tt` behaves exactly like `tt -h` (click's default usage-error exit 2
    # is overridden centrally in cli.py).
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_bare_command_group_shows_help_and_exits_zero(runner):
    for group in ("device", "model", "report"):
        result = runner.invoke(app, [group])
        assert result.exit_code == 0, f"tt {group}: {result.output}"
        assert f"tt {group} [OPTIONS] COMMAND" in result.output.replace("\n", "")


def test_bare_leaf_command_with_required_args_shows_help_and_exits_zero(runner):
    for argv in (["serve"], ["model", "info"], ["model", "pull"],
                 ["config", "get"], ["config", "set"], ["model", "compile"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"tt {' '.join(argv)}: {result.output}"
        assert "Usage" in result.output


def test_partial_args_still_a_usage_error(runner):
    # only a fully bare invocation gets the help treatment; a wrong invocation
    # with some args keeps the explicit usage error and exit 2.
    result = runner.invoke(app, ["serve", "--workflow", "benchmarks"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["config", "set", "only.a.key"])
    assert result.exit_code == 2


def test_version(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unknown_command_is_usage_error(runner):
    result = runner.invoke(app, ["frobnicate"])
    assert result.exit_code == 2


def _fake_pager(tmp_path):
    import sys

    sink = tmp_path / "paged.txt"
    cmd = (
        f'"{sys.executable}" -c "import sys, pathlib; '
        f"pathlib.Path(r'{sink}').write_bytes(sys.stdin.buffer.read())\""
    )
    return cmd, sink


def test_help_advertises_no_pager(runner):
    assert "--no-pager" in runner.invoke(app, ["--help"]).output.replace("\n", "")


def test_help_pages_on_a_short_terminal(runner, monkeypatch, tmp_path):
    """`tt --help` and `tt model --help` scroll off a small tmux pane; on a
    terminal they go through the pager. The captured help keeps Typer's own
    rendering — the pager sees the same Usage line a direct print would show."""
    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    monkeypatch.setattr("tenstorrent.output._stdout_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.output._terminal_lines", lambda: 5)
    for argv in (["--help"], ["model", "--help"]):
        sink.unlink(missing_ok=True)
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        # The pager owns the screen; click's own echo of the (empty) formatter
        # still adds one newline, exactly as it does for an unpaged help.
        assert result.output.strip() == ""
        assert "Usage: " in sink.read_text()


def test_help_no_pager_flag_prints_directly(runner, monkeypatch, tmp_path):
    cmd, sink = _fake_pager(tmp_path)
    monkeypatch.setenv("TT_PAGER", cmd)
    monkeypatch.setattr("tenstorrent.output._stdout_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.output._terminal_lines", lambda: 5)
    # --help is eager, so the flag is read off argv before any callback runs.
    monkeypatch.setattr("sys.argv", ["tt", "--no-pager", "--help"])
    result = runner.invoke(app, ["--no-pager", "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
    assert not sink.exists()
