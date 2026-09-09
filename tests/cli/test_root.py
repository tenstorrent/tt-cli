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
