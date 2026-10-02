# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt agent`: Claude Code detection/installation, goal selection, plugin install, hand-off.

Every test drives the fake `claude` in tests/fakes/bin through TT_TOOL_BIN_CLAUDE and
captures the exec hand-off, so nothing here touches ~/.claude, PATH, or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.commands.agent import (
    CLAUDE_INSTALL_COMMAND,
    GOALS,
    MARKETPLACE_NAME,
)
from tenstorrent.errors import ExitCode

FAKE_CLAUDE = Path(__file__).parent.parent / "fakes" / "bin" / "claude"


@pytest.fixture
def claude_bin(monkeypatch, tmp_path) -> Path:
    """The fake claude, wired in by path; returns its argv log."""
    log = tmp_path / "claude-argv.jsonl"
    monkeypatch.setenv("TT_TOOL_BIN_CLAUDE", str(FAKE_CLAUDE))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    return log


@pytest.fixture
def no_claude(monkeypatch, tmp_path):
    """No Claude Code anywhere tt looks: not on PATH, not in ~/.local/bin."""
    monkeypatch.delenv("TT_TOOL_BIN_CLAUDE", raising=False)
    monkeypatch.setattr("tenstorrent.commands.agent.shutil.which", lambda name: None)
    # isolated_dirs already points HOME at tmp_path/home, so ~/.local/bin/claude is absent.


@pytest.fixture(autouse=True)
def execed(monkeypatch):
    """Capture the exec hand-off instead of replacing the test process."""
    calls: list[list[str]] = []

    def fake_execvpe(file, argv, env):
        calls.append(list(argv))
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    return calls


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.agent._stdin_isatty", lambda: True)


def argv_log(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line)["argv"] for line in log.read_text().splitlines()]


def plugin_steps(log: Path) -> list[list[str]]:
    """Everything the command asked claude to do, minus the read-only listing."""
    return [a for a in argv_log(log) if a[:3] != ["plugin", "marketplace", "list"]]


# -- goals ---------------------------------------------------------------------------------
def test_goal_plugins_are_the_documented_sets():
    assert GOALS["deploy"].plugins == ("tt-deploy",)
    # tt-model-bringup requires tt-autodebug, so the dependency is installed first.
    assert GOALS["bringup"].plugins == ("tt-autodebug", "tt-model-bringup")
    assert GOALS["develop"].plugins == ("tt-skills", "tt-review-skills", "tt-autodebug")


@pytest.mark.parametrize("goal", ["deploy", "bringup", "develop"])
def test_installs_goal_plugins_and_hands_over(runner, claude_bin, execed, goal):
    result = runner.invoke(app, ["agent", goal])
    assert result.exit_code == 0, result.output
    steps = plugin_steps(claude_bin)
    # Fresh machine: the marketplace is registered from the default GitHub repo …
    assert steps[0] == ["plugin", "marketplace", "add", "tenstorrent/skills"]
    # … then exactly the goal's plugins, in order, keyed on the repo's marketplace name.
    assert steps[1:] == [
        ["plugin", "install", f"{name}@{MARKETPLACE_NAME}"] for name in GOALS[goal].plugins
    ]
    assert execed == [[str(FAKE_CLAUDE)]]


@pytest.mark.parametrize("spelling", ["a", "1", "DEPLOY"])
def test_goal_accepts_letter_number_and_case(runner, claude_bin, execed, spelling):
    result = runner.invoke(app, ["agent", spelling])
    assert result.exit_code == 0, result.output
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] in plugin_steps(claude_bin)


def test_unknown_goal_is_usage_error(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "bogus"])
    assert result.exit_code == ExitCode.USAGE
    assert "deploy" in result.output and "bringup" in result.output
    assert plugin_steps(claude_bin) == []
    assert execed == []


def test_interactive_picker(runner, claude_bin, execed, tty):
    result = runner.invoke(app, ["agent"], input="2\n")
    assert result.exit_code == 0, result.output
    assert "What are you looking to do today?" in result.output
    assert "a. Deploy a model on your Tenstorrent hardware" in result.output
    assert "b. Bring up a new model" in result.output
    assert "c. Actively develop" in result.output
    installed = [a[2] for a in plugin_steps(claude_bin) if a[:2] == ["plugin", "install"]]
    assert installed == [f"tt-autodebug@{MARKETPLACE_NAME}", f"tt-model-bringup@{MARKETPLACE_NAME}"]
    assert execed == [[str(FAKE_CLAUDE)]]


def test_picker_default_is_deploy(runner, claude_bin, execed, tty):
    result = runner.invoke(app, ["agent"], input="\n")
    assert result.exit_code == 0, result.output
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] in plugin_steps(claude_bin)


def test_picker_needs_a_terminal(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent"])  # CliRunner stdin is not a TTY
    assert result.exit_code == ExitCode.USAGE
    assert "tt agent <goal>" in result.output
    assert execed == []


def test_picker_refused_in_json_mode(runner, claude_bin, execed, tty):
    result = runner.invoke(app, ["--json", "agent"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.output)["error"]["code"] == "USAGE"


# -- marketplace handling ---------------------------------------------------------------------
def test_existing_marketplace_is_updated_not_re_added(runner, claude_bin, execed, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CLAUDE_MARKETPLACES",
        json.dumps([{"name": MARKETPLACE_NAME, "source": "github", "repo": "tenstorrent/skills"}]),
    )
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    steps = plugin_steps(claude_bin)
    assert steps[0] == ["plugin", "marketplace", "update", MARKETPLACE_NAME]
    assert ["plugin", "marketplace", "add", "tenstorrent/skills"] not in steps


def test_marketplace_from_different_source_warns_and_uses_it(runner, claude_bin, execed, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CLAUDE_MARKETPLACES",
        json.dumps([{"name": MARKETPLACE_NAME, "source": "directory", "path": "/src/skills"}]),
    )
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    assert "warning:" in result.output and "/src/skills" in result.output
    assert plugin_steps(claude_bin)[0] == ["plugin", "marketplace", "update", MARKETPLACE_NAME]


def test_failed_marketplace_refresh_is_not_fatal(runner, claude_bin, execed, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CLAUDE_MARKETPLACES",
        json.dumps([{"name": MARKETPLACE_NAME, "source": "github", "repo": "tenstorrent/skills"}]),
    )
    monkeypatch.setenv("FAKE_CLAUDE_UPDATE_FAIL", "1")
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    assert "could not refresh" in result.output
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] in plugin_steps(claude_bin)
    assert execed == [[str(FAKE_CLAUDE)]]


def test_marketplace_source_from_env_and_config(runner, claude_bin, execed, monkeypatch, tmp_path):
    checkout = tmp_path / "skills"
    checkout.mkdir()
    result = runner.invoke(app, ["config", "set", "agent.marketplace_source", str(checkout)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    assert plugin_steps(claude_bin)[0] == ["plugin", "marketplace", "add", str(checkout)]

    claude_bin.unlink()
    monkeypatch.setenv("TT_AGENT_MARKETPLACE", "git@github.com:tenstorrent/skills.git")
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    assert plugin_steps(claude_bin)[0] == [
        "plugin", "marketplace", "add", "git@github.com:tenstorrent/skills.git"
    ]


def test_plugin_install_failure_is_tool_failed(runner, claude_bin, execed, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_INSTALL_FAIL", "1")
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert execed == []


def test_offline_with_unregistered_remote_marketplace(runner, claude_bin, execed):
    result = runner.invoke(app, ["--offline", "agent", "deploy"])
    assert result.exit_code == ExitCode.OFFLINE
    assert plugin_steps(claude_bin) == []


def test_offline_skips_refresh_but_installs(runner, claude_bin, execed, monkeypatch):
    monkeypatch.setenv(
        "FAKE_CLAUDE_MARKETPLACES",
        json.dumps([{"name": MARKETPLACE_NAME, "source": "github", "repo": "tenstorrent/skills"}]),
    )
    result = runner.invoke(app, ["--offline", "agent", "deploy"])
    assert result.exit_code == 0, result.output
    steps = plugin_steps(claude_bin)
    assert ["plugin", "marketplace", "update", MARKETPLACE_NAME] not in steps
    assert steps == [["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"]]


# -- output modes and hand-off -------------------------------------------------------------------
def test_dry_run_changes_nothing(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "bringup", "--dry-run", "--", "--resume"])
    assert result.exit_code == 0, result.output
    assert plugin_steps(claude_bin) == []
    assert execed == []
    assert "plugin marketplace add tenstorrent/skills" in result.output
    assert f"plugin install tt-autodebug@{MARKETPLACE_NAME}" in result.output
    assert f"plugin install tt-model-bringup@{MARKETPLACE_NAME}" in result.output
    assert f"{FAKE_CLAUDE} --resume" in result.output


def test_dry_run_json(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "develop", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["goal"] == "develop"
    assert payload["plugins"] == ["tt-skills", "tt-review-skills", "tt-autodebug"]
    assert payload["marketplace"] == MARKETPLACE_NAME
    assert payload["dry_run"] is True
    assert len(payload["steps"]) == 4
    assert execed == []


def test_json_mode_installs_but_never_launches(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "deploy", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["launch"] is None
    assert payload["steps"] == [
        f"{FAKE_CLAUDE} plugin marketplace add tenstorrent/skills",
        f"{FAKE_CLAUDE} plugin install tt-deploy@{MARKETPLACE_NAME}",
    ]
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] in plugin_steps(claude_bin)
    assert execed == []


def test_no_launch(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "deploy", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] in plugin_steps(claude_bin)
    assert execed == []
    assert "Start Claude Code with" in result.output


def test_extra_args_go_to_claude(runner, claude_bin, execed):
    result = runner.invoke(app, ["agent", "develop", "--", "--resume", "--model", "opus"])
    assert result.exit_code == 0, result.output
    assert execed == [[str(FAKE_CLAUDE), "--resume", "--model", "opus"]]


# -- Claude Code missing --------------------------------------------------------------------------
def test_missing_claude_noninteractive_is_tool_missing(runner, no_claude, execed):
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert CLAUDE_INSTALL_COMMAND in result.output
    assert execed == []


def test_missing_claude_json_error_carries_install_command(runner, no_claude, execed):
    result = runner.invoke(app, ["--json", "agent", "deploy"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    err = json.loads(result.output)["error"]
    assert err["code"] == "TOOL_MISSING"
    assert err["details"]["install_command"] == CLAUDE_INSTALL_COMMAND


def test_missing_claude_declined_install(runner, no_claude, execed, tty, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(
        "tenstorrent.commands.agent.confirm", lambda question: asked.append(question) and False
    )
    ran: list[list[str]] = []
    monkeypatch.setattr(
        "tenstorrent.tools.runner.Runner.stream", lambda self, argv, **kw: ran.append(list(argv)) or 0
    )
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert asked == ["Run it now?"]
    assert CLAUDE_INSTALL_COMMAND in result.output
    assert ran == []
    assert execed == []


def test_missing_claude_accepted_install_runs_documented_command(
    runner, no_claude, execed, tty, monkeypatch, tmp_path
):
    """The user says yes: tt runs the installer from the Claude Code docs, finds the
    launcher it drops in ~/.local/bin, and carries on with the goal."""
    monkeypatch.setattr("tenstorrent.commands.agent.confirm", lambda _: True)
    home_bin = Path(tmp_path / "home" / ".local" / "bin")
    real_stream = None
    ran: list[list[str]] = []

    def fake_stream(self, argv, **kw):
        ran.append(list(argv))
        if argv[:2] == ["bash", "-c"]:
            # "the installer" drops a launcher that behaves like the fake claude
            home_bin.mkdir(parents=True, exist_ok=True)
            launcher = home_bin / "claude"
            launcher.write_text(f"#!/bin/sh\nexec {FAKE_CLAUDE} \"$@\"\n")
            launcher.chmod(0o755)
        return 0

    monkeypatch.setattr("tenstorrent.tools.runner.Runner.stream", fake_stream)
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == 0, result.output
    assert ran[0] == ["bash", "-c", CLAUDE_INSTALL_COMMAND]
    launcher = str(home_bin / "claude")
    assert ["plugin", "install", f"tt-deploy@{MARKETPLACE_NAME}"] == ran[-1][1:]
    assert ran[-1][0] == launcher
    assert execed == [[launcher]]


def test_missing_claude_yes_flag_installs_without_prompt(runner, no_claude, execed, monkeypatch, tmp_path):
    ran: list[list[str]] = []
    home_bin = Path(tmp_path / "home" / ".local" / "bin")

    def fake_stream(self, argv, **kw):
        ran.append(list(argv))
        if argv[:2] == ["bash", "-c"]:
            home_bin.mkdir(parents=True, exist_ok=True)
            (home_bin / "claude").write_text("#!/bin/sh\nexit 0\n")
            (home_bin / "claude").chmod(0o755)
        return 0

    monkeypatch.setattr("tenstorrent.tools.runner.Runner.stream", fake_stream)
    result = runner.invoke(app, ["agent", "deploy", "--yes", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert ran[0] == ["bash", "-c", CLAUDE_INSTALL_COMMAND]
    assert "Run it now?" not in result.output


def test_missing_claude_offline_is_offline_error(runner, no_claude, execed, tty):
    result = runner.invoke(app, ["--offline", "agent", "deploy"])
    assert result.exit_code == ExitCode.OFFLINE


def test_installer_that_leaves_no_binary_is_reported(runner, no_claude, execed, tty, monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.agent.confirm", lambda _: True)
    monkeypatch.setattr("tenstorrent.tools.runner.Runner.stream", lambda self, argv, **kw: 0)
    result = runner.invoke(app, ["agent", "deploy"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "installer finished" in result.output


def test_dry_run_without_claude_still_shows_plan(runner, no_claude, execed):
    result = runner.invoke(app, ["agent", "deploy", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert CLAUDE_INSTALL_COMMAND in result.output
    assert "not installed" in result.output
    assert execed == []


def test_help_lists_agent_under_workloads(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "deploy, bringup, develop" in result.output
