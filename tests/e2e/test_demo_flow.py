# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""End-to-end: the demo flow through the real entry point (`python -m tenstorrent`),
real subprocesses, fake tools. This exercises main()'s exit-code mapping, which
CliRunner-based tests bypass."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent
FAKES = REPO / "tests" / "fakes"

# The demo flow wires the fakes explicitly and drives a full `tt update` against the
# fake installer; it exercises main()'s exit-code mapping, not real hardware.
pytestmark = pytest.mark.fakes_only


@pytest.fixture
def env(tmp_path):
    env = dict(os.environ)
    env.update(
        TT_CONFIG_DIR=str(tmp_path / "config"),
        TT_DATA_DIR=str(tmp_path / "data"),
        TT_CACHE_DIR=str(tmp_path / "cache"),
        TT_TOOL_BIN_TT_SMI=str(FAKES / "bin" / "tt-smi"),
        TT_TOOL_BIN_TT_INSTALLER=str(FAKES / "install.sh"),
        TT_TOOL_BIN_TT_INFERENCE_SERVER=str(FAKES / "inference-repo" / "run.py"),
        TT_UV_BIN=str(FAKES / "bin" / "uv"),
        FAKE_INSTALLER_LOG=str(tmp_path / "installer.log"),
    )
    env.pop("FAKE_SMI_SCENARIO", None)
    return env


def tt(env, *args, check=True):
    proc = subprocess.run(
        [sys.executable, "-m", "tenstorrent", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"tt {' '.join(args)} exited {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def test_completion_paths_exit_cleanly(env):
    # Regression for #34: typer 0.27.2 moved Abort out of typer._click.exceptions,
    # and main()'s except clause on the old path raised AttributeError while *any*
    # exception was propagating — including the SystemExit(0) the completion
    # callbacks and the shell-complete protocol exit through.
    # (prog name under `python -m tenstorrent` isn't `tt`, so don't assert on the
    # _TT_COMPLETE spelling — just that the script came out and exit was clean)
    # By default --show-completion ignores its argument and detects the shell from
    # the process tree, which has no shell at all on CI ("Shell sh not supported",
    # exit 1). Typer's detection-disable seam makes the option take the shell
    # value, so the call behaves the same everywhere.
    env = dict(env, _TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION="1")
    show = tt(env, "--show-completion", "bash")
    assert "COMP_WORDS" in show.stdout

    # The completion env var is derived from the prog name (_TT_COMPLETE for the
    # installed `tt`; under `python -m tenstorrent` it's "_PYTHON _M
    # TENSTORRENT_COMPLETE", spaces included) — take it from the script rather
    # than hardcoding a spelling that only fits one entry point.
    match = re.search(r"(_[A-Z][A-Z_ ]*_COMPLETE)=complete_bash", show.stdout)
    assert match, show.stdout
    complete_var = match.group(1)

    def complete(words):
        complete_env = dict(env)
        complete_env[complete_var] = "complete_bash"
        complete_env.update(COMP_WORDS=words, COMP_CWORD=str(len(words.split()) - 1))
        proc = subprocess.run(
            [sys.executable, "-m", "tenstorrent"],
            capture_output=True,
            text=True,
            env=complete_env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    assert "model" in complete("tt mod")
    # Model-name completion from the bundled spec, no network (isolated dirs).
    assert "Llama-3.1-8B-Instruct" in complete("tt serve Llama-3.1")
    assert "Llama-3.1-8B-Instruct" in complete("tt model info Llama-3.1")


def test_demo_flow(env, tmp_path):
    # config
    tt(env, "config", "set", "telemetry.enabled", "false")
    tt(env, "config", "set", "tools.sudo_command", "")
    listing = tt(env, "config", "list")
    assert "telemetry.enabled = false" in listing.stdout

    # device status, human and json (stdout must be clean JSON)
    status = tt(env, "device", "status")
    assert "p300c" in status.stdout
    status_json = tt(env, "device", "status", "--json")
    payload = json.loads(status_json.stdout)
    assert payload["devices"][0]["board_type"] == "p300c"

    # reset with --yes
    reset = tt(env, "device", "reset", "0", "--yes")
    assert reset.returncode == 0

    # update: dry-run then real (fake installer + fake uv)
    plan = tt(env, "update", "--dry-run", "--json")
    assert json.loads(plan.stdout)["dry_run"] is True
    tt(env, "update", "--yes")
    installer_log = (tmp_path / "installer.log").read_text()
    assert "--mode-non-interactive" in installer_log
    assert "--versions=release" in installer_log

    # model list (nothing cached in the isolated env)
    models = tt(env, "model", "list", "--json")
    names = {m["name"] for m in json.loads(models.stdout)["models"]}
    assert "Llama-3.1-8B-Instruct" in names

    # serve via fake run.py (docker check must pass or fail visibly)
    serve = tt(env, "serve", "Llama-3.1-8B-Instruct", check=False)
    if serve.returncode == 0:
        assert "listening" in serve.stdout
    else:
        assert serve.returncode == 4  # TOOL_MISSING: no docker on this machine
        assert "container runtime" in serve.stderr

    # stubs exit UNSUPPORTED (7), errors on stderr
    stub = tt(env, "compile", "x", check=False)
    assert stub.returncode == 7

    # report issue builds a prefilled GitHub URL (--no-browser: no xdg-open here)
    issue = tt(env, "report", "issue", "--no-browser", "--json")
    assert "tenstorrent/tt-cli/issues/new" in json.loads(issue.stdout)["url"]

    # exit-code contract through the real main(): unknown key → CONFIG (9)
    bad = tt(env, "config", "get", "bogus.key", check=False)
    assert bad.returncode == 9

    # bare `tt` and bare groups print help and exit 0 (like -h), through real main()
    bare = tt(env)
    assert "Usage" in bare.stdout or "Usage" in bare.stderr
    group = tt(env, "device")
    assert "status" in group.stdout
