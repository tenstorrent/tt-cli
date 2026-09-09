# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt self update`, the end-of-command notice, and the `tt update` hand-off.

The layout is patched in (detection is unit-tested against real dist-info trees in
tests/unit/test_self_update_layout.py); the version source is a local file; the
installer is the fake uv, which records its argv."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tenstorrent import __version__
from tenstorrent.cli import app
from tenstorrent.config.paths import get_paths
from tenstorrent.selfupdate import check as C
from tenstorrent.selfupdate import layout as L
from tenstorrent.selfupdate import update as U

NEWER = "99.0.0"


def _doc(*versions):
    return {"releases": {v: [{"filename": "x.whl", "requires_python": ">=3.10", "yanked": False}] for v in versions}}


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Point the check at a local PyPI-shaped file; returns a setter."""
    path = tmp_path / "pypi.json"

    def set_versions(*versions):
        path.write_text(json.dumps(_doc(*versions)))
        return path

    set_versions(NEWER)
    monkeypatch.setenv(C.SOURCE_ENV, str(path))
    monkeypatch.delenv(C.DISABLE_ENV, raising=False)
    return set_versions


def _layout(kind, tmp_path, **extra):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    return L.InstallLayout(
        kind, prefix, "pip", __version__, str(prefix / "bin" / "python"),
        detail=f"fake {kind} layout", extra=extra,
    )


@pytest.fixture
def uv_tool_layout(tmp_path, monkeypatch):
    bin_dir = tmp_path / "home" / "bin"
    layout = _layout(L.KIND_UV_TOOL, tmp_path, tool_dir=str(tmp_path / "tools"), bin_dir=str(bin_dir))
    monkeypatch.setattr(U, "detect_layout", lambda: layout)
    monkeypatch.setattr(C, "detect_layout", lambda: layout)
    # The `tt` a shell would run is this install (nothing else on PATH).
    monkeypatch.setattr(U.shutil, "which", lambda name: None)
    return layout


@pytest.fixture
def shared_layout(tmp_path, monkeypatch):
    layout = L.InstallLayout(
        L.KIND_SHARED_VENV, tmp_path / "venv", "pip", __version__, "/venv/bin/python",
        tenants=("tt-smi", "requests"), detail="tt shares /venv with other packages.",
    )
    monkeypatch.setattr(U, "detect_layout", lambda: layout)
    return layout


def _json(result):
    return json.loads(result.output[result.output.index("{"):])


# -- tt self update -------------------------------------------------------------------
def test_check_reports_an_available_release(runner, source, uv_tool_layout):
    result = runner.invoke(app, ["self", "update", "--check", "--json"])
    assert result.exit_code == 0, result.output
    data = _json(result)
    assert data["status"] == "available"
    assert data["latest"] == NEWER and data["current"] == __version__
    assert data["hint"] == "tt self update"
    assert data["layout"]["kind"] == "uv-tool"


def test_up_to_date_exits_zero_and_changes_nothing(runner, source, uv_tool_layout, fake_bin):
    source("0.0.1")
    result = runner.invoke(app, ["self", "update", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    assert _json(result)["status"] == "up-to-date"


def test_shared_venv_refuses_with_the_exact_command(runner, source, shared_layout):
    result = runner.invoke(app, ["self", "update", "--yes"])
    assert result.exit_code == 7, result.output
    assert "cannot upgrade itself" in result.output
    assert "/venv/bin/python -m pip install --upgrade tenstorrent" in result.output
    # --check is still allowed: it only reports, and tells the user what to run.
    result = runner.invoke(app, ["self", "update", "--check", "--json"])
    assert result.exit_code == 0
    assert _json(result)["hint"].endswith("-m pip install --upgrade tenstorrent")


def test_uv_tool_upgrade_runs_uv_with_the_receipt_dirs(runner, source, uv_tool_layout, uv_bin, monkeypatch):
    monkeypatch.setattr(U, "installed_version", lambda layout: NEWER)
    result = runner.invoke(app, ["self", "update", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    data = _json(result)
    assert data["status"] == "upgraded" and data["to"] == NEWER
    argv = [json.loads(line) for line in uv_bin.read_text().splitlines()]
    assert argv == [["tool", "install", "--force", f"tenstorrent=={NEWER}"]]
    # A fresh check is recorded against the new version: no stale notice afterwards.
    assert C.UpdateState(get_paths()).pending(__version__) is None


def test_upgrade_env_points_uv_at_this_tool_venv(uv_tool_layout, monkeypatch):
    monkeypatch.setenv("TT_UV_BIN", "/fake/uv")
    argv, env = U.upgrade_command(uv_tool_layout, NEWER)
    assert argv == ["/fake/uv", "tool", "install", "--force", f"tenstorrent=={NEWER}"]
    assert env["UV_TOOL_DIR"] == uv_tool_layout.extra["tool_dir"]
    assert env["UV_TOOL_BIN_DIR"] == uv_tool_layout.extra["bin_dir"]


def test_pipx_upgrade_uses_pipx_home(tmp_path, monkeypatch):
    layout = L.InstallLayout(
        L.KIND_PIPX, tmp_path / "pipx" / "venvs" / "tenstorrent", "pip", __version__, "python",
    )
    monkeypatch.setattr(U.shutil, "which", lambda name: "/usr/bin/pipx" if name == "pipx" else None)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    argv, env = U.upgrade_command(layout, NEWER)
    assert argv == ["/usr/bin/pipx", "upgrade", "tenstorrent"]
    assert env["PIPX_HOME"] == str(tmp_path / "pipx")
    monkeypatch.setattr(U.shutil, "which", lambda name: None)
    with pytest.raises(Exception) as info:
        U.upgrade_command(layout, NEWER)
    assert info.value.exit_code == 4


def test_sole_venv_uses_the_installer_that_made_it(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_UV_BIN", "/fake/uv")
    pip_venv = L.InstallLayout(L.KIND_VENV, tmp_path, "pip", __version__, "/v/bin/python")
    monkeypatch.setattr(U, "_has_pip", lambda: True)
    assert U.upgrade_command(pip_venv, NEWER)[0] == ["/v/bin/python", "-m", "pip", "install", f"tenstorrent=={NEWER}"]
    # No pip module (a uv-made venv, or pip stripped): go through uv against that python.
    monkeypatch.setattr(U, "_has_pip", lambda: False)
    assert U.upgrade_command(pip_venv, NEWER)[0] == ["/fake/uv", "pip", "install", "--python", "/v/bin/python", f"tenstorrent=={NEWER}"]
    uv_venv = L.InstallLayout(L.KIND_VENV, tmp_path, "uv", __version__, "/v/bin/python")
    monkeypatch.setattr(U, "_has_pip", lambda: True)
    assert U.upgrade_command(uv_venv, NEWER)[0][:3] == ["/fake/uv", "pip", "install"]


def test_non_interactive_without_yes_is_a_usage_error(runner, source, uv_tool_layout, uv_bin):
    result = runner.invoke(app, ["self", "update"])
    assert result.exit_code == 2, result.output
    assert "--yes" in result.output
    assert not uv_bin.exists()


def test_offline_is_refused(runner, source, uv_tool_layout):
    result = runner.invoke(app, ["--offline", "self", "update", "--yes"])
    assert result.exit_code == 8, result.output


def test_offline_check_uses_the_cache_when_there_is_one(runner, source, uv_tool_layout):
    result = runner.invoke(app, ["--offline", "self", "update", "--check", "--json"])
    assert result.exit_code == 8
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)
    result = runner.invoke(app, ["--offline", "self", "update", "--check", "--json"])
    assert result.exit_code == 0, result.output
    assert _json(result)["status"] == "available"


def test_a_different_tt_on_path_blocks_the_upgrade(runner, source, uv_tool_layout, uv_bin, monkeypatch):
    monkeypatch.setattr(U.shutil, "which", lambda name: "/opt/other/bin/tt" if name == "tt" else None)
    result = runner.invoke(app, ["self", "update", "--yes"])
    assert result.exit_code == 7, result.output
    assert "/opt/other/bin/tt" in result.output
    assert not uv_bin.exists()


def test_unreachable_source_is_an_error_not_a_crash(runner, uv_tool_layout, monkeypatch, tmp_path):
    monkeypatch.delenv(C.DISABLE_ENV, raising=False)
    monkeypatch.setenv(C.SOURCE_ENV, str(tmp_path / "nope.json"))
    result = runner.invoke(app, ["self", "update", "--yes"])
    assert result.exit_code == 1, result.output
    assert "latest tt release" in result.output


def test_version_mismatch_after_install_is_reported(runner, source, uv_tool_layout, uv_bin, monkeypatch):
    monkeypatch.setattr(U, "installed_version", lambda layout: "98.0.0")
    result = runner.invoke(app, ["self", "update", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    assert _json(result)["status"] == "mismatch"
    assert "Expected tt 99.0.0 but 98.0.0" in result.output


def test_installed_version_ignores_a_failing_interpreter(tmp_path):
    """A non-zero exit must read as "unknown", never as whatever stale stdout held."""
    ok = L.InstallLayout(L.KIND_VENV, tmp_path, "pip", __version__, "/bin/sh")
    bad = L.InstallLayout(L.KIND_VENV, tmp_path, "pip", __version__, str(tmp_path / "missing"))
    assert U.installed_version(bad) is None
    # /bin/sh -c "<python code>" exits non-zero: still None, not the shell's chatter.
    assert U.installed_version(ok) is None


def test_editable_checkout_is_unsupported(runner, source, tmp_path, monkeypatch):
    layout = L.InstallLayout(L.KIND_EDITABLE, tmp_path, "uv", __version__, "python", detail="editable")
    monkeypatch.setattr(U, "detect_layout", lambda: layout)
    result = runner.invoke(app, ["self", "update", "--yes"])
    assert result.exit_code == 7
    assert "git pull" in result.output


# -- the end-of-command notice ----------------------------------------------------------
@pytest.fixture
def spawns(monkeypatch):
    launched = []
    monkeypatch.setattr(C, "spawn_check", lambda paths, **kw: launched.append(paths) or True)
    return launched


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.setattr(C, "_interactive", lambda: True)


def test_pending_notice_is_printed_on_a_tty_at_most_once_a_day(runner, source, uv_tool_layout, interactive, spawns):
    state = C.UpdateState(get_paths())
    state.record(latest=NEWER, current=__version__)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert f"A new release of tt is available: {__version__} → {NEWER}" in result.output
    assert "tt self update" in result.output
    assert spawns == []  # fresh state: no background check
    # Told once; a user who chose not to upgrade is not nagged on the next command...
    result = runner.invoke(app, ["config", "path"])
    assert "A new release" not in result.output
    # ...until a day has passed...
    state.mark_notified(NEWER, now=time.time() - C.NOTICE_INTERVAL_S)
    assert "A new release" in runner.invoke(app, ["config", "path"]).output
    # ...or a different newer version turns up, which is news.
    state.record(latest="100.0.0", current=__version__)
    assert "→ 100.0.0" in runner.invoke(app, ["config", "path"]).output


def test_notice_names_the_manual_command_in_a_shared_venv(runner, source, shared_layout, interactive, spawns, monkeypatch):
    monkeypatch.setattr(C, "detect_layout", lambda: shared_layout)
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)
    result = runner.invoke(app, ["config", "path"])
    assert "/venv/bin/python -m pip install --upgrade tenstorrent" in result.output
    assert "tt self update" not in result.output


@pytest.mark.parametrize("argv", [["--json", "config", "path"], ["--quiet", "config", "path"], ["--offline", "config", "path"]])
def test_no_notice_and_no_spawn_under_json_quiet_or_offline(runner, source, uv_tool_layout, interactive, spawns, argv):
    """Scripts call tt with --json/--quiet: no notice they could not read, and no
    background process going to the network on their behalf either."""
    # Stale state AND a pending newer version: both the spawn and the notice are live.
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__, now=time.time() - C.CHECK_INTERVAL_S)
    result = runner.invoke(app, argv)
    assert "A new release" not in result.output
    assert spawns == []
    # The same state on a plain run does both — so the gate, not the state, is why.
    result = runner.invoke(app, ["config", "path"])
    assert "A new release" in result.output
    assert len(spawns) == 1


def test_no_notice_without_a_tty(runner, source, uv_tool_layout, spawns):
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)
    result = runner.invoke(app, ["config", "path"])
    assert "A new release" not in result.output


def test_stale_state_spawns_one_background_check(runner, source, uv_tool_layout, spawns):
    runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1
    # Once recorded (by the check itself), nothing is spawned again for a day.
    C.UpdateState(get_paths()).record(latest=None, current=__version__)
    runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_no_spawn_while_a_background_check_holds_the_lock(runner, source, uv_tool_layout, spawns):
    """A slow lookup leaves the state stale for its whole duration; commands run in the
    meantime must not each start another one (mirrors telemetry's drainer lock)."""
    state = C.UpdateState(get_paths())
    with state.lock() as acquired:
        assert acquired
        runner.invoke(app, ["config", "path"])
        runner.invoke(app, ["config", "path"])
    assert spawns == []
    # Lock released: the next command spawns exactly one.
    runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_the_background_check_steps_aside_when_another_holds_the_lock(runner, source, uv_tool_layout):
    state = C.UpdateState(get_paths())
    with state.lock():
        result = runner.invoke(app, ["self", "check-update", "--json"])
    assert result.exit_code == 0, result.output
    data = _json(result)
    assert data["busy"] is True and data["latest"] is None
    # Nothing was recorded: the running check owns the state file.
    assert not get_paths().self_update_file.exists()
    assert state.is_stale()


def test_the_lock_is_released_after_a_check(runner, source, uv_tool_layout):
    state = C.UpdateState(get_paths())
    runner.invoke(app, ["self", "check-update"])
    assert not state.check_in_progress()
    assert not state.is_stale()


def test_no_check_for_a_dev_checkout_or_a_system_install(runner, source, tmp_path, spawns, interactive, monkeypatch):
    for kind in (L.KIND_EDITABLE, L.KIND_SYSTEM):
        layout = L.InstallLayout(kind, tmp_path, None, __version__, "python")
        monkeypatch.setattr(C, "detect_layout", lambda layout=layout: layout)
        C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)
        result = runner.invoke(app, ["config", "path"])
        assert "A new release" not in result.output
    assert spawns == []


@pytest.mark.parametrize("switch", [("env", C.DISABLE_ENV), ("ci", "CI"), ("config", None)])
def test_every_off_switch_silences_notice_and_check(runner, source, uv_tool_layout, interactive, spawns, monkeypatch, switch):
    kind, var = switch
    if var:
        monkeypatch.setenv(var, "1")
    else:
        from tenstorrent.config.store import ConfigStore

        ConfigStore(get_paths()).set(C.CONFIG_KEY, False)
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)
    runner.invoke(app, ["config", "path"])
    result = runner.invoke(app, ["config", "path"])
    assert "A new release" not in result.output
    assert spawns == []


def test_the_background_check_itself_never_triggers_another(runner, source, uv_tool_layout, spawns):
    result = runner.invoke(app, ["self", "check-update", "--json"])
    assert result.exit_code == 0, result.output
    assert _json(result)["latest"] == NEWER
    assert spawns == []
    assert not C.UpdateState(get_paths()).is_stale()


def test_the_detached_check_command_releases_stdio(monkeypatch):
    """Same contract as the telemetry drainer: an inherited stdout would make
    `$(tt ...)` wait for the background check to finish."""
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            seen["argv"] = argv
            seen.update(kwargs)

    monkeypatch.setattr(C.subprocess, "Popen", FakePopen)
    assert C.spawn_check(get_paths()) is True
    assert seen["argv"][1:] == ["-m", "tenstorrent", "self", "check-update"]
    devnull = C.subprocess.DEVNULL
    assert (seen["stdin"], seen["stdout"], seen["stderr"]) == (devnull, devnull, devnull)
    assert seen["start_new_session"] is True and seen["close_fds"] is True


# -- tt update hands off to a newer tt first --------------------------------------------
@pytest.fixture
def cached(source):
    """A fresh cached check saying NEWER exists, so `tt update` needs no lookup."""
    C.UpdateState(get_paths()).record(latest=NEWER, current=__version__)


def test_update_offers_the_upgrade_on_a_tty_and_stops_after_it(runner, cached, uv_tool_layout, uv_bin, monkeypatch):
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(U, "installed_version", lambda layout: NEWER)
    result = runner.invoke(app, ["update", "--dry-run"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Upgrade tt before updating the system?" in result.output
    assert "Re-run `tt update`" in result.output
    assert "Update plan" not in result.output  # the new tt should run the update
    assert uv_bin.exists()


def test_update_continues_when_the_offer_is_declined(runner, cached, uv_tool_layout, uv_bin, monkeypatch):
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["update", "--dry-run"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Update plan" in result.output
    assert not uv_bin.exists()


def test_update_without_a_tty_only_mentions_the_release(runner, cached, uv_tool_layout, uv_bin):
    result = runner.invoke(app, ["update", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert "A new release of tt is available" in result.output
    assert "Upgrade tt before" not in result.output
    assert "Update plan" in result.output
    assert not uv_bin.exists()


def test_update_quiet_neither_prompts_nor_mentions(runner, cached, uv_tool_layout, uv_bin, monkeypatch):
    """--quiet hides the notice, so a prompt after it would appear without context."""
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["update", "--dry-run", "--quiet"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Upgrade tt before" not in result.output
    assert "A new release" not in result.output
    assert not uv_bin.exists()


def test_update_json_stays_a_single_document(runner, cached, uv_tool_layout):
    result = runner.invoke(app, ["update", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    assert "A new release" not in result.output
    assert json.loads(result.output)["dry_run"] is True


def test_update_in_a_shared_venv_prints_the_manual_command(runner, cached, shared_layout, monkeypatch):
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "-m pip install --upgrade tenstorrent" in result.output
    assert "Upgrade tt before" not in result.output


def test_update_is_unaffected_when_current(runner, source, uv_tool_layout, monkeypatch):
    source("0.0.1")
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["update"])
    assert "A new release" not in result.output


def test_update_dry_run_never_looks_up_the_release(runner, source, uv_tool_layout, monkeypatch):
    """A preview must not wait on the network: with nothing cached it says nothing,
    even though the source knows a newer version."""
    monkeypatch.setattr(U, "_stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "A new release" not in result.output
    assert not get_paths().self_update_file.exists()


def test_update_uses_a_fresh_cache_instead_of_fetching(runner, cached, uv_tool_layout, monkeypatch, tmp_path):
    monkeypatch.setenv(C.SOURCE_ENV, str(tmp_path / "unreachable.json"))  # a fetch would fail
    result = runner.invoke(app, ["update", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert f"→ {NEWER}" in result.output
    # ...and today's notice is marked shown, so the next command does not repeat it.
    assert not C.UpdateState(get_paths()).notice_due(NEWER)


def test_update_refreshes_a_stale_cache_once(runner, source, uv_tool_layout, monkeypatch):
    state = C.UpdateState(get_paths())
    state.record(latest=None, current=__version__, now=time.time() - C.CHECK_INTERVAL_S)
    assert state.is_stale()
    result = runner.invoke(app, ["update", "--yes"])
    assert f"→ {NEWER}" in result.output
    assert not state.is_stale() and state.pending(__version__) == NEWER
