# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from pathlib import Path

import pytest

from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import ConfigStore
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.installers import InstallResult
from tenstorrent.tools.manifest import ToolSpec
from tenstorrent.tools.registry import ToolRegistry, env_var_for
from tenstorrent.tools.runner import Runner
from tenstorrent.tools.state import ToolState


class FakeInstaller:
    def __init__(self, paths):
        self.paths = paths
        self.calls = []

    def install(self, spec, *, offline=False):
        self.calls.append((spec.name, offline))
        path = self.paths.tool_bin_dir / spec.bin_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        return InstallResult(spec.name, spec.golden_version, path)


@pytest.fixture(autouse=True)
def fixture_golden(monkeypatch, fakes_dir):
    """Pure unit tests: pin TT_GOLDEN_PATH to the fixture ourselves so golden
    versions are known even under --hardware, where conftest deliberately leaves
    it unset (ensure()/status() assertions depend on a known tt-smi pin)."""
    monkeypatch.setenv("TT_GOLDEN_PATH", str(fakes_dir / "data" / "golden.json"))


@pytest.fixture
def registry(isolated_dirs):
    paths = get_paths()
    fake = FakeInstaller(paths)
    reg = ToolRegistry(
        paths,
        ConfigStore(paths),
        runner=Runner(sudo_command=""),
        installers={"uv-tool": fake, "script": fake, "git-venv": fake},
    )
    reg._fake_installer = fake
    return reg


def test_env_var_naming():
    assert env_var_for("tt-smi") == "TT_TOOL_BIN_TT_SMI"


def test_resolve_env_wins(registry, monkeypatch, tmp_path):
    fake_smi = tmp_path / "custom-smi"
    fake_smi.write_text("")
    monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(fake_smi))
    assert registry.resolve("tt-smi") == fake_smi


def test_resolve_config_override_second(registry, tmp_path):
    override = tmp_path / "override-smi"
    override.write_text("")
    registry.config.set("tools.override.tt-smi", str(override))
    assert registry.resolve("tt-smi") == override


def test_resolve_installed_state_third(registry):
    installed = registry.paths.tool_bin_dir / "tt-smi"
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_text("")
    ToolState(registry.paths).record("tt-smi", version="5.3.0", path=installed)
    assert registry.resolve("tt-smi") == installed


def _installer_venv_tool(name: str) -> Path:
    """Drop a fake entry point where tt-installer's managed venv would have it. HOME is
    the per-test temp home (isolated_dirs), so this never touches the real machine."""
    path = Path.home() / ".tenstorrent-venv" / "bin" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    return path


def test_resolve_falls_back_to_the_installer_venv_for_tt_smi(registry):
    """A box set up by tt-installer has ~/.tenstorrent-venv/bin/tt-smi before `tt update`
    ever ran; `tt model list` used to warn "device detection skipped (Required tool
    'tt-smi' is not installed.)" on exactly that machine."""
    venv_smi = _installer_venv_tool("tt-smi")
    assert registry.resolve("tt-smi") == venv_smi
    assert {r.name: r for r in registry.status()}["tt-smi"].source == "installer"


def test_installed_state_beats_the_installer_venv(registry):
    _installer_venv_tool("tt-smi")
    installed = registry.paths.tool_bin_dir / "tt-smi"
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_text("")
    ToolState(registry.paths).record("tt-smi", version="5.3.0", path=installed)
    assert registry.resolve("tt-smi") == installed


def test_installer_venv_is_not_probed_for_tools_the_installer_does_not_own(registry):
    # A stray binary in that venv must never stand in for a pinned tool.
    _installer_venv_tool("tt-model")
    with pytest.raises(TTError) as exc:
        registry.resolve("tt-model")
    assert exc.value.exit_code == ExitCode.TOOL_MISSING


def test_ensure_still_installs_the_pin_when_only_the_installer_venv_has_the_tool(registry):
    _installer_venv_tool("tt-smi")
    path = registry.ensure("tt-smi")
    assert registry._fake_installer.calls == [("tt-smi", False)]
    assert path == registry.paths.tool_bin_dir / "tt-smi"


def test_resolve_stale_state_entry_is_missing(registry):
    ToolState(registry.paths).record(
        "tt-smi", version="5.3.0", path=Path("/nonexistent/tt-smi")
    )
    with pytest.raises(TTError) as exc:
        registry.resolve("tt-smi")
    assert exc.value.exit_code == ExitCode.TOOL_MISSING


def test_resolve_missing_tool_points_at_tt_update(registry):
    with pytest.raises(TTError) as exc:
        registry.resolve("tt-smi")
    err = exc.value
    assert err.exit_code == ExitCode.TOOL_MISSING
    assert "tt update" in err.next_step


def test_resolve_unknown_tool_is_config_error(registry):
    with pytest.raises(TTError) as exc:
        registry.resolve("not-a-tool")
    assert exc.value.exit_code == ExitCode.CONFIG


def test_ensure_installs_and_records(registry):
    path = registry.ensure("tt-smi")
    assert path.exists()
    assert registry._fake_installer.calls == [("tt-smi", False)]
    assert registry.state.get("tt-smi").version == registry.spec("tt-smi").golden_version
    # second ensure resolves without reinstalling
    registry.ensure("tt-smi")
    assert len(registry._fake_installer.calls) == 1


def test_ensure_reinstalls_when_installed_version_is_stale(registry):
    # a golden-pin bump must take effect for on-demand tools (script/git-venv):
    # state says an old version is installed → ensure() reinstalls at the pin.
    stale = registry.paths.tool_bin_dir / "run.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("")
    ToolState(registry.paths).record("tt-inference-server", version="v0.0.1", path=stale)
    registry.ensure("tt-inference-server")
    assert registry._fake_installer.calls == [("tt-inference-server", False)]
    golden = registry.spec("tt-inference-server").golden_version
    assert registry.state.get("tt-inference-server").version == golden


def test_ensure_leaves_env_and_override_resolutions_alone(registry, monkeypatch, tmp_path):
    custom = tmp_path / "my-run.py"
    custom.write_text("")
    monkeypatch.setenv("TT_TOOL_BIN_TT_INFERENCE_SERVER", str(custom))
    assert registry.ensure("tt-inference-server") == custom
    assert registry._fake_installer.calls == []


def test_status_reports_source(registry, monkeypatch, tmp_path):
    rows = {r.name: r for r in registry.status()}
    assert rows["tt-smi"].source == "missing"
    assert rows["tt-smi"].golden_version == registry.spec("tt-smi").golden_version
    fake = tmp_path / "smi"
    fake.write_text("")
    monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(fake))
    rows = {r.name: r for r in registry.status()}
    assert rows["tt-smi"].source == "env"
