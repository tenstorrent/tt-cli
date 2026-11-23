# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from __future__ import annotations

from pathlib import Path

from tenstorrent.backends.serving.studio import runtime_root


def test_runtime_root_defaults_to_the_wheels_home_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TT_STUDIO_HOME", raising=False)
    assert runtime_root() == tmp_path / ".tt-studio"


def test_runtime_root_follows_tt_studio_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_STUDIO_HOME", str(tmp_path / "studio"))
    assert runtime_root() == tmp_path / "studio"


def test_runtime_root_expands_a_tilde_and_ignores_a_blank_value(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TT_STUDIO_HOME", "~/custom")
    assert runtime_root() == tmp_path / "custom"
    monkeypatch.setenv("TT_STUDIO_HOME", "  ")
    assert runtime_root() == Path(tmp_path / ".tt-studio")


def _backend(tools_dir, containers=([], [])):
    from types import SimpleNamespace

    from tenstorrent.backends.serving.studio import StudioBackend

    notes: list[str] = []
    output = SimpleNamespace(status=notes.append, warn=notes.append)
    registry = SimpleNamespace(paths=SimpleNamespace(tools_dir=tools_dir))
    backend = StudioBackend(registry, None, None, output)
    backend._docker = lambda: "docker"
    backend._containers = lambda docker: containers
    return backend, notes


def _old_checkout(tools_dir, ref="v2.12.0"):
    src = tools_dir / f"tt-studio-{ref}" / "src"
    src.mkdir(parents=True)
    (src / "run.py").write_text("")
    return src.parent


def test_old_git_checkouts_are_removed_but_the_pypi_install_is_not(tmp_path):
    old = _old_checkout(tmp_path)
    venv = tmp_path / "tt-studio"  # the uv tool venv of the PyPI install
    venv.mkdir()
    other = tmp_path / "tt-studio-notes"  # no src/run.py: not ours to delete
    other.mkdir()
    backend, notes = _backend(tmp_path)
    backend._remove_old_checkouts()
    assert not old.exists() and venv.exists() and other.exists()
    assert any("Removed the old TT-Studio checkout" in n for n in notes)


def test_old_checkouts_stay_while_a_studio_stack_is_running(tmp_path):
    old = _old_checkout(tmp_path)
    backend, _ = _backend(tmp_path, containers=(["tt_studio_backend"], []))
    backend._remove_old_checkouts()
    assert old.exists()


def test_old_checkouts_stay_when_docker_cannot_be_asked(tmp_path):
    old = _old_checkout(tmp_path)
    backend, _ = _backend(tmp_path)

    def boom():
        raise RuntimeError("no docker")

    backend._docker = boom
    backend._remove_old_checkouts()
    assert old.exists()
