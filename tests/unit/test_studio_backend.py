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
