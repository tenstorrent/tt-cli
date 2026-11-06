# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""huggingface_hub draws its own tqdm bars outside OutputManager, so the
`--quiet` / `--json` contract has to be enforced explicitly."""

import pytest
from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars

from tenstorrent.modelhub.hub import _honor_output_mode
from tenstorrent.output import OutputManager


@pytest.fixture(autouse=True)
def restore_progress_bars():
    # disable_progress_bars() is process-global; don't leak it into other tests.
    yield
    enable_progress_bars()


@pytest.mark.parametrize(
    "kwargs",
    [{"quiet": True}, {"json_mode": True}, {"quiet": True, "json_mode": True}],
)
def test_quiet_and_json_silence_the_hub_progress_bars(kwargs):
    _honor_output_mode(OutputManager(**kwargs))
    assert are_progress_bars_disabled()


def test_human_mode_keeps_the_download_bar():
    # A live download bar earns its space; only the machine-readable and
    # silenced modes lose it.
    _honor_output_mode(OutputManager())
    assert not are_progress_bars_disabled()


def test_no_output_manager_is_a_no_op():
    _honor_output_mode(None)
    assert not are_progress_bars_disabled()



def test_hf_cache_resolution_precedence(tmp_path, monkeypatch):
    """Precedence: paths.hf_model_cache_directory -> HF_HUB_CACHE -> HF_HOME -> default."""
    from pathlib import Path
    from tenstorrent.config.paths import get_paths
    from tenstorrent.config.store import ConfigStore
    from tenstorrent.modelhub.hub import hf_cache_dir, hf_home_dir

    config = ConfigStore(get_paths())
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    # 1. Default when nothing is set
    assert hf_home_dir(config) == Path.home() / ".cache" / "huggingface"
    assert hf_cache_dir(config) is None

    # 2. HF_HOME set
    hf_home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert hf_home_dir(config) == hf_home
    assert hf_cache_dir(config) == str(hf_home / "hub")

    # 3. HF_HUB_CACHE set outside HF_HOME overrides HF_HOME
    hub_cache = tmp_path / "shared_hub_cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    assert hf_home_dir(config) == hub_cache
    assert hf_cache_dir(config) == str(hub_cache)

    # 4. paths.hf_model_cache_directory overrides both HF_HUB_CACHE and HF_HOME
    configured = tmp_path / "configured_root"
    config.set("paths.hf_model_cache_directory", str(configured))
    assert hf_home_dir(config) == configured
    assert hf_cache_dir(config) == str(configured / "hub")


def test_tt_model_env_overrides_inherited_hf_hub_cache_when_configured(tmp_path, monkeypatch):
    """When paths.hf_model_cache_directory is set, an inherited HF_HUB_CACHE in
    os.environ must not bypass the configured folder for tt-model commands."""
    from tenstorrent.backends.serving.model_manager import ModelManagerBackend
    from tenstorrent.config.paths import get_paths
    from tenstorrent.config.store import ConfigStore
    from tenstorrent.output import OutputManager

    config = ConfigStore(get_paths())
    configured = tmp_path / "configured_hf"
    config.set("paths.hf_model_cache_directory", str(configured))
    monkeypatch.setenv("HF_HUB_CACHE", "/opt/tenstorrent/models/hub")

    backend = ModelManagerBackend(
        registry=None, runner=None, config=config, output=OutputManager()
    )
    env = backend._env()
    assert env["HF_HOME"] == str(configured)
    assert env["HF_HUB_CACHE"] == str(configured / "hub")
