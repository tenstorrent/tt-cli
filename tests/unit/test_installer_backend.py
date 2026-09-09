# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""InstallerBackend: the golden-tag guard (a golden install.sh whose
TTIS_GOLDEN_VERSIONS_TAG disagrees with the [golden] tag we pin must refuse to
run) and refresh_goldens (fetch + verify + cache of golden.json at the pin).
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from tenstorrent.backends.installer import INSTALLER_TOOL, InstallerBackend
from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import ConfigStore
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.output import OutputManager
from tenstorrent.tools.manifest import golden_cache_write
from tenstorrent.tools.registry import ToolRegistry
from tenstorrent.tools.state import ToolState


class RecordingRunner:
    def __init__(self):
        self.streamed = []

    def stream(self, argv, *, tool=None, cwd=None):
        self.streamed.append(list(argv))
        return 0


def script_with_tag(tag: str | None) -> str:
    line = f'readonly TTIS_GOLDEN_VERSIONS_TAG="{tag}"\n' if tag else ""
    return f"#!/bin/sh\n{line}exit 0\n"


@pytest.fixture(autouse=True)
def fixture_golden(monkeypatch, fakes_dir):
    """Pure unit tests: pin TT_GOLDEN_PATH to the fixture ourselves so they behave
    identically under --hardware, where conftest deliberately leaves it unset.
    The refresh tests delete it again to engage the fetch/cache layer."""
    monkeypatch.setenv("TT_GOLDEN_PATH", str(fakes_dir / "data" / "golden.json"))


@pytest.fixture
def backend(isolated_dirs):
    paths = get_paths()
    registry = ToolRegistry(paths, ConfigStore(paths), runner=RecordingRunner())
    runner = RecordingRunner()
    backend = InstallerBackend(
        registry, runner, paths, OutputManager(quiet=True)
    )
    backend._recording_runner = runner
    return backend


def install_managed_script(backend: InstallerBackend, text: str) -> Path:
    """Record a golden-pinned install.sh in installed state, the source the
    guard applies to (env/config overrides are the user's business)."""
    paths = backend.paths
    script = paths.tool_bin_dir / "install.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    script.chmod(0o755)
    golden = backend.registry.spec(INSTALLER_TOOL).golden_version
    ToolState(paths).record(INSTALLER_TOOL, version=golden, path=script)
    return script


# -- the golden-tag guard ---------------------------------------------------------------
def test_matching_tag_runs_the_installer(backend):
    tag = backend.registry.manifest.golden_tag
    assert tag  # the bundled supplement records the pinned release
    install_managed_script(backend, script_with_tag(tag))
    assert backend._run_system_installer(offline=False) is True
    assert backend._recording_runner.streamed


def test_mismatched_tag_refuses_to_run(backend):
    install_managed_script(backend, script_with_tag("v9.9.9"))
    with pytest.raises(TTError) as exc:
        backend._run_system_installer(offline=False)
    assert exc.value.exit_code == ExitCode.CONFIG
    assert backend._recording_runner.streamed == []


def test_script_without_a_tag_refuses_to_run(backend):
    # The pinned golden install.sh is known to declare its tag; a managed script
    # without one means the pin and the guard have drifted apart.
    install_managed_script(backend, script_with_tag(None))
    with pytest.raises(TTError) as exc:
        backend._run_system_installer(offline=False)
    assert exc.value.exit_code == ExitCode.CONFIG


def test_env_override_script_is_not_checked(backend, monkeypatch, tmp_path):
    # An env-supplied install.sh is the user's business (same rule as ensure()) —
    # and it is how the fake-mode test suite wires the fake installer.
    script = tmp_path / "my-install.sh"
    script.write_text(script_with_tag("v0.0.0-whatever"))
    script.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_TT_INSTALLER", str(script))
    assert backend._run_system_installer(offline=False) is True
    assert backend._recording_runner.streamed


def test_user_requested_version_is_not_checked(backend, monkeypatch, tmp_path):
    # `tt update <semver>` runs a non-golden release, which legitimately pins a
    # different tt-sw-manifest tag.
    def fake_fetch(url):
        return script_with_tag("v0.0.1-old").encode()

    monkeypatch.setattr(
        "tenstorrent.tools.installers.ScriptInstaller._fetch_https",
        staticmethod(fake_fetch),
    )
    assert backend._run_system_installer(offline=False, version="3.1.0") is True
    assert backend._recording_runner.streamed


# -- refresh_goldens -------------------------------------------------------------------
GOLDEN = {"smi": "6.1.0", "flash": "3.10.0", "kmd": "2.10.0", "firmware": "19.13.1"}


@pytest.fixture
def fetching_backend(backend, monkeypatch):
    """A backend whose golden.json 'download' returns GOLDEN, with the recorded
    sha256 patched to match and TT_GOLDEN_PATH cleared so the cache layer engages."""
    monkeypatch.delenv("TT_GOLDEN_PATH")
    blob = json.dumps(GOLDEN).encode()
    fetches = []

    def fake_fetch(url):
        fetches.append(url)
        return blob

    monkeypatch.setattr("tenstorrent.backends.installer.fetch_https", fake_fetch)
    backend.registry._manifest = dataclasses.replace(
        backend.registry.manifest, golden_sha256=hashlib.sha256(blob).hexdigest()
    )
    backend._fetches = fetches
    return backend


def test_refresh_fetches_verifies_and_caches(fetching_backend):
    backend = fetching_backend
    expected_url = backend.registry.manifest.golden_url
    backend.refresh_goldens(offline=False)
    assert backend._fetches == [expected_url]
    # cache written, manifest reloaded with the fetched pins
    manifest = backend.registry.manifest
    assert manifest.spec("tt-smi").golden_version == "6.1.0"
    assert manifest.system["kmd"] == "2.10.0"
    # a second refresh is a no-op: the pin names an immutable release
    backend.refresh_goldens(offline=False)
    assert backend._fetches == [expected_url]


def test_refresh_rejects_a_checksum_mismatch(fetching_backend):
    backend = fetching_backend
    backend.registry._manifest = dataclasses.replace(
        backend.registry.manifest, golden_sha256="0" * 64
    )
    with pytest.raises(TTError) as exc:
        backend.refresh_goldens(offline=False)
    assert exc.value.exit_code == ExitCode.TOOL_FAILED
    assert not backend.paths.golden_file.exists()  # nothing cached


def test_refresh_offline_without_cache_is_offline_error(backend, monkeypatch):
    monkeypatch.delenv("TT_GOLDEN_PATH")
    with pytest.raises(TTError) as exc:
        backend.refresh_goldens(offline=True)
    assert exc.value.exit_code == ExitCode.OFFLINE


def test_refresh_offline_with_valid_cache_is_fine(backend, monkeypatch):
    monkeypatch.delenv("TT_GOLDEN_PATH")
    tag = backend.registry.manifest.golden_tag
    golden_cache_write(backend.paths, tag, GOLDEN)
    backend.registry.reload_manifest()
    backend.refresh_goldens(offline=True)  # does not raise, no network involved
    assert backend.registry.manifest.spec("tt-smi").golden_version == "6.1.0"


def test_refresh_skips_a_golden_path_override(backend, monkeypatch):
    # TT_GOLDEN_PATH (pointing at the fixture) is the user's business: no fetch,
    # no cache, no error — exactly how the fake suite stays network-free.
    def boom(url):
        raise AssertionError("must not fetch")

    monkeypatch.setattr("tenstorrent.backends.installer.fetch_https", boom)
    backend.refresh_goldens(offline=False)
    assert not backend.paths.golden_file.exists()
