# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json

import pytest

from tenstorrent.config.paths import get_paths
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.manifest import (
    LocalManifestSource,
    golden_cache_read,
    golden_cache_write,
)


@pytest.fixture(autouse=True)
def fixture_golden(monkeypatch, fakes_dir):
    """These are pure unit tests of the manifest layer: pin TT_GOLDEN_PATH to the
    fixture ourselves so they behave identically under --hardware, where conftest
    deliberately leaves it unset (the real fetch path). Tests exercising the
    cache/unknown paths delete it again."""
    monkeypatch.setenv("TT_GOLDEN_PATH", str(fakes_dir / "data" / "golden.json"))


def load_manifest():
    return LocalManifestSource(get_paths()).load()


def test_manifest_loads_with_fixture_golden(isolated_dirs):
    # TT_GOLDEN_PATH points at a verbatim captured golden.json (v1.0.0).
    manifest = load_manifest()
    assert manifest.schema_version == 1
    smi = manifest.spec("tt-smi")
    assert smi.kind == "uv-tool"
    assert smi.package == "tt-smi"
    assert smi.golden_version  # pinned by golden.json
    assert smi.needs_sudo is True
    flash = manifest.spec("tt-flash")
    assert flash.kind == "uv-tool"
    installer = manifest.spec("tt-installer")
    assert installer.kind == "script"
    assert installer.url and installer.url.startswith("https://")
    inference = manifest.spec("tt-inference-server")
    assert inference.kind == "git-venv"
    assert inference.entry == "run.py"
    assert inference.deps == ("pyyaml", "packaging")  # run.py bootstrap imports
    # golden.json system/firmware pins surface for `tt update`
    assert "kmd" in manifest.system
    assert manifest.firmware
    # smi/flash/installer/test-sha are not system components
    assert not {"smi", "flash", "installer", "test-sha"} & manifest.system.keys()


def test_supplement_records_the_golden_pin(isolated_dirs):
    # The [golden] table is the single pin everything hangs off: the tag `tt update`
    # fetches at (and checks install.sh against), the digest the fetch is verified
    # with, and the URL template it uses.
    manifest = load_manifest()
    assert manifest.golden_tag
    assert manifest.golden_sha256
    assert manifest.golden_url_template and "{tag}" in manifest.golden_url_template
    assert manifest.golden_url == manifest.golden_url_template.format(
        tag=manifest.golden_tag
    )


def test_golden_path_override(tmp_path, monkeypatch):
    golden = {"smi": "9.0.0", "kmd": "9.9.9", "firmware": "42.0.0"}
    path = tmp_path / "custom-golden.json"
    path.write_text(json.dumps(golden))
    monkeypatch.setenv("TT_GOLDEN_PATH", str(path))
    manifest = load_manifest()
    assert manifest.spec("tt-smi").golden_version == "9.0.0"
    assert manifest.firmware == "42.0.0"
    assert manifest.system == {"kmd": "9.9.9"}
    # supplement tools still merged in
    assert manifest.spec("tt-inference-server").kind == "git-venv"


def test_missing_override_path_is_config_error(monkeypatch):
    monkeypatch.setenv("TT_GOLDEN_PATH", "/nonexistent/golden.json")
    with pytest.raises(TTError) as exc:
        load_manifest()
    assert exc.value.exit_code == ExitCode.CONFIG


def test_invalid_golden_override_is_config_error(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "a", "map"]))
    monkeypatch.setenv("TT_GOLDEN_PATH", str(path))
    with pytest.raises(TTError) as exc:
        load_manifest()
    assert exc.value.exit_code == ExitCode.CONFIG


def test_without_cache_golden_versions_are_unknown(monkeypatch):
    # Before the first `tt update`, tt still knows its tools — just not their pins.
    monkeypatch.delenv("TT_GOLDEN_PATH")
    manifest = load_manifest()
    assert manifest.spec("tt-smi").golden_version == ""
    assert manifest.system == {}
    assert "run `tt update`" in manifest.origin
    # the pin itself is still known (it lives in the supplement, not the cache)
    assert manifest.golden_tag


def test_cache_is_used_when_its_tag_matches_the_pin(monkeypatch):
    monkeypatch.delenv("TT_GOLDEN_PATH")
    paths = get_paths()
    tag = load_manifest().golden_tag
    golden_cache_write(paths, tag, {"smi": "7.0.0", "kmd": "3.0.0"})
    manifest = load_manifest()
    assert manifest.spec("tt-smi").golden_version == "7.0.0"
    assert "cached" in manifest.origin


def test_stale_cache_from_another_tag_is_ignored(monkeypatch):
    # Upgrading tt moves the [golden] tag; a cache written for the old tag must not
    # keep supplying old pins.
    monkeypatch.delenv("TT_GOLDEN_PATH")
    paths = get_paths()
    golden_cache_write(paths, "v0.0.9-old", {"smi": "1.0.0"})
    manifest = load_manifest()
    assert manifest.spec("tt-smi").golden_version == ""
    assert golden_cache_read(paths, manifest.golden_tag) is None


def test_corrupt_cache_degrades_to_unknown(monkeypatch):
    # A damaged cache must never break unrelated commands.
    monkeypatch.delenv("TT_GOLDEN_PATH")
    paths = get_paths()
    paths.golden_file.parent.mkdir(parents=True, exist_ok=True)
    paths.golden_file.write_text("{ not json")
    manifest = load_manifest()
    assert manifest.spec("tt-smi").golden_version == ""


def test_unknown_tool_is_config_error():
    manifest = load_manifest()
    with pytest.raises(TTError) as exc:
        manifest.spec("tt-nonsense")
    assert exc.value.exit_code == ExitCode.CONFIG
