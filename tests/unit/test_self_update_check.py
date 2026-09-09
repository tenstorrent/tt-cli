# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The background version check: release selection, the state file, and the fetch."""

from __future__ import annotations

import json

import pytest

from tenstorrent import __version__
from tenstorrent.config.paths import get_paths
from tenstorrent.selfupdate import check as C


def pypi_doc(**releases):
    """Build a PyPI-shaped project document. Each value is a list of file dicts."""
    return {"info": {"name": "tenstorrent"}, "releases": releases}


def file(requires_python=">=3.10", yanked=False):
    return {"filename": "x.whl", "requires_python": requires_python, "yanked": yanked}


def test_latest_release_skips_pre_yanked_empty_and_incompatible():
    doc = pypi_doc(**{
        "0.1.0": [file()],
        "0.1.2": [file()],
        "0.1.5": [],  # no files: never installable
        "0.2.0": [file(yanked=True)],
        "0.2.5": [file(requires_python=">=3.14")],
        "0.3.0rc1": [file()],
        "1.0.0.dev1": [file()],
        "not-a-version": [file()],
    })
    assert C.latest_release(doc, python_version="3.12.3") == "0.1.2"
    assert C.latest_release(doc, python_version="3.14.0") == "0.2.5"


def test_one_unyanked_file_is_enough():
    doc = pypi_doc(**{"0.2.0": [file(yanked=True), file(requires_python=None)]})
    assert C.latest_release(doc, python_version="3.12.0") == "0.2.0"


def test_no_releases_means_none():
    assert C.latest_release({"releases": {}}) is None
    assert C.latest_release({}) is None


@pytest.mark.parametrize(
    "candidate, current, expected",
    [("0.2.0", "0.1.0", True), ("0.1.0", "0.1.0", False), ("0.1.0", "0.1.0.dev0", True),
     (None, "0.1.0", False), ("0.2.0", None, False), ("junk", "0.1.0", False)],
)
def test_is_newer(candidate, current, expected):
    assert C.is_newer(candidate, current) is expected


def test_state_is_stale_until_recorded_and_pending_only_for_the_same_version():
    state = C.UpdateState(get_paths())
    assert state.is_stale()
    assert state.pending("0.1.0") is None
    state.record(latest="0.2.0", current="0.1.0", now=1_000_000.0)
    assert not state.is_stale(now=1_000_000.0 + C.CHECK_INTERVAL_S - 1)
    assert state.is_stale(now=1_000_000.0 + C.CHECK_INTERVAL_S)
    assert state.pending("0.1.0") == "0.2.0"
    # Upgraded since the check: the cached "latest" was compared against another tt.
    assert state.pending("0.2.0") is None
    assert state.pending("0.1.9") is None


def test_state_with_no_latest_is_not_pending_but_is_fresh():
    state = C.UpdateState(get_paths())
    state.record(latest=None, current="0.1.0", now=5.0)
    assert not state.is_stale(now=6.0)
    assert state.pending("0.1.0") is None


def test_run_check_reads_a_local_file_source(tmp_path, monkeypatch):
    source = tmp_path / "pypi.json"
    source.write_text(json.dumps(pypi_doc(**{"99.0.0": [file()]})))
    monkeypatch.setenv(C.SOURCE_ENV, str(source))
    result = C.run_check(get_paths(), current="0.1.0")
    assert result.latest == "99.0.0" and result.newer and result.error is None
    assert C.UpdateState(get_paths()).pending("0.1.0") == "99.0.0"


def test_failed_lookup_is_recorded_so_the_next_command_does_not_retry_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv(C.SOURCE_ENV, str(tmp_path / "missing.json"))
    result = C.run_check(get_paths(), current="0.1.0")
    assert result.latest is None and result.error
    state = C.UpdateState(get_paths())
    assert not state.is_stale()
    assert state.pending("0.1.0") is None


def test_unparseable_state_file_is_treated_as_empty():
    paths = get_paths()
    paths.self_update_file.parent.mkdir(parents=True, exist_ok=True)
    paths.self_update_file.write_text("this is not toml = = =")
    state = C.UpdateState(paths)
    assert state.load() == {}
    assert state.is_stale()


def test_check_enabled_switches(monkeypatch):
    from tenstorrent.config.store import ConfigStore

    config = ConfigStore(get_paths())
    monkeypatch.delenv(C.DISABLE_ENV, raising=False)
    assert C.check_enabled(config)
    assert not C.check_enabled(config, offline=True)
    monkeypatch.setenv("CI", "true")
    assert not C.check_enabled(config)
    monkeypatch.delenv("CI")
    monkeypatch.setenv(C.DISABLE_ENV, "1")
    assert not C.check_enabled(config)
    monkeypatch.setenv(C.DISABLE_ENV, "0")  # "0" means unset, like the other switches
    assert C.check_enabled(config)
    config.set(C.CONFIG_KEY, False)
    assert not C.check_enabled(config)


def test_default_source_is_pypi_for_this_package():
    assert C.DEFAULT_SOURCE == "https://pypi.org/pypi/tenstorrent/json"
    assert __version__  # the check compares against the running version
