# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Tag classification and manifest reads in modelhub/bundles.py.

`_classify` covers the tags tt-model-manager writes today (see cli.py's
`mesh_topology.lower()` and build.py's `_card_tags` upstream); `hardware_for`
covers the same data read back from a pulled bundle's own manifest."""

from __future__ import annotations

from tenstorrent.modelhub.bundles import (
    _classify,
    _hardware_chips,
    _is_hardware_tag,
    hardware_for,
    hardware_satisfies,
)


def test_is_hardware_tag_accepts_known_boards_with_or_without_a_count():
    for tag in ("p150", "p150x4", "p300x2", "n300", "n300x4", "e150"):
        assert _is_hardware_tag(tag), tag


def test_is_hardware_tag_rejects_arch_and_unrelated_tags():
    for tag in ("blackhole", "wormhole_b0", "vllm", "region:us", "q200x4"):
        assert not _is_hardware_tag(tag), tag


def test_classify_splits_arch_from_hardware_tags():
    kind, engine, arch, hardware = _classify(
        ["tt-model-container", "blackhole", "vllm-plugin", "p150x4", "region:us"]
    )
    assert kind == "container"
    assert engine == "vllm-plugin"
    assert arch == ["blackhole"]
    assert hardware == ["p150x4"]


def test_classify_sorts_multiple_hardware_tags():
    _, _, _, hardware = _classify(["p300x2", "p150x4"])
    assert hardware == ["p150x4", "p300x2"]


def test_hardware_for_reads_the_default_serve_block(monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles._manifest_for",
        lambda repo_id, entry: {"container": {"serve": {"hardware": "P150x4"}}},
    )
    assert hardware_for("ns/repo", {}) == ["p150x4"]


def test_hardware_for_collects_every_serve_profile(monkeypatch):
    manifest = {
        "container": {
            "serve": {"hardware": "p150x4"},
            "serve_profiles": [
                {"name": "default"},  # inherits the flat serve block
                {"name": "big-mesh", "hardware": "p300x2"},
            ],
        }
    }
    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles._manifest_for", lambda repo_id, entry: manifest
    )
    assert hardware_for("ns/repo", {}) == ["p150x4", "p300x2"]


def test_hardware_for_returns_empty_without_a_manifest(monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles._manifest_for", lambda repo_id, entry: None
    )
    assert hardware_for("ns/repo", {}) == []


def test_hardware_chips_counts_boards_times_mesh_multiplier():
    assert _hardware_chips("p150") == ("blackhole", 1)
    assert _hardware_chips("p300") == ("blackhole", 2)
    assert _hardware_chips("p150x4") == ("blackhole", 4)
    assert _hardware_chips("n300x2") == ("wormhole_b0", 4)


def test_hardware_chips_is_none_for_an_unrecognised_tag():
    assert _hardware_chips("galaxy") is None
    assert _hardware_chips("q200x4") is None


def test_hardware_satisfies_allows_a_bundle_needing_fewer_chips_of_the_same_arch():
    # a p150 (1 blackhole chip) bundle runs on anything with >= 1 blackhole chip
    assert hardware_satisfies("p150", "p150")
    assert hardware_satisfies("p150", "p300")  # 2 chips, same arch
    assert hardware_satisfies("p150", "p150x4")  # 4 chips, same arch
    assert hardware_satisfies("p150", "p300x2")  # 4 chips, different board
    # P150x4 and P300x2 are both a (1, 4) mesh — same chip budget, either board
    assert hardware_satisfies("p300x2", "p150x4")


def test_hardware_satisfies_rejects_too_few_chips_or_a_different_arch():
    assert not hardware_satisfies("p150x4", "p150")  # needs more chips than offered
    assert not hardware_satisfies("n150", "p150")  # wormhole_b0 vs blackhole


def test_hardware_satisfies_falls_back_to_an_exact_match_for_unknown_tags():
    assert hardware_satisfies("galaxy", "galaxy")
    assert not hardware_satisfies("galaxy", "p150")
    assert not hardware_satisfies("p150", "galaxy")
