# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Tag classification and manifest reads in modelhub/bundles.py.

`_classify` covers the tags tt-model-manager writes today (see cli.py's
`mesh_topology.lower()` and build.py's `_card_tags` upstream); `hardware_for`
covers the same data read back from a pulled bundle's own manifest;
`hardware_from_hub_manifest` covers a never-pulled bundle whose tags carry
none, by fetching that same manifest from the Hub instead."""

from __future__ import annotations

import json

from tenstorrent.modelhub.bundles import (
    MANIFEST_NAME,
    _classify,
    _hardware_chips,
    _is_hardware_tag,
    drop_superseded_hardware,
    hardware_for,
    hardware_from_hub_manifest,
    hardware_satisfies,
    search_community,
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


def test_drop_superseded_hardware_keeps_only_the_smallest_matching_tag():
    # speecht5_tts: same capability on p150 and p300x2, so p300x2 adds nothing.
    kept = drop_superseded_hardware({"p150": ("media",), "p300x2": ("media",)})
    assert kept == ["p150"]


def test_drop_superseded_hardware_keeps_a_bigger_tag_with_a_real_difference():
    # Llama: p300 unlocks a longer context than p150, so both stay.
    kept = drop_superseded_hardware({"p150": (65536,), "p300": (131072,)})
    assert kept == ["p150", "p300"]


def test_drop_superseded_hardware_keeps_a_bigger_tag_with_its_own_tuning():
    # Same max_context, but p300x2 ships its own trace_region_size — a real
    # integration, not a copy of p300's listing.
    kept = drop_superseded_hardware({"p150": (65536, 56000000), "p300x2": (65536, 155000000)})
    assert kept == ["p150", "p300x2"]


def test_drop_superseded_hardware_keeps_equal_sized_siblings():
    # p150x4 and p300x2 are both a 4-chip blackhole mesh, so neither supersedes
    # the other — mesh-equivalent alternates, not a smaller/bigger pair.
    kept = drop_superseded_hardware({"p150x4": ("x",), "p300x2": ("x",)})
    assert kept == ["p150x4", "p300x2"]


def test_drop_superseded_hardware_leaves_tags_outside_the_board_grammar_alone():
    kept = drop_superseded_hardware({"p150": ("a",), "galaxy": ("a",)})
    assert kept == ["galaxy", "p150"]


def test_hardware_from_hub_manifest_reads_the_fetched_file(tmp_path, monkeypatch):
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps({"container": {"serve": {"hardware": "P150"}}}))

    def fake_download(repo_id, filename):
        assert (repo_id, filename) == ("ns/untagged", MANIFEST_NAME)
        return str(manifest_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert hardware_from_hub_manifest("ns/untagged") == ["p150"]


def test_hardware_from_hub_manifest_is_empty_on_any_failure(monkeypatch):
    def boom(repo_id, filename):
        raise OSError("404")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    assert hardware_from_hub_manifest("ns/gone") == []


class _Repo:
    def __init__(self, id, tags):
        self.id = id
        self.tags = tags
        self.downloads = 0


def test_search_community_fetches_the_manifest_for_an_untagged_bundle(monkeypatch):
    """No board tag and never pulled here: the manifest on the Hub is the only
    source left, so it is fetched — but only for this one bundle, not the
    tagged one beside it."""
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_models",
        lambda self, **kw: iter(
            [_Repo("ns/untagged", ["blackhole"]), _Repo("ns/tagged", ["blackhole", "p300"])]
        ),
    )
    calls = []

    def fake_hardware_from_hub_manifest(repo_id):
        calls.append(repo_id)
        return ["p150"]

    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles.hardware_from_hub_manifest",
        fake_hardware_from_hub_manifest,
    )
    found = {b.name: b.hardware for b in search_community()}
    assert found == {"ns/untagged": ["p150"], "ns/tagged": ["p300"]}
    assert calls == ["ns/untagged"]  # never called for the already-tagged bundle


def test_search_community_skips_the_hub_manifest_for_an_installed_bundle(
    tmp_path, monkeypatch
):
    """A bundle pulled here already has a local manifest — hardware_for reads
    that, so the untagged fallback must not also hit the Hub."""
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = tmp_path / "tt-model"
    pulled = root / "pulled" / "ns__untagged"
    pulled.mkdir(parents=True)
    (pulled / MANIFEST_NAME).write_text(
        json.dumps({"container": {"serve": {"hardware": "p300x2"}}})
    )
    (root / "installed.json").write_text(json.dumps({"ns/untagged": {"repo_id": "ns/untagged"}}))

    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_models",
        lambda self, **kw: iter([_Repo("ns/untagged", ["blackhole"])]),
    )

    def boom(repo_id):  # pragma: no cover - must never run
        raise AssertionError("the Hub manifest fallback ran for an installed bundle")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.hardware_from_hub_manifest", boom)
    (found,) = bundles.search_community()
    assert found.hardware == ["p300x2"]
