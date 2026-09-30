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

import pytest

from tenstorrent.modelhub import bundles
from tenstorrent.modelhub.bundles import (
    MANIFEST_NAME,
    _classify,
    _hardware_chips,
    drop_superseded_hardware,
    hardware_for,
    hardware_from_hub_manifest,
    hardware_satisfies,
    is_hardware_tag,
    search_community,
)


def test_is_hardware_tag_accepts_known_boards_with_or_without_a_count():
    for tag in ("p150", "p150x4", "p300x2", "n300", "n300x4", "e150"):
        assert is_hardware_tag(tag), tag


def test_is_hardware_tag_rejects_arch_and_unrelated_tags():
    for tag in ("blackhole", "wormhole_b0", "vllm", "region:us", "q200x4", "p250", "p150x0"):
        assert not is_hardware_tag(tag), tag


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


def test_hardware_satisfies_rejects_a_different_single_card_of_equal_chip_count():
    # p100 and p150 are both one blackhole chip, but different products —
    # neither substitutes for the other just because the chip count matches.
    assert not hardware_satisfies("p150", "p100")
    assert not hardware_satisfies("p100", "p150")


def test_hardware_satisfies_falls_back_to_an_exact_match_for_unknown_tags():
    assert hardware_satisfies("galaxy", "galaxy")
    assert not hardware_satisfies("galaxy", "p150")
    assert not hardware_satisfies("p150", "galaxy")


def test_hardware_satisfies_resolves_t3k_to_its_n300x4_equivalent():
    # t3k is run.py's catalog id for 4 n300 boards; a bundle tagged the board
    # form directly must still match a detected/explicit --hw t3k, and vice versa.
    assert hardware_satisfies("n300x4", "t3k")
    assert hardware_satisfies("t3k", "n300x4")
    assert hardware_satisfies("n300", "t3k")  # fewer chips than the target
    assert not hardware_satisfies("p150", "t3k")  # different arch


def test_hardware_tag_rejects_a_zero_mesh_multiplier():
    assert _hardware_chips("p150x0") is None
    assert not hardware_satisfies("p150", "p150x0")


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


def test_search_community_fetches_the_manifest_for_an_untagged_bundle(
    curated_catalog, monkeypatch
):
    """No board tag and never pulled here: the manifest on the Hub is the only
    source left, so it is fetched — but only for this one bundle, not the
    tagged one beside it."""
    curated_catalog(
        {"repo": "ns/untagged", "arch": "blackhole"},
        {"repo": "ns/tagged", "arch": "blackhole", "hardware": ["p300"]},
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
    tmp_path, curated_catalog, monkeypatch
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

    curated_catalog({"repo": "ns/untagged", "arch": "blackhole"})

    def boom(repo_id):  # pragma: no cover - must never run
        raise AssertionError("the Hub manifest fallback ran for an installed bundle")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.hardware_from_hub_manifest", boom)
    (found,) = bundles.search_community()
    assert found.hardware == ["p300x2"]


# -- curated community catalog ------------------------------------------------------
def test_the_bundled_community_catalog_loads():
    from tenstorrent.modelhub import bundles

    bundles._load_curated()  # a malformed shipped file would raise here


def test_search_community_lists_only_curated_bundles_filtered_by_query(curated_catalog):
    curated_catalog("ns/Alpha-7B", "ns/beta", "other/alpha-2")
    assert [b.name for b in search_community()] == ["ns/Alpha-7B", "ns/beta", "other/alpha-2"]
    assert [b.name for b in search_community(query="ALPHA")] == ["ns/Alpha-7B", "other/alpha-2"]
    assert [b.name for b in search_community(limit=1)] == ["ns/Alpha-7B"]


def test_curated_ids_are_lowercased(curated_catalog):
    from tenstorrent.modelhub.bundles import curated_ids

    curated_catalog("NS/Mixed")
    assert curated_ids() == {"ns/mixed"}


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": 2, "bundles": []}',
        '{"schema_version": 1, "bundles": [{"kind": "container"}]}',
        '{"schema_version": 1, "bundles": [',
    ],
    ids=["schema", "entry", "json"],
)
def test_a_malformed_community_catalog_is_a_config_error(tmp_path, monkeypatch, text):
    from tenstorrent.errors import ExitCode, TTError
    from tenstorrent.modelhub.bundles import curated_ids

    path = tmp_path / "catalog.json"
    path.write_text(text)
    monkeypatch.setenv("TT_COMMUNITY_CATALOG_PATH", str(path))
    with pytest.raises(TTError) as err:
        curated_ids()
    assert err.value.exit_code == ExitCode.CONFIG


def test_a_missing_community_catalog_override_is_a_config_error(tmp_path, monkeypatch):
    from tenstorrent.errors import ExitCode, TTError
    from tenstorrent.modelhub.bundles import curated_ids

    monkeypatch.setenv("TT_COMMUNITY_CATALOG_PATH", str(tmp_path / "absent.json"))
    with pytest.raises(TTError) as err:
        curated_ids()
    assert err.value.exit_code == ExitCode.CONFIG


# -- unverified bundles (the Hub's community catalog) -------------------------------
class _HubRepo:
    def __init__(self, id, tags, downloads=0):
        self.id, self.tags, self.downloads = id, tags, downloads


def test_search_unverified_lists_hub_bundles_outside_the_curated_catalog(
    curated_catalog, monkeypatch
):
    curated_catalog("ns/Verified")
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_models",
        lambda self, **kw: iter([
            _HubRepo("NS/verified", ["p150"]),
            _HubRepo("ns/other", ["tt-model-container", "vllm-plugin", "blackhole", "p300x2"], 7),
        ]),
    )
    (found,) = bundles.search_unverified()
    assert (found.name, found.kind, found.engine, found.arch, found.hardware) == (
        "ns/other", "container", "vllm-plugin", ["blackhole"], ["p300x2"]
    )
    assert (found.downloads, found.verified) == (7, False)


def test_search_community_rows_are_verified(curated_catalog):
    curated_catalog({"repo": "ns/v", "hardware": ["p150"]})
    assert [b.verified for b in search_community()] == [True]


def test_local_bundles_are_verified_by_the_curated_catalog(
    tmp_path, curated_catalog, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-model").mkdir()
    (tmp_path / "tt-model" / "installed.json").write_text(
        json.dumps({"ns/v": {"repo_id": "ns/v"}, "ns/u": {"repo_id": "ns/u"}})
    )
    curated_catalog("NS/V")
    assert {b.name: b.verified for b in bundles.local_bundles()} == {"ns/v": True, "ns/u": False}


def test_search_unverified_is_a_tt_error_when_the_hub_is_unreachable(
    curated_catalog, monkeypatch
):
    from tenstorrent.errors import TTError

    curated_catalog()

    def boom(self, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.HfApi.list_models", boom)
    with pytest.raises(TTError, match="Could not reach the Hugging Face Hub"):
        bundles.search_unverified()


def test_search_unverified_limit_counts_only_unverified_bundles(curated_catalog, monkeypatch):
    """Verified rows are filtered out after the Hub applies its limit, so they must
    not use up slots the unverified rows need."""
    curated_catalog("ns/v1", "ns/v2")
    hub = [_HubRepo(f"ns/{name}", ["p150"]) for name in ("v1", "v2", "u1", "u2", "u3")]
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_models", lambda self, limit, **kw: iter(hub[:limit])
    )
    assert [b.name for b in bundles.search_unverified(limit=2)] == ["ns/u1", "ns/u2"]


# -- Tenstorrent copies (tt-model-manager `verify`) -------------------------------------
# The real function: conftest stubs the module attribute so the suite stays offline.
from tenstorrent.modelhub.bundles import search_verified_copies as _real_copies  # noqa: E402


class _Card(dict):
    """ModelCardData answers .get, as the Hub's listing returns it."""


def test_a_repo_in_the_tenstorrent_org_is_verified_by_its_id():
    assert bundles.is_verified("Tenstorrent/Qwen3-32B", set())
    assert bundles.is_verified("tenstorrent/qwen3-32b", set())
    assert bundles.is_verified("ns/curated", {"ns/curated"})
    assert not bundles.is_verified("ns/other", {"ns/curated"})
    assert not bundles.is_verified("Tenstorrent-fan/x", set())


def test_search_verified_copies_asks_the_hub_for_the_orgs_listed_repos(monkeypatch):
    seen = {}

    def list_models(self, **kw):
        seen.update(kw)
        repo = _HubRepo("Tenstorrent/Qwen3-32B", ["tt-model-container", "vllm-plugin", "p150x4"])
        repo.card_data = _Card(tt_verified_source="someone/qwen3-32b-p150x4")
        return iter([repo])

    monkeypatch.setattr("huggingface_hub.HfApi.list_models", list_models)
    (copy,) = _real_copies()
    assert (seen["author"], seen["filter"], seen["cardData"]) == (
        bundles.VERIFIED_ORG, bundles.CATALOG_TAG, True
    )
    assert (copy.name, copy.verified, copy.copy_of, copy.hardware) == (
        "Tenstorrent/Qwen3-32B", True, "someone/qwen3-32b-p150x4", ["p150x4"]
    )


def test_copy_of_is_ignored_outside_the_org_and_when_malformed():
    """Anyone can write the key into their own card."""
    card = _Card(tt_verified_source="someone/original")
    assert bundles._copy_of("ns/fake-copy", card) is None
    assert bundles._copy_of("Tenstorrent/x", card) == "someone/original"
    assert bundles._copy_of("Tenstorrent/x", _Card(tt_verified_source="no-slash")) is None
    assert bundles._copy_of("Tenstorrent/x", _Card(tt_verified_source=3)) is None
    assert bundles._copy_of("Tenstorrent/x", None) is None


def test_search_unverified_leaves_out_tenstorrent_copies(curated_catalog, monkeypatch):
    curated_catalog()
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_models",
        lambda self, **kw: iter([_HubRepo("Tenstorrent/x", ["p150"]), _HubRepo("ns/u", ["p150"])]),
    )
    assert [b.name for b in bundles.search_unverified()] == ["ns/u"]


def test_search_verified_copies_is_a_tt_error_when_the_hub_is_unreachable(monkeypatch):
    from tenstorrent.errors import TTError

    def boom(self, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.HfApi.list_models", boom)
    with pytest.raises(TTError, match="Could not reach the Hugging Face Hub"):
        _real_copies()


def test_an_installed_tenstorrent_copy_is_verified(tmp_path, curated_catalog, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-model").mkdir()
    (tmp_path / "tt-model" / "installed.json").write_text(
        json.dumps({"tenstorrent/x": {"repo_id": "Tenstorrent/x"}})
    )
    curated_catalog()
    assert {b.name: b.verified for b in bundles.local_bundles()} == {"Tenstorrent/x": True}


def test_describe_finds_a_tenstorrent_copy(curated_catalog, monkeypatch):
    curated_catalog()
    asked = []

    def copies(**kw):
        asked.append(kw["query"])
        return [bundles.BundleInfo(name="Tenstorrent/X", verified=True, copy_of="ns/x")]

    monkeypatch.setattr(bundles, "search_verified_copies", copies)
    found = bundles.describe("tenstorrent/x")
    assert (found.name, found.copy_of, asked) == ("Tenstorrent/X", "ns/x", ["x"])
