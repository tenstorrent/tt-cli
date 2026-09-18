# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""ModelCatalog / model_support.json loader and parser.

Reads tests/fakes/data/model_support_small.json — a hand-maintained 4-model
stand-in for the shipped file, so exact-list assertions do not move every time
the real one is regenerated. Keep it in step with the schema in
scripts/build_model_support.py: a mark, a device fallback and a serve override
are all represented there on purpose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.models.model import ModelInfo
from tenstorrent.modelhub.catalog import (
    ModelCatalog,
    ModelSupportSource,
    parse_model_support,
)



SMALL_SUPPORT = (
    Path(__file__).parent.parent / "fakes" / "data" / "model_support_small.json"
)


@pytest.fixture(autouse=True)
def small_spec(monkeypatch):
    """Pin the catalog to the 4-model fixture so exact assertions stay stable when
    the shipped model_support.json is regenerated."""
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))


@pytest.fixture(autouse=True)
def no_hf_scan(monkeypatch):
    monkeypatch.setattr("tenstorrent.modelhub.catalog.scan_hf_cache", lambda: {})


# -- loader ---------------------------------------------------------------------------
def test_support_path_override_is_honored(monkeypatch, tmp_path):
    path = tmp_path / "support.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_version": "0.0.0",
                "models": [
                    {
                        "name": "only-model",
                        "hf_repo": "org/only-model",
                        "devices": {"n150": {"engines": ["vLLM"], "status": "COMPLETE"}},
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(path))
    source = ModelSupportSource()
    assert source.origin == str(path)
    assert [e.name for e in source.entries()] == ["only-model"]


def test_the_bundled_list_is_what_is_read_by_default(monkeypatch):
    monkeypatch.delenv("TT_MODEL_SUPPORT_PATH", raising=False)
    source = ModelSupportSource()
    assert source.origin == "bundled model_support.json"
    assert len(source.entries()) > 50


def test_missing_override_path_is_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(TTError) as err:
        ModelSupportSource()
    assert err.value.exit_code == ExitCode.CONFIG
    assert "TT_MODEL_SUPPORT_PATH" in err.value.what


def test_malformed_json_is_config_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(bad))
    with pytest.raises(TTError) as err:
        ModelSupportSource()
    assert err.value.exit_code == ExitCode.CONFIG


def test_wrong_schema_version_is_config_error(monkeypatch, tmp_path):
    support = tmp_path / "future.json"
    support.write_text(json.dumps({"schema_version": 99, "models": []}))
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(support))
    with pytest.raises(TTError) as err:
        ModelSupportSource()
    assert err.value.exit_code == ExitCode.CONFIG
    assert "schema_version" in err.value.what


def test_an_unsupported_device_without_a_reason_is_a_config_error():
    """Marks must carry their reason: `tt serve` refuses on them and has to be
    able to say why, and a bare False would hide a model with no explanation."""
    doc = {
        "schema_version": 1,
        "release_version": "0.0.0",
        "models": [
            {
                "name": "x",
                "hf_repo": "o/x",
                "devices": {"n150": {"engines": ["vLLM"], "supported": False}},
            }
        ],
    }
    with pytest.raises(TTError) as err:
        parse_model_support(json.dumps(doc), "test list")
    assert err.value.exit_code == ExitCode.CONFIG
    assert "without reason, details, verified_on" in err.value.what


# -- parsing (small fixture) ----------------------------------------------------------
def entries_by_name() -> dict[str, ModelInfo]:
    return {e.name: e for e in ModelSupportSource().entries()}


def test_vllm_model_is_servable_with_canonical_hardware_order():
    llama = entries_by_name()["Llama-3.1-8B-Instruct"]
    assert llama.tt_model_id == "Llama-3.1-8B-Instruct"
    assert llama.hf_repo == "meta-llama/Llama-3.1-8B-Instruct"
    assert llama.model_type == "llm"
    assert llama.engines == ["vLLM"]
    # canonical order, lowercase; p150x4 is borrowed from the p300x2 spec
    assert llama.hardware == ["n150", "p150x4", "p300", "p300x2"]
    assert llama.param_count == 8
    assert llama.min_disk_gb == 36
    assert llama.min_ram_gb == 20.0


def test_per_device_status_and_max_context():
    devices = entries_by_name()["Llama-3.1-8B-Instruct"].devices
    assert devices["n150"].status == "COMPLETE"
    assert devices["n150"].max_context == 65536
    assert devices["p300"].status == "FUNCTIONAL"
    assert devices["p300"].max_context == 131072


def test_media_and_forge_models_are_servable():
    """Every spec entry carries the name run.py --model takes, whatever the
    engine — the vLLM-only restriction was tt's, not the server's."""
    models = entries_by_name()
    assert models["whisper-large-v3"].tt_model_id == "whisper-large-v3"
    assert models["whisper-large-v3"].engines == ["media"]
    assert models["whisper-large-v3"].model_type == "audio"
    assert models["resnet-50"].tt_model_id == "resnet-50"
    assert models["resnet-50"].engines == ["forge"]
    assert models["resnet-50"].hf_repo == "resnet-50"  # bare, non-HF label


def test_default_impl_wins_over_sort_order():
    # Qwen3-32B on GALAXY has two impls; the default_impl one (tt_transformers,
    # COMPLETE) sorts after the other, so this pins that the marker is honored.
    qwen = entries_by_name()["Qwen3-32B"]
    assert qwen.devices["galaxy"].status == "COMPLETE"
    assert qwen.devices["galaxy"].max_context == 131072


# -- ModelCatalog ---------------------------------------------------------------------
def test_get_is_case_insensitive_on_name_and_hf_repo():
    catalog = ModelCatalog()
    assert catalog.get("llama-3.1-8b-instruct").name == "Llama-3.1-8B-Instruct"
    assert catalog.get("META-LLAMA/Llama-3.1-8B-Instruct").name == "Llama-3.1-8B-Instruct"


def test_get_unknown_model_is_usage_error():
    with pytest.raises(TTError) as err:
        ModelCatalog().get("gpt-17")
    assert err.value.exit_code == ExitCode.USAGE


def test_cache_merge_keys_on_hf_repo():
    models = {
        m.name: m
        for m in ModelCatalog().list(cached_sizes={"openai/whisper-large-v3": 123})
    }
    assert models["whisper-large-v3"].cached is True
    assert models["whisper-large-v3"].cache_size_bytes == 123
    assert models["Llama-3.1-8B-Instruct"].cached is False


class _StubSource:
    def __init__(self, origin: str, entries: list[ModelInfo]) -> None:
        self.origin = origin
        self._entries = entries

    def entries(self) -> list[ModelInfo]:
        return self._entries


def test_later_sources_override_earlier_by_name():
    # The source-merge seam: an appended source's entry replaces the released
    # spec's entry of the same name.
    base = _StubSource("base", [ModelInfo(name="m", hf_repo="a/m", model_type="llm")])
    override = _StubSource(
        "local", [ModelInfo(name="m", hf_repo="local/m", model_type="llm")]
    )
    catalog = ModelCatalog(sources=[base, override])
    assert catalog.origin == "base + local"
    assert catalog.get("m", cached_sizes={}).hf_repo == "local/m"


# -- community bundle parsing (tt-model repo tags) ---------------------------------
def test_bundle_tags_are_classified_into_kind_engine_and_arch():
    from tenstorrent.modelhub.bundles import _classify

    kind, engine, arch = _classify(
        ["blackhole", "tt-model-cache", "tt-model-catalog", "tt-model-container",
         "vllm-plugin", "region:us", "1x4"]
    )
    assert (kind, engine) == ("container", "vllm-plugin")  # upstream's own name
    # tt-model's own tags, region:, and the 1x4 mesh shape are all dropped —
    # arch is the architecture family, see the test below
    assert arch == ["blackhole"]


def test_arch_is_the_architecture_family_and_nothing_else():
    """Repo tags are free text, so "everything left over is arch" reported
    `tenstorrent`, `tt-model` and `vllm-fork` as hardware — all three are live in
    the catalog today. Board and mesh tags are dropped too: `p300x2` is one
    publisher's wording for a configuration, while the family is what says
    whether a bundle can run on your machine at all. A tag tt does not recognise
    is left out rather than guessed at, so a new family belongs in _ARCH_TAGS."""
    from tenstorrent.modelhub.bundles import _classify

    assert _classify(["blackhole", "p300x2", "tenstorrent", "tt-model"])[2] == ["blackhole"]
    assert _classify(["wormhole_b0", "1x4", "vllm-fork"])[2] == ["wormhole_b0"]
    assert _classify(["quasar", "tt-model-cache"])[2] == []


def test_installed_bundle_ids_reads_tt_models_index(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-model").mkdir()
    (tmp_path / "tt-model" / "installed.json").write_text('{"NS/Alpha": {}}')
    assert bundles.installed_bundle_ids() == {"ns/alpha"}


def test_installed_bundle_ids_prefers_the_legacy_tt_kernel_dir(tmp_path, monkeypatch):
    """tt-model keeps using an already-populated pre-rename dir; so must we."""
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-kernel").mkdir()
    (tmp_path / "tt-kernel" / "installed.json").write_text('{"ns/legacy": {}}')
    assert bundles.installed_bundle_ids() == {"ns/legacy"}


def test_installed_bundle_ids_is_empty_when_tt_model_never_ran(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert bundles.installed_bundle_ids() == set()


def test_weights_repo_comes_from_the_pulled_manifest(tmp_path, monkeypatch):
    """A bundle references weights rather than shipping them; the pulled manifest is
    the only local record of which repo."""
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pulled = tmp_path / "tt-model" / "pulled" / "ns__alpha"
    pulled.mkdir(parents=True)
    (pulled / "tt_kernel_manifest.json").write_text(
        json.dumps({"schema_version": "5.1", "weights": {"repo_id": "org/weights"}})
    )
    # no "manifest" key in the index entry: fall back to the conventional path
    assert bundles.weights_repo_for("ns/alpha", {}) == "org/weights"


def test_weights_repo_honors_the_recorded_manifest_path(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    elsewhere = tmp_path / "somewhere" / "tt_kernel_manifest.json"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(json.dumps({"weights": {"repo": "org/aliased"}}))
    # "repo" is the JSON alias of the repo_id field upstream
    assert bundles.weights_repo_for("ns/alpha", {"manifest": str(elsewhere)}) == "org/aliased"


def test_weights_repo_is_none_without_a_manifest(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert bundles.weights_repo_for("ns/alpha", {}) is None


def test_engine_comes_from_the_manifests_container_kind(tmp_path, monkeypatch):
    """Bundles are not all vLLM: an LLM package is vllm-plugin, a diffusion one
    tt-dit-server. The engine is read, never assumed."""
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pulled = tmp_path / "tt-model" / "pulled" / "ns__dit"
    pulled.mkdir(parents=True)
    (pulled / "tt_kernel_manifest.json").write_text(
        json.dumps({"schema_version": "5.1", "container": {"kind": "tt-dit-server"}})
    )
    assert bundles.engine_for("ns/dit", {}) == "tt-dit-server"


def test_engine_is_none_for_an_unpulled_bundle(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert bundles.engine_for("ns/absent", {}) is None


def test_local_bundle_engine_is_read_from_disk(tmp_path, monkeypatch):
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = tmp_path / "tt-model"
    pulled = root / "pulled" / "ns__dit"
    pulled.mkdir(parents=True)
    (pulled / "tt_kernel_manifest.json").write_text(
        json.dumps({"container": {"kind": "tt-dit-server"}, "weights": {"repo_id": "org/w"}})
    )
    (root / "installed.json").write_text(
        json.dumps({"ns/dit": {"repo_id": "ns/dit", "container": True, "arch": "blackhole"}})
    )
    (found,) = bundles.local_bundles()
    assert (found.name, found.engine, found.kind) == ("ns/dit", "tt-dit-server", "container")
    assert found.weights_repo == "org/w"


def test_engine_is_read_from_the_repo_tag_without_a_manifest():
    """`tt-model package` writes container.kind into the model card as a repo tag,
    so a bundle nobody has pulled still reports its engine."""
    from tenstorrent.modelhub.bundles import _classify

    _, engine, arch = _classify(["blackhole", "tt-model-container", "tt-dit-server"])
    assert engine == "tt-dit-server"
    assert arch == ["blackhole"]  # not misfiled as an arch


def test_an_unknown_future_engine_kind_is_still_recognized():
    """A kind upstream adds later must not land in the arch column."""
    from tenstorrent.modelhub.bundles import _classify

    for tag in ("sglang-plugin", "tt-quasar-server"):
        _, engine, arch = _classify(["blackhole", tag])
        assert (engine, arch) == (tag, ["blackhole"]), tag


# -- the whitelist: a namespace, not an index -----------------------------------------
# `tt-model whitelist` copies a reviewed bundle into the Tenstorrent org, so "reviewed"
# is answerable from the repo id alone — no extra request, works offline, works for an
# installed bundle. The copy's card names what it was made from, which is how the
# listing knows to show the copy instead of both.


@pytest.mark.parametrize(("repo_id", "expected"), [
    ("Tenstorrent/Qwen3-32B", True),
    ("tenstorrent/qwen3-32b", True),      # tt-model's index lowercases its keys
    ("TENSTORRENT/Qwen3-32B", True),
    ("someauthor/qwen3-32b", False),
    ("tenstorrent-labs/qwen3-32b", False),  # a prefix test would get this wrong
    ("no-namespace", False),
    ("", False),
])
def test_whitelisted_is_decided_by_the_namespace(repo_id, expected):
    from tenstorrent.modelhub.bundles import is_whitelisted

    assert is_whitelisted(repo_id) is expected


def test_a_bundle_info_derives_its_review_state_from_its_name():
    """Derived, not passed: the table can never disagree with --json, and there is no
    'cannot tell' state to render."""
    from tenstorrent.modelhub.bundles import BundleInfo

    assert BundleInfo(name="Tenstorrent/Foo").whitelisted is True
    assert BundleInfo(name="ns/foo").whitelisted is False
    assert BundleInfo(name="tenstorrent/foo", source="local").whitelisted is True


def test_the_json_field_order_is_the_contract():
    """`whitelisted` and `whitelist_source` are appended last, in that order — the
    module says field order IS the --json contract, so pin it rather than trust it."""
    import dataclasses

    from tenstorrent.modelhub.bundles import BundleInfo

    keys = list(dataclasses.asdict(BundleInfo(name="ns/x")).keys())
    assert keys[-2:] == ["whitelisted", "whitelist_source"]
    assert keys[0] == "name"


class _Card(dict):
    """Stands in for ModelCardData, which is dict-like via .get()."""


def _repo(repo_id, *, tags=("blackhole",), card=None, downloads=0):
    return type("R", (), {"id": repo_id, "tags": list(tags),
                          "downloads": downloads, "card_data": card})()


def _search(monkeypatch, repos):
    from tenstorrent.modelhub import bundles

    seen = {}

    def fake(self, **kw):
        seen.update(kw)
        seen["calls"] = seen.get("calls", 0) + 1
        return iter(repos)

    monkeypatch.setattr("huggingface_hub.HfApi.list_models", fake)
    return bundles.search_community(), seen


def test_the_listing_asks_for_card_data_on_its_single_request(monkeypatch, tmp_path):
    """cardData rides the same request. One round trip for the whole listing is the
    property this module protects — a per-repo fetch would be O(catalog)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, seen = _search(monkeypatch, [_repo("ns/a"), _repo("ns/b"), _repo("ns/c")])
    assert seen["cardData"] is True
    assert seen["calls"] == 1
    assert len(rows) == 3


def test_a_tenstorrent_copy_reports_the_bundle_it_was_made_from(monkeypatch, tmp_path):
    from tenstorrent.modelhub.bundles import REVIEW_SOURCE_KEY

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, _ = _search(monkeypatch, [
        _repo("Tenstorrent/Qwen3-32B", card=_Card({REVIEW_SOURCE_KEY: "ns/qwen-v51"})),
    ])
    assert rows[0].whitelist_source == "ns/qwen-v51"
    assert rows[0].whitelisted is True


def test_a_community_repo_cannot_claim_to_supersede_anything(monkeypatch, tmp_path):
    """A card is author-written. Without this, anyone could put a popular model's id in
    their own frontmatter and hide it from every listing — the mirror of a forged review,
    and the reason only a namespace nobody else can write may make this claim."""
    from tenstorrent.modelhub.bundles import REVIEW_SOURCE_KEY

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, _ = _search(monkeypatch, [
        _repo("randomguy/evil", card=_Card({REVIEW_SOURCE_KEY: "microsoft/phi-4"})),
    ])
    assert rows[0].whitelist_source is None


@pytest.mark.parametrize("value", [
    None, 123, ["ns/a"], {"repo": "ns/a"},          # frontmatter is arbitrary YAML
    "not-a-repo-id", "too/many/parts", "/leading", "trailing/", "",
])
def test_an_unusable_source_claim_is_ignored_rather_than_guessed_at(
    monkeypatch, tmp_path, value
):
    from tenstorrent.modelhub.bundles import REVIEW_SOURCE_KEY

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, _ = _search(monkeypatch, [
        _repo("Tenstorrent/X", card=_Card({REVIEW_SOURCE_KEY: value})),
    ])
    assert rows[0].whitelist_source is None


@pytest.mark.parametrize("card", [None, _Card({}), "not-card-like"])
def test_a_repo_with_no_usable_card_is_still_listed(monkeypatch, tmp_path, card):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, _ = _search(monkeypatch, [_repo("Tenstorrent/X", card=card)])
    assert rows[0].whitelist_source is None
    assert rows[0].whitelisted is True          # the namespace still decides


def test_a_padded_source_claim_is_trimmed(monkeypatch, tmp_path):
    from tenstorrent.modelhub.bundles import REVIEW_SOURCE_KEY

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows, _ = _search(monkeypatch, [
        _repo("Tenstorrent/X", card=_Card({REVIEW_SOURCE_KEY: "  ns/a\n"})),
    ])
    assert rows[0].whitelist_source == "ns/a"


# -- the collapse ---------------------------------------------------------------------


def _rows(*specs):
    from tenstorrent.modelhub.bundles import BundleInfo

    return [BundleInfo(name=n, source=s, whitelist_source=src) for n, s, src in specs]


def _names(rows):
    return [(r.name, r.source) for r in rows]


def test_a_reviewed_copy_replaces_the_bundle_it_was_made_from():
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", "ns/foo"), ("ns/foo", "HF", None))
    assert _names(collapse_whitelisted(rows)) == [("Tenstorrent/Foo", "HF")]


def test_a_claim_naming_a_bundle_outside_the_listing_hides_nothing():
    """Listing-relative by design: resolving the other half would need the per-repo
    fetch this listing avoids."""
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", "ns/delisted"), ("ns/other", "HF", None))
    assert len(collapse_whitelisted(rows)) == 2


def test_a_bundle_on_this_machine_is_never_hidden():
    """A local row is a fact about this machine — `tt serve` works on it right now, so
    hiding it would be a lie. The listing shows what you have next to what we recommend."""
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", "ns/foo"), ("ns/foo", "HF", None),
                 ("ns/foo", "local", None))
    assert _names(collapse_whitelisted(rows)) == [
        ("Tenstorrent/Foo", "HF"), ("ns/foo", "local")
    ]


def test_both_copies_installed_keeps_both_local_rows():
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", "ns/foo"), ("Tenstorrent/Foo", "local", None),
                 ("ns/foo", "HF", None), ("ns/foo", "local", None))
    assert _names(collapse_whitelisted(rows)) == [
        ("Tenstorrent/Foo", "HF"), ("Tenstorrent/Foo", "local"), ("ns/foo", "local")
    ]


def test_a_copy_that_records_no_source_hides_nothing():
    """A hand-made repo in the org is reviewed by the namespace rule, but it must not
    silently hide someone's bundle on the strength of no evidence."""
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", None), ("ns/foo", "HF", None))
    assert len(collapse_whitelisted(rows)) == 2


def test_a_copy_never_collapses_itself_or_another_copy():
    """A self-referential claim — from a re-whitelist or a writer bug — would otherwise
    make a reviewed model vanish, the worst failure this feature could have."""
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    itself = _rows(("Tenstorrent/Foo", "HF", "Tenstorrent/Foo"))
    assert len(collapse_whitelisted(itself)) == 1

    chain = _rows(("Tenstorrent/B", "HF", "Tenstorrent/A"), ("Tenstorrent/A", "HF", "ns/a"))
    assert len(collapse_whitelisted(chain)) == 2


def test_a_forged_claim_from_a_community_repo_hides_nothing():
    """Belt and braces with the parse-time guard: a caller building rows by hand cannot
    reintroduce the hiding attack."""
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("randomguy/evil", "HF", "microsoft/phi-4"), ("microsoft/phi-4", "HF", None))
    assert len(collapse_whitelisted(rows)) == 2


def test_two_copies_naming_one_bundle_collapse_it_once():
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/A", "HF", "ns/foo"), ("Tenstorrent/B", "HF", "ns/foo"),
                 ("ns/foo", "HF", None))
    assert _names(collapse_whitelisted(rows)) == [("Tenstorrent/A", "HF"), ("Tenstorrent/B", "HF")]


def test_the_claim_is_matched_case_insensitively():
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    rows = _rows(("Tenstorrent/Foo", "HF", "SomeAuthor/Foo"), ("someauthor/foo", "HF", None))
    assert _names(collapse_whitelisted(rows)) == [("Tenstorrent/Foo", "HF")]


def test_the_collapse_is_idempotent_and_handles_an_empty_listing():
    from tenstorrent.modelhub.bundles import collapse_whitelisted

    assert collapse_whitelisted([]) == []
    rows = _rows(("Tenstorrent/Foo", "HF", "ns/foo"), ("ns/foo", "HF", None))
    once = collapse_whitelisted(rows)
    assert _names(collapse_whitelisted(once)) == _names(once)


def test_a_local_install_is_whitelisted_by_its_namespace_with_no_network(tmp_path, monkeypatch):
    """The index lowercases its keys and an older entry has no `repo_id`, so the row
    arrives as `tenstorrent/foo` — which must still read as reviewed."""
    from tenstorrent.modelhub import bundles

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-model").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tt-model" / "installed.json").write_text(
        json.dumps({"Tenstorrent/Foo": {}, "ns/bar": {}})
    )
    rows = {b.name: b.whitelisted for b in bundles.local_bundles()}
    assert rows == {"tenstorrent/foo": True, "ns/bar": False}
