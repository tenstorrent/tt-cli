# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The studio catalog source and its merge with the support list."""

from pathlib import Path

import pytest

from tenstorrent.modelhub import hub
from tenstorrent.modelhub.catalog import ModelCatalog, ModelSupportSource
from tenstorrent.modelhub.studio import (
    STUDIO_OWN_MODELS,
    StudioModel,
    StudioModelsSource,
    studio_model_info,
)

DATA = Path(__file__).parent.parent / "fakes" / "data"
SMALL_SUPPORT = DATA / "model_support_small.json"


@pytest.fixture
def small_support(monkeypatch):
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))


def _by_name(entries):
    return {m.name: m for m in entries}


def _own(name):
    return next(m for m in STUDIO_OWN_MODELS if m.name == name)


def test_studio_own_models_take_studios_shape():
    qwen = studio_model_info(_own("Qwen3.5-9B"))
    assert qwen.backends == ["studio"]
    assert qwen.tt_model_id is None  # that field is tt-inference-server's id
    assert qwen.hf_repo == "Qwen/Qwen3.5-9B"
    assert qwen.model_type == "llm"
    assert qwen.engines == ["vLLM"]
    assert qwen.param_count == 9
    assert qwen.devices["p150"].status == "EXPERIMENTAL"


def test_single_chip_models_are_widened_onto_the_multi_card_boards():
    """Studio's backend runs a P150 model on one chip of a P300x2/P300Cx4/P150X4/
    P150X8 box; listing only the p150 board would hide Qwen3.5-9B on a QuietBox."""
    qwen = studio_model_info(_own("Qwen3.5-9B"))
    assert set(qwen.hardware) == {"p150", "p300x2", "p300cx4", "p150x4", "p150x8"}
    assert qwen.devices["p150"].support_source == "spec"
    widened = qwen.devices["p300x2"]
    assert widened.support_source == "single-chip"
    assert widened.supported and "one chip" in widened.note
    # a model with its own P300x2 entry keeps it, and is not widened further
    big = studio_model_info(StudioModel("big", "org/big", ("p300x2",), "EXPERIMENTAL", 27))
    assert big.hardware == ["p300x2"]
    assert big.devices["p300x2"].support_source == "spec"


def test_studio_serves_the_whole_support_list_and_its_own_models(small_support):
    """Studio builds its catalog from tt's model_support.json at the same
    tt-inference-server release, so every support-list model is a studio model."""
    support = ModelSupportSource()
    names = [m.name for m in StudioModelsSource(support).entries()]
    assert set(names) == {m.name for m in support.entries()} | {
        m.name for m in STUDIO_OWN_MODELS
    }


def test_an_own_model_the_support_list_gains_is_not_listed_twice(small_support):
    support = ModelSupportSource()
    clash = StudioModel("Qwen3-32B", "Qwen/Qwen3-32B", ("p150",), "EXPERIMENTAL", 32)
    entries = StudioModelsSource(support, own=(clash,)).entries()
    assert [m.name for m in entries].count("Qwen3-32B") == 1


def test_the_support_list_wins_and_studio_is_kept_as_a_second_path(small_support):
    """A shared name keeps the support list's entry wholesale — its device marks
    and serve overrides — but studio stays in `backends`: studio deploys
    tt-inference-server's models too, and `tt serve X --studio` has to be able
    to ask for that. inference-server is listed first, as the default."""
    catalog = ModelCatalog()
    assert catalog.origin.endswith("model_support_small.json")
    models = _by_name(catalog.list(cached_sizes={}))
    llama = models["Llama-3.1-8B-Instruct"]
    assert llama.backends == ["inference-server", "studio"]
    assert llama.tt_model_id == "Llama-3.1-8B-Instruct"
    assert llama.devices["p300x2"].tool_call_parser == "llama3_json"
    assert models["Qwen3-32B"].backends == ["inference-server", "studio"]
    assert models["Qwen3.5-9B"].backends == ["studio"]
    # support-list marks (whisper broken on p300x2) survive the merge
    assert models["whisper-large-v3"].backends == ["inference-server", "studio"]
    assert models["whisper-large-v3"].devices["p300x2"].supported is False


def test_merge_backends_orders_by_preference_and_dedupes():
    from tenstorrent.modelhub.catalog import merge_backends

    assert merge_backends(["studio"], ["inference-server"]) == ["inference-server", "studio"]
    assert merge_backends(["inference-server"], ["inference-server"]) == ["inference-server"]
    assert merge_backends(["studio"], ["other"]) == ["studio", "other"]


def test_find_resolves_a_studio_only_model_by_name_or_repo(small_support):
    catalog = ModelCatalog()
    assert catalog.find("qwen3.5-9b", cached_sizes={}).name == "Qwen3.5-9B"
    assert catalog.find("Qwen/Qwen3.5-9B", cached_sizes={}).name == "Qwen3.5-9B"


def test_studio_own_models_are_absent_from_the_bundled_support_list(monkeypatch):
    """The reason STUDIO_OWN_MODELS exists — once the spec carries one, drop it."""
    monkeypatch.delenv("TT_MODEL_SUPPORT_PATH", raising=False)
    bundled = {m.name for m in ModelSupportSource().entries()}
    assert not bundled & {m.name for m in STUDIO_OWN_MODELS}


# -- HF token seeding -----------------------------------------------------------------


class _Config:
    def get(self, key):
        return ""


def test_hf_token_prefers_the_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "token").write_text("from-file\n")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hub.hf_token(_Config()) == ("from-env", "env")


def test_hf_token_falls_back_to_the_login_store(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "token").write_text("hf_abc\n")
    assert hub.hf_token(_Config()) == ("hf_abc", "hf-login")


def test_hf_token_honors_hf_token_path(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    custom = tmp_path / "elsewhere"
    custom.write_text("hf_custom")
    monkeypatch.setenv("HF_TOKEN_PATH", str(custom))
    assert hub.hf_token(_Config()) == ("hf_custom", "hf-login")


def test_hf_token_is_none_without_any_source(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hub.hf_token(_Config()) is None
    assert hub.hf_token(None) is None
