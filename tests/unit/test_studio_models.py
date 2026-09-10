# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The studio catalog source and its merge with the support list."""

import json
from pathlib import Path

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.modelhub import hub
from tenstorrent.modelhub.catalog import ModelCatalog
from tenstorrent.modelhub.studio import StudioModelsSource, parse_studio_models

DATA = Path(__file__).parent.parent / "fakes" / "data"
SMALL_STUDIO = DATA / "studio_models_small.json"
SMALL_SUPPORT = DATA / "model_support_small.json"


@pytest.fixture
def small_catalogs(monkeypatch):
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))
    monkeypatch.setenv("TT_STUDIO_MODELS_PATH", str(SMALL_STUDIO))


def _by_name(entries):
    return {m.name: m for m in entries}


def test_studio_entries_take_studios_shape():
    models = _by_name(parse_studio_models(SMALL_STUDIO.read_text(), "fixture"))
    qwen = models["Qwen3.5-9B"]
    assert qwen.backends == ["studio"]
    assert qwen.tt_model_id is None  # that field is tt-inference-server's id
    assert qwen.hf_repo == "Qwen/Qwen3.5-9B"
    assert qwen.model_type == "llm"
    assert qwen.engines == ["vLLM"]
    assert qwen.param_count == 9
    assert qwen.devices["p150"].status == "EXPERIMENTAL"
    whisper = models["whisper-large-v3"]
    assert whisper.model_type == "audio" and whisper.engines == ["media"]
    # device names are lowercased to the support list's vocabulary
    assert set(_by_name(models.values())["Llama-3.1-8B-Instruct"].hardware) >= {
        "n150", "p150", "p300x2", "t3k"
    }


def test_single_chip_models_are_widened_onto_the_multi_card_boards():
    """Studio's backend runs a P150 model on one chip of a P300x2/P300Cx4/P150X4/
    P150X8 box; the catalog file alone would hide Qwen3.5-9B on this QuietBox."""
    models = _by_name(parse_studio_models(SMALL_STUDIO.read_text(), "fixture"))
    qwen = models["Qwen3.5-9B"]
    assert set(qwen.hardware) == {"p150", "p300x2", "p300cx4", "p150x4", "p150x8"}
    assert qwen.devices["p150"].support_source == "spec"
    widened = qwen.devices["p300x2"]
    assert widened.support_source == "single-chip"
    assert widened.supported and "one chip" in widened.note
    # a model with its own P300x2 entry keeps it, and is not widened further
    big = models["Qwen3.8-27B"]
    assert big.hardware == ["p300x2"]
    assert big.devices["p300x2"].support_source == "spec"


def test_entries_without_boards_are_skipped():
    text = json.dumps(
        {"models": [{"model_name": "x", "device_configurations": []}, {"model_name": "y"}]}
    )
    assert parse_studio_models(text, "t") == []


def test_malformed_studio_list_is_a_config_error(monkeypatch, tmp_path):
    bad = tmp_path / "studio.json"
    bad.write_text("{not json")
    monkeypatch.setenv("TT_STUDIO_MODELS_PATH", str(bad))
    with pytest.raises(TTError) as err:
        StudioModelsSource()
    assert err.value.exit_code == ExitCode.CONFIG


def test_missing_override_path_is_a_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_STUDIO_MODELS_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(TTError) as err:
        StudioModelsSource()
    assert err.value.exit_code == ExitCode.CONFIG


def test_the_support_list_wins_and_studio_only_adds_what_it_lacks(small_catalogs):
    """A model tt-inference-server serves is never offered through studio, so a
    shared name keeps the support list's entry wholesale — backend included."""
    catalog = ModelCatalog()
    assert catalog.origin.endswith("model_support_small.json")
    models = _by_name(catalog.list(cached_sizes={}))
    llama = models["Llama-3.1-8B-Instruct"]
    assert llama.backends == ["inference-server"]
    assert llama.tt_model_id == "Llama-3.1-8B-Instruct"
    assert llama.devices["p300x2"].tool_call_parser == "llama3_json"
    assert "t3k" not in llama.hardware  # studio-only boards do not leak in
    assert models["Qwen3.5-9B"].backends == ["studio"]
    assert models["Qwen3-32B"].backends == ["inference-server"]
    # whisper is in both fixtures: support-list marks (broken on p300x2) survive
    assert models["whisper-large-v3"].backends == ["inference-server"]
    assert models["whisper-large-v3"].devices["p300x2"].supported is False


def test_find_resolves_a_studio_only_model_by_name_or_repo(small_catalogs):
    catalog = ModelCatalog()
    assert catalog.find("qwen3.5-9b", cached_sizes={}).name == "Qwen3.5-9B"
    assert catalog.find("Qwen/Qwen3.5-9B", cached_sizes={}).name == "Qwen3.5-9B"


def test_the_bundled_studio_catalog_parses_and_adds_only_what_the_spec_lacks(monkeypatch):
    monkeypatch.delenv("TT_STUDIO_MODELS_PATH", raising=False)
    monkeypatch.delenv("TT_MODEL_SUPPORT_PATH", raising=False)
    source = StudioModelsSource()
    assert source.origin == "bundled studio_models.json"
    assert len(source.entries()) == 60
    studio_only = [
        m.name for m in ModelCatalog().list(cached_sizes={}) if m.backends == ["studio"]
    ]
    # the reason this source exists — re-check when either bundle is refreshed
    assert {"Qwen3.5-9B", "Qwen3.8-27B"} <= set(studio_only)


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
