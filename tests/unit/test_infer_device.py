# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from tenstorrent.backends.serving.inference_server import infer_device_config
from tenstorrent.backends.smi import parse_snapshot
from tenstorrent.models.device import DeviceSnapshot

DATA = Path(__file__).parent.parent / "fakes" / "data"


def devices(name):
    return parse_snapshot(json.loads((DATA / f"snapshot_{name}.json").read_text())).devices


def test_real_quietbox_snapshot_is_p300x2():
    # 4 per-asic entries, 2 unique board_ids → 2x P300 boards
    assert infer_device_config(devices("multi")) == "p300x2"


def test_single_p300_board():
    assert infer_device_config(devices("normal")) == "p300"


def test_empty_and_unknown_are_none():
    assert infer_device_config(devices("empty")) is None
    assert infer_device_config([DeviceSnapshot(index=0, board_type="frobnicator")]) is None
    assert infer_device_config([DeviceSnapshot(index=0)]) is None  # no board_type


@pytest.mark.parametrize(
    "board_type, ids, expected",
    [
        ("n150 L", ["a"], "n150"),
        ("n150 L", ["a", "b", "c", "d"], "n150x4"),
        ("n300 R", ["a", "a"], "n300"),  # 2 asic entries sharing 1 board_id = 1 board
        ("n300 L", ["a", "a", "b", "b", "c", "c", "d", "d"], "t3k"),
        ("p150b", ["a", "b", "c", "d"], "p150x4"),
        ("p300c", ["a", "a", "b", "b", "c", "c"], None),  # 3x p300: unmapped count
    ],
)
def test_board_family_and_count_mapping(board_type, ids, expected):
    devs = [
        DeviceSnapshot(index=i, board_type=board_type, board_id=bid)
        for i, bid in enumerate(ids)
    ]
    assert infer_device_config(devs) == expected


def test_mixed_board_types_are_none():
    devs = [
        DeviceSnapshot(index=0, board_type="n150 L", board_id="a"),
        DeviceSnapshot(index=1, board_type="p300c", board_id="b"),
    ]
    assert infer_device_config(devs) is None


# -- identifying a running server container -------------------------------------------
def _identity(mounts):
    from tenstorrent.backends.serving.inference_server import _identity_from_inspect

    return _identity_from_inspect({"Mounts": mounts})


def test_a_named_volume_identifies_a_container():
    hf_repo, volume = _identity(
        [{"Name": "volume_id_tt_transformers-Qwen3-32B-v0.17.0", "Source": "/var/lib/docker/…"}]
    )
    assert volume == "volume_id_tt_transformers-Qwen3-32B-v0.17.0"


def test_a_bind_mounted_volume_identifies_a_container():
    """--host-volume gives a bind whose Name is empty and whose Source ends with
    the same volume_id_ directory, so reading Name alone loses the container —
    `tt model stop` then cannot find what it just started."""
    hf_repo, volume = _identity(
        [
            {
                "Name": "",
                "Source": "/data/tt-cache/volume_id_tt_transformers-Qwen3-32B-v0.17.0",
            }
        ]
    )
    assert volume == "volume_id_tt_transformers-Qwen3-32B-v0.17.0"


@pytest.mark.parametrize(
    "volume",
    [
        # docker named volume: versionless, so an image upgrade reuses it
        "volume_id_tt_transformers-Qwen3-32B",
        # --host-volume subdirectory: setup_host appends the spec version
        "volume_id_tt_transformers-Qwen3-32B-v0.17.0",
    ],
)
def test_a_volume_matches_the_model_in_either_naming_scheme(volume):
    """Both shapes are real and produced by the same server for the same model,
    depending on whether tt passed --host-volume."""
    from tenstorrent.backends.serving.inference_server import ServerContainer
    from tenstorrent.models.model import ModelInfo

    container = ServerContainer(
        id="abc", name="tt-inference-server-1", image="img", volume=volume
    )
    assert container.matches(
        ModelInfo(name="Qwen3-32B", hf_repo="Qwen/Qwen3-32B", model_type="llm")
    )
    assert not container.matches(
        ModelInfo(name="Qwen3-8B", hf_repo="Qwen/Qwen3-8B", model_type="llm")
    )


# -- MODEL_SOURCE for run.py -----------------------------------------------------------
def _model(name, hf_repo, model_type):
    from tenstorrent.models.model import ModelInfo

    return ModelInfo(name=name, hf_repo=hf_repo, model_type=model_type)


def _backend():
    from tenstorrent.backends.serving.inference_server import InferenceServerBackend

    return InferenceServerBackend(None, None, None, None)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_model("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "llm"), "huggingface"),
        # STT/TTS and forge containers fetch their own weights; `huggingface`
        # would download the repo to the host first and never read it.
        (_model("whisper-large-v3", "openai/whisper-large-v3", "audio"), "noaction"),
        (_model("speecht5_tts", "microsoft/speecht5_tts", "text_to_speech"), "noaction"),
        (_model("resnet-50", "resnet-50", "cnn"), "noaction"),
    ],
)
def test_model_source_matches_who_actually_fetches_the_weights(model, expected):
    assert _backend()._env(model)["MODEL_SOURCE"] == expected


def test_an_explicit_model_source_is_left_alone(monkeypatch):
    """Someone pointing the server at a local folder has said so deliberately."""
    monkeypatch.setenv("MODEL_SOURCE", "local")
    model = _model("whisper-large-v3", "openai/whisper-large-v3", "audio")
    assert _backend()._env(model)["MODEL_SOURCE"] == "local"
