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


# -- Galaxy chassis -------------------------------------------------------------------
def _galaxy(count=32, board_type="tt-galaxy-wh L", board_id="0100035100000000"):
    """bifroest's real shape: 32 asics, one board_type, and one board_id shared by
    every asic — the whole chassis reports a single board."""
    return [
        DeviceSnapshot(index=i, board_type=board_type, board_id=board_id)
        for i in range(count)
    ]


def test_galaxy_chassis_is_galaxy():
    # Would be read as one board by the unique-board_id rule, so the chassis rule
    # has to count asics instead.
    assert infer_device_config(_galaxy()) == "galaxy"


def test_blackhole_galaxy_chassis_is_blackhole_galaxy():
    devs = _galaxy(board_type="tt-galaxy-bh L")
    assert infer_device_config(devs) == "blackhole_galaxy"


@pytest.mark.parametrize("count", [1, 4, 16, 31, 33])
def test_partial_or_oversized_chassis_is_unmapped(count):
    # Consistent with 3x p300: an unmapped count is None, so serve warns and asks
    # for --device instead of passing a device run.py would fail on.
    assert infer_device_config(_galaxy(count=count)) is None


def test_galaxy_sharing_the_box_with_a_card_is_none():
    devs = _galaxy() + [DeviceSnapshot(index=32, board_type="n150 L", board_id="a")]
    assert infer_device_config(devs) is None


# -- reading the runtime's inspect output ----------------------------------------------
# podman ignores `inspect --format '{{json .}}'` and pretty-prints one multi-line JSON
# array. Inner lines that are complete JSON values on their own — a trailing element of
# an Env array, a bare integer — parse as scalars, so a line-by-line reader that trusts
# json.loads hands its caller a str and every later `.get` is an AttributeError. This is
# the real shape from podman 3.4.4 (`docker` on this box is its shim).

_PODMAN_INSPECT = """[
    {
        "Id": "abc123def456",
        "Name": "tt-inference-server-6af6ab4d",
        "Config": {
            "Env": [
                "ARCH_NAME=wormhole_b0",
                "TZ=America/Los_Angeles"
            ]
        },
        "Mounts": [
            {
                "Name": "",
                "Source": "/home/u/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/abc123456789"
            }
        ]
    }
]
"""


def test_podman_pretty_output_yields_only_records():
    """No scalars in, no AttributeError out: `tt model stop` walks these records."""
    from tenstorrent.backends.serving.inference_server import parse_inspect_records

    records = parse_inspect_records(_PODMAN_INSPECT)
    assert [r["Id"] for r in records] == ["abc123def456"]


def test_podman_pretty_output_still_identifies_a_container():
    from tenstorrent.backends.serving.inference_server import (
        _identity_from_inspect,
        parse_inspect_records,
    )

    (record,) = parse_inspect_records(_PODMAN_INSPECT)
    assert _identity_from_inspect(record) == ("openai/gpt-oss-120b", None)


def test_one_object_per_line_is_still_read():
    from tenstorrent.backends.serving.inference_server import parse_inspect_records

    stdout = '{"Id": "a1"}\nnot json\n{"Id": "b2"}\n'
    assert [r["Id"] for r in parse_inspect_records(stdout)] == ["a1", "b2"]


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
