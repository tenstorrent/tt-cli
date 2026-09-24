# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import pytest

from tenstorrent.backends.serving import chips
from tenstorrent.backends.serving.inference_server import choose_device
from tenstorrent.models.model import DeviceSupport, ModelInfo

IDS = [0, 1, 2, 3]


def _container(*, ipc="host", devices=(), mounts=(), privileged=False, labels=None):
    return {
        "Config": {"Labels": labels or {}},
        "HostConfig": {
            "IpcMode": ipc,
            "Privileged": privileged,
            "Devices": [{"PathOnHost": d} for d in devices],
        },
        "Mounts": [{"Source": m} for m in mounts],
    }


@pytest.mark.parametrize(
    "info, held",
    [
        (_container(devices=["/dev/tenstorrent"]), {0, 1, 2, 3}),
        (_container(devices=["/dev/tenstorrent/2"]), {2}),
        (_container(mounts=["/dev/", "/tmp"]), {0, 1, 2, 3}),
        (_container(privileged=True), {0, 1, 2, 3}),
        (_container(ipc="private", devices=["/dev/tenstorrent"]), set()),
        (_container(ipc="private", labels={"org.tenstorrent.tt-model.devices": "1,3"}), {1, 3}),
        (_container(mounts=["/dev/tenstorrent-foo", ""]), set()),
    ],
)
def test_a_container_claims_what_it_was_granted(info, held):
    assert chips._claims(info, IDS) == held


def _model(model_type, *devices, mesh=None):
    return ModelInfo(
        name="m", hf_repo="org/m", model_type=model_type,
        devices={
            d: DeviceSupport(engines=["vLLM"], status="COMPLETE", mesh_graph_desc=mesh)
            for d in devices
        },
    )


@pytest.mark.parametrize(
    "model, board, chosen",
    [
        (_model("audio", "p150", "p300x2", mesh="p150.textproto"), "p300x2", "p150"),
        (_model("llm", "p150", "p300x2"), "p300x2", "p300x2"),
        (_model("llm", "p150", "p150x4"), "p150x4", "p150"),
        (_model("image", "p300x2"), "p300x2", "p300x2"),
        (_model("llm", "n300", "t3k"), "t3k", "t3k"),
        (_model("cnn", "n300", "t3k"), "t3k", "n300"),
        (_model("audio", "n150", "n300", "t3k"), "t3k", "n150"),
        (_model("llm", "p150"), "p150", "p150"),
    ],
)
def test_the_device_follows_tt_studio(model, board, chosen):
    assert choose_device(model, board) == chosen
