# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""What TT-Studio can deploy, as a ModelCatalog source.

TT-Studio builds its catalog from tt's own model_support.json, at the same
tt-inference-server release tt pins, so every model in the support list is also
a studio model and needs no copy of studio's catalog here. The only models studio
adds are the few it ships its own images for (STUDIO_OWN_MODELS); keep that list
in step with the entries marked `hand_owned` in tt-studio's
app/backend/shared_config/models_from_inference_server.json, skipping any the
support list already carries, when the tt-studio pin moves.

Entries come out with backends=["studio"] and no tt_model_id (that field is
tt-inference-server's id). ModelCatalog lets the support list win by name but
keeps studio in `backends`, so a support-list model is offered through either
path (`tt serve X --studio` deploys it with studio, which runs
tt-inference-server's own images) and studio's own models come through with
studio alone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..models.model import DeviceSupport, ModelInfo

BACKEND = "studio"

# Studio's own board→device widening (app/backend/docker_control/views.py): a
# multi-card box also runs the single-chip models of its card family, one chip at
# a time. Mirrored here so `tt model list` on a p300x2 shows what studio will
# deploy — a plain `p150` entry would otherwise hide Qwen3.5-9B on this box.
_SINGLE_CHIP_WIDENING = {
    "p300x2": ("p150", "p300"),
    "p300cx4": ("p150", "p300"),
    "p150x4": ("p150",),
    "p150x8": ("p150",),
}
_SINGLE_CHIP_NOTE = "single-chip model; studio runs it on one chip of the board"


@dataclass(frozen=True)
class StudioModel:
    name: str
    hf_repo: str
    boards: tuple[str, ...]  # studio's device_configurations, lowercased
    status: str
    param_count: int | None
    model_type: str = "llm"
    engine: str = "vLLM"


# Models studio deploys from its own images, absent from the tt-inference-server
# spec. From tt-studio v2.12.0's catalog (its `hand_owned` entries, minus
# Llama-3.1-8B-Instruct and Qwen3.8-27B, which the spec carries).
STUDIO_OWN_MODELS: tuple[StudioModel, ...] = (
    StudioModel("Qwen3.5-9B", "Qwen/Qwen3.5-9B", ("p150",), "EXPERIMENTAL", 9),
)


def studio_model_info(model: StudioModel) -> ModelInfo:
    engines = [model.engine]
    devices = {
        board: DeviceSupport(engines=list(engines), status=model.status)
        for board in model.boards
    }
    for board, singles in _SINGLE_CHIP_WIDENING.items():
        if board not in devices and any(s in devices for s in singles):
            devices[board] = DeviceSupport(
                engines=list(engines),
                status=model.status,
                support_source="single-chip",
                note=_SINGLE_CHIP_NOTE,
            )
    return ModelInfo(
        name=model.name,
        hf_repo=model.hf_repo,
        model_type=model.model_type,
        engines=list(engines),
        hardware=list(devices),
        devices=devices,
        param_count=model.param_count,
        backends=[BACKEND],
    )


class StudioModelsSource:
    """Studio's catalog: the support list it builds from, plus its own models.

    `support` is the support-list source the catalog also reads. Only the names
    matter here: ModelCatalog keeps that source's entry and merges in studio as a
    second serving path.
    """

    origin = "tt-studio"

    def __init__(self, support, own: tuple[StudioModel, ...] = STUDIO_OWN_MODELS) -> None:
        self._support = support
        self._own = own

    def entries(self) -> list[ModelInfo]:
        known = {e.name for e in self._support.entries()}
        return [dataclasses.replace(e, backends=[BACKEND]) for e in self._support.entries()] + [
            studio_model_info(m) for m in self._own if m.name not in known
        ]


def studio_only(model: ModelInfo) -> bool:
    return model.backends == [BACKEND]
