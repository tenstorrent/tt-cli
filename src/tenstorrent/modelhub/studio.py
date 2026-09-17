# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""TT-Studio's model catalog as a ModelCatalog source.

studio_models.json is a verbatim copy of tt-studio's
app/backend/shared_config/models_from_inference_server.json at the pinned
branch (see tools/supplement.toml). Studio has no `list` command and no --json,
and reading the file out of an installed checkout would force a clone just to
run `tt model list`, so the file ships with tt. `TT_STUDIO_MODELS_PATH`
overrides it. Re-copy the file whenever the tt-studio pin moves.

Entries come out with backends=["studio"] and no tt_model_id (that field is
tt-inference-server's id). ModelCatalog lets the support list win by name, so a
model tt-inference-server serves is offered through it alone and studio only
ever adds the models tt-inference-server cannot serve.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from ..errors import ExitCode, TTError
from ..models.model import DeviceSupport, ModelInfo

STUDIO_PATH_ENV = "TT_STUDIO_MODELS_PATH"
BACKEND = "studio"

# Studio's own board→device widening (app/backend/docker_control/views.py): a
# multi-card box also runs the single-chip models of its card family, one chip at
# a time. Mirrored here so `tt model list` on a p300x2 shows what studio will
# deploy — a plain `P150` entry would otherwise hide Qwen3.5-9B on this box.
_SINGLE_CHIP_WIDENING = {
    "p300x2": ("p150", "p300"),
    "p300cx4": ("p150", "p300"),
    "p150x4": ("p150",),
    "p150x8": ("p150",),
}
_SINGLE_CHIP_NOTE = "single-chip model; studio runs it on one chip of the board"


def _load_text() -> tuple[str, str]:
    override = os.environ.get(STUDIO_PATH_ENV)
    if override:
        path = Path(override)
        if not path.exists():
            raise TTError(
                f"{STUDIO_PATH_ENV} points at {path}, which does not exist.",
                next_step=f"Fix or unset {STUDIO_PATH_ENV} to use the bundled list.",
                exit_code=ExitCode.CONFIG,
            )
        return path.read_text(), str(path)
    return (
        (resources.files("tenstorrent.modelhub") / "studio_models.json").read_text(),
        "bundled studio_models.json",
    )


def parse_studio_models(text: str, origin: str) -> list[ModelInfo]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TTError(
            f"Studio model list at {origin} is not valid JSON.",
            why=str(exc),
            next_step=f"Reinstall tt, or fix {STUDIO_PATH_ENV}.",
            exit_code=ExitCode.CONFIG,
        ) from exc
    entries = []
    for raw in (doc.get("models") if isinstance(doc, dict) else None) or []:
        name = raw.get("model_name")
        boards = [str(b).lower() for b in raw.get("device_configurations") or []]
        if not name or not boards:
            continue
        engines = [str(raw["inference_engine"])] if raw.get("inference_engine") else []
        status = str(raw.get("status", ""))
        devices = {
            board: DeviceSupport(engines=list(engines), status=status) for board in boards
        }
        for board, singles in _SINGLE_CHIP_WIDENING.items():
            if board not in devices and any(s in devices for s in singles):
                devices[board] = DeviceSupport(
                    engines=list(engines),
                    status=status,
                    support_source="single-chip",
                    note=_SINGLE_CHIP_NOTE,
                )
        entries.append(
            ModelInfo(
                name=str(name),
                hf_repo=str(raw.get("hf_model_id") or ""),
                # display_model_type already uses tt's type vocabulary (LLM, VLM,
                # IMAGE, CNN, EMBEDDING, AUDIO, TEXT_TO_SPEECH, VIDEO), just upper.
                model_type=str(raw.get("display_model_type") or raw.get("model_type") or "").lower(),
                engines=list(engines),
                hardware=list(devices),
                devices=devices,
                param_count=raw.get("param_count"),
                backends=[BACKEND],
            )
        )
    return entries


class StudioModelsSource:
    """The bundled copy of studio's catalog (or TT_STUDIO_MODELS_PATH)."""

    def __init__(self) -> None:
        text, self.origin = _load_text()
        self._entries = parse_studio_models(text, self.origin)

    def entries(self) -> list[ModelInfo]:
        return self._entries


def studio_only(model: ModelInfo) -> bool:
    return model.backends == [BACKEND]

