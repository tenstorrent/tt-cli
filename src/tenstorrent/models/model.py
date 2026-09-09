# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Typed model-hub models — the stable contract behind `tt model` and its --json.

Derived 1:1 from tt-inference-server's release_model_spec.json (see
modelhub/catalog.py). One ModelInfo per model; per-device support lives in
`devices`. Field order IS the --json contract."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Unsupported:
    """Why a model does not work on a device, from model_support_overrides.toml."""

    reason: str  # "broken" (a bug) | "unsupported" (tt cannot drive it)
    details: str
    verified_on: str  # ISO date the failure was last seen
    source: str = ""  # who recorded it, e.g. "tt-studio"


@dataclass(frozen=True)
class DeviceSupport:
    """Support detail for one device config (the keys of ModelInfo.devices)."""

    engines: list[str]  # subset of ("vLLM", "media", "forge"), spec casing
    status: str  # EXPERIMENTAL | FUNCTIONAL | COMPLETE
    max_context: int | None = None  # device_model_spec.max_context, when published
    supported: bool = True
    unsupported: Unsupported | None = None  # set iff supported is False
    docker_image: str | None = None  # the spec's own image, before any override
    # Halves of the server's volume directory name, volume_id_<impl_id>-<name>-v<version>.
    impl_id: str | None = None
    version: str | None = None
    # vLLM launch settings the server publishes but does not apply itself; tt
    # passes them as --vllm-override-args.
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    # The spec's own tt-config. Informational: the server already applies it.
    override_tt_config: dict | None = None
    # Flags tt must pass because the spec's value is wrong or missing —
    # docker_image and/or override_tt_config.
    serve_overrides: dict | None = None
    # Device name to send as `run.py --device` instead of this one, when the
    # model is reached through a device_fallback rather than its own spec.
    serve_as: str | None = None
    # How this entry was reached: "spec" is the model's own, "mesh-equivalent"
    # borrows the other four-chip Blackhole mesh, "single-chip" runs it on one
    # chip of a larger board.
    support_source: str = "spec"
    note: str = ""  # why this entry differs from the model's own spec


@dataclass(frozen=True)
class ModelInfo:
    name: str  # spec model_name — exactly what run.py --model takes
    hf_repo: str  # spec hf_model_repo (a bare label for forge-only models)
    model_type: str  # lowercase spec model_type: llm, vlm, cnn, embedding, …
    engines: list[str] = field(default_factory=list)  # union across devices
    tt_model_id: str | None = None
    hardware: list[str] = field(default_factory=list)  # lowercase device_type keys
    devices: dict[str, DeviceSupport] = field(default_factory=dict)  # keys == hardware
    param_count: int | None = None  # billions of parameters
    min_disk_gb: int | None = None
    min_ram_gb: float | None = None
    cached: bool = False
    cache_size_bytes: int | None = None
