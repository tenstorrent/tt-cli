# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Typed device models. These are the stable internal contract between commands
and backends, and they define the public `--json` schema — swapping a delegated
backend for a native one must not change this shape."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceSnapshot:
    index: int
    board_type: str | None = None
    board_id: str | None = None
    bus_id: str | None = None
    coords: str | None = None
    dram_status: str | None = None
    pcie_speed: str | None = None
    pcie_width: str | None = None
    voltage_v: float | None = None
    current_a: float | None = None
    power_w: float | None = None
    aiclk_mhz: float | None = None
    temperature_c: float | None = None
    firmware: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSnapshot:
    host: dict[str, str] = field(default_factory=dict)
    devices: list[DeviceSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # tolerant-parse notes


@dataclass(frozen=True)
class ResetResult:
    ok: bool
    devices: list[int] | None  # None = all devices
    message: str | None = None
