# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Delegated device backend: shells out to tt-smi and tolerantly parses its
snapshot JSON into typed models.

tt-smi's snapshot has no schema contract, so parsing never KeyErrors: missing or
malformed fields become None plus a warning. `tt-smi -s -f <file>` is used
because some tt-smi versions write the snapshot to a file rather than stdout;
we read the file back and fall back to stdout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ..config.paths import Paths
from ..errors import ExitCode, TTError
from ..models.device import DeviceSnapshot, ResetResult, SystemSnapshot
from ..tools.registry import ToolRegistry
from ..tools.runner import Runner

TOOL = "tt-smi"


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _clean_float(value: Any, field: str, index: int, warnings: list[str]) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        warnings.append(f"device {index}: unparseable {field} {value!r}")
        return None


def parse_snapshot(data: dict) -> SystemSnapshot:
    """Tolerant parse: anything missing or odd degrades to None + a warning,
    never an exception."""
    warnings: list[str] = []
    if not isinstance(data, dict):
        return SystemSnapshot(warnings=["snapshot root is not an object"])
    host_raw = data.get("host_info")
    host = (
        {str(k): str(v) for k, v in host_raw.items()}
        if isinstance(host_raw, dict)
        else {}
    )
    if host_raw is not None and not isinstance(host_raw, dict):
        warnings.append("host_info is not an object")
    devices: list[DeviceSnapshot] = []
    raw_devices = data.get("device_info")
    if raw_devices is None:
        warnings.append("snapshot has no device_info section")
        raw_devices = []
    if not isinstance(raw_devices, list):
        warnings.append("device_info is not a list")
        raw_devices = []
    for i, dev in enumerate(raw_devices):
        if not isinstance(dev, dict):
            warnings.append(f"device {i}: entry is not an object")
            devices.append(DeviceSnapshot(index=i))
            continue
        board = dev.get("board_info") if isinstance(dev.get("board_info"), dict) else {}
        telem = dev.get("telemetry") if isinstance(dev.get("telemetry"), dict) else {}
        fw_raw = dev.get("firmwares") if isinstance(dev.get("firmwares"), dict) else {}
        for section, value in (("board_info", board), ("telemetry", telem)):
            if not value:
                warnings.append(f"device {i}: missing {section}")
        devices.append(
            DeviceSnapshot(
                index=i,
                board_type=_clean_str(board.get("board_type")),
                board_id=_clean_str(board.get("board_id")),
                bus_id=_clean_str(board.get("bus_id")),
                coords=_clean_str(board.get("coords")),
                dram_status=_clean_str(board.get("dram_status")),
                pcie_speed=_clean_str(board.get("pcie_speed")),
                pcie_width=_clean_str(board.get("pcie_width")),
                voltage_v=_clean_float(telem.get("voltage"), "voltage", i, warnings),
                current_a=_clean_float(telem.get("current"), "current", i, warnings),
                power_w=_clean_float(telem.get("power"), "power", i, warnings),
                aiclk_mhz=_clean_float(telem.get("aiclk"), "aiclk", i, warnings),
                temperature_c=_clean_float(
                    telem.get("asic_temperature"), "asic_temperature", i, warnings
                ),
                firmware={str(k): str(v) for k, v in fw_raw.items()},
            )
        )
    return SystemSnapshot(host=host, devices=devices, warnings=warnings)


class SmiDelegatedBackend:
    def __init__(self, registry: ToolRegistry, runner: Runner, paths: Paths) -> None:
        self.registry = registry
        self.runner = runner
        self.paths = paths

    def _smi_bin(self) -> str:
        return str(self.registry.resolve(TOOL))

    def raw_snapshot(self) -> dict:
        smi = self._smi_bin()
        with tempfile.TemporaryDirectory(prefix="tt-smi-snapshot-") as tmp:
            out_file = Path(tmp) / "snapshot.json"
            result = self.runner.capture([smi, "-s", "-f", str(out_file)], tool=TOOL)
            text = (
                out_file.read_text()
                if out_file.exists() and out_file.stat().st_size
                else result.stdout
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TTError(
                "tt-smi produced a snapshot this CLI could not parse.",
                why=f"invalid JSON: {exc}",
                next_step="Run `tt update` to align tool versions, or `tt-smi -s` directly to inspect.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": TOOL},
            ) from exc

    def snapshot(self) -> SystemSnapshot:
        return parse_snapshot(self.raw_snapshot())

    def reset(self, indices: Sequence[int] | None, *, allow_prompt: bool = True) -> ResetResult:
        smi = self._smi_bin()
        argv = [smi, "-r"]
        if indices:
            argv.append(",".join(str(i) for i in indices))
        self.runner.stream(argv, sudo=True, allow_prompt=allow_prompt, tool=TOOL)
        targets = list(indices) if indices else None
        return ResetResult(ok=True, devices=targets, message="reset complete")

    def top_argv(self) -> list[str]:
        return [self._smi_bin()]
