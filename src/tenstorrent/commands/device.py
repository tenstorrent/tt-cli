# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt device` — hardware inventory, health, reset, and the tt-smi TUI."""

from __future__ import annotations

import dataclasses
import json
import sys

import typer
from rich.table import Table

from ..backends.device import get_device_backend
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..models.device import SystemSnapshot

device_app = typer.Typer(
    help="Hardware inventory, health checks, and resets.", no_args_is_help=True
)


def _no_devices_error() -> TTError:
    return TTError(
        "No Tenstorrent devices detected.",
        why="The driver may not be installed, or no card is seated.",
        next_step="Run `tt update` to install the driver stack, then `tt device status` again.",
        exit_code=ExitCode.NO_DEVICES,
    )


def _snapshot_payload(snap: SystemSnapshot) -> dict:
    return dataclasses.asdict(snap)


def _fmt(value: object, suffix: str = "") -> str:
    return f"{value}{suffix}" if value is not None else "—"


def _status_table(payload: dict) -> Table:
    driver = payload["host"].get("Driver", "unknown")
    table = Table(title=f"Tenstorrent devices (driver: {driver})")
    for column in ("#", "Board", "Bus ID", "Temp", "Power", "AIClk", "Voltage", "DRAM"):
        table.add_column(column)
    for dev in payload["devices"]:
        table.add_row(
            str(dev["index"]),
            _fmt(dev["board_type"]),
            _fmt(dev["bus_id"]),
            _fmt(dev["temperature_c"], " °C"),
            _fmt(dev["power_w"], " W"),
            _fmt(dev["aiclk_mhz"], " MHz"),
            _fmt(dev["voltage_v"], " V"),
            _fmt(dev["dram_status"]),
        )
    return table


@device_app.command("status")
@handle_tt_errors
def status(
    ctx: typer.Context,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
    raw: bool = typer.Option(
        False, "--raw", help="Dump tt-smi's own snapshot JSON (no schema promise)."
    ),
) -> None:
    """Show detected devices: board, temperature, power, clock, DRAM state."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    backend = get_device_backend(appctx)
    if raw:
        print(json.dumps(backend.raw_snapshot(), indent=2))
        return
    snap = backend.snapshot()
    for warning in snap.warnings:
        appctx.output.warn(f"snapshot: {warning}")
    if not snap.devices:
        raise _no_devices_error()
    appctx.output.emit(_snapshot_payload(snap), renderer=_status_table)


def _info_tables(payload: dict) -> Table:
    table = Table(title="Device details")
    table.add_column("field")
    for dev in payload["devices"]:
        table.add_column(f"device {dev['index']}")
    rows = [
        ("board_type", "board_type"),
        ("board_id", "board_id"),
        ("bus_id", "bus_id"),
        ("coords", "coords"),
        ("pcie_speed", "pcie_speed"),
        ("pcie_width", "pcie_width"),
        ("dram_status", "dram_status"),
    ]
    for label, key in rows:
        table.add_row(label, *(_fmt(dev[key]) for dev in payload["devices"]))
    fw_keys = sorted({k for dev in payload["devices"] for k in dev["firmware"]})
    for key in fw_keys:
        table.add_row(
            f"fw.{key}", *(_fmt(dev["firmware"].get(key)) for dev in payload["devices"])
        )
    return table


@device_app.command("info")
@handle_tt_errors
def info(
    ctx: typer.Context,
    index: list[int] = typer.Argument(None, help="Device index(es); default all."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Show device metadata: PCI IDs, board revision, serial, firmware versions."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    snap = get_device_backend(appctx).snapshot()
    for warning in snap.warnings:
        appctx.output.warn(f"snapshot: {warning}")
    if not snap.devices:
        raise _no_devices_error()
    devices = snap.devices
    if index:
        known = {d.index for d in devices}
        bad = [i for i in index if i not in known]
        if bad:
            raise TTError(
                f"No such device index: {', '.join(map(str, bad))}.",
                why=f"Detected devices are 0–{max(known)}.",
                next_step="Run `tt device status` to list devices.",
                exit_code=ExitCode.USAGE,
            )
        devices = [d for d in devices if d.index in index]
    payload = {"host": snap.host, "devices": [dataclasses.asdict(d) for d in devices]}
    appctx.output.emit(payload, renderer=_info_tables)


@device_app.command("reset")
@handle_tt_errors
def reset(
    ctx: typer.Context,
    index: list[int] = typer.Argument(None, help="Device index(es); default all."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Reset one or more devices (interrupts anything running on them)."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    target = ", ".join(map(str, index)) if index else "ALL devices"
    if not yes:
        if appctx.output.json_mode or not sys.stdin.isatty():
            raise TTError(
                "Device reset needs confirmation.",
                why="This interrupts any workload running on the device.",
                next_step="Re-run with --yes to confirm.",
                exit_code=ExitCode.USAGE,
            )
        if not typer.confirm(f"Reset {target}? Running workloads will be interrupted."):
            raise typer.Exit(int(ExitCode.OK))
    appctx.output.status(f"Resetting {target} …")
    backend = get_device_backend(appctx)
    result = backend.reset(index or None, allow_prompt=not appctx.output.json_mode)
    appctx.output.emit(
        dataclasses.asdict(result),
        renderer=lambda d: f"Reset complete ({target}).",
    )


@device_app.command("top")
@handle_tt_errors
def top(ctx: typer.Context) -> None:
    """Open the interactive tt-smi TUI (hands over this terminal)."""
    appctx = get_app_context(ctx)
    backend = get_device_backend(appctx)
    appctx.runner.exec_tty(backend.top_argv())
