# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt update` — converge system software and managed tools onto the golden set."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterable

import typer
from rich.table import Table

from ..backends.device import get_device_backend
from ..backends.installer import InstallerBackend, UpdatePlan
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import get_app_context
from ..errors import ExitCode, TTError
from ..selfupdate.update import offer_before_update


@dataclasses.dataclass(frozen=True, order=True)
class FirmwareSemVer:
    major: int
    minor: int
    patch: int
    build: int = 0

    @classmethod
    def parse(cls, value: str | None) -> "FirmwareSemVer | None":
        """Parse and normalize tt-smi/golden firmware version strings."""
        if value is None:
            return None
        raw = value.strip().removeprefix("v")
        parts = raw.split(".")
        if len(parts) not in (3, 4) or any(not part.isdigit() for part in parts):
            return None
        version = [int(part) for part in parts]
        if len(version) == 3:
            version.append(0)
        # Old bundles used 80.major.minor.patch; mirror tt-flash's normalization.
        if version[0] == 80:
            version = [version[1], version[2], version[3], 0]
        return cls(*version)


def is_any_fw_semver_higher(
    target: FirmwareSemVer | None,
    installed: Iterable[FirmwareSemVer | None],
) -> bool:
    """Return whether the target is higher than any known installed firmware."""
    return target is not None and any(
        version is not None and target > version for version in installed
    )


def _reset_notice(appctx, plan: UpdatePlan, *, offline: bool) -> str | None:
    """Explain why this update may reset devices, or None when it will not."""
    if offline:
        return None  # the system installer, including firmware, is skipped

    try:
        snapshot = get_device_backend(appctx).snapshot()
    except TTError:
        return (
            "Current device firmware could not be determined. Continuing may flash "
            "and reset TT devices, interrupting any running AI model workloads."
        )

    if not snapshot.devices:
        return None
    if plan.force:
        return (
            "This forced update will reflash firmware and reset TT devices, even if "
            "their firmware is already current. Any running AI model workloads will "
            "be interrupted."
        )

    target = FirmwareSemVer.parse(plan.firmware)
    current: list[FirmwareSemVer | None] = []
    for device in snapshot.devices:
        bundle = next(
            (
                value
                for key, value in device.firmware.items()
                if key.lower() == "fw_bundle_version"
            ),
            None,
        )
        current.append(FirmwareSemVer.parse(bundle))

    if is_any_fw_semver_higher(target, current):
        return (
            f"Firmware {plan.firmware} is newer than the firmware on one or more "
            "TT devices. Continuing will flash and reset those devices, interrupting "
            "any running AI model workloads."
        )
    if target is None or any(version is None for version in current):
        return (
            "Current and target firmware versions could not be compared for every "
            "TT device. Continuing may flash and reset devices, interrupting any "
            "running AI model workloads."
        )
    return None


def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _confirm_reset(appctx, notice: str, *, yes: bool) -> None:
    if not yes and (appctx.output.json_mode or not _stdin_is_interactive()):
        raise TTError(
            "Firmware update needs confirmation.",
            why=notice,
            next_step="Stop AI model workloads, then re-run with --yes to confirm.",
            exit_code=ExitCode.USAGE,
        )
    question = "Continue with tt update?"
    if appctx.output.quiet:
        # A destructive confirmation must retain its context even in quiet mode.
        question = f"{notice}\n{question}"
    elif not appctx.output.json_mode:
        appctx.output.warn(notice)
    if not yes and not typer.confirm(question):
        raise typer.Exit(int(ExitCode.OK))


def _plan_table(payload: dict) -> Table:
    table = Table(title=f"Update plan (goldens: {payload['manifest_origin']})")
    for column in ("component", "kind", "installed", "golden", "action"):
        table.add_column(column)
    for item in payload["items"]:
        table.add_row(
            item["name"],
            item["kind"],
            item["current"] or "—",
            item["target"],
            item["action"],
        )
    return table


def _result_summary(payload: dict) -> str:
    result = payload["result"]
    parts = [f"{len(result['updated'])} tool(s) updated"]
    if result["up_to_date"]:
        parts.append(f"{len(result['up_to_date'])} already current")
    if result["skipped"]:
        parts.append(f"{len(result['skipped'])} optional (--include-lazy)")
    system = "converged" if result["installer_ran"] else "skipped (offline)"
    parts.append(f"system stack {system}")
    summary = "Update complete: " + ", ".join(parts) + "."
    if result["failed"]:
        names = ", ".join(item["tool"] for item in result["failed"])
        summary = summary.replace("Update complete", "Update finished with errors")
        summary += f" Not updated: {names}."
    return summary


@handle_tt_errors
def update(
    ctx: typer.Context,
    version: str = typer.Argument(
        None,
        metavar="[INSTALLER_VERSION]",
        help="Run a specific tt-installer release (e.g. 3.1.0) instead of the "
        "bundled golden one — use this to downgrade the system stack. Implies "
        "--force (we assume you know what you're doing), and the script is "
        "fetched unpinned (no bundled sha256 check).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Let the installer move the system stack to the goldens even when "
        "that means a downgrade (passes --update-firmware=force).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm a firmware update that may reset TT devices.",
    ),
    include_lazy: bool = typer.Option(
        False,
        "--include-lazy",
        help="Also install optional tools that are not present yet — the ones tt "
        "fetches on first use (tt-inference-server).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change without changing it."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use only pre-seeded local caches; never download."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Get the latest stable, tested "golden" versions for this system."""
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    offline = offline or appctx.offline
    force = force or version is not None  # an explicit version means "I know what I'm doing"
    # A newer tt may carry newer tool pins, so it goes first: on a TTY, offer to
    # upgrade tt and stop (the new tt runs the update); otherwise just say so.
    if offer_before_update(appctx, offline=offline, dry_run=dry_run):
        return
    backend = InstallerBackend(
        appctx.registry, appctx.runner, appctx.paths, appctx.output
    )
    # Golden versions live in tt-sw-manifest's golden.json, fetched at the pinned
    # tag and cached; nothing version-shaped is bundled. A valid cache makes this
    # a no-op, so only the very first update (per tag) needs the network for it.
    backend.refresh_goldens(offline=offline)
    plan: UpdatePlan = backend.plan(
        version=version, force=force, include_lazy=include_lazy
    )
    plan_payload = dataclasses.asdict(plan)
    # Before either path prints: --dry-run is the best moment to learn that stale
    # copies of these tools are sitting in ~/.local/lib.
    backend.warn_unmanaged_app_clones()
    if dry_run:
        appctx.output.emit({"dry_run": True, **plan_payload}, renderer=_plan_table)
        return
    if not appctx.output.json_mode:  # in JSON mode everything lands in one document
        appctx.output.emit(plan_payload, renderer=_plan_table)
    reset_notice = _reset_notice(appctx, plan, offline=offline)
    if reset_notice is not None:
        _confirm_reset(appctx, reset_notice, yes=yes)
    result = backend.apply(plan, offline=offline, version=version, force=force)
    appctx.output.emit(
        {**plan_payload, "result": dataclasses.asdict(result)},
        renderer=_result_summary,
    )
    if result.failed:
        # Everything else converged, so the summary above is the report; exiting
        # via typer keeps `--json` to the single document it promises.
        raise typer.Exit(ExitCode.TOOL_FAILED)
