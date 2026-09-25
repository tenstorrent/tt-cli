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

# A fixed roadmap: the count must never drift with flags, which is what makes
# `k/3` worth trusting. --offline skips System; it does not remove it.
PHASES = ["Checks", "Tools", "System"]


def _plan_summary(plan: UpdatePlan) -> str:
    """`3 to install, 1 up to date` — the one-line gist for the step's suffix."""
    actionable = sum(
        1 for item in plan.items if item.action in ("install", "upgrade", "converge")
    )
    current = sum(1 for item in plan.items if item.action == "up-to-date")
    parts = []
    if actionable:
        parts.append(f"{actionable} to change")
    if current:
        parts.append(f"{current} up to date")
    return ", ".join(parts) or "nothing to do"


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
    ui = appctx.output.ui
    # A newer tt may carry newer tool pins, so it goes first: on a TTY, offer to
    # upgrade tt and stop (the new tt runs the update); otherwise just say so.
    if offer_before_update(appctx, offline=offline, dry_run=dry_run):
        return
    backend = InstallerBackend(
        appctx.registry, appctx.runner, appctx.paths, appctx.output
    )

    # --dry-run reports and returns, so it never enters the phase flow: the phases
    # describe work being done, and a dry run does none.
    if dry_run:
        backend.refresh_goldens(offline=offline)
        plan = backend.plan(version=version, force=force, include_lazy=include_lazy)
        backend.warn_unmanaged_app_clones()
        appctx.output.emit(
            {"dry_run": True, **dataclasses.asdict(plan)}, renderer=_plan_table
        )
        return

    # A FIXED three-phase run. The count never varies with flags — --offline skips
    # the System phase rather than removing it, so `k/3` stays trustworthy.
    ui.register_phases(PHASES)

    with ui.phase("Checks"):
        # Golden versions live in tt-sw-manifest's golden.json, fetched at the
        # pinned tag and cached; nothing version-shaped is bundled. A valid cache
        # makes this a no-op, so only the very first update (per tag) needs the
        # network for it.
        backend.refresh_goldens(offline=offline)
        with ui.step("Comparing installed versions to the goldens") as step:
            plan: UpdatePlan = backend.plan(
                version=version, force=force, include_lazy=include_lazy
            )
            step.detail(_plan_summary(plan))
        # Stale copies of these tools sitting in ~/.local/lib are actionable, so
        # this is never folded.
        backend.warn_unmanaged_app_clones()

    plan_payload = dataclasses.asdict(plan)
    if not appctx.output.json_mode:  # in JSON mode everything lands in one document
        appctx.output.emit(plan_payload, renderer=_plan_table)

    # The prompt lives between phases, never inside one: a live row would repaint
    # over it and the CLI would look hung rather than merely wrong.
    reset_notice = _reset_notice(appctx, plan, offline=offline)
    if reset_notice is not None:
        with ui.prompting():
            _confirm_reset(appctx, reset_notice, yes=yes)

    with ui.phase("Tools") as phase:
        result = backend.apply_tools(plan, offline=offline)
        # The command exits non-zero when a tool failed, so an all-green stepper
        # would be dishonest even though the run deliberately carried on.
        if result.failed:
            phase.fail()

    if offline:
        ui.skip_phase(
            "System",
            "tt-installer needs the network to fetch golden versions and distro "
            "packages; re-run without --offline",
        )
        installer_ran = False
    else:
        with ui.phase("System"):
            installer_ran = backend.apply_system(
                offline=offline, version=version, force=force
            )
    result = dataclasses.replace(result, installer_ran=installer_ran)

    ui.final_stepper()
    appctx.output.emit(
        {
            **plan_payload,
            "result": dataclasses.asdict(result),
            **ui.timings.as_payload(),
        },
        renderer=_result_summary,
    )
    if result.failed:
        # Everything else converged, so the summary above is the report; exiting
        # via typer keeps `--json` to the single document it promises.
        raise typer.Exit(ExitCode.TOOL_FAILED)
