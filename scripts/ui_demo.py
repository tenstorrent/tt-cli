#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""A tour of the tt CLI's output language. Doubles as a smoke test.

    python3 scripts/ui_demo.py            # the calm default
    python3 scripts/ui_demo.py -v         # folded detail returns
    python3 scripts/ui_demo.py --fail     # the failure paths
    python3 scripts/ui_demo.py | cat -v   # non-TTY: expect zero escape codes

Nothing here touches hardware, the network, or Docker.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenstorrent.output import OutputManager  # noqa: E402
from tenstorrent.ui import (  # noqa: E402
    failure_card,
    fmt_bytes,
    interrupted_panel,
    progress_bar,
    ready_panel,
)

PHASES = ["Checks", "Tools", "System"]

# Tests that don't assert on durations run the tour at a fraction of the pace.
SPEED = float(os.environ.get("TT_UI_DEMO_SPEED", "1"))


def pause(seconds: float) -> None:
    time.sleep(seconds * SPEED)


def main() -> int:
    fail = "--fail" in sys.argv
    out = OutputManager(verbose=("-v" in sys.argv or "--verbose" in sys.argv))
    ui = out.ui
    ui.register_phases(PHASES)

    with ui.phase("Checks"):
        with ui.step("Fetching golden versions") as step:
            pause(1.2)
            step.detail("v1.0.0")
        with ui.step("Comparing installed versions to the goldens") as step:
            pause(0.4)
            step.detail("3 to install, 1 up to date")
        ui.alert("tt-metal is managed outside tt — an override is in effect")

    with ui.phase("Tools"):
        for name, version, took in (("tt-smi", "3.0.30", 1.4), ("tt-flash", "0.5.2", 0.9)):
            with ui.step(f"Installing {name} {version}"):
                pause(took)
        with ui.step("Installing tt-topology 1.2.0") as step:
            pause(0.3)
            step.skip("already up to date")
        if out.ui.show_detail():
            ui.note("tt-luwen 0.7.1 is optional — `tt update --include-lazy` installs it")

    with ui.phase("System") as phase:
        # One live row standing in for a stream we don't control: an exact
        # denominator (packages) with bytes as a counter, never a fake percentage.
        with ui.activity("Running the system installer") as row:
            total, done_bytes = 24, 0
            for i in range(1, total + 1):
                done_bytes += 17_000_000
                row.set(
                    "Running the system installer  "
                    f"{progress_bar(i, total)}  {i}/{total} packages · {fmt_bytes(done_bytes)}"
                )
                if i in (8, 16):
                    row.milestone(f"kmd {i // 8}.4.0 installed")
                pause(0.08)
        if fail:
            phase.fail()

    if fail:
        ui.card(
            failure_card(
                "tt-installer",
                {
                    "cause": "the firmware bundle didn't verify",
                    "detail": "The downloaded bundle's checksum did not match the manifest.",
                    "evidence": "ERROR: sha256 mismatch for fw_pack-19.13.1.fwbundle",
                    "actions": ["tt update --refresh", "tt report issue"],
                },
                log_path="~/.local/share/tenstorrent/logs/tt-installer.log",
                consequence="The tools above are installed; the system stack was left unchanged.",
            )
        )
        ui.card(interrupted_panel("tt update", cleanup="tt update --dry-run"))
        return 1

    ui.final_stepper()
    ui.card(
        ready_panel(
            "tt is up to date",
            [
                ("Tools", "tt-smi 3.0.30, tt-flash 0.5.2", "installed"),
                ("System", "kmd 2.4.0 · fw 19.13.1", "converged"),
                ("Mode", "Local + TT hardware"),
            ],
            [
                f"[muted]Ready in {ui.timings.to_dict()['total_seconds']:.0f}s · "
                f"{len(PHASES)} phases[/muted]",
                "[muted]Inspect · tt self tools[/muted]",
                "[muted]Devices · tt device status[/muted]",
            ],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        OutputManager().ui.card(interrupted_panel("tt update"))
        raise SystemExit(130)
