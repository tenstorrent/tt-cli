# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""DEVELOPER TOOL — reset the telemetry consent prompt so it is asked again.

Solely for developers iterating on the opt-in flow (wording, styling, plumbing).
It is not shipped with the CLI, not a supported user command, and never the answer
to a user question — users manage telemetry with `tt config set telemetry.enabled
true|false` and are, by design, asked for consent at most once per install.

What it does (and nothing else):
  * drops `optin_prompt_answered` from <data_dir>/telemetry.toml, keeping the
    anonymous `instance_id` (deleting the whole file would rotate it);
  * sets `telemetry.enabled = false` in config.toml via ConfigStore, since the
    prompt only fires while no consent is in force.

Honors TT_DATA_DIR / TT_CONFIG_DIR, same as `tt` itself. After running, any
interactive `tt` command re-prompts — provided none of the usual suppressors are
active (--quiet/--json/--offline, CI, DO_NOT_TRACK, TT_TELEMETRY_DISABLED, no real
TTYs). The script warns about the ones it can detect from here.

    uv run scripts/reset_telemetry_prompt.py
"""

from __future__ import annotations

import os
import sys

import tomlkit

from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import ConfigStore
from tenstorrent.telemetry.session import resolve_endpoint
from tenstorrent.telemetry.state import TelemetryState


def main() -> int:
    paths = get_paths()

    # Drop the answered flag, preserving everything else in telemetry.toml.
    state_file = paths.telemetry_file
    if state_file.exists():
        doc = tomlkit.parse(state_file.read_text())
        table = doc.get("telemetry")
        if table is not None and "optin_prompt_answered" in table:
            del table["optin_prompt_answered"]
            state_file.write_text(tomlkit.dumps(doc))
            print(f"cleared optin_prompt_answered in {state_file}")
        else:
            print(f"no optin_prompt_answered flag in {state_file} (already unanswered)")
    else:
        print(f"{state_file} does not exist (already unanswered)")

    # Withdraw any recorded consent; the prompt only runs while telemetry is off.
    config = ConfigStore(paths)
    config.set("telemetry.enabled", False)
    print(f"set telemetry.enabled = false in {paths.config_file}")

    if not TelemetryState(paths).prompt_answered():
        print("prompt state reset: the next eligible interactive `tt` command will ask.")
    else:
        print("ERROR: prompt still reads as answered — reset did not take.", file=sys.stderr)
        return 1

    # The prompt also skips silently when nothing could be sent or a kill switch is
    # set; flag the causes visible from this process so a dev isn't left guessing.
    endpoint, token = resolve_endpoint(config)
    if not endpoint or not token:
        print(
            "WARNING: endpoint/key resolve empty (a config.toml predating the bundled "
            "key may pin posthog_project_key = \"\") — the prompt will NOT appear.",
            file=sys.stderr,
        )
    for var in ("TT_TELEMETRY_DISABLED", "DO_NOT_TRACK"):
        if os.environ.get(var):
            print(f"WARNING: {var} is set in this shell — the prompt will NOT appear.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
