# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Telemetry state: <data_dir>/telemetry.toml (anonymous install id + prompt flag).

Mirrors tools/state.py: a small tomlkit round-trip under the data dir. Holds the
random per-install `instance_id` (a rotation-safe pseudonym, never derived from
hardware/MAC) and whether the one-time opt-in consent prompt has been answered.
"""

from __future__ import annotations

import uuid

import tomlkit

from ..config.paths import Paths


class TelemetryState:
    def __init__(self, paths: Paths) -> None:
        self._file = paths.telemetry_file

    def _load(self) -> dict:
        if not self._file.exists():
            return {}
        return tomlkit.parse(self._file.read_text()).unwrap().get("telemetry", {})

    def _save(self, data: dict) -> None:
        doc = tomlkit.document()
        doc["telemetry"] = data
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(tomlkit.dumps(doc))

    def instance_id(self) -> str:
        """Return the anonymous install id, generating and persisting it on first use."""
        data = self._load()
        existing = data.get("instance_id")
        if existing:
            return str(existing)
        new_id = str(uuid.uuid4())
        data["instance_id"] = new_id
        self._save(data)
        return new_id

    def prompt_answered(self) -> bool:
        """Has the one-time opt-in prompt been answered (either way)?

        Keyed as `optin_prompt_answered`, deliberately NOT the pre-opt-in-era
        `notice_shown`: installs that only ever saw the old opt-out notice were never
        asked for consent, so they are asked once when they next run interactively.
        """
        return bool(self._load().get("optin_prompt_answered", False))

    def mark_prompt_answered(self) -> None:
        data = self._load()
        data["optin_prompt_answered"] = True
        self._save(data)
