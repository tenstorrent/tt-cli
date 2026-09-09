# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Installed-tool state: <data_dir>/installed.toml (name → version/path/timestamp)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import tomlkit

from ..config.paths import Paths


@dataclass(frozen=True)
class InstalledTool:
    name: str
    version: str
    path: Path
    installed_at: str


class ToolState:
    def __init__(self, paths: Paths) -> None:
        self._file = paths.state_file

    def _load(self) -> dict:
        if not self._file.exists():
            return {}
        return tomlkit.parse(self._file.read_text()).unwrap().get("tools", {})

    def get(self, name: str) -> InstalledTool | None:
        entry = self._load().get(name)
        if not entry:
            return None
        return InstalledTool(
            name=name,
            version=str(entry.get("version", "")),
            path=Path(entry.get("path", "")),
            installed_at=str(entry.get("installed_at", "")),
        )

    def all(self) -> dict[str, InstalledTool]:
        return {name: self.get(name) for name in self._load()}  # type: ignore[misc]

    def record(self, name: str, *, version: str, path: Path) -> InstalledTool:
        tools = self._load()
        tools[name] = {
            "version": version,
            "path": str(path),
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        doc = tomlkit.document()
        doc["tools"] = tools
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(tomlkit.dumps(doc))
        return self.get(name)  # type: ignore[return-value]

    def forget(self, name: str) -> None:
        tools = self._load()
        tools.pop(name, None)
        doc = tomlkit.document()
        doc["tools"] = tools
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(tomlkit.dumps(doc))
