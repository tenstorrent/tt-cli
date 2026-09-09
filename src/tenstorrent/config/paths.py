# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Filesystem locations. XDG-compliant via platformdirs; TT_*_DIR env vars override
everything (this is also the primary isolation seam for tests)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import platformdirs

APP_NAME = "tenstorrent"


@dataclass(frozen=True)
class Paths:
    config_dir: Path
    data_dir: Path
    cache_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def tools_dir(self) -> Path:
        """UV_TOOL_DIR — one isolated venv per wrapped tool lives here."""
        return self.data_dir / "tools"

    @property
    def tool_bin_dir(self) -> Path:
        """UV_TOOL_BIN_DIR — where uv symlinks tool entry points."""
        return self.data_dir / "bin"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "installed.toml"

    @property
    def golden_file(self) -> Path:
        """Cached golden.json fetched by `tt update` at the pinned tt-sw-manifest
        tag ({"tag": ..., "data": {...}}); ignored when the tag no longer matches."""
        return self.data_dir / "golden.json"

    @property
    def telemetry_file(self) -> Path:
        """Anonymous install id + answered-once opt-in-prompt flag for usage telemetry."""
        return self.data_dir / "telemetry.toml"

    @property
    def self_update_file(self) -> Path:
        """Last background version check: when, what it found, which tt was running."""
        return self.data_dir / "self-update.toml"

    @property
    def self_update_lock(self) -> Path:
        """flock held by a running background version check (cf. telemetry's spool.lock)."""
        return self.data_dir / "self-update.lock"

    @property
    def telemetry_dir(self) -> Path:
        """Span spool + drain lock for persist-then-drain telemetry (see telemetry/spool.py)."""
        return self.data_dir / "telemetry"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def installer_work_dir(self) -> Path:
        """CWD for `tt update`'s install.sh run. install.sh takes no meaningful input
        from its working directory (every path default is absolute), but something in
        its install path drops a `wget-log` there — keep that out of the user's CWD."""
        return self.data_dir / "installer-work"


def get_paths(env: Mapping[str, str] | None = None) -> Paths:
    env = os.environ if env is None else env

    def pick(var: str, default: str) -> Path:
        return Path(env.get(var) or default).expanduser()

    return Paths(
        config_dir=pick("TT_CONFIG_DIR", platformdirs.user_config_dir(APP_NAME)),
        data_dir=pick("TT_DATA_DIR", platformdirs.user_data_dir(APP_NAME)),
        cache_dir=pick("TT_CACHE_DIR", platformdirs.user_cache_dir(APP_NAME)),
    )
