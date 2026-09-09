# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Device backend protocol + factory — the native-vs-delegated seam.

Commands talk only to this protocol and receive typed models; all wrapped-tool
parsing lives inside the concrete backend. A future native backend (pyluwen)
plugs in via the `device.backend` config key without touching commands or the
`--json` schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from ..errors import ExitCode, TTError
from ..models.device import ResetResult, SystemSnapshot

if TYPE_CHECKING:  # pragma: no cover
    from ..context import AppContext


class DeviceBackend(Protocol):
    def snapshot(self) -> SystemSnapshot: ...

    def raw_snapshot(self) -> dict:
        """The wrapped tool's own snapshot JSON (escape hatch, no schema promise)."""
        ...

    def reset(self, indices: Sequence[int] | None, *, allow_prompt: bool = True) -> ResetResult: ...

    def top_argv(self) -> list[str]:
        """argv to exec for the interactive TUI (delegation-only by design)."""
        ...


def get_device_backend(appctx: "AppContext") -> DeviceBackend:
    kind = str(appctx.config.get("device.backend"))
    if kind == "smi":
        from .smi import SmiDelegatedBackend

        return SmiDelegatedBackend(appctx.registry, appctx.runner, appctx.paths)
    raise TTError(
        f"Unknown device backend {kind!r}.",
        why='Only "smi" (delegate to tt-smi) exists today; a native backend is planned.',
        next_step="tt config set device.backend smi",
        exit_code=ExitCode.CONFIG,
    )
