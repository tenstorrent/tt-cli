# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""AppContext: everything a command needs, hung on ctx.obj by the root callback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import typer

from .config.paths import Paths, get_paths
from .config.store import ConfigStore
from .output import OutputManager

if TYPE_CHECKING:  # pragma: no cover
    from .telemetry import TelemetrySession
    from .tools.registry import ToolRegistry
    from .tools.runner import Runner


@dataclass
class AppContext:
    paths: Paths
    config: ConfigStore
    output: OutputManager
    offline: bool = False
    _extras: dict = field(default_factory=dict)
    # Run just before a command replaces this process (Runner.exec_tty). Anything
    # relying on the command returning — the telemetry span, above all — has to be
    # closed out here, or it dies with the process. @handle_tt_errors registers it.
    before_exec: list = field(default_factory=list)

    def run_before_exec(self) -> None:
        """Guarded: a hook must never stop the hand-off it precedes."""
        for hook in self.before_exec:
            try:
                hook()
            except Exception:
                pass

    @property
    def runner(self) -> "Runner":
        if "runner" not in self._extras:
            from .tools.runner import Runner

            self._extras["runner"] = Runner(
                sudo_command=str(self.config.get("tools.sudo_command")),
                before_exec=self.run_before_exec,
            )
        return self._extras["runner"]

    @property
    def registry(self) -> "ToolRegistry":
        if "registry" not in self._extras:
            from .tools.registry import ToolRegistry

            self._extras["registry"] = ToolRegistry(
                self.paths, self.config, runner=self.runner
            )
        return self._extras["registry"]

    @property
    def telemetry(self) -> "TelemetrySession":
        """Usage-telemetry session (a no-op sentinel when disabled/unconfigured).
        Built once per process; reads config lazily so the no-op path costs nothing."""
        if "telemetry" not in self._extras:
            from .telemetry import TelemetrySession

            self._extras["telemetry"] = TelemetrySession.create(
                self.paths, self.config, offline=self.offline, output=self.output
            )
        return self._extras["telemetry"]

    @classmethod
    def create(
        cls,
        *,
        json_mode: bool = False,
        quiet: bool = False,
        verbose: bool = False,
        no_color: bool = False,
        offline: bool = False,
    ) -> "AppContext":
        paths = get_paths()
        output = OutputManager(
            json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color
        )
        return cls(
            paths=paths,
            # on_warning: settings the file declares but this version can't act on are
            # reported on stderr wherever config is read, not only under `tt config`.
            config=ConfigStore(paths, on_warning=output.warn),
            output=output,
            offline=offline,
        )


def get_app_context(ctx: typer.Context) -> AppContext:
    """Fetch the AppContext, creating a default one if the callback didn't run
    (defensive: keeps commands usable under direct invocation in tests)."""
    if ctx.obj is None:
        ctx.obj = AppContext.create()
    return ctx.obj
