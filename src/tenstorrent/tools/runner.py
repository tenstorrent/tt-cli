# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Process execution for wrapped tools: capture (parse output), stream (inherit
stdio), exec_tty (hand the terminal over). Sudo wrapping with a fail-fast probe.

Constructor seams (`spawn`, `exec_fn`, `isatty_fn`) exist for tests; passing
`sudo_command=""` disables sudo entirely.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, NoReturn, Sequence

from ..errors import ExitCode, TTError


@dataclass(frozen=True)
class CaptureResult:
    returncode: int
    stdout: str
    stderr: str


class Runner:
    def __init__(
        self,
        *,
        sudo_command: str = "sudo",
        spawn: Callable[..., subprocess.CompletedProcess] | None = None,
        exec_fn: Callable[..., NoReturn] | None = None,
        isatty_fn: Callable[[], bool] | None = None,
        before_exec: Callable[[], None] | None = None,
    ) -> None:
        self.sudo_command = sudo_command
        self._spawn = spawn or subprocess.run
        self._exec_fn = exec_fn or os.execvpe
        self._isatty = isatty_fn or (lambda: sys.stdin.isatty())
        self._before_exec = before_exec

    # -- sudo ---------------------------------------------------------------------
    def _sudo_prefix(self) -> list[str]:
        return shlex.split(self.sudo_command) if self.sudo_command else []

    def wrap_sudo(self, argv: Sequence[str], *, allow_prompt: bool = True) -> list[str]:
        """Prefix argv with sudo when needed. Probes `sudo -n true` first and fails
        fast with NEEDS_SUDO (carrying the exact rerun command) instead of letting a
        password prompt deadlock non-interactive/--json contexts."""
        argv = list(argv)
        if os.geteuid() == 0 or not self._sudo_prefix():
            return argv
        probe = self._spawn(
            [*self._sudo_prefix(), "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode != 0 and not (allow_prompt and self._isatty()):
            rerun = shlex.join([*self._sudo_prefix(), *argv])
            raise TTError(
                "This operation needs elevated privileges.",
                why="sudo requires a password and this session cannot prompt for one.",
                next_step=f"Run it yourself: {rerun}",
                exit_code=ExitCode.NEEDS_SUDO,
                details={"command": rerun},
            )
        return [*self._sudo_prefix(), *argv]

    # -- modes ----------------------------------------------------------------------
    def capture(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        check: bool = True,
        tool: str | None = None,
    ) -> CaptureResult:
        try:
            proc = self._spawn(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(env) if env is not None else None,
                cwd=cwd,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise TTError(
                f"Cannot run {argv[0]!r}: executable not found.",
                next_step="Run `tt update` to install managed tools.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": tool or str(argv[0])},
            ) from exc
        except PermissionError as exc:
            raise TTError(
                f"Cannot run {argv[0]!r}: permission denied.",
                why="The file exists but is not executable.",
                next_step=f"chmod +x {argv[0]} or re-run `tt update` to reinstall it.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0])},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TTError(
                f"{tool or argv[0]} timed out after {timeout}s.",
                next_step="Retry; if it persists, run the tool directly to see where it hangs.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0])},
            ) from exc
        result = CaptureResult(proc.returncode, proc.stdout or "", proc.stderr or "")
        if check and result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-8:])
            raise TTError(
                f"{tool or argv[0]} exited with status {result.returncode}.",
                why=tail or None,
                next_step="Re-run with --verbose for the full command line.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0]), "returncode": result.returncode},
            )
        return result

    def stream(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        sudo: bool = False,
        allow_prompt: bool = True,
        check: bool = True,
        tool: str | None = None,
    ) -> int:
        """Run inheriting stdio (installers, resets — anything interactive/long)."""
        if sudo:
            argv = self.wrap_sudo(argv, allow_prompt=allow_prompt)
        try:
            proc = self._spawn(
                list(argv), env=dict(env) if env is not None else None, cwd=cwd
            )
        except FileNotFoundError as exc:
            raise TTError(
                f"Cannot run {argv[0]!r}: executable not found.",
                next_step="Run `tt update` to install managed tools.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": tool or str(argv[0])},
            ) from exc
        except PermissionError as exc:
            raise TTError(
                f"Cannot run {argv[0]!r}: permission denied.",
                why="The file exists but is not executable.",
                next_step=f"chmod +x {argv[0]} or re-run `tt update` to reinstall it.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0])},
            ) from exc
        if check and proc.returncode != 0:
            raise TTError(
                f"{tool or argv[0]} exited with status {proc.returncode}.",
                next_step="Scroll up for the tool's own output.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0]), "returncode": proc.returncode},
            )
        return proc.returncode

    def exec_tty(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> NoReturn:
        """Replace this process (TUI hand-off). Never returns.

        Nothing after this runs — not the caller, not @handle_tt_errors' `finally` —
        so the hook closes out anything that would otherwise die with the process.
        """
        if self._before_exec is not None:
            self._before_exec()
        final_env = dict(os.environ)
        if env:
            final_env.update(env)
        self._exec_fn(str(argv[0]), list(argv), final_env)
        raise AssertionError("exec_fn returned")  # pragma: no cover
