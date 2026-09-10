# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Process execution for wrapped tools.

Four modes:

- `capture`       pipe and buffer; the caller parses the result.
- `stream`        inherit stdio; the child owns the terminal.
- `stream_parsed` pipe, read line by line, tee to a log, feed a parser.
- `exec_tty`      replace this process; never returns.

`stream_parsed` is what lets a long tool show one live line instead of thousands
of its own: raw output goes to a log file, a pure parser turns lines into
milestones and an activity label, and on failure the error carries the log path.

Note the env asymmetry, which is deliberate and load-bearing: `capture` and
`stream` *replace* the environment (their callers already spread `{**os.environ}`,
and `inference_server`/`model_manager` depend on that), while `stream_parsed` and
`exec_tty` *merge* over it. New code should prefer the merging form.

Constructor seams (`spawn`, `popen`, `exec_fn`, `isatty_fn`) exist for tests;
passing `sudo_command=""` disables sudo entirely.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from ..errors import ExitCode, TTError
from .runlog import open_run_log


@dataclass(frozen=True)
class CaptureResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class StreamResult:
    returncode: int
    output: str  # merged stdout+stderr, trimmed to the last `keep_lines`
    log_path: Path | None
    truncated: bool


class Runner:
    def __init__(
        self,
        *,
        sudo_command: str = "sudo",
        spawn: Callable[..., subprocess.CompletedProcess] | None = None,
        popen: Callable[..., Any] | None = None,
        exec_fn: Callable[..., NoReturn] | None = None,
        isatty_fn: Callable[[], bool] | None = None,
        before_exec: Callable[[], None] | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.sudo_command = sudo_command
        self._spawn = spawn or subprocess.run
        # A separate seam from `spawn`: streaming needs a live process object, so
        # a CompletedProcess stand-in can't express it.
        self._popen = popen or subprocess.Popen
        self._exec_fn = exec_fn or os.execvpe
        self._isatty = isatty_fn or (lambda: sys.stdin.isatty())
        self._before_exec = before_exec
        self.log_dir = log_dir

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
                next_step="The tool printed its own output above.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": tool or str(argv[0]), "returncode": proc.returncode},
            )
        return proc.returncode

    def stream_parsed(
        self,
        argv: Sequence[str],
        *,
        on_line: Callable[[str], None] | None = None,
        env_extra: Mapping[str, str] | None = None,
        cwd: str | None = None,
        sudo: bool = False,
        allow_prompt: bool = True,
        check: bool = True,
        tool: str | None = None,
        log: bool = True,
        keep_lines: int = 500,
        stdin_devnull: bool = True,
    ) -> StreamResult:
        """Pipe a child, read it line by line, tee it to a log, feed `on_line`.

        stderr is merged into stdout so the parser sees everything, and — more
        importantly — so the child can't paint over a live spinner row.

        stdin defaults to /dev/null: a piped child that inherits stdin is how you
        get a half-visible password prompt underneath a repainting row. A caller
        that genuinely needs the tty passes `stdin_devnull=False` and suspends the
        UI around it (`ui.prompting()`).

        `on_line` exceptions are swallowed: a parser bug must never take down the
        command it was only describing.
        """
        if sudo:
            argv = self.wrap_sudo(argv, allow_prompt=allow_prompt)
        argv = list(argv)
        name = tool or argv[0]
        run_log = open_run_log(self.log_dir if log else None, name, argv)
        log_path = run_log.path if run_log is not None else None

        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)

        try:
            process = self._popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL if stdin_devnull else None,
                text=True,
                bufsize=1,
                errors="replace",
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            if run_log is not None:
                run_log.close()
            raise TTError(
                f"Cannot run {argv[0]!r}: executable not found.",
                next_step="Run `tt update` to install managed tools.",
                exit_code=ExitCode.TOOL_MISSING,
                details={"tool": name},
            ) from exc
        except PermissionError as exc:
            if run_log is not None:
                run_log.close()
            raise TTError(
                f"Cannot run {argv[0]!r}: permission denied.",
                why="The file exists but is not executable.",
                next_step=f"chmod +x {argv[0]} or re-run `tt update` to reinstall it.",
                exit_code=ExitCode.TOOL_FAILED,
                details={"tool": name},
            ) from exc

        kept: list = []
        truncated = False
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    if run_log is not None:
                        run_log.write(line)
                    kept.append(line)
                    if len(kept) > keep_lines:
                        kept.pop(0)
                        truncated = True
                    if on_line is not None:
                        try:
                            on_line(line.rstrip("\n"))
                        except Exception:
                            pass
            returncode = process.wait()
        except KeyboardInterrupt:
            # The child shares our process group, so it already saw the SIGINT.
            # Give it a moment to exit on its own before escalating.
            self._reap(process)
            raise
        finally:
            if run_log is not None:
                run_log.close()

        output = "".join(kept)
        if check and returncode != 0:
            tail = "\n".join(
                line for line in output.strip().splitlines()[-8:] if line.strip()
            )
            raise TTError(
                f"{name} exited with status {returncode}.",
                why=tail or None,
                next_step=(
                    f"Full output: {log_path}"
                    if log_path is not None
                    else "Re-run with --verbose to see the tool's own output."
                ),
                exit_code=ExitCode.TOOL_FAILED,
                details={
                    "tool": name,
                    "returncode": returncode,
                    **({"log_path": str(log_path)} if log_path is not None else {}),
                },
            )
        return StreamResult(returncode, output, log_path, truncated)

    @staticmethod
    def _reap(process: Any, grace: float = 5.0) -> None:
        """Terminate, then kill: never leave an orphan holding the terminal.

        Catches BaseException, not Exception, on purpose: a second Ctrl-C arriving
        mid-cleanup must not abort the cleanup — that is exactly how an orphaned
        child ends up owning the user's terminal.
        """
        for finish in (lambda: process.wait(grace), process.terminate, process.kill):
            try:
                finish()
                if process.poll() is not None:
                    return
            except BaseException:
                continue

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
