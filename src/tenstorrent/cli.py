# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Root Typer app for `tt`, global flags, and central TTError → exit-code handling.

Every leaf command is wrapped in @handle_tt_errors so a raised TTError becomes a
rendered error plus the documented exit code — both under the real binary and under
Typer's CliRunner in tests (which never reaches main()'s safety net).
"""

from __future__ import annotations

import contextlib
import functools
import sys
from typing import Annotated, Callable, TypeVar

import typer

from . import __version__
from ._compat import Abort, click
from .context import AppContext
from .errors import ExitCode, TTError, render_error
from .output import OutputManager
from .telemetry import NULL_SESSION
from .ui import failure_card, interrupted_panel

F = TypeVar("F", bound=Callable)

# `--help` is documentation, so flags and commands live in a FIXED set of panels:
# a new command has an obvious home and the help stays skimmable.
# Typer groups items by panel name, but the rendered panel order is not fully controllable.
PANEL_DEVICES = "Devices"
PANEL_SOFTWARE = "Software"
PANEL_MODELS = "Models"
PANEL_WORKLOADS = "Workloads"
PANEL_CONFIG = "Configuration"
PANEL_TROUBLESHOOTING = "Troubleshooting"
PANEL_OUTPUT = "Output"
PANEL_GLOBAL = "Global"

# Shared leaf-command flags: `tt device status --json` and `tt --json device status`
# must both work, so these are declared once and reused by every command.
JsonFlag = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Emit machine-readable JSON on stdout.",
        rich_help_panel=PANEL_OUTPUT,
    ),
]
QuietFlag = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress everything except errors.",
        rich_help_panel=PANEL_OUTPUT,
    ),
]
# Declared per-leaf for the same reason as --json/-q: Typer's leaf parser rejects
# an unknown short option, so `tt update -v` needs the alias here — sniffing argv
# in the root callback is too late.
VerboseFlag = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Show the detail a normal run folds away.",
        rich_help_panel=PANEL_OUTPUT,
    ),
]
NoColorFlag = Annotated[
    bool,
    typer.Option(
        "--no-color",
        help="Disable colour and styling (also honours NO_COLOR).",
        rich_help_panel=PANEL_OUTPUT,
    ),
]


def _find_context(args, kwargs):
    """Locate the command's own typer.Context (whose .obj is the AppContext).
    Duck-typed (any object with an AppContext .obj) so it works across click/typer
    versions — the global click context stack can't be trusted."""
    for value in (*args, *kwargs.values()):
        if isinstance(getattr(value, "obj", None), AppContext):
            return value
    return None


def handle_tt_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = _find_context(args, kwargs)
        appctx = ctx.obj if ctx is not None else None
        # One usage span per command invocation, keyed on the same AppContext the
        # error handling uses. NULL_SESSION when telemetry is off/unavailable.
        session = appctx.telemetry if appctx is not None else NULL_SESSION
        try:
            with contextlib.ExitStack() as stack:
                span = stack.enter_context(session.command_span(ctx))
                if appctx is not None:
                    # A command that hands the terminal over (exec_tty) never returns,
                    # so neither the span nor the flush below would ever run. Close
                    # them out first; both are idempotent, so the normal path is
                    # unaffected.
                    def _finish_before_exec() -> None:
                        span.set_exit_code(ExitCode.OK)
                        stack.close()
                        session.flush()

                    appctx.before_exec.append(_finish_before_exec)
                try:
                    result = fn(*args, **kwargs)
                    span.set_exit_code(ExitCode.OK)
                    return result
                except TTError as err:
                    span.record_error(err)
                    output = appctx.output if appctx is not None else OutputManager()
                    render_error(err, output)
                    raise typer.Exit(int(err.exit_code)) from err
                except typer.Exit as exit_exc:
                    span.set_exit_code(getattr(exit_exc, "exit_code", 0) or 0)
                    raise
                except Exception:
                    span.set_exit_code(ExitCode.ERROR)
                    raise
        finally:
            session.flush()
            if appctx is not None:
                # Update notice / background version check. Guarded internally; the
                # stale-state spawn is the same detached-process trick telemetry uses.
                from .selfupdate.check import after_command

                after_command(appctx, ctx)

    return wrapper  # type: ignore[return-value]


app = typer.Typer(
    name="tt",
    help="Tenstorrent CLI: the single entry point to the Tenstorrent software stack.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    # A traceback is not a user-facing error message, and Typer's pretty version
    # renders local variables — which here can include environment values and
    # tokens. main() turns an unexpected exception into a card and writes the
    # traceback to a log instead; -v prints it.
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Bare `tt` / `tt <group>` must behave exactly like `-h`: print help, exit 0.
# click >= 8.2 raises NoArgsIsHelpError with the usage exit code (2); overriding
# the class attribute fixes it centrally for the real binary AND CliRunner tests
# (which never reach main()). Sub-apps opt in via no_args_is_help=True.
click.exceptions.NoArgsIsHelpError.exit_code = 0


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tt {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Never touch the network; fail with guidance instead.",
            rich_help_panel=PANEL_GLOBAL,
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the CLI version and exit.",
            callback=_version_callback,
            is_eager=True,
            rich_help_panel=PANEL_GLOBAL,
        ),
    ] = False,
) -> None:
    ctx.obj = AppContext.create(
        json_mode=json_mode,
        quiet=quiet,
        verbose=verbose,
        no_color=no_color,
        offline=offline,
    )


def _register_commands() -> None:
    # Imported here so `import tenstorrent.cli` stays cheap and cycle-free.
    from .commands.config_cmd import config_app
    from .commands.device import device_app
    from .commands.launch import launch_app
    from .commands.model import model_app
    from .commands.self_cmd import self_app
    from .commands.report import report_app
    from .commands.serve import serve
    from .commands.stubs import compile_, train
    from .commands.update import update

    app.add_typer(device_app, name="device", rich_help_panel=PANEL_DEVICES)
    app.command("update", rich_help_panel=PANEL_SOFTWARE)(update)
    app.add_typer(model_app, name="model", rich_help_panel=PANEL_MODELS)
    # no_args_is_help on leaf commands: bare invocation of a command that cannot
    # run without arguments shows help (exit 0, like -h) instead of a usage error.
    app.command(
        "serve",
        no_args_is_help=True,
        rich_help_panel=PANEL_WORKLOADS,
        # Unknown options are collected into ctx.args and forwarded to tt-model
        # for a bundle id (rejected for a catalog model) — see commands/serve.py.
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(serve)
    app.add_typer(launch_app, name="launch", rich_help_panel=PANEL_WORKLOADS)
    # Hidden until the workflows behind them ship: they stay registered so
    # `tt train` / `tt compile` answer with the exit-7 stub instead of a usage
    # error, but `tt --help` advertises only what works today. They keep their
    # panel so unhiding stays the one-word change it was meant to be.
    app.command("train", hidden=True, rich_help_panel=PANEL_WORKLOADS)(train)
    app.command("compile", hidden=True, rich_help_panel=PANEL_WORKLOADS)(compile_)
    app.add_typer(config_app, name="config", rich_help_panel=PANEL_CONFIG)
    app.add_typer(report_app, name="report", rich_help_panel=PANEL_TROUBLESHOOTING)
    app.add_typer(self_app, name="self", rich_help_panel=PANEL_SOFTWARE)

    # Muscle-memory alias for tt-smi users: `tt smi` == `tt device top`.
    from .commands.device import top

    app.command("smi", hidden=True)(top)


_register_commands()


def _argv_verbose() -> bool:
    """Whether -v/--verbose was asked for, read straight from argv.

    main() runs outside any command, so there is no AppContext to consult — and
    this is exactly the moment (an unexpected crash) when we most need to know.
    """
    return any(arg in ("-v", "--verbose") for arg in sys.argv[1:])


def _resume_command() -> str:
    """Reconstruct the command so an interrupt card can offer to resume it."""
    args = [arg for arg in sys.argv[1:] if arg not in ("-v", "--verbose")]
    return "tt " + " ".join(args) if args else "tt"


def _report_unexpected(exc: BaseException) -> None:
    """Render an unexpected exception as a card, and keep the traceback on disk.

    The traceback is evidence, not UI: it goes to a log the card points at, and
    only reaches the terminal under -v.
    """
    import traceback

    output = OutputManager()
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path = None
    try:
        from .config.paths import get_paths
        from .tools.runlog import open_run_log

        run_log = open_run_log(get_paths().logs_dir, "crash", sys.argv)
        if run_log is not None:
            run_log.write(detail)
            run_log.close()
            log_path = run_log.path
    except Exception:
        pass

    actions = [f"tt {' '.join(sys.argv[1:])} -v" if len(sys.argv) > 1 else "tt -v"]
    actions.append("tt report issue")
    output.ui.card(
        failure_card(
            "tt hit an unexpected error",
            {
                "cause": type(exc).__name__,
                "detail": str(exc) or "No message was attached to the exception.",
                "evidence": detail.strip().splitlines()[-1] if detail.strip() else "",
                "actions": actions,
            },
            log_path=log_path,
            consequence="Nothing further was attempted; the command stopped here.",
        )
    )
    if _argv_verbose():
        output.status_console.print(detail, style="muted")


def main() -> None:
    """Console entry point: run the app, mapping every failure to a documented exit code."""
    try:
        result = app(standalone_mode=False)
        code = result if isinstance(result, int) else 0
    except TTError as err:  # safety net; commands normally handle this via decorator
        render_error(err, OutputManager())
        code = int(err.exit_code)
    except click.exceptions.NoArgsIsHelpError as err:
        # Typer already rendered the rich help before raising; don't print it twice.
        # exit_code is 0 here (patched above): bare `tt <group>` == `tt <group> -h`.
        code = err.exit_code
    except click.ClickException as err:  # usage errors etc. (exit code 2)
        err.show()
        code = err.exit_code
    except Abort:
        click.echo("Aborted.", err=True)
        code = 130
    except KeyboardInterrupt:
        # Ctrl-C is never a cliff: say how to pick up where we left off. Until now
        # this fell through to a raw traceback.
        OutputManager().ui.card(interrupted_panel(_resume_command()))
        code = 130
    except Exception as exc:
        _report_unexpected(exc)
        code = int(ExitCode.ERROR)
    raise SystemExit(code)
