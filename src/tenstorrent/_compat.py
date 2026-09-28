# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Import the click that Typer actually uses.

Typer >= 0.27 vendors click as `typer._click`; importing the standalone `click`
package gives a *different* module whose context stack and exception classes
don't match what Typer raises. Everything in this codebase must import click
from here.
"""

try:  # Typer with vendored click
    import typer._click as click  # type: ignore[import-not-found]
except ModuleNotFoundError:  # older Typer that depends on real click
    import click  # type: ignore[no-redef]

# Typer re-exports these at its top level from whichever click it uses, and
# that surface is stable; the vendored submodule paths are not (0.27.2 moved
# Abort from typer._click.exceptions to typer.exceptions). IntRange has no
# typer re-export but lives in <click>.types in both vendored and real click.
from typer import Abort, confirm, prompt, style

IntRange = click.types.IntRange


def _exit_exceptions() -> tuple:
    """Every exception class that means "exit with this code", not "something broke".

    `typer.Exit` is stable at typer's top level, but click's own `Exit` (what
    `ctx.exit()` raises) is not re-exported by Typer and is absent from some
    vendored layouts entirely — so probe for it rather than importing a path.

    This matters because typer.Exit subclasses RuntimeError: an `except Exception`
    that means to catch crashes will swallow a deliberate exit code unless this
    tuple is caught first.
    """
    import typer

    classes: list = [typer.Exit]
    for holder in (getattr(click, "exceptions", None), click):
        found = getattr(holder, "Exit", None)
        if isinstance(found, type) and found not in classes:
            classes.append(found)
    return tuple(classes)


EXIT_EXCEPTIONS = _exit_exceptions()

__all__ = [
    "click",
    "Abort",
    "EXIT_EXCEPTIONS",
    "IntRange",
    "confirm",
    "prompt",
    "style",
]
