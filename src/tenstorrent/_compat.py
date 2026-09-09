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

__all__ = ["click", "Abort", "IntRange", "confirm", "prompt", "style"]
