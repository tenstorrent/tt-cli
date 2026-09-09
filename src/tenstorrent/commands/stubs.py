# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Documented stubs: commands that are part of the stable surface but not yet
implemented. They describe the intended flow and exit UNSUPPORTED (7) so scripts
never mistake a stub for success."""

from __future__ import annotations

import typer

from ..cli import handle_tt_errors
from ..errors import ExitCode, TTError


@handle_tt_errors
def train(
    ctx: typer.Context,
    args: list[str] = typer.Argument(None, help="Reserved for the future command."),
) -> None:
    """[stub] Train models on Tenstorrent hardware."""
    raise TTError(
        "`tt train` is not available yet.",
        why="Training workflows (tt-train / tt-forge) will land behind this command.",
        next_step="Track progress: https://github.com/tenstorrent/tt-metal",
        exit_code=ExitCode.UNSUPPORTED,
    )


@handle_tt_errors
def compile_(
    ctx: typer.Context,
    args: list[str] = typer.Argument(None, help="Reserved for the future command."),
) -> None:
    """[stub] Compile models from tt-forge."""
    raise TTError(
        "`tt compile` is not available yet.",
        why="Standalone tt-forge compilation will land behind this command "
        "(model-cache-aware compilation ships first as `tt model compile`).",
        next_step="Track progress: https://github.com/tenstorrent/tt-forge",
        exit_code=ExitCode.UNSUPPORTED,
    )
