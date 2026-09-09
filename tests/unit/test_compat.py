# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from tenstorrent import _compat


def test_abort_matches_the_click_implementation_typer_uses():
    if _compat.click.__name__ == "typer._click":
        from typer.exceptions import Abort
    else:
        from click import Abort

    assert _compat.Abort is Abort


def test_prompt_helpers_are_available():
    assert callable(_compat.confirm)
    assert callable(_compat.prompt)
    assert callable(_compat.style)
