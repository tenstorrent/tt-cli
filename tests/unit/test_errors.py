# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from tenstorrent.errors import ExitCode, TTError


def test_exit_code_values_are_the_documented_contract():
    assert ExitCode.OK == 0
    assert ExitCode.ERROR == 1
    assert ExitCode.USAGE == 2
    assert ExitCode.NO_DEVICES == 3
    assert ExitCode.TOOL_MISSING == 4
    assert ExitCode.TOOL_FAILED == 5
    assert ExitCode.NEEDS_SUDO == 6
    assert ExitCode.UNSUPPORTED == 7
    assert ExitCode.OFFLINE == 8
    assert ExitCode.CONFIG == 9


def test_tterror_to_dict_round_trip():
    err = TTError(
        "Thing failed.",
        why="because",
        next_step="do this",
        exit_code=ExitCode.TOOL_FAILED,
        details={"tool": "tt-smi"},
    )
    payload = err.to_dict()["error"]
    assert payload["what"] == "Thing failed."
    assert payload["why"] == "because"
    assert payload["next_step"] == "do this"
    assert payload["exit_code"] == 5
    assert payload["code"] == "TOOL_FAILED"
    assert payload["details"] == {"tool": "tt-smi"}


def test_tterror_defaults():
    err = TTError("boom")
    assert err.exit_code == ExitCode.ERROR
    assert err.why is None and err.next_step is None and err.details == {}
