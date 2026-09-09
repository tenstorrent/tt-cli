# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""huggingface_hub draws its own tqdm bars outside OutputManager, so the
`--quiet` / `--json` contract has to be enforced explicitly."""

import pytest
from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars

from tenstorrent.modelhub.hub import _honor_output_mode
from tenstorrent.output import OutputManager


@pytest.fixture(autouse=True)
def restore_progress_bars():
    # disable_progress_bars() is process-global; don't leak it into other tests.
    yield
    enable_progress_bars()


@pytest.mark.parametrize(
    "kwargs",
    [{"quiet": True}, {"json_mode": True}, {"quiet": True, "json_mode": True}],
)
def test_quiet_and_json_silence_the_hub_progress_bars(kwargs):
    _honor_output_mode(OutputManager(**kwargs))
    assert are_progress_bars_disabled()


def test_human_mode_keeps_the_download_bar():
    # A live download bar earns its space; only the machine-readable and
    # silenced modes lose it.
    _honor_output_mode(OutputManager())
    assert not are_progress_bars_disabled()


def test_no_output_manager_is_a_no_op():
    _honor_output_mode(None)
    assert not are_progress_bars_disabled()
