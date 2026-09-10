# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Rendering guards that need a real terminal, plus the stdout/stderr contract.

Anything animated is invisible to CliRunner, so the live paths are exercised by
spawning scripts/ui_demo.py under a PTY and asserting on the captured bytes.
"""

from __future__ import annotations

import os
import pty
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "scripts" / "ui_demo.py"
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

pytestmark = pytest.mark.fakes_only


# The tour sleeps to make the animation visible. Scale it down everywhere except
# the two tests that assert on the durations themselves.
FAST = "0.06"
REAL = "1"


def run_in_pty(argv: list, speed: str = FAST) -> str:
    """Run argv on a real pty and return everything it wrote."""
    buf = bytearray()
    previous = os.environ.get("TT_UI_DEMO_SPEED")
    os.environ["TT_UI_DEMO_SPEED"] = speed

    def read(fd: int) -> bytes:
        data = os.read(fd, 4096)
        buf.extend(data)
        return data

    try:
        pty.spawn([sys.executable, *argv], read)
    finally:
        if previous is None:
            os.environ.pop("TT_UI_DEMO_SPEED", None)
        else:
            os.environ["TT_UI_DEMO_SPEED"] = previous
    return bytes(buf).decode("utf-8", "replace")


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def run_piped(argv: list, speed: str = FAST) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "TT_UI_DEMO_SPEED": speed},
    )


# -- non-TTY -------------------------------------------------------------------
def test_piped_output_has_zero_escape_codes():
    """The hard rule: a pipe or a CI log must never see cursor motion or colour."""
    result = run_piped([str(DEMO)])
    assert result.returncode == 0
    assert "\x1b" not in result.stdout
    assert "\x1b" not in result.stderr


def test_piped_output_is_readable_and_collapses_each_step_once():
    result = run_piped([str(DEMO)])
    combined = result.stdout + result.stderr
    assert combined.count("Installing tt-smi 3.0.30") == 1
    assert "✓ Installing tt-smi 3.0.30" in combined
    assert "○ Installing tt-topology 1.2.0" in combined
    # No spinner leftovers and no phase chrome when piped.
    assert not any(frame in combined for frame in SPINNER_FRAMES)
    assert "Phase 1/3" not in combined


def test_piped_body_lines_have_no_trailing_whitespace():
    result = run_piped([str(DEMO)])
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("│") or line.startswith("╭") or line.startswith("╰"):
            continue  # panel rows are padded to the border by design
        assert line == line.rstrip(), repr(line)


def test_verbose_unfolds_detail_that_a_normal_run_hides():
    normal = run_piped([str(DEMO)])
    verbose = run_piped([str(DEMO), "-v"])
    note = "tt-luwen 0.7.1 is optional"
    assert note not in normal.stdout + normal.stderr
    assert note in verbose.stdout + verbose.stderr


def test_failure_path_renders_a_diagnosis_card_not_a_log_dump():
    result = run_piped([str(DEMO), "--fail"])
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "the firmware bundle didn't verify" in combined
    assert "Try:" in combined
    assert "tt update --refresh" in combined
    # The consequence is named: the reader learns whether the run continued.
    assert "left unchanged" in combined
    # Evidence is one line, not a tail of the log.
    assert combined.count("ERROR:") == 1


@pytest.mark.parametrize("columns", ["40", "80", "120"])
def test_renders_at_several_terminal_widths(columns):
    result = subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "COLUMNS": columns,
            "PYTHONIOENCODING": "utf-8",
            "TT_UI_DEMO_SPEED": FAST,
        },
    )
    assert result.returncode == 0
    assert "tt is up to date" in result.stdout + result.stderr


# -- real terminal -------------------------------------------------------------
def test_spinner_advances_through_its_frames():
    # Real pace: the ticker runs at 10fps, so a scaled-down tour would finish
    # before it had a chance to cycle.
    text = run_in_pty([str(DEMO)], speed=REAL)
    seen = {frame for frame in SPINNER_FRAMES if frame in text}
    assert len(seen) >= 4, f"spinner barely moved: {seen}"


def test_live_row_is_erased_before_a_result_is_printed():
    """`\\r\\x1b[2K` rewrites the whole row, so a stray write self-heals and the
    result line never lands on top of a spinner."""
    text = run_in_pty([str(DEMO)])
    assert text.count("\r\x1b[2K") > 5


def test_terminal_is_left_clean_on_exit():
    text = run_in_pty([str(DEMO)])
    assert "\x1b[?25h" in text, "cursor was not restored"
    # We deliberately do not install a DECSTBM scroll region; if that changes,
    # every exit path must reset it with \x1b[r.
    assert "\x1b[3;" not in text


def test_stepper_shows_progress_and_finishes_all_green():
    text = plain(run_in_pty([str(DEMO)]))
    assert "◉" in text, "no active phase marker while running"
    assert "✓ Checks ── ✓ Tools ── ✓ System" in text


def test_phases_collapse_with_a_timing_on_a_real_terminal():
    text = plain(run_in_pty([str(DEMO)], speed=REAL))
    assert len(re.findall(r"Phase \d/3", text)) == 3
    assert re.search(r"Phase 1/3 · Checks\s+\d+\.\d+s", text)


def test_steps_report_elapsed_time_past_the_threshold():
    text = plain(run_in_pty([str(DEMO)], speed=REAL))
    # "Fetching golden versions" sleeps 1.2s, so it must carry a duration.
    assert re.search(r"Fetching golden versions\s+v1\.0\.0\s+1\.\ds", text)
    # The 0.3s skipped step never gets one.
    assert re.search(r"○ Installing tt-topology 1\.2\.0\s+already up to date", text)


def test_activity_row_shows_an_exact_denominator_and_a_byte_counter():
    text = plain(run_in_pty([str(DEMO)]))
    assert re.search(r"\d+/24 packages · [\d.]+ MB", text)
    assert "▕" in text and "█" in text
