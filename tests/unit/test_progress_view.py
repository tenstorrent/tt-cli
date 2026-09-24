# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The checklist as it renders: live on a terminal, plain lines everywhere else."""

import io
import time

import pytest
from rich.console import Console

from tenstorrent.output import OutputManager
from tenstorrent.progress import Checklist, format_bytes, format_duration, ready_panel
from tenstorrent.progress.view import _SPINNER as SPINNER_FRAMES


def make_output(*, terminal=False, **flags):
    output = OutputManager(**flags)
    output.status_console = Console(file=io.StringIO(), force_terminal=terminal, width=100)
    return output


def rendered(output):
    return output.status_console.file.getvalue()


def test_a_piped_run_records_each_row_once_instead_of_redrawing():
    output = make_output()
    with Checklist(output) as view:
        view.instant("host ready", "docker 28.1")
        view.begin("pulling the container image")
        view.progress(12, 34)
        view.done("image ready", "vllm:0.22.0")
    text = rendered(output)
    assert "✓ host ready" in text
    assert "✓ image ready" in text
    assert "vllm:0.22.0" in text
    assert text.count("image ready") == 1  # settled once, never repainted


def test_quiet_and_json_runs_print_no_checklist_at_all():
    """The status channel is where progress lives; --json must leave stdout and
    stderr alone for the payload and the error panel."""
    for flags in ({"quiet": True}, {"json_mode": True}):
        output = make_output(**flags)
        with Checklist(output) as view:
            view.begin("pulling the container image")
            view.done()
        assert rendered(output) == ""


def test_an_unfinished_row_settles_when_the_block_ends():
    output = make_output()
    with Checklist(output) as view:
        view.begin("warming up the model")
    assert "✓ warming up the model" in rendered(output)


def test_an_interrupted_wait_is_marked_stopped_not_failed():
    output = make_output()
    with pytest.raises(KeyboardInterrupt):
        with Checklist(output) as view:
            view.begin("waiting for the model server")
            raise KeyboardInterrupt
    assert "○ waiting for the model server" in rendered(output)
    assert "interrupted" in rendered(output)


def test_a_placeholder_row_leaves_no_tick_for_work_that_never_happened():
    """"waiting for the model server" only exists to keep the spinner honest
    until something reports; a ✓ for it would claim a step that never ran."""
    output = make_output()
    with Checklist(output) as view:
        view.begin("waiting for the model server", placeholder=True)
        view.begin("opening the Tenstorrent device")
        view.done("Tenstorrent device opened")
    text = rendered(output)
    assert "waiting for the model server" not in text
    assert "✓ Tenstorrent device opened" in text


def test_the_live_block_keeps_moving_while_a_step_reports_nothing():
    """Weights download for twenty minutes without a word. Live re-renders the
    object it was handed, so a pre-built frame repaints identically forever —
    the spinner and the elapsed clock have to be recomputed on every tick, or
    the whole view reads as hung exactly when it matters most."""
    output = make_output(terminal=True)
    with Checklist(output) as view:
        view.begin("downloading weights")
        time.sleep(0.4)  # no state change whatsoever
    painted = {char for char in rendered(output) if char in SPINNER_FRAMES}
    assert len(painted) > 1, "the spinner never advanced"


def test_a_live_run_repaints_one_block_rather_than_appending_rows():
    output = make_output(terminal=True)
    with Checklist(output) as view:
        view.begin("loading weights")
        view.progress(16, 64)
        view.done("weights loaded")
    text = rendered(output)
    assert "weights loaded" in text
    assert "\x1b[" in text  # cursor control, i.e. a repainted block


@pytest.mark.parametrize(
    "done, total, is_bytes, expected",
    [
        (16, 64, False, "16/64 · 25%"),
        (1.68e9, 4.98e9, True, "1.68 GB / 4.98 GB · 34%"),
        # The denominator grows as docker reveals layers, so the ratio can
        # briefly exceed 1; the bar clamps rather than overflowing.
        (5, 4, False, "5/4 · 100%"),
    ],
)
def test_progress_reads_as_counts_or_sizes(done, total, is_bytes, expected):
    from tenstorrent.progress.view import _Row

    row = _Row("x", done=done, total=total, is_bytes=is_bytes)
    assert expected in row.progress_text()


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0 B"), (999, "999 B"), (1500, "1.50 kB"), (15_900_000_000, "15.9 GB")],
)
def test_sizes_are_decimal_to_match_what_hugging_face_reports(value, expected):
    assert format_bytes(value) == expected


@pytest.mark.parametrize(
    "seconds, expected", [(42, "42s"), (64, "1m 04s"), (4020, "1h 07m")]
)
def test_durations_read_in_the_unit_that_fits(seconds, expected):
    assert format_duration(seconds) == expected


def test_the_ready_card_names_the_endpoint_and_the_next_command():
    console = Console(file=io.StringIO(), width=100)
    console.print(
        ready_panel(
            "Llama-3.1-8B-Instruct ready",
            [("endpoint", "http://127.0.0.1:20000/v1")],
            footer="tt model logs Llama-3.1-8B-Instruct --follow",
        )
    )
    text = console.file.getvalue()
    assert "http://127.0.0.1:20000/v1" in text
    assert "tt model logs" in text


def frame(view):
    console = Console(file=io.StringIO(), width=120, color_system=None)
    console.print(view._render())
    return console.file.getvalue()


def test_the_footer_counts_the_steps():
    view = Checklist(make_output())
    view.plan(["pulling the image", "downloading weights", "opening the device", "ready"])
    view.begin("pulling the image")
    text = frame(view)
    assert "step 1 of 4" in text and "total" in text


def test_a_placeholder_is_not_counted_as_a_step():
    view = Checklist(make_output())
    view.plan(["pulling the image", "ready"])
    view.begin("preparing", placeholder=True)
    assert "step 1 of 2" in frame(view)


def test_a_step_the_run_skips_is_dropped_from_the_count():
    view = Checklist(make_output())
    view.plan(["pulling the image", "downloading weights", "opening the device", "ready"])
    view.begin("pulling the image")
    view.begin("opening the device")
    text = frame(view)
    assert "step 2 of 3" in text


def test_the_live_area_stays_one_line_however_many_steps_finish():
    """A frame taller than the terminal cannot be redrawn in place and leaves a
    copy of itself per refresh, so finished rows are printed, not redrawn."""
    output = make_output(terminal=True)
    with Checklist(output) as view:
        view.plan([f"step {i}" for i in range(30)])
        for i in range(30):
            view.begin(f"step {i}")
        assert len(frame(view).splitlines()) == 1
    text = rendered(output)
    assert "Ctrl-C stops watching" in text
    assert text.count("step 0 ") >= 1 and "step 28" in text


def test_nothing_is_left_live_once_the_checklist_closes():
    view = Checklist(make_output())
    view.plan(["pulling the image", "ready"])
    view.begin("pulling the image")
    view.close()
    assert frame(view).strip() == ""


def test_the_bar_moves_in_eighths():
    view = Checklist(make_output())
    view.begin("downloading weights")
    view.progress(1, 96)
    assert "▕▏" in frame(view)
    view.progress(3, 64)
    assert "▕▌" in frame(view)


def test_a_settled_row_keeps_its_result_not_a_stale_bar():
    view = Checklist(make_output())
    view.begin("warming up the model")
    view.progress(2, 444)
    view.done("model warmed up")
    assert "▕" not in frame(view) and "2/444" not in frame(view)


def test_the_live_line_stays_short_enough_not_to_rewrap_on_a_resize():
    view = Checklist(make_output(terminal=True))
    view.plan(["a", "b"])
    view.begin("weights Qwen/Qwen-Image-2.1@790b3a1 into a very long label indeed")
    view.progress(3.2e9, 16e9, is_bytes=True)
    line = frame(view).rstrip("\n")
    assert "\n" not in line and len(line) <= 80


def test_a_line_the_terminal_rewrapped_on_a_resize_is_cleared_first():
    output = make_output(terminal=True)
    view = Checklist(output)
    view.begin("opening the Tenstorrent device")
    view._drawn = 78
    output.status_console.width = 40
    view._render()
    assert rendered(output).endswith("\r\x1b[2K\x1b[1A\x1b[2K")
    before = rendered(output)
    view._render()
    assert rendered(output) == before  # once per resize, not per frame
