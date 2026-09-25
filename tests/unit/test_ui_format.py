# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure formatting helpers — no terminal, no state, no hardware."""

from tenstorrent.ui.format import (
    elide,
    fmt_bytes,
    fmt_clock,
    fmt_duration,
    progress_bar,
    show_detail,
)
from tenstorrent.ui.theme import ELAPSED_THRESHOLD_S


def test_fmt_duration_switches_units_at_the_right_boundaries():
    assert fmt_duration(0.42) == "420ms"
    assert fmt_duration(0.999) == "999ms"
    assert fmt_duration(1.0) == "1.0s"
    assert fmt_duration(5.0) == "5.0s"
    assert fmt_duration(59.9) == "59.9s"
    assert fmt_duration(60.0) == "1m 0s"
    # The shape the ready card reports a real run in.
    assert fmt_duration(214.3) == "3m 34s"


def test_elapsed_threshold_is_below_a_second_in_both_directions():
    """A step the user never waited for stays clean; a slow one earns its suffix."""
    assert ELAPSED_THRESHOLD_S == 0.8
    assert 0.79 < ELAPSED_THRESHOLD_S
    assert 0.81 > ELAPSED_THRESHOLD_S


def test_fmt_clock_counts_up_in_minutes_and_seconds():
    assert fmt_clock(0) == "0:00"
    assert fmt_clock(42) == "0:42"
    assert fmt_clock(727) == "12:07"


def test_fmt_bytes_uses_decimal_units_like_docker_and_uv():
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(1_500) == "1.5 kB"
    assert fmt_bytes(412_000_000) == "412.0 MB"
    assert fmt_bytes(2_400_000_000) == "2.4 GB"


def test_progress_bar_is_empty_when_the_total_is_unknown():
    """The anti-fake-percentage guard: no total, no bar. Callers show a counter."""
    assert progress_bar(3, 0) == ""
    assert progress_bar(3, -1) == ""


def test_progress_bar_fills_proportionally_and_clamps():
    assert progress_bar(0, 4, width=4) == "▕░░░░▏"
    assert progress_bar(2, 4, width=4) == "▕██░░▏"
    assert progress_bar(4, 4, width=4) == "▕████▏"
    # Overshoot can't run past the end of the bar.
    assert progress_bar(9, 4, width=4) == "▕████▏"


def test_show_detail_folds_only_inside_a_phase_on_a_normal_run():
    assert show_detail(verbose=False, in_phase=True) is False
    assert show_detail(verbose=True, in_phase=True) is True
    # Outside a phase there is no collapsed line to act as the confirmation.
    assert show_detail(verbose=False, in_phase=False) is True
    assert show_detail(verbose=True, in_phase=False) is True


def test_elide_keeps_one_line_so_evidence_cannot_become_a_log_viewer():
    assert elide("") == ""
    assert elide("  first\nsecond\nthird  ") == "first"
    long = "x" * 200
    assert len(elide(long, limit=40)) == 40
    assert elide(long, limit=40).endswith("…")


def test_tilde_shortens_a_path_under_home(monkeypatch, tmp_path):
    from tenstorrent.ui.format import tilde

    monkeypatch.setenv("HOME", str(tmp_path))
    assert tilde(tmp_path / "logs" / "x.log") == "~/logs/x.log"
    assert tilde("/etc/hosts") == "/etc/hosts"
