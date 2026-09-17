# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Ui behaviour that must hold without a TTY: stream discipline, mode gating,
zero escape codes when piped, and a teardown that cannot leave the terminal dirty.

The animated paths need a real PTY and live in tests/cli/test_cli_output.py.
"""

import pytest

from tenstorrent.output import OutputManager
from tenstorrent.ui.console import _ACTIVE_LIVE, Ui


def test_ui_is_lazy_and_cached_on_the_output_manager():
    out = OutputManager()
    assert "_ui" not in out.__dict__
    assert out.ui is out.ui
    assert isinstance(out.ui, Ui)


def test_release_ui_does_not_construct_a_ui_that_never_existed():
    out = OutputManager()
    out.release_ui()
    assert "_ui" not in out.__dict__


def test_steps_render_on_stderr_and_never_on_stdout(capsys):
    out = OutputManager()
    with out.ui.step("Fetching golden versions") as step:
        step.detail("v1.0.0")
    captured = capsys.readouterr()
    assert "Fetching golden versions" in captured.err
    assert "v1.0.0" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("mode", [{"json_mode": True}, {"quiet": True}])
def test_ui_is_completely_silent_in_json_and_quiet_modes(capsys, mode):
    """The one-document --json contract: the UI layer must emit nothing at all."""
    out = OutputManager(**mode)
    assert out.ui.enabled is False
    out.ui.register_phases(["Checks", "Tools"])
    with out.ui.phase("Checks"):
        with out.ui.step("Doing a thing") as step:
            step.detail("detail")
        out.ui.note("a note")
        out.ui.milestone("a milestone")
        out.ui.alert("an alert")
    with out.ui.activity("Working") as row:
        row.set("still working")
    out.ui.final_stepper()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_handles_still_work_while_disabled_so_call_sites_stay_branch_free():
    out = OutputManager(quiet=True)
    with out.ui.step("Silent thing") as step:
        step.detail("x")
        step.skip("nothing to do")
    assert out.ui.timings.to_dict()["steps"][0]["status"] == "skipped"


def test_piped_output_contains_no_escape_codes(capsys):
    out = OutputManager()
    out.ui.register_phases(["Checks", "Tools"])
    with out.ui.phase("Checks"):
        with out.ui.step("A step") as step:
            step.detail("v1")
        out.ui.note("a note about state")
        out.ui.milestone("something finished")
        out.ui.alert("something needs attention")
    with out.ui.activity("Working") as row:
        row.set("label change")
    captured = capsys.readouterr()
    assert "\x1b" not in captured.err
    assert "\x1b" not in captured.out


def test_piped_step_label_appears_exactly_once(capsys):
    """Without cursor motion there is no pre-line to overwrite, so printing
    `label…` first would leave the label in the output twice."""
    out = OutputManager()
    with out.ui.step("Installing tt-smi"):
        pass
    err = capsys.readouterr().err
    assert err.count("Installing tt-smi") == 1
    assert "…" not in err


def test_piped_steps_omit_the_elapsed_suffix_so_output_is_deterministic(capsys):
    out = OutputManager()
    with out.ui.step("Slow thing") as step:
        step.start -= 5.0  # pretend it took five seconds
    err = capsys.readouterr().err
    assert "Slow thing" in err
    assert "5.0s" not in err
    # The duration is still recorded for --json, where machines want it.
    assert out.ui.timings.to_dict()["steps"][0]["seconds"] >= 5.0


def test_piped_body_lines_have_no_trailing_whitespace(capsys):
    """Padding would expand to the console width and pad every line."""
    out = OutputManager()
    out.ui.note("a state worth explaining")
    out.ui.milestone("a milestone")
    for line in capsys.readouterr().err.splitlines():
        assert line == line.rstrip(), repr(line)


def test_phase_chrome_is_suppressed_when_piped_but_steps_survive(capsys):
    out = OutputManager()
    out.ui.register_phases(["Checks", "Tools"])
    with out.ui.phase("Checks"):
        with out.ui.step("Inner step"):
            pass
    err = capsys.readouterr().err
    assert "Inner step" in err
    assert "Phase 1/2" not in err
    assert "──" not in err


def test_a_failing_step_is_marked_and_the_exception_propagates(capsys):
    out = OutputManager()
    with pytest.raises(RuntimeError):
        with out.ui.step("Risky thing"):
            raise RuntimeError("boom")
    err = capsys.readouterr().err
    assert "✗ Risky thing" in err
    assert out.ui.timings.to_dict()["steps"][0]["status"] == "failed"


def test_a_failing_phase_marks_the_stepper_and_propagates():
    out = OutputManager()
    out.ui.register_phases(["Checks"])
    with pytest.raises(RuntimeError):
        with out.ui.phase("Checks"):
            raise RuntimeError("boom")
    assert out.ui._phases[0]["status"] == "failed"
    assert out.ui.timings.to_dict()["phases"][0]["status"] == "failed"


def test_show_detail_tracks_phase_membership():
    out = OutputManager()
    out.ui.register_phases(["Checks"])
    assert out.ui.show_detail() is True  # outside any phase
    with out.ui.phase("Checks"):
        assert out.ui.in_phase() is True
        assert out.ui.show_detail() is False
    assert out.ui.show_detail() is True


def test_verbose_unfolds_detail_inside_a_phase():
    out = OutputManager(verbose=True)
    out.ui.register_phases(["Checks"])
    with out.ui.phase("Checks"):
        assert out.ui.show_detail() is True


def test_skip_phase_keeps_the_count_and_says_why(capsys):
    out = OutputManager()
    out.ui.register_phases(["Checks", "Tools", "System"])
    out.ui.skip_phase("System", "tt-installer needs the network")
    assert len(out.ui._phases) == 3
    assert out.ui._phases[2]["status"] == "skipped"
    assert "tt-installer needs the network" in capsys.readouterr().err


def test_rename_phase_changes_the_title_but_never_the_count():
    out = OutputManager()
    out.ui.register_phases(["Checks", "Pull", "Serve"])
    out.ui.rename_phase(1, "Build")
    assert [p["title"] for p in out.ui._phases] == ["Checks", "Build", "Serve"]
    out.ui.rename_phase(99, "Nope")  # out of range is a no-op
    assert len(out.ui._phases) == 3


def test_stepper_line_marks_done_current_and_pending():
    out = OutputManager()
    out.ui.register_phases(["Checks", "Tools", "System"])
    out.ui._phases[0]["status"] = "done"
    out.ui._phases[1]["status"] = "active"
    plain = out.ui.stepper_line().plain
    assert plain == "✓ Checks ── ◉ Tools ── ○ System"


def test_release_is_idempotent_and_leaves_no_live_display():
    out = OutputManager()
    out.ui.release()
    out.ui.release()
    assert _ACTIVE_LIVE == []


def test_prompting_suspends_motion():
    out = OutputManager()
    with out.ui.prompting():
        assert out.ui.live is False
