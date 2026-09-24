# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Stream parsers, driven by real captured tool output.

Fixtures in tests/fixtures/streams/ are verbatim captures — see the README there.
Mocked input would only prove the parser handles the wording we imagined.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tenstorrent.ui.parsers import (
    GitCloneProgress,
    UvPipProgress,
    parse_git_line,
    parse_size,
    parse_uv_line,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "streams"


def feed_all(parser, name):
    """Run a whole capture through a parser, collecting its milestones."""
    milestones = []
    for line in (FIXTURES / name).read_text().splitlines():
        got = parser.feed(line)
        if got:
            milestones.append(got)
    return milestones


# -- sizes ---------------------------------------------------------------------
def test_parse_size_handles_uvs_binary_units():
    """uv reports MiB; Docker and curl report MB. Both must land as bytes."""
    assert parse_size("16.0MiB") == pytest.approx(16.0 * 1024**2)
    assert parse_size("1.5 GiB") == pytest.approx(1.5 * 1024**3)
    assert parse_size("512B") == pytest.approx(512)
    assert parse_size("12.11MB") == pytest.approx(12.11e6)


def test_parse_size_rejects_nonsense():
    assert parse_size("") is None
    assert parse_size("lots") is None
    assert parse_size("MiB") is None


# -- uv ------------------------------------------------------------------------
def test_parse_uv_line_recognises_the_real_vocabulary():
    assert parse_uv_line("Resolved 9 packages in 71ms") == ("resolved", 9)
    assert parse_uv_line("Prepared 9 packages in 155ms") == ("prepared", 9)
    assert parse_uv_line("Installed 9 packages in 5ms") == ("installed", 9)
    assert parse_uv_line(" Downloaded numpy") == ("downloaded", "numpy")
    assert parse_uv_line(" + anyio==4.15.1") == ("added", ("anyio", "4.15.1"))
    assert parse_uv_line("Using CPython 3.10.21") == ("python", "CPython 3.10.21")
    kind, payload = parse_uv_line("Downloading numpy (16.0MiB)")
    assert kind == "downloading"
    assert payload[0] == "numpy"
    assert payload[1] == pytest.approx(16.0 * 1024**2)


def test_parse_uv_line_ignores_noise():
    assert parse_uv_line("") is None
    assert parse_uv_line("Activate with: source capvenv/bin/activate") is None


def test_uv_progress_over_a_real_cold_install():
    parser = UvPipProgress("Installing tt-inference-server dependencies")
    milestones = feed_all(parser, "uv_pip_install.txt")
    assert parser.total == 9
    assert parser.added == 9
    assert parser.bytes == pytest.approx(16.0 * 1024**2)
    assert milestones == ["9 packages installed"]
    activity = parser.activity()
    assert "9/9 packages" in activity
    assert "16.8 MB" in activity  # 16.0 MiB, reported in decimal like every other size


def test_uv_progress_over_a_real_cached_install_has_no_bytes():
    """A warm cache prints no Downloading lines, so the counter must stay absent
    rather than showing a misleading 0 B."""
    parser = UvPipProgress()
    milestones = feed_all(parser, "uv_pip_cached.txt")
    assert parser.total == 13
    assert parser.bytes == 0
    assert milestones == ["13 packages installed"]
    assert "·" not in parser.activity()


def test_uv_activity_says_resolving_before_a_total_is_known():
    """No denominator yet, so no bar — the rule against invented percentages."""
    parser = UvPipProgress("Installing x")
    assert parser.activity() == "Installing x · resolving"
    assert "▕" not in parser.activity()


def test_uv_singular_package_reads_correctly():
    parser = UvPipProgress()
    assert parser.feed("Installed 1 package in 5ms") == "1 package installed"


# -- git -----------------------------------------------------------------------
def test_parse_git_line_recognises_remote_and_local_stages():
    assert parse_git_line("Cloning into 'gitcap'...") == ("cloning", "gitcap")
    assert parse_git_line("remote: Counting objects:  22% (37/164)") == (
        "stage",
        "Counting objects",
        37,
        164,
        None,
    )
    # Local stages carry no `remote:` prefix.
    assert parse_git_line("Resolving deltas: 100% (4/4), done.") == (
        "stage",
        "Resolving deltas",
        4,
        4,
        None,
    )


def test_parse_git_line_reads_a_total_without_a_fraction():
    assert parse_git_line("remote: Enumerating objects: 164, done.") == (
        "stage",
        "Enumerating objects",
        0,
        164,
        None,
    )


def test_parse_git_line_extracts_received_bytes():
    event = parse_git_line("Receiving objects:  57% (94/164), 1.20 MiB | 2.40 MiB/s")
    assert event[1] == "Receiving objects"
    assert (event[2], event[3]) == (94, 164)
    assert event[4] == pytest.approx(1.20 * 1024**2)


def test_parse_git_line_ignores_noise():
    assert parse_git_line("") is None
    assert parse_git_line("Note: switching to 'abc'.") is None


def test_git_progress_over_a_real_clone():
    parser = GitCloneProgress("Cloning tt-inference-server")
    milestones = feed_all(parser, "git_clone.txt")
    assert any("received 164 objects" in m for m in milestones)
    # One milestone per stage at most — not one per progress tick.
    assert len(milestones) <= len(("counting", "compressing", "receiving", "resolving"))


def test_git_progress_never_regresses_to_an_earlier_stage():
    """git interleaves remote and local progress, so a late `remote:` line must
    not drag the label back to counting once we're receiving."""
    parser = GitCloneProgress()
    parser.feed("Receiving objects:  50% (82/164)")
    assert parser.stage == "Receiving objects"
    parser.feed("remote: Counting objects: 100% (164/164)")
    assert parser.stage == "Receiving objects"


def test_git_received_bytes_keep_the_max_rather_than_accumulating():
    """Each line restates the running total; summing them would race away."""
    parser = GitCloneProgress()
    parser.feed("Receiving objects:  10% (16/164), 1.00 MiB | 2.00 MiB/s")
    parser.feed("Receiving objects:  50% (82/164), 4.00 MiB | 2.00 MiB/s")
    assert parser.bytes == pytest.approx(4.0 * 1024**2)


def test_git_activity_says_connecting_before_anything_arrives():
    parser = GitCloneProgress("Cloning x")
    assert parser.activity() == "Cloning x · connecting"


def test_uv_shows_no_bar_until_something_has_actually_completed():
    """uv prints its `+ name==version` list only after installing, so a bar would
    sit at 0/N for the whole run and then snap to full. Name the stage instead."""
    parser = UvPipProgress("Installing x")
    parser.feed("Resolved 12 packages in 71ms")
    activity = parser.activity()
    assert "▕" not in activity
    assert "0/12" not in activity
    assert "downloading 12 packages" in activity

    parser.feed("Downloading numpy (16.0MiB)")
    assert "16.8 MB" in parser.activity()

    # Once a real count exists, the bar earns its place.
    parser.feed(" Downloaded numpy")
    assert "▕" in parser.activity()
    assert "1/12 packages" in parser.activity()


def test_uv_names_the_installing_stage_once_prepared():
    parser = UvPipProgress("Installing x")
    parser.feed("Resolved 3 packages in 10ms")
    parser.feed("Prepared 3 packages in 20ms")
    assert "installing 3 packages" in parser.activity()
