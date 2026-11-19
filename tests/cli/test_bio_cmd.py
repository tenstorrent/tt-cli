# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt bio` — forwarding, and the chip grant that is the whole point of the wrapper."""

from __future__ import annotations

import json

import pytest

from tenstorrent.cli import app
from tenstorrent.commands.bio import GRANT_ENV
from tenstorrent.errors import ExitCode


@pytest.fixture(autouse=True)
def _handoff(bio_handoff):
    """autouse so no test in this file can reach a real os.execvpe."""
    return bio_handoff


# -- forwarding -----------------------------------------------------------------------
def test_unknown_flags_reach_tt_bio(runner, bio_bin, bio_handoff, bio_docker, fake_bin):
    """A tt-bio option tt has never heard of must still arrive, verbatim and in order."""
    result = runner.invoke(
        app, ["bio", "predict", "prot.yaml", "--model", "boltz2", "--affinity_mw_correction"]
    )
    assert result.exit_code == 0
    argv = bio_handoff["argv"]
    assert argv[0] == str(fake_bin / "tt-bio")
    assert argv[1] == "predict"
    assert argv[2:6] == ["prot.yaml", "--model", "boltz2", "--affinity_mw_correction"]


def test_run_forwards_an_unlisted_subcommand(runner, bio_bin, bio_handoff, bio_docker, fake_bin):
    """`tt bio run` is the escape hatch for commands this tt release does not name."""
    result = runner.invoke(app, ["bio", "run", "evaluate", "--metrics_tags", "design"])
    assert result.exit_code == 0
    assert bio_handoff["argv"][1:] == ["evaluate", "--metrics_tags", "design"]


def test_run_without_a_command_shows_help(runner, bio_bin, bio_docker):
    """The repo's contract for an arg-requiring command: help, exit 0, like -h."""
    result = runner.invoke(app, ["bio", "run"])
    assert result.exit_code == 0
    assert "Usage" in result.output


# -- the chip grant -------------------------------------------------------------------
def test_free_chips_are_granted_and_passed(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, bio_chips):
    """With nothing running, a fold is granted every chip — as a flag and as the env
    grant tt-bio enforces at its own device open."""
    bio_chips(4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert bio_handoff["argv"][-2:] == ["--devices", "0,1,2,3"]
    assert bio_handoff["env"][GRANT_ENV] == "0,1,2,3"


def test_a_chip_held_by_a_served_model_is_withheld(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, bio_chips, bio_container_on):
    """The case tt exists for: tt-bio's lease cannot see a container, so tt must."""
    bio_chips(4)
    bio_docker(bio_container_on(2))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert bio_handoff["argv"][-2:] == ["--devices", "0,1,3"]
    assert bio_handoff["env"][GRANT_ENV] == "0,1,3"


def test_a_chip_another_tt_bio_run_holds_is_withheld(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, tt_bio_lease, bio_chips, bio_container_on):
    """tt-bio's own lease counts too, so a second run is not granted a card the first
    one is folding on (it would only block behind it at the device open)."""
    bio_chips(4)
    bio_docker(bio_container_on(0))
    tt_bio_lease(1)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert bio_handoff["env"][GRANT_ENV] == "2,3"


def test_every_chip_busy_stops_the_run_with_the_way_out(
    runner, bio_bin, bio_docker, monkeypatch, tmp_path, bio_chips, bio_container_on):
    bio_chips(2)
    bio_docker(bio_container_on(0), bio_container_on(1))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "tt model ps" in result.output


def test_asking_for_a_busy_chip_is_refused_by_name(
    runner, bio_bin, bio_docker, monkeypatch, tmp_path, bio_chips, bio_container_on):
    """An explicit --devices is checked, not obeyed: the collision is the point."""
    bio_chips(4)
    bio_docker(bio_container_on(2))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml", "--devices", "1,2"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "2" in result.output


@pytest.mark.parametrize("spelling", (["--devices", "0,1"], ["--devices=0,1"], ["--device_ids", "0,1"]))
def test_an_explicit_free_choice_is_honoured_and_not_widened(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, spelling, bio_chips):
    """Both `--flag value` and `--flag=value` are read, and the grant never exceeds
    what the user asked for."""
    bio_chips(4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml", *spelling])
    assert result.exit_code == 0
    assert bio_handoff["env"][GRANT_ENV] == "0,1"
    assert bio_handoff["argv"].count("--devices") <= 1


def test_a_device_free_command_is_not_given_a_grant(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, bio_chips, bio_container_on):
    """`weights` never opens a card, so it must not be constrained to one."""
    bio_chips(4)
    bio_docker(bio_container_on(0), bio_container_on(1), bio_container_on(2), bio_container_on(3))
    result = runner.invoke(app, ["bio", "weights", "--download"])
    assert result.exit_code == 0
    assert GRANT_ENV not in bio_handoff["env"]
    assert "--devices" not in bio_handoff["argv"]


def test_no_chips_present_leaves_tt_bio_unconstrained(
    runner, bio_bin, bio_handoff, bio_docker, monkeypatch, tmp_path, bio_chips):
    """A device-free host is not an error here: tt-bio still runs its CPU paths."""
    bio_chips(0)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert GRANT_ENV not in bio_handoff["env"]


def test_an_unreadable_runtime_grants_nothing_rather_than_everything(
    runner, bio_bin, bio_handoff, bio_no_runtime, monkeypatch, tmp_path, bio_chips):
    """With no container runtime tt cannot know who holds what, so it must not
    pretend: the run is forwarded unconstrained instead of handed every chip."""
    bio_chips(4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert GRANT_ENV not in bio_handoff["env"]
    assert "--devices" not in bio_handoff["argv"]


# -- status ---------------------------------------------------------------------------
def test_status_json_reports_free_and_held(
    runner, bio_bin, bio_docker, monkeypatch, tmp_path, bio_chips, bio_container_on):
    bio_chips(4)
    bio_docker(bio_container_on(3))
    result = runner.invoke(app, ["bio", "status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["chips"]["free"] == [0, 1, 2]
    assert payload["chips"]["held"] == [3]
    assert payload["tool"]["installed"] is True


def test_status_counts_a_live_tt_bio_lease_as_held(
    runner, bio_bin, bio_docker, monkeypatch, tmp_path, tt_bio_lease, bio_chips):
    bio_chips(4)
    tt_bio_lease(2)
    payload = json.loads(runner.invoke(app, ["bio", "status", "--json"]).output)
    assert payload["chips"]["held"] == [2]
    assert payload["chips"]["free"] == [0, 1, 3]


def test_bio_is_listed_in_help(runner):
    assert " bio " in runner.invoke(app, ["--help"]).output
