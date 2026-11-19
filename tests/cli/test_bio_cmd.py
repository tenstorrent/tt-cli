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
def handoff(monkeypatch):
    """Capture the exec_tty hand-off instead of replacing the test process.

    autouse, and deliberately so: `tt bio` ends in os.execvpe, so a test that
    reaches the hand-off without this replaces the *pytest* process and the run
    ends mid-suite with no report. Catching it here makes that failure mode
    impossible rather than a trap for the next test added to this file.
    """
    recorded: dict = {}

    def fake_exec(file, argv, env):
        recorded["argv"] = argv
        recorded["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_exec)
    return recorded


def _chips(monkeypatch, tmp_path, count: int) -> None:
    """Present `count` chips under a redirected /dev/tenstorrent."""
    root = tmp_path / "dev-tenstorrent"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / str(index)).write_text("")
    monkeypatch.setenv("TT_DEVICE_ROOT", str(root))


def _container_on(chip: int) -> dict:
    """An inspect record for a container holding one chip, in the shape chips.py reads."""
    return {
        "Id": f"{chip:012d}",
        "Name": f"/tt-model-demo-{chip}",
        "State": {"Running": True, "Status": "running"},
        "Config": {"Image": "tt-model/demo:1", "Labels": {}},
        "HostConfig": {
            "IpcMode": "host",
            "Devices": [{"PathOnHost": f"/dev/tenstorrent/{chip}"}],
        },
        "Mounts": [],
    }


@pytest.fixture
def docker_bin(monkeypatch, fake_bin):
    """Point `tt bio`'s runtime lookup at tests/fakes/bin/docker and seed it.

    The runtime is found with shutil.which inside the command module, so the patch
    goes there rather than on PATH — same approach as the `fake_docker` fixture, and
    it leaves the interpreter the fake's own shebang needs reachable.
    """
    monkeypatch.setattr(
        "tenstorrent.commands.bio.shutil.which",
        lambda name: str(fake_bin / "docker") if name == "docker" else None,
    )

    def seed(*records: dict) -> None:
        monkeypatch.setenv("FAKE_DOCKER_CONTAINERS", json.dumps(list(records)))

    seed()
    return seed


@pytest.fixture
def no_runtime(monkeypatch):
    """A host with neither docker nor podman."""
    monkeypatch.setattr("tenstorrent.commands.bio.shutil.which", lambda name: None)


# -- forwarding -----------------------------------------------------------------------
def test_unknown_flags_reach_tt_bio(runner, bio_bin, handoff, docker_bin, fake_bin):
    """A tt-bio option tt has never heard of must still arrive, verbatim and in order."""
    result = runner.invoke(
        app, ["bio", "predict", "prot.yaml", "--model", "boltz2", "--affinity_mw_correction"]
    )
    assert result.exit_code == 0
    argv = handoff["argv"]
    assert argv[0] == str(fake_bin / "tt-bio")
    assert argv[1] == "predict"
    assert argv[2:6] == ["prot.yaml", "--model", "boltz2", "--affinity_mw_correction"]


def test_run_forwards_an_unlisted_subcommand(runner, bio_bin, handoff, docker_bin, fake_bin):
    """`tt bio run` is the escape hatch for commands this tt release does not name."""
    result = runner.invoke(app, ["bio", "run", "evaluate", "--metrics_tags", "design"])
    assert result.exit_code == 0
    assert handoff["argv"][1:] == ["evaluate", "--metrics_tags", "design"]


def test_run_without_a_command_shows_help(runner, bio_bin, docker_bin):
    """The repo's contract for an arg-requiring command: help, exit 0, like -h."""
    result = runner.invoke(app, ["bio", "run"])
    assert result.exit_code == 0
    assert "Usage" in result.output


# -- the chip grant -------------------------------------------------------------------
def test_free_chips_are_granted_and_passed(
    runner, bio_bin, handoff, docker_bin, monkeypatch, tmp_path
):
    """With nothing running, a fold is granted every chip — as a flag and as the env
    grant tt-bio enforces at its own device open."""
    _chips(monkeypatch, tmp_path, 4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert handoff["argv"][-2:] == ["--devices", "0,1,2,3"]
    assert handoff["env"][GRANT_ENV] == "0,1,2,3"


def test_a_chip_held_by_a_served_model_is_withheld(
    runner, bio_bin, handoff, docker_bin, monkeypatch, tmp_path
):
    """The case tt exists for: tt-bio's lease cannot see a container, so tt must."""
    _chips(monkeypatch, tmp_path, 4)
    docker_bin(_container_on(2))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert handoff["argv"][-2:] == ["--devices", "0,1,3"]
    assert handoff["env"][GRANT_ENV] == "0,1,3"


def test_every_chip_busy_stops_the_run_with_the_way_out(
    runner, bio_bin, docker_bin, monkeypatch, tmp_path
):
    _chips(monkeypatch, tmp_path, 2)
    docker_bin(_container_on(0), _container_on(1))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "tt model ps" in result.output


def test_asking_for_a_busy_chip_is_refused_by_name(
    runner, bio_bin, docker_bin, monkeypatch, tmp_path
):
    """An explicit --devices is checked, not obeyed: the collision is the point."""
    _chips(monkeypatch, tmp_path, 4)
    docker_bin(_container_on(2))
    result = runner.invoke(app, ["bio", "predict", "prot.yaml", "--devices", "1,2"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "2" in result.output


@pytest.mark.parametrize("spelling", (["--devices", "0,1"], ["--devices=0,1"], ["--device_ids", "0,1"]))
def test_an_explicit_free_choice_is_honoured_and_not_widened(
    runner, bio_bin, handoff, docker_bin, monkeypatch, tmp_path, spelling
):
    """Both `--flag value` and `--flag=value` are read, and the grant never exceeds
    what the user asked for."""
    _chips(monkeypatch, tmp_path, 4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml", *spelling])
    assert result.exit_code == 0
    assert handoff["env"][GRANT_ENV] == "0,1"
    assert handoff["argv"].count("--devices") <= 1


def test_a_device_free_command_is_not_given_a_grant(
    runner, bio_bin, handoff, docker_bin, monkeypatch, tmp_path
):
    """`weights` never opens a card, so it must not be constrained to one."""
    _chips(monkeypatch, tmp_path, 4)
    docker_bin(_container_on(0), _container_on(1), _container_on(2), _container_on(3))
    result = runner.invoke(app, ["bio", "weights", "--download"])
    assert result.exit_code == 0
    assert GRANT_ENV not in handoff["env"]
    assert "--devices" not in handoff["argv"]


def test_no_chips_present_leaves_tt_bio_unconstrained(
    runner, bio_bin, handoff, docker_bin, monkeypatch, tmp_path
):
    """A device-free host is not an error here: tt-bio still runs its CPU paths."""
    _chips(monkeypatch, tmp_path, 0)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert GRANT_ENV not in handoff["env"]


def test_an_unreadable_runtime_grants_nothing_rather_than_everything(
    runner, bio_bin, handoff, no_runtime, monkeypatch, tmp_path
):
    """With no container runtime tt cannot know who holds what, so it must not
    pretend: the run is forwarded unconstrained instead of handed every chip."""
    _chips(monkeypatch, tmp_path, 4)
    result = runner.invoke(app, ["bio", "predict", "prot.yaml"])
    assert result.exit_code == 0
    assert GRANT_ENV not in handoff["env"]
    assert "--devices" not in handoff["argv"]


# -- status ---------------------------------------------------------------------------
def test_status_json_reports_free_and_held(
    runner, bio_bin, docker_bin, monkeypatch, tmp_path
):
    _chips(monkeypatch, tmp_path, 4)
    docker_bin(_container_on(3))
    result = runner.invoke(app, ["bio", "status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["chips"]["free"] == [0, 1, 2]
    assert payload["chips"]["held"] == [3]
    assert payload["tool"]["installed"] is True


def test_bio_is_listed_in_help(runner):
    assert " bio " in runner.invoke(app, ["--help"]).output
