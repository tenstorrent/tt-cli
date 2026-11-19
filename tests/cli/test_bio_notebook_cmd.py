# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt bio notebook` — the seeded notebook, and handing the terminal to jupyter."""

from __future__ import annotations

import json

import pytest

from tenstorrent.cli import app
from tenstorrent.commands.bio import GRANT_ENV
from tenstorrent.commands.bio_notebook import (
    STARTER_NAME,
    starter_notebook,
    write_starter,
)
from tenstorrent.errors import ExitCode


@pytest.fixture(autouse=True)
def _handoff(bio_handoff):
    """autouse so no test in this file can reach a real os.execvpe."""
    return bio_handoff


@pytest.fixture
def jupyter_beside_tt_bio(bio_bin, fake_bin, monkeypatch, tmp_path):
    """tt-bio resolves to a venv that also has `jupyter`, as `--with` produces."""
    venv_bin = tmp_path / "tool-venv" / "bin"
    venv_bin.mkdir(parents=True)
    tt_bio = venv_bin / "tt-bio"
    tt_bio.write_text((fake_bin / "tt-bio").read_text())
    tt_bio.chmod(0o755)
    jupyter = venv_bin / "jupyter"
    jupyter.write_text("#!/bin/sh\nexit 0\n")
    jupyter.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_TT_BIO", str(tt_bio))
    return venv_bin


# -- the seeded notebook ---------------------------------------------------------------
def test_the_starter_notebook_is_valid_nbformat():
    doc = starter_notebook([0, 1])
    assert doc["nbformat"] == 4
    assert doc["metadata"]["kernelspec"]["name"] == "python3"
    for cell in doc["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        # nbformat 4.5+ requires a unique cell id and warns that its absence becomes
        # a hard error; without this the seeded file rots into an invalid notebook.
        assert cell["id"]
        if cell["cell_type"] == "code":
            # nbformat requires these on a code cell; a notebook missing them opens
            # as "not trusted" or fails to load outright.
            assert "outputs" in cell and "execution_count" in cell
    ids = [c["id"] for c in doc["cells"]]
    assert len(set(ids)) == len(ids), "cell ids must be unique"


def test_the_granted_chips_are_named_in_the_notebook():
    text = json.dumps(starter_notebook([2, 3]))
    assert "chip(s) 2, 3" in text


def test_without_a_grant_the_notebook_says_so_instead_of_lying():
    text = json.dumps(starter_notebook(None))
    assert "No chip grant" in text
    assert "chip(s)" not in text.split("No chip grant")[0][-200:]


def test_the_code_cells_are_syntactically_valid_python():
    """A seeded notebook whose cells do not compile is worse than no notebook."""
    for cell in starter_notebook([0])["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<cell>", "exec")


def test_write_starter_never_clobbers_an_edited_notebook(tmp_path):
    path = tmp_path / STARTER_NAME
    assert write_starter(path, [0]) is True
    path.write_text('{"cells": ["my own work"]}')
    assert write_starter(path, [0]) is False
    assert "my own work" in path.read_text()


# -- the command -----------------------------------------------------------------------
def test_it_seeds_then_hands_over_to_jupyter(
    runner, bio_handoff, bio_docker, bio_chips, jupyter_beside_tt_bio, tmp_path
):
    bio_chips(4)
    workdir = tmp_path / "nb"
    result = runner.invoke(app, ["bio", "notebook", "--dir", str(workdir)])
    assert result.exit_code == 0
    assert (workdir / STARTER_NAME).exists()
    argv = bio_handoff["argv"]
    assert argv[0].endswith("jupyter")
    assert argv[1] == "lab"
    assert f"--notebook-dir={workdir}" in argv


def test_the_notebook_server_inherits_the_chip_grant(
    runner, bio_handoff, bio_docker, bio_chips, bio_container_on, jupyter_beside_tt_bio, tmp_path
):
    """The grant has to reach jupyter's environment, or every fold run from a cell
    would be unconstrained — the exact collision `tt bio` exists to prevent."""
    bio_chips(4)
    bio_docker(bio_container_on(0), bio_container_on(1))
    result = runner.invoke(app, ["bio", "notebook", "--dir", str(tmp_path / "nb")])
    assert result.exit_code == 0
    assert bio_handoff["env"][GRANT_ENV] == "2,3"


def test_a_busy_box_still_opens_the_notebook_with_a_warning(
    runner, bio_handoff, bio_docker, bio_chips, bio_container_on, jupyter_beside_tt_bio, tmp_path
):
    """A notebook is a workspace, not one job: it opens so you can read it and edit a
    sequence, and a chip may free up while it is open. `predict` still refuses."""
    bio_chips(2)
    bio_docker(bio_container_on(0), bio_container_on(1))
    result = runner.invoke(app, ["bio", "notebook", "--dir", str(tmp_path / "nb")])
    assert result.exit_code == 0
    assert (tmp_path / "nb" / STARTER_NAME).exists()
    assert "Every chip is in use" in result.output
    # No grant at all: tt-bio reads an empty one as "take the whole box".
    assert GRANT_ENV not in bio_handoff["env"]


def test_predict_still_refuses_on_a_busy_box(
    runner, bio_bin, bio_docker, bio_chips, bio_container_on
):
    bio_chips(2)
    bio_docker(bio_container_on(0), bio_container_on(1))
    result = runner.invoke(app, ["bio", "predict", "x.yaml"])
    assert result.exit_code == ExitCode.TOOL_FAILED


def test_a_missing_jupyter_triggers_one_reinstall(
    runner, bio_bin, bio_docker, bio_chips, monkeypatch, tmp_path
):
    """tt-bio from before the notebook extras were pinned: the version is unchanged,
    so ensure() will not re-resolve it. The command must install() rather than send
    the user off to a command that would not fix it either."""
    bio_chips(4)
    calls: list = []

    def fake_install(self, spec, *, offline=False):
        calls.append(spec.name)
        raise SystemExit(99)  # stop here; we only care that it was reached

    monkeypatch.setattr("tenstorrent.tools.registry.ToolRegistry.install", fake_install)
    runner.invoke(app, ["bio", "notebook", "--dir", str(tmp_path / "nb")])
    assert calls == ["tt-bio"]


def test_a_reinstall_that_still_lacks_jupyter_is_a_clear_error(
    runner, bio_bin, bio_docker, bio_chips, monkeypatch, tmp_path
):
    """If uv reports success and jupyter still is not there, say so plainly rather
    than exec'ing a path that does not exist."""
    bio_chips(4)
    from tenstorrent.tools.installers import InstallResult

    monkeypatch.setattr(
        "tenstorrent.tools.registry.ToolRegistry.install",
        lambda self, spec, *, offline=False: InstallResult(
            spec.name, spec.golden_version, tmp_path / "nowhere" / "tt-bio"
        ),
    )
    result = runner.invoke(app, ["bio", "notebook", "--dir", str(tmp_path / "nb")])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "tt report issue" in result.output


def test_json_reports_what_it_would_open(
    runner, bio_docker, bio_chips, bio_container_on, jupyter_beside_tt_bio, tmp_path
):
    bio_chips(4)
    bio_docker(bio_container_on(3))
    workdir = tmp_path / "nb"
    result = runner.invoke(app, ["bio", "notebook", "--dir", str(workdir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["chips"] == [0, 1, 2]
    assert payload["created"] is True
    assert payload["notebook"].endswith(STARTER_NAME)


def test_no_browser_is_forwarded(
    runner, bio_handoff, bio_docker, bio_chips, jupyter_beside_tt_bio, tmp_path
):
    bio_chips(4)
    result = runner.invoke(
        app, ["bio", "notebook", "--dir", str(tmp_path / "nb"), "--no-browser"]
    )
    assert result.exit_code == 0
    assert "--no-browser" in bio_handoff["argv"]
