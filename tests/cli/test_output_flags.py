# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The output flags are a per-leaf contract, so walk the command tree and enforce it.

`tt --json device status` and `tt device status --json` must both work, which means
every leaf re-declares the output flags. Typer's leaf parser rejects an unknown
short option, so a missing alias is a hard usage error rather than a cosmetic gap —
`tt update -v` was exactly that until these were added.
"""

from __future__ import annotations

from typer.main import get_command
from typer.testing import CliRunner

from tenstorrent.cli import app

runner = CliRunner()

# Leaves that legitimately have no output of ours to shape:
#   stubs raise UNSUPPORTED before printing anything, and the TUI hand-offs
#   replace this process with tt-smi's own full-screen UI.
NO_OUTPUT_OF_OURS = {
    "train",
    "compile",
    "report feedback",
    "device top",
    "smi",
}
# Leaves that emit data but have nothing worth suppressing separately.
NO_QUIET = {"self check-update", "self send-telemetry"}


def leaf_commands():
    """Every runnable leaf in the tree, as (dotted path, click command)."""
    found = []

    def walk(command, trail):
        # Duck-typed rather than isinstance(click.Group): Typer's vendored click
        # doesn't re-export Group at its top level, and the submodule paths it
        # does live at are not stable across Typer versions.
        children = getattr(command, "commands", None)
        if children:
            for name, child in children.items():
                walk(child, trail + [name])
            return
        found.append((" ".join(trail), command))

    walk(get_command(app), [])
    return found


def option_names(command) -> set:
    names = set()
    for param in command.params:
        names.update(getattr(param, "opts", []) or [])
    return names


def test_the_tree_has_the_leaves_we_expect():
    paths = {path for path, _ in leaf_commands()}
    assert "update" in paths
    assert "device status" in paths
    assert "model pull" in paths
    assert len(paths) > 15


def test_every_leaf_accepts_the_output_flags():
    """Guards the next command against forgetting them."""
    missing = {}
    for path, command in leaf_commands():
        if path in NO_OUTPUT_OF_OURS:
            continue
        opts = option_names(command)
        gaps = [flag for flag in ("--json", "--verbose", "-v", "--no-color") if flag not in opts]
        if gaps:
            missing[path] = gaps
    assert not missing, f"leaves missing output flags: {missing}"


def test_the_exemption_list_names_only_leaves_that_still_exist():
    """A renamed command must not silently inherit an exemption."""
    paths = {path for path, _ in leaf_commands()}
    assert NO_OUTPUT_OF_OURS <= paths, NO_OUTPUT_OF_OURS - paths
    assert NO_QUIET <= paths, NO_QUIET - paths


def test_quiet_is_present_wherever_it_already_was():
    """--quiet is not universal, but every leaf that emits data must offer it."""
    for path, command in leaf_commands():
        opts = option_names(command)
        if "--json" in opts and path not in NO_QUIET:
            assert "--quiet" in opts and "-q" in opts, path


def test_verbose_parses_on_a_leaf_not_only_on_the_root():
    """The regression: `tt update -v` used to exit 2 with "No such option"."""
    for argv in (["update", "-v", "--dry-run"], ["update", "--verbose", "--dry-run"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{argv} → {result.exit_code}\n{result.output}"
        assert "No such option" not in result.output


def test_no_color_parses_on_a_leaf():
    result = runner.invoke(app, ["update", "--no-color", "--dry-run"])
    assert result.exit_code == 0
    assert "No such option" not in result.output


def test_output_flags_are_grouped_in_the_output_help_panel():
    result = runner.invoke(app, ["update", "--help"])
    assert result.exit_code == 0
    assert "─ Output ─" in result.output
    assert "--verbose" in result.output
    assert "--no-color" in result.output
