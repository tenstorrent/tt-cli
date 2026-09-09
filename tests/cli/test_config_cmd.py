# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json

from tenstorrent.cli import app
from tenstorrent.config import schema
from tenstorrent.errors import ExitCode


def test_config_set_get(runner):
    result = runner.invoke(app, ["config", "set", "telemetry.enabled", "false"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "get", "telemetry.enabled"])
    assert result.exit_code == 0
    assert result.output.strip() == "false"


def test_config_get_json_leaf_flag(runner):
    result = runner.invoke(app, ["config", "get", "telemetry.enabled", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "key": "telemetry.enabled",
        "value": False,
        "source": "default",
    }


def test_config_list_human(runner):
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "telemetry.enabled = false" in result.output
    assert "tools.sudo_command" in result.output


def test_config_list_root_json_flag(runner):
    # global --json before the subcommand must work too
    result = runner.invoke(app, ["--json", "config", "list"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["telemetry.enabled"] is False


def test_config_unknown_key_exits_config_code(runner):
    result = runner.invoke(app, ["config", "get", "bogus.key"])
    assert result.exit_code == ExitCode.CONFIG
    assert "Unknown config key" in result.output


def test_config_unknown_key_json_error_payload(runner):
    result = runner.invoke(app, ["--json", "config", "get", "bogus.key"])
    assert result.exit_code == ExitCode.CONFIG
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "CONFIG"


def test_config_path_points_into_isolated_dir(runner, isolated_dirs):
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert str(isolated_dirs / "config") in result.output


def test_config_quiet_suppresses_output(runner):
    result = runner.invoke(app, ["config", "list", "--quiet"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_bare_config_creates_file_and_opens_editor(runner, isolated_dirs):
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert (isolated_dirs / "config" / "config.toml").exists()


# -- provenance in the human listing -------------------------------------------------
def test_config_list_marks_defaults_and_overrides(runner):
    runner.invoke(app, ["config", "set", "device.backend", "native"])
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    lines = {
        line.split(" = ")[0]: line for line in result.output.splitlines() if " = " in line
    }
    assert "(set in config.toml)" in lines["device.backend"]
    assert "(default)" in lines["tools.sudo_command"]


def test_config_list_json_stays_flat_by_default(runner):
    """`tt config list --json` is a published shape: {key: value}, not {key: {...}}."""
    result = runner.invoke(app, ["config", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["telemetry.enabled"] is False
    assert data["tools.sudo_command"] == "sudo"


def test_config_list_sources_flag_reports_provenance(runner):
    runner.invoke(app, ["config", "set", "device.backend", "native"])
    result = runner.invoke(app, ["config", "list", "--json", "--sources"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["device.backend"] == {"value": "native", "source": "config.toml"}
    assert data["tools.sudo_command"] == {"value": "sudo", "source": "default"}


# -- the unrecognized-key warning ----------------------------------------------------
def _write_stray_key(isolated_dirs):
    """A config.toml with `flush_mode` hand-added at the end, so TOML files it under the
    last table ([device]) instead of [telemetry]."""
    path = isolated_dirs / "config" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '[telemetry]\nenabled = true\n\n[device]\nbackend = "smi"\nflush_mode = "sync"\n'
    )
    return path


def test_stray_key_is_reported_with_a_suggestion(runner, isolated_dirs):
    _write_stray_key(isolated_dirs)
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "device.flush_mode" in result.output
    assert "telemetry.flush_mode" in result.output  # did-you-mean
    assert "unrecognized" in result.output


def test_stray_key_is_reported_on_unrelated_commands_too(runner, isolated_dirs):
    """The mistake breaks whatever command relied on the setting, so the warning has to
    follow the mistake rather than wait for someone to run `tt config list`."""
    _write_stray_key(isolated_dirs)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "does not" in result.output or "no effect" in result.output
    assert "device.flush_mode" in result.output


def test_quiet_suppresses_the_warning(runner, isolated_dirs):
    _write_stray_key(isolated_dirs)
    result = runner.invoke(app, ["--quiet", "config", "path"])
    assert result.exit_code == 0
    assert "device.flush_mode" not in result.output


def test_warning_does_not_corrupt_json_output(runner, isolated_dirs):
    """The warning goes to stderr; stdout must stay parseable for `tt --json | jq`."""
    _write_stray_key(isolated_dirs)
    result = runner.invoke(app, ["--json", "config", "get", "telemetry.flush_mode"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "key": "telemetry.flush_mode",
        "value": "async",
        "source": "default",
    }


# -- tt config sync ------------------------------------------------------------------
def _write_old_file(isolated_dirs):
    """A config.toml predating telemetry.flush_mode and the whole [paths] section."""
    path = isolated_dirs / "config" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[telemetry]\nenabled = true\n\n[device]\nbackend = "smi"\n')
    return path


def test_config_sync_adds_missing_settings(runner, isolated_dirs):
    path = _write_old_file(isolated_dirs)
    result = runner.invoke(app, ["config", "sync"])
    assert result.exit_code == 0
    assert "telemetry.flush_mode" in result.output
    text = path.read_text()
    assert 'flush_mode = "async"' in text
    assert "# How spans are delivered" in text  # arrives with its documentation


def test_config_sync_dry_run_touches_nothing(runner, isolated_dirs):
    path = _write_old_file(isolated_dirs)
    before = path.read_text()
    result = runner.invoke(app, ["config", "sync", "--dry-run"])
    assert result.exit_code == 0
    assert "Would add" in result.output
    assert path.read_text() == before


def test_config_sync_reports_a_clean_file(runner):
    runner.invoke(app, ["config", "sync"])
    result = runner.invoke(app, ["config", "sync"])
    assert result.exit_code == 0
    assert "already up to date" in result.output


def test_config_sync_json_lists_what_it_added(runner, isolated_dirs):
    _write_old_file(isolated_dirs)
    result = runner.invoke(app, ["config", "sync", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "telemetry.flush_mode" in data["added"]
    assert data["dry_run"] is False


def test_sync_then_the_setting_is_editable_in_place(runner, isolated_dirs):
    """The point of sync: after it, the key is visible in the file where the user expects
    to edit it — and editing it there actually takes effect."""
    path = _write_old_file(isolated_dirs)
    runner.invoke(app, ["config", "sync"])
    path.write_text(path.read_text().replace('flush_mode = "async"', 'flush_mode = "sync"'))
    result = runner.invoke(app, ["config", "get", "telemetry.flush_mode"])
    assert result.output.strip() == "sync"


def test_bare_config_points_at_sync_when_settings_are_missing(runner, isolated_dirs):
    _write_old_file(isolated_dirs)
    result = runner.invoke(app, ["config"])  # EDITOR=true via isolated_dirs
    assert result.exit_code == 0
    assert "tt config sync" in result.output


def test_bare_config_says_nothing_when_the_file_is_complete(runner):
    runner.invoke(app, ["config", "sync"])
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "tt config sync" not in result.output


# -- tt config reset -----------------------------------------------------------------
def test_config_reset_needs_confirmation_when_non_interactive(runner, isolated_dirs):
    path = _write_old_file(isolated_dirs)
    result = runner.invoke(app, ["config", "reset"])
    assert result.exit_code == ExitCode.USAGE
    assert "--yes" in result.output
    assert path.read_text().startswith("[telemetry]")  # untouched


def test_config_reset_yes_writes_defaults_and_backs_up(runner, isolated_dirs):
    runner.invoke(app, ["config", "set", "telemetry.posthog_project_key", "phc_mine"])
    result = runner.invoke(app, ["config", "reset", "--yes"])
    assert result.exit_code == 0
    backup = isolated_dirs / "config" / "config.toml.bak"
    assert backup.exists()
    assert "phc_mine" in backup.read_text()
    # Back to the bundled default (the live key), not the user's override.
    assert (
        runner.invoke(app, ["config", "get", "telemetry.posthog_project_key"]).output.strip()
        == schema.default_for("telemetry.posthog_project_key")
    )


def test_config_reset_json_reports_the_backup(runner, isolated_dirs):
    _write_old_file(isolated_dirs)
    result = runner.invoke(app, ["config", "reset", "--yes", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["backup"].endswith("config.toml.bak")


def test_config_reset_clears_a_stray_key(runner, isolated_dirs):
    """Reset is the escape hatch for a file that has drifted: afterwards there is
    nothing unrecognized left to warn about."""
    _write_stray_key(isolated_dirs)
    assert "device.flush_mode" in runner.invoke(app, ["config", "path"]).output
    runner.invoke(app, ["config", "reset", "--yes"])
    assert "device.flush_mode" not in runner.invoke(app, ["config", "path"]).output
