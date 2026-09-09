# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import pytest
import tomlkit

from tenstorrent.config import schema
from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import (
    SOURCE_DEFAULT,
    SOURCE_FILE,
    SOURCE_UNRECOGNIZED,
    ConfigStore,
    coerce_value,
    suggest_key,
)
from tenstorrent.errors import ExitCode, TTError


@pytest.fixture
def store(isolated_dirs):
    return ConfigStore(get_paths())


def test_coerce_value():
    assert coerce_value("true") is True
    assert coerce_value("False") is False
    assert coerce_value("42") == 42
    assert coerce_value("1.5") == 1.5
    assert coerce_value("hello") == "hello"
    assert coerce_value("") == ""


def test_get_falls_back_to_defaults_with_no_file(store):
    assert store.get("telemetry.enabled") is False  # telemetry is opt-in
    assert store.get("tools.sudo_command") == "sudo"
    assert not store.paths.config_file.exists()


def test_set_then_get_round_trip(store):
    store.set("telemetry.enabled", True)  # True: distinguishable from the default
    assert store.get("telemetry.enabled") is True


def test_set_preserves_template_comments(store):
    store.set("telemetry.enabled", True)
    text = store.paths.config_file.read_text()
    assert "enabled = true" in text
    assert "# Anonymous usage data" in text  # template comment survived the edit


def test_unknown_key_is_a_config_error(store):
    with pytest.raises(TTError) as exc:
        store.get("nonsense.key")
    assert exc.value.exit_code == ExitCode.CONFIG
    with pytest.raises(TTError):
        store.set("telemetry.bogus", True)


def test_tools_override_is_a_dynamic_table(store):
    assert store.get("tools.override.tt-smi") is None  # known but unset
    store.set("tools.override.tt-smi", "/opt/bin/tt-smi")
    assert store.get("tools.override.tt-smi") == "/opt/bin/tt-smi"


def test_list_flat_merges_file_over_defaults(store):
    flat = store.list_flat()
    assert flat["telemetry.enabled"] is False
    store.set("device.backend", "native")
    flat = store.list_flat()
    assert flat["device.backend"] == "native"
    assert flat["tools.sudo_command"] == "sudo"


def test_invalid_toml_is_a_config_error(store):
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text("this is [not toml")
    with pytest.raises(TTError) as exc:
        store.get("telemetry.enabled")
    assert exc.value.exit_code == ExitCode.CONFIG
    assert "tt config" in (exc.value.next_step or "")


def test_ensure_file_writes_template_once(store):
    store.ensure_file()
    first = store.paths.config_file.read_text()
    assert "[telemetry]" in first
    store.set("telemetry.enabled", True)  # True: the template says false
    store.ensure_file()  # must not clobber the edit
    assert "enabled = true" in store.paths.config_file.read_text()


# -- new keys added to an existing config.toml ---------------------------------------
# A config file written by an older tt has none of the settings added since. That must
# keep working (reads fall back to the schema default, `set` inserts into the right
# table) — and where it does NOT work, it has to say so instead of going quiet.
OLD_FILE = """\
[telemetry]
enabled = true

[tools]
sudo_command = "sudo"

[device]
backend = "smi"
"""


@pytest.fixture
def old_file(store):
    """A config.toml predating a newly added key (telemetry.flush_mode)."""
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text(OLD_FILE)
    return store


def test_key_missing_from_an_old_file_resolves_to_its_default(old_file):
    assert old_file.get("telemetry.flush_mode") == "async"


def test_set_inserts_a_new_key_into_the_right_table(old_file):
    old_file.set("telemetry.flush_mode", "sync")
    assert old_file.get("telemetry.flush_mode") == "sync"
    # Under [telemetry], not appended to whatever table happened to be last.
    parsed = tomlkit.parse(old_file.paths.config_file.read_text()).unwrap()
    assert parsed["telemetry"]["flush_mode"] == "sync"
    assert "flush_mode" not in parsed["device"]


def test_a_key_appended_to_the_end_of_the_file_is_reported(old_file):
    """The incident this exists for: TOML assigns a bare key appended to the end of a
    file to the *last* table, so hand-adding `flush_mode = "sync"` silently became
    `device.flush_mode` and the real setting kept its default."""
    with old_file.paths.config_file.open("a") as handle:
        handle.write('flush_mode = "sync"\n')

    assert old_file.unknown_keys() == ["device.flush_mode"]
    assert old_file.get("telemetry.flush_mode") == "async"  # unchanged, as before
    entries = old_file.list_entries()
    assert entries["device.flush_mode"] == ("sync", SOURCE_UNRECOGNIZED)
    assert entries["telemetry.flush_mode"] == ("async", SOURCE_DEFAULT)


def test_unknown_keys_is_empty_for_a_clean_file(store):
    store.ensure_file()
    assert store.unknown_keys() == []


def test_unknown_keys_is_empty_with_no_file(store):
    assert store.unknown_keys() == []


def test_dynamic_tool_overrides_are_not_flagged(store):
    store.set("tools.override.tt-smi", "/opt/bin/tt-smi")
    assert store.unknown_keys() == []


def test_unparseable_file_does_not_raise_from_unknown_keys(store):
    """unknown_keys runs on the way to reporting a warning; the TOML error belongs to
    _read_document, which raises a proper CONFIG error of its own."""
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text("this is [not toml")
    assert store.unknown_keys() == []


# -- the warning ---------------------------------------------------------------------
def test_unrecognized_key_warns_once_with_a_suggestion(isolated_dirs):
    warnings: list[str] = []
    store = ConfigStore(get_paths(), on_warning=warnings.append)
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text(OLD_FILE + 'flush_mode = "sync"\n')

    # Several reads, one warning: every command reads config, often more than once.
    store.get("telemetry.enabled")
    store.get("device.backend")
    store.list_entries()

    assert len(warnings) == 1
    assert "device.flush_mode" in warnings[0]
    assert "telemetry.flush_mode" in warnings[0]  # did-you-mean
    assert "no effect" in warnings[0]


def test_no_warning_without_a_sink(isolated_dirs):
    """A store built without on_warning must stay silent, not print or raise."""
    store = ConfigStore(get_paths())
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text(OLD_FILE + 'flush_mode = "sync"\n')
    assert store.get("telemetry.enabled") is True


def test_a_clean_file_warns_about_nothing(isolated_dirs):
    warnings: list[str] = []
    store = ConfigStore(get_paths(), on_warning=warnings.append)
    store.ensure_file()
    store.list_entries()
    assert warnings == []


# -- suggestions ---------------------------------------------------------------------
def test_suggest_matches_a_misplaced_key_by_leaf_name():
    assert suggest_key("device.flush_mode") == "telemetry.flush_mode"
    assert suggest_key("flush_mode") == "telemetry.flush_mode"


def test_suggest_matches_a_typo():
    assert suggest_key("telemetry.enabeld") == "telemetry.enabled"


def test_suggest_gives_up_on_nonsense():
    assert suggest_key("wildly.unrelated.nonsense") is None


# -- provenance ----------------------------------------------------------------------
def test_sources_distinguish_defaults_from_real_overrides(store):
    store.ensure_file()  # template restates every default explicitly
    entries = store.list_entries()
    # Present in the file at its default value is still a default: otherwise a fresh
    # config would report every key as user-set and the signal would be worthless.
    assert entries["device.backend"] == ("smi", SOURCE_DEFAULT)
    assert store.source_of("device.backend") == SOURCE_DEFAULT

    store.set("device.backend", "native")
    assert store.list_entries()["device.backend"] == ("native", SOURCE_FILE)
    assert store.source_of("device.backend") == SOURCE_FILE


def test_dynamic_override_counts_as_set(store):
    store.set("tools.override.tt-smi", "/opt/bin/tt-smi")
    assert store.list_entries()["tools.override.tt-smi"][1] == SOURCE_FILE


def test_list_flat_still_returns_plain_values(store):
    """list_flat is a published shape (`tt config list --json`); it must stay flat even
    though it is now derived from list_entries."""
    store.set("device.backend", "native")
    flat = store.list_flat()
    assert flat["device.backend"] == "native"
    assert all(not isinstance(value, tuple) for value in flat.values())


# -- sync: making newly added settings visible ---------------------------------------
def test_sync_adds_a_missing_key_with_its_comments(old_file):
    added = old_file.sync()
    assert "telemetry.flush_mode" in added
    text = old_file.paths.config_file.read_text()
    assert 'flush_mode = "async"' in text
    # The documentation comes with it, which is the whole point of syncing rather than
    # telling people to add the key themselves.
    assert "# How spans are delivered" in text
    # And it lands in the right table, not appended to the last one.
    parsed = tomlkit.parse(text).unwrap()
    assert parsed["telemetry"]["flush_mode"] == "async"
    assert "flush_mode" not in parsed["device"]


def test_sync_preserves_existing_values_and_comments(store):
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text(
        "# my own note\n[telemetry]\nenabled = false\nposthog_project_key = \"phc_mine\"\n"
    )
    store.sync()
    text = store.paths.config_file.read_text()
    assert "# my own note" in text
    assert 'posthog_project_key = "phc_mine"' in text
    assert store.get("telemetry.enabled") is False


def test_sync_creates_a_whole_missing_section(store):
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text("[telemetry]\nenabled = true\n")
    added = store.sync()
    assert "paths.hf_model_cache_directory" in added
    parsed = tomlkit.parse(store.paths.config_file.read_text()).unwrap()
    assert "paths" in parsed


def test_sync_is_idempotent(old_file):
    assert old_file.sync()
    assert old_file.sync() == []


def test_sync_dry_run_changes_nothing_and_agrees_with_the_real_run(old_file):
    before = old_file.paths.config_file.read_text()
    planned = old_file.sync(dry_run=True)
    assert planned  # there is something to do
    assert old_file.paths.config_file.read_text() == before
    assert old_file.sync() == planned


def test_sync_with_no_file_writes_the_template(store):
    added = store.sync()
    assert added == schema.known_keys()
    assert "[telemetry]" in store.paths.config_file.read_text()


def test_sync_leaves_unrecognized_keys_alone(old_file):
    """Syncing is additive. Removing keys we don't recognize would delete settings
    belonging to a newer tt that the user also has installed."""
    with old_file.paths.config_file.open("a") as handle:
        handle.write('flush_mode = "sync"\n')
    old_file.sync()
    assert old_file.unknown_keys() == ["device.flush_mode"]


def test_synced_file_still_parses_and_resolves(old_file):
    old_file.sync()
    # The acid test: whatever tomlkit wrote has to be readable by us afterwards.
    assert old_file.get("telemetry.flush_mode") == "async"
    assert old_file.get("telemetry.enabled") is True
    assert old_file.unknown_keys() == []


def test_missing_keys_lists_only_absent_documented_keys(old_file):
    missing = old_file.missing_keys()
    assert "telemetry.flush_mode" in missing
    assert "telemetry.enabled" not in missing  # present in the file
    assert all(schema.is_known_key(key) for key in missing)


# -- reset ---------------------------------------------------------------------------
def test_reset_writes_defaults_and_keeps_a_backup(store):
    store.set("telemetry.posthog_project_key", "phc_mine")
    backup = store.reset()

    assert backup is not None and backup.exists()
    assert 'phc_mine' in backup.read_text()  # nothing destroyed without a copy
    # Back to the bundled default (the live key), not the user's override.
    assert store.get("telemetry.posthog_project_key") == schema.default_for(
        "telemetry.posthog_project_key"
    )
    assert "# Anonymous usage data" in store.paths.config_file.read_text()


def test_reset_with_no_existing_file_takes_no_backup(store):
    assert store.reset() is None
    assert store.paths.config_file.exists()


def test_reset_output_is_a_clean_template(store):
    store.paths.config_dir.mkdir(parents=True, exist_ok=True)
    store.paths.config_file.write_text('[device]\nbogus_key = 1\n')
    store.reset()
    assert store.unknown_keys() == []
    assert store.missing_keys() == []
