# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Usage-telemetry tests: event contents, the anonymization policy, the opt-in gate and
its one-time consent prompt, opt-out paths, delivery, and the never-break-the-CLI
guarantee.

All `fakes_only`: they assert exact event properties and inject an in-memory transport,
so nothing here touches the network except a loopback collector the test owns. The
autouse `isolated_dirs` fixture sets TT_TELEMETRY_DISABLED=1; tests that want telemetry
active delete it explicitly — and, since telemetry is opt-in, also record consent via
`_opt_in()` (the schema default is enabled=false, so without it every session is the
null one).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from tenstorrent.cli import app
from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import ConfigStore
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.launchers.base import RunningModel
from tenstorrent.output import OutputManager
from tenstorrent.telemetry.attributes import (
    EVENT_PROPERTY_NAMES,
    build_event,
    command_properties,
    error_properties,
    install_properties,
)
from tenstorrent.telemetry import session as session_module
from tenstorrent.telemetry import spool as spool_module
from tenstorrent.telemetry.drain import build_batch, drain
from tenstorrent.telemetry.session import TelemetrySession
from tenstorrent.telemetry.spool import Spool
from tenstorrent.telemetry.state import TelemetryState

pytestmark = pytest.mark.fakes_only

INSTANCE = "11111111-2222-3333-4444-555555555555"


def _opt_in() -> None:
    """Record durable consent the way the prompt (or the user) would."""
    ConfigStore(get_paths()).set("telemetry.enabled", True)


@pytest.fixture
def collected(monkeypatch):
    """Opt in, enable telemetry, and capture events in memory instead of over HTTP.

    Forces sync mode: the injected transport stands in for the in-process POST, which
    only sync mode uses. What these tests assert — event contents and the anonymization
    policy — is identical in both modes, since the two differ only in delivery. Async
    delivery has its own tests below.
    """
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    events: list[dict] = []
    monkeypatch.setattr(TelemetrySession, "_transport", staticmethod(lambda config: events.extend))
    return events


@pytest.fixture
def spooling(monkeypatch):
    """Enable telemetry in async (default) mode, pointed at an endpoint nobody serves.

    Async mode has no transport seam to inject: it spools to disk, and whether an upload
    is even configured is decided from the endpoint + key. So these tests set a real-
    looking endpoint and assert on the spool, never on the network — nothing here can
    reach it, because handing off is what these tests control.
    """
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/batch/")
    monkeypatch.setenv("TT_TELEMETRY_POSTHOG_KEY", "phc_test_key")
    _opt_in()
    return Spool(get_paths())


def _only(events: list[dict]) -> dict:
    assert len(events) == 1, events
    return events[0]


def _props(events: list[dict]) -> dict:
    return _only(events)["properties"]


# -- pure property builders ----------------------------------------------------------
def test_install_properties_are_anonymous():
    props = install_properties()
    assert set(props) == {"tt_version", "os_type", "os_arch", "python_version", "ci"}
    assert props["ci"] is False


def test_error_properties_carry_category_only():
    assert error_properties(ExitCode.NO_DEVICES) == {"exit_code": 3, "exit_code_name": "NO_DEVICES"}


def test_command_properties_record_names_not_values():
    """Baseline for a command with no allowlist entry: names only, no values at all."""

    class FakeSource:
        def __init__(self, name):
            self.name = name

    class FakeCtx:
        command_path = "tt compile"  # deliberately not in _SAFE_VALUES
        params = {"args": ["./private_model.py"], "json_mode": False}

        def get_parameter_source(self, name):
            # user supplied `args`; `json_mode` left at its default
            return FakeSource("COMMANDLINE" if name == "args" else "DEFAULT")

    props = command_properties(FakeCtx())
    assert props["command"] == "tt compile"
    assert props["options_set"] == ["args"]  # the NAME, never the value
    assert "private_model" not in str(props)


# -- the event envelope --------------------------------------------------------------
def test_event_carries_a_v4_uuid_the_install_id_and_a_utc_capture_time():
    before = datetime.now(timezone.utc)
    event = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK, duration_ms=7)
    after = datetime.now(timezone.utc)

    assert event["event"] == "tt_command"
    assert uuid.UUID(event["uuid"]).version == 4
    assert event["distinct_id"] == INSTANCE
    # tz-aware UTC, stamped at capture. Events sit in the spool for minutes to days, and
    # a naive local time would shift every one of them by the user's UTC offset.
    stamp = datetime.fromisoformat(event["timestamp"])
    assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
    assert before <= stamp <= after
    assert event["properties"]["duration_ms"] == 7


def test_every_event_gets_its_own_uuid():
    a = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)
    b = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)
    assert a["uuid"] != b["uuid"]


def test_person_properties_describe_the_install_only():
    props = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)["properties"]
    assert set(props["$set"]) == {"tt_version", "os_type", "os_arch", "python_version"}
    assert set(props["$set_once"]) == {"first_seen_version", "first_seen_os_type"}
    assert props["$lib"] == "tt-cli"


def test_every_event_forbids_geoip_enrichment():
    """No location, not even country-level: `$geoip_disable` tells PostHog not to derive
    anything from the request address (the project setting discards the address itself).
    Pinned as exactly True — a falsy or missing value would silently re-enable it."""
    props = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)["properties"]
    assert props["$geoip_disable"] is True


def test_the_property_set_is_closed():
    """Every name an event may carry is enumerated in attributes.py, and this literal
    pins it: growing the set is a reviewed change, and neither an `$ip` nor anything
    SDK-shaped can appear by accident."""
    assert EVENT_PROPERTY_NAMES == {
        "command",
        "options_set",
        "exit_code",
        "exit_code_name",
        "reason",
        "exception_type",
        "duration_ms",
        "tt_version",
        "os_type",
        "os_arch",
        "python_version",
        "ci",
        "$lib",
        "$lib_version",
        "$geoip_disable",
        "$set",
        "$set_once",
        # the argument-value allowlist
        "model",
        "model_type",
        "hardware",
        "workflow",
        "device_config",
        "config_key",
        "installer_version",
        "device_count",
    }


# -- end-to-end through the decorator seam -------------------------------------------
def test_command_emits_one_event_with_ok(runner, collected):
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    event = _only(collected)
    assert event["event"] == "tt_command"
    assert event["distinct_id"] == TelemetryState(get_paths()).instance_id()
    props = event["properties"]
    assert props["command"] == "tt config path"
    assert props["exit_code"] == 0
    assert props["exit_code_name"] == "OK"
    assert isinstance(props["duration_ms"], int) and props["duration_ms"] >= 0
    assert set(props) == {
        "command", "exit_code", "exit_code_name", "duration_ms", "tt_version", "os_type",
        "os_arch", "python_version", "ci", "$lib", "$lib_version", "$geoip_disable",
        "$set", "$set_once",
    }


def test_distinct_id_is_stable_across_commands(runner, collected):
    for _ in range(2):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    ids = {event["distinct_id"] for event in collected}
    assert len(collected) == 2 and len(ids) == 1
    assert ids == {TelemetryState(get_paths()).instance_id()}


def test_ambient_otel_environment_never_reaches_an_event(runner, collected, monkeypatch):
    """The OpenTelemetry-based predecessor had to guard against the SDK merging a user's
    OTEL_RESOURCE_ATTRIBUTES into what it sent. attributes.py is now the only producer,
    but the property is worth keeping pinned: nothing from the environment rides along."""
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "user.name=someone,deployment.environment=prod-cluster"
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "not-tt")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    props = _props(collected)
    assert set(props) <= EVENT_PROPERTY_NAMES
    assert "someone" not in json.dumps(collected)
    assert "prod-cluster" not in json.dumps(collected)


def test_error_command_records_its_exit_code(runner, collected):
    # `tt compile <x>` is a stub that exits UNSUPPORTED (7).
    result = runner.invoke(app, ["compile", "somemodel"])
    assert result.exit_code == 7
    props = _props(collected)
    assert props["exit_code"] == 7
    assert props["exit_code_name"] == "UNSUPPORTED"


def test_group_callback_does_not_add_a_second_event(runner, collected):
    """`tt config` is a decorated group callback that fires before `tt config get`.

    Only the leaf represents what the user ran; an event for the group too would
    over-count `tt config` by the volume of all its subcommands.
    """
    result = runner.invoke(app, ["config", "get", "telemetry.enabled"])
    assert result.exit_code == 0
    assert [e["properties"]["command"] for e in collected] == ["tt config get"]


def test_bare_group_invocation_keeps_its_event(runner, collected):
    """With no subcommand, `tt config` is itself the leaf (it opens $EDITOR) and counts.

    isolated_dirs sets EDITOR=true, so this exercises the real invoke_without_command
    path without launching anything.
    """
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert [e["properties"]["command"] for e in collected] == ["tt config"]


# -- errors: category, reason slug, crash class --------------------------------------
def test_a_reason_slug_is_recorded_but_never_the_message():
    err = TTError(
        "Required tool 'tt-smi' is not installed at /home/someone/.local/bin.",
        why="looked in /home/someone/.local/bin",
        exit_code=ExitCode.TOOL_MISSING,
        reason="tool.missing.tt_smi",
    )
    props = error_properties(ExitCode.TOOL_MISSING, err)
    assert props == {"exit_code": 4, "exit_code_name": "TOOL_MISSING", "reason": "tool.missing.tt_smi"}
    assert "someone" not in str(props)


@pytest.mark.parametrize(
    "bad",
    ["Tool Missing", "tool-missing", "/home/someone/x", "tool_missing.", ".x", "a" * 65, "", None, 3],
)
def test_a_reason_that_is_not_a_slug_is_dropped(bad):
    err = TTError("x", exit_code=ExitCode.ERROR)
    err.reason = bad  # bypass the constructor's typing: the grammar is the gate
    props = error_properties(ExitCode.ERROR, err)
    assert "reason" not in props
    assert props == {"exit_code": 1, "exit_code_name": "ERROR"}


def test_a_crash_records_the_exception_class_only():
    exc = FileNotFoundError(2, "No such file", "/home/someone/secret.toml")
    props = error_properties(ExitCode.ERROR, exc)
    assert props == {"exit_code": 1, "exit_code_name": "ERROR", "exception_type": "FileNotFoundError"}
    assert "someone" not in str(props) and "secret" not in str(props)


def test_a_tterror_never_contributes_an_exception_type():
    props = error_properties(ExitCode.CONFIG, TTError("bad key", exit_code=ExitCode.CONFIG))
    assert "exception_type" not in props and "reason" not in props


def test_an_uncaught_exception_in_a_command_is_recorded_by_class(runner, collected, monkeypatch):
    """The bare `except Exception` in @handle_tt_errors is the crash path: the event
    still goes out, naming the class and nothing the message said."""

    def boom(self, *args, **kwargs):
        raise RuntimeError("could not write /home/someone/.config/tenstorrent/config.toml")

    monkeypatch.setattr(OutputManager, "emit", boom)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code != 0
    props = _props(collected)
    assert props["exit_code_name"] == "ERROR"
    assert props["exception_type"] == "RuntimeError"
    assert "someone" not in json.dumps(collected)
    assert "config.toml" not in json.dumps(collected)


def test_unknown_model_failure_carries_its_reason(runner, collected):
    result = runner.invoke(app, ["model", "info", "/home/someone/private-finetune"])
    assert result.exit_code == 2
    props = _props(collected)
    assert props["reason"] == "model.unknown"
    assert "someone" not in json.dumps(collected)


# -- the argument-value allowlist ----------------------------------------------------
def test_catalog_model_name_is_recorded(runner, collected):
    result = runner.invoke(app, ["model", "info", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0
    assert _props(collected)["model"] == "Llama-3.1-8B-Instruct"


def test_non_catalog_model_name_is_dropped(runner, collected):
    """A name outside the catalog vocabulary must not be exported, even verbatim argv.

    This is the path that would leak a local filesystem path or a private finetune name.
    """
    result = runner.invoke(app, ["model", "info", "/home/someone/private-finetune"])
    assert result.exit_code == 2  # unknown model
    props = _props(collected)
    assert "model" not in props
    assert "private-finetune" not in str(props)
    assert "someone" not in str(props)


def test_a_terminal_handoff_still_records_its_event(runner, collected, monkeypatch, tmp_path):
    """exec_tty replaces the process, so nothing after it runs — not the event's
    context manager, not @handle_tt_errors' finally. Without the before_exec hook
    every *successful* `tt launch <terminal client>` would go unrecorded, leaving
    only the failures and making the command look like it never works."""
    exe = tmp_path / "opencode"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_OPENCODE", str(exe))
    monkeypatch.setattr("tenstorrent.commands.launch._stdin_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: True)
    # Patch where the name is *used*: commands.launch imported discover directly, so
    # patching the discovery module would leave the real one bound — and this test
    # would then quietly pass or fail on whether the machine happens to be serving.
    # The id is deliberately not a catalog model, so no support-list entry is needed.
    monkeypatch.setattr(
        "tenstorrent.commands.launch.discover",
        lambda base_url: [RunningModel(served_id="not-a-catalog-model", base_url=base_url)],
    )

    def fake_execvpe(file, argv, env):
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0, result.output
    # Exactly one: the early close records it, and the normal exit path must not again.
    props = _props(collected)
    assert props["command"] == "tt launch opencode"
    assert props["exit_code"] == 0


def test_launch_names_the_client_in_the_event_but_never_the_endpoint(runner, collected):
    """Each client is its own leaf command, so the command path identifies it with
    no value allowlist. --url is free text and must not be exported."""
    result = runner.invoke(
        app, ["launch", "opencode", "--url", "http://secret-box.corp:8000/v1"]
    )
    assert result.exit_code != 0  # nothing is serving there
    props = _props(collected)
    assert props["command"] == "tt launch opencode"
    assert "secret-box" not in json.dumps(collected)


def test_config_set_records_the_key_but_never_the_value(runner, collected):
    """The schema's own keys include a secret, so values must never be exported."""
    result = runner.invoke(
        app, ["config", "set", "telemetry.posthog_project_key", "phc_live_supersecret"]
    )
    assert result.exit_code == 0
    props = _props(collected)
    assert props["config_key"] == "telemetry.posthog_project_key"
    assert "phc_live_supersecret" not in json.dumps(collected)


def test_tools_override_key_collapses_to_the_namespace(runner, collected):
    result = runner.invoke(app, ["config", "set", "tools.override.tt-smi", "/opt/bin/tt-smi"])
    assert result.exit_code == 0
    props = _props(collected)
    assert props["config_key"] == "tools.override.*"
    assert "/opt/bin" not in json.dumps(collected)


def test_stub_argv_is_never_recorded(runner, collected):
    """`tt compile` takes permissive argv, i.e. arbitrary local filenames."""
    result = runner.invoke(app, ["compile", "./proprietary_model.py"])
    assert result.exit_code == 7
    props = _props(collected)
    assert "proprietary_model" not in json.dumps(collected)
    # that args were passed, not what they were
    assert props["options_set"] == ["args"]


def test_bounded_filters_and_enums_are_recorded(runner, collected):
    result = runner.invoke(app, ["model", "list", "--type", "llm", "--hw", "p300"])
    assert result.exit_code == 0
    props = _props(collected)
    assert props["model_type"] == "llm"
    assert props["hardware"] == "p300"


def test_unknown_filter_values_are_dropped(runner, collected):
    result = runner.invoke(app, ["model", "list", "--hw", "definitely-not-a-board"])
    assert result.exit_code == 0
    props = _props(collected)
    assert "hardware" not in props
    assert "definitely-not-a-board" not in json.dumps(collected)


def test_installer_version_must_look_like_semver(runner, collected):
    runner.invoke(app, ["update", "not-a-version", "--dry-run"])
    props = _props(collected)
    assert "installer_version" not in props
    assert "not-a-version" not in json.dumps(collected)


def test_path_values_never_leak_into_events(runner, collected):
    """Config values are the leakiest surface: paths carry the user's home directory."""
    result = runner.invoke(
        app, ["config", "set", "paths.hf_model_cache_directory", "/home/someone/models"]
    )
    assert result.exit_code == 0
    props = _props(collected)
    assert props["config_key"] == "paths.hf_model_cache_directory"
    assert "/home/someone" not in json.dumps(collected)
    assert "someone" not in json.dumps(collected)


# -- opt-out paths -------------------------------------------------------------------
def test_disabled_env_emits_nothing(runner, monkeypatch):
    # TT_TELEMETRY_DISABLED=1 is set by the isolated_dirs fixture; it must silence
    # even an install that opted in.
    _opt_in()
    events: list[dict] = []
    monkeypatch.setattr(TelemetrySession, "_transport", staticmethod(lambda config: events.extend))
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert events == []


def test_config_disabled_emits_nothing(runner, collected):
    ConfigStore(get_paths()).set("telemetry.enabled", False)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert collected == []


def test_offline_flag_emits_nothing(runner, collected):
    result = runner.invoke(app, ["--offline", "config", "path"])
    assert result.exit_code == 0
    assert collected == []


def test_no_key_configured_is_inert(runner, monkeypatch):
    # Real transport path, but no project key -> nothing to send to, no events, no crash.
    # The bundled default key is live, so the keyless state must be established explicitly.
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    ConfigStore(get_paths()).set("telemetry.posthog_project_key", "")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0


# -- the opt-in gate and its one-time consent prompt ---------------------------------
def test_not_opted_in_collects_nothing(runner, monkeypatch):
    """THE opt-in property: with the kill switch lifted and the live default endpoint +
    key in place, a fresh install still spools and sends nothing, because nobody said
    yes yet."""
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert not Spool(get_paths()).path.exists()


@pytest.fixture
def interactive(monkeypatch):
    """Pretend a person is at the terminal, and let each test script the answer.

    Returns a dict: set answers["value"] for the reply, read answers["asked"] for how
    many times the prompt fired. `Ellipsis` as value raises click's Abort (Ctrl-C/EOF).
    """
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(session_module, "_interactive", lambda: True)
    answers = {"value": False, "asked": 0}

    def fake_confirm(*args, **kwargs):
        answers["asked"] += 1
        # Recorded, not asserted: an assert raised in here would be swallowed by the
        # prompt's never-break-the-CLI guard. The yes test checks it.
        answers["default"] = kwargs.get("default", "missing")
        if answers["value"] is Ellipsis:
            raise session_module.Abort()
        return answers["value"]

    monkeypatch.setattr(session_module, "confirm", fake_confirm)
    return answers


def _create_session():
    paths = get_paths()
    config = ConfigStore(paths)
    return TelemetrySession.create(paths, config, offline=False, output=OutputManager())


def test_first_interactive_run_prompts_and_yes_opts_in(interactive, capsys):
    interactive["value"] = True
    session = _create_session()
    assert interactive["asked"] == 1
    # Consent must be an explicit y/n: default=None makes a bare Enter re-ask instead
    # of silently answering for the user.
    assert interactive["default"] is None
    # The consent is persisted where the user can see and revoke it.
    assert ConfigStore(get_paths()).get("telemetry.enabled") is True
    assert TelemetryState(get_paths()).prompt_answered() is True
    assert session is not session_module.NULL_SESSION
    # Flattened: Rich wraps stderr at terminal width, splitting phrases arbitrarily.
    err = " ".join(capsys.readouterr().err.split())
    # The disclosure points at the full document as a URL the user can click through.
    # Matched against the despaced output: a narrow terminal can fold even a URL.
    url = "https://github.com/tenstorrent/tt-cli/blob/main/TELEMETRY.md"
    assert url in err.replace(" ", "")
    assert "telemetry.enabled false" in err  # how to change their mind


def test_the_disclosure_url_is_a_clickable_hyperlink():
    """Consent is asked for in the same breath as the pointer to what's collected, so
    the document has to be one click away: on a terminal the URL carries an OSC 8
    hyperlink. Where that isn't supported — a pipe, a log, a CI transcript — it has to
    degrade to the plain URL rather than leaking escape sequences into the text.
    """
    url = session_module._TELEMETRY_DOC_URL

    tty = StringIO()
    Console(file=tty, force_terminal=True, highlight=False, width=100).print(
        session_module._PROMPT_INTRO
    )
    linked = tty.getvalue()
    # OSC 8 wraps the visible text: the URL appears in the escape and again as the text.
    assert "\x1b]8;" in linked
    assert linked.count(url) == 2

    piped = StringIO()
    Console(file=piped, highlight=False, width=100).print(session_module._PROMPT_INTRO)
    plain = piped.getvalue()
    assert "\x1b" not in plain
    assert url in plain


def test_declining_stays_off_and_is_never_asked_again(interactive):
    session = _create_session()
    assert interactive["asked"] == 1
    assert ConfigStore(get_paths()).get("telemetry.enabled") is False
    assert session is session_module.NULL_SESSION
    _create_session()
    assert interactive["asked"] == 1  # the answer was remembered


def test_aborting_the_prompt_is_not_an_answer(interactive):
    """Ctrl-C/EOF at the prompt means "not now": stay off, but ask again next time."""
    interactive["value"] = Ellipsis
    assert _create_session() is session_module.NULL_SESSION
    assert TelemetryState(get_paths()).prompt_answered() is False
    interactive["value"] = False
    _create_session()
    assert interactive["asked"] == 2


def test_no_prompt_without_a_tty(interactive, monkeypatch):
    """Scripts and pipelines are never interrogated (and CliRunner has no TTY, which is
    what keeps every other test in this file prompt-free)."""
    monkeypatch.setattr(session_module, "_interactive", lambda: False)
    assert _create_session() is session_module.NULL_SESSION
    assert interactive["asked"] == 0


@pytest.mark.parametrize("mode", ["quiet", "json_mode"])
def test_no_prompt_under_quiet_or_json(interactive, mode):
    paths = get_paths()
    session = TelemetrySession.create(
        paths, ConfigStore(paths), offline=False, output=OutputManager(**{mode: True})
    )
    assert session is session_module.NULL_SESSION
    assert interactive["asked"] == 0


def test_no_prompt_when_a_per_run_switch_says_no(interactive, monkeypatch):
    """--offline, DO_NOT_TRACK and CI each mean the answer could not matter this run
    (and on CI nobody is there to answer)."""
    paths = get_paths()
    config = ConfigStore(paths)
    TelemetrySession.create(paths, config, offline=True, output=OutputManager())
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    TelemetrySession.create(paths, config, offline=False, output=OutputManager())
    monkeypatch.delenv("DO_NOT_TRACK")
    monkeypatch.setenv("CI", "true")
    TelemetrySession.create(paths, config, offline=False, output=OutputManager())
    assert interactive["asked"] == 0


def test_no_prompt_when_nothing_could_be_sent(interactive):
    """With no endpoint/key there is nothing to consent to, so nobody is asked — and
    the not-asked state survives, in case a key is configured later."""
    ConfigStore(get_paths()).set("telemetry.posthog_project_key", "")
    assert _create_session() is session_module.NULL_SESSION
    assert interactive["asked"] == 0
    assert TelemetryState(get_paths()).prompt_answered() is False


def test_opting_in_by_hand_skips_the_prompt(interactive):
    """`tt config set telemetry.enabled true` is consent; asking again would be noise."""
    _opt_in()
    session = _create_session()
    assert interactive["asked"] == 0
    assert session is not session_module.NULL_SESSION


# -- the endpoint --------------------------------------------------------------------
def test_legacy_traces_endpoint_in_config_maps_to_the_batch_endpoint():
    """tt <= 1.0.1 materialized PostHog's OTLP traces URL into config.toml. Events posted
    there would be accepted and discarded, so the old default is mapped to the new one —
    for both regions, and however the trailing slash was spelled."""
    store = ConfigStore(get_paths())
    for legacy, batch in (
        ("https://us.i.posthog.com/i/v1/traces", "https://us.i.posthog.com/batch/"),
        ("https://us.i.posthog.com/i/v1/traces/", "https://us.i.posthog.com/batch/"),
        ("https://eu.i.posthog.com/i/v1/traces", "https://eu.i.posthog.com/batch/"),
    ):
        store.set("telemetry.endpoint", legacy)
        assert session_module.resolve_endpoint(store)[0] == batch


def test_a_custom_endpoint_passes_through_untouched(monkeypatch):
    """Self-hosted PostHog, a local sink: the user's URL is the user's business."""
    store = ConfigStore(get_paths())
    store.set("telemetry.endpoint", "https://posthog.example.internal/batch/")
    assert session_module.resolve_endpoint(store)[0] == "https://posthog.example.internal/batch/"
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:9/anything")
    assert session_module.resolve_endpoint(store)[0] == "http://127.0.0.1:9/anything"


def test_the_default_endpoint_is_posthog_batch_capture():
    assert session_module.resolve_endpoint(ConfigStore(get_paths()))[0] == "https://us.i.posthog.com/batch/"


# -- resilience: telemetry never breaks the CLI --------------------------------------
def test_unreachable_endpoint_does_not_stall_the_command(runner, monkeypatch):
    """An endpoint that never answers must not add its stall to the command.

    session.flush() bounds the in-process POST externally (a daemon thread it stops
    waiting on). This transport blocks far longer than the budget; the command must
    still return promptly.

    Sync mode only. Async mode cannot stall at all — that is the point of it — but this
    ceiling still has to hold, because CI forces sync and developers select it by hand.
    """
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    monkeypatch.setattr(
        TelemetrySession, "_transport", staticmethod(lambda config: lambda batch: time.sleep(30.0))
    )
    started = time.monotonic()
    result = runner.invoke(app, ["config", "path"])
    elapsed = time.monotonic() - started

    assert result.exit_code == 0
    # Generous headroom over the 1.5s budget so this can't flake on a loaded machine,
    # while still failing loudly if the bound is lost (it would take 30 seconds).
    assert elapsed < 10, f"telemetry stalled the command for {elapsed:.1f}s"


def test_flush_never_raises_even_if_its_own_setup_fails(monkeypatch):
    """flush() runs in the decorator's `finally`, so anything escaping it breaks the
    command it was supposed to be invisible to.

    Regression: threading.Event() was constructed outside the guard, so a failure there
    propagated to the caller and every command exited 1 with a NameError.
    """

    class Unusable:
        def __getattr__(self, name):
            raise RuntimeError("threading unavailable")

    monkeypatch.setattr(session_module, "threading", Unusable())
    session = TelemetrySession(instance_id=INSTANCE, transport=lambda batch: None)
    # A pending event: without one the flush has nothing to post and never touches
    # threading, so the regression would not be exercised at all.
    session._pending.append({"event": "tt_command"})
    session.flush()  # must return quietly


def test_transport_failure_does_not_break_command(runner, monkeypatch):
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()

    def boom(batch):
        raise RuntimeError("boom")

    monkeypatch.setattr(TelemetrySession, "_transport", staticmethod(lambda config: boom))
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0  # command succeeds despite the transport blowing up


def test_a_broken_event_builder_does_not_break_command(runner, collected, monkeypatch):
    monkeypatch.setattr(
        session_module.attributes, "build_event", lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
    )
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert collected == []


# -- async delivery: the spool -------------------------------------------------------
def _spool_lines(spool: Spool) -> list[str]:
    return [line for line in spool.path.read_text().splitlines() if line.strip()]


def test_async_mode_spools_the_event_instead_of_posting(runner, spooling):
    """The default path writes to disk and makes no HTTP request at all.

    `spooling` points the endpoint at a closed port, so if this ever regressed to an
    in-process post the command would still pass — which is why the assertion is on
    the spool file, not on timing.
    """
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    lines = _spool_lines(spooling)
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "tt_command"
    assert record["properties"]["command"] == "tt config path"
    assert uuid.UUID(record["uuid"]).version == 4


def test_async_mode_never_opens_a_socket(runner, spooling, monkeypatch):
    """Spool appends run inline as the command's context closes; the live path never
    reaches httpx. If an in-process post ever crept back onto this path, this would
    catch it."""
    import httpx

    def _boom(*args, **kwargs):
        raise AssertionError("async mode must not make an HTTP request")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr(httpx.Client, "send", _boom)
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert len(_spool_lines(spooling)) == 1


def test_spool_is_inert_without_an_endpoint(runner, monkeypatch):
    """No key configured means no spool: accumulating events that can never be delivered
    would be worse than collecting nothing. The bundled default key is live, so the
    keyless state must be established explicitly."""
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    ConfigStore(get_paths()).set("telemetry.posthog_project_key", "")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert not Spool(get_paths()).path.exists()


def test_events_accumulate_across_commands(runner, spooling):
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert len(_spool_lines(spooling)) == 3
    assert spooling.stats().events == 3


def test_an_oversized_event_is_dropped_rather_than_written_torn():
    """Lock-free concurrent appends are safe only while one line is one write() below
    the atomicity bound; a line that would exceed it must not go down in pieces."""
    spool = Spool(get_paths())
    huge = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)
    huge["properties"]["blob"] = "x" * spool_module.MAX_LINE_BYTES
    assert spool.append(huge) is False
    assert not spool.path.exists()
    assert spool.append(build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)) is True
    assert spool.stats().events == 1


def test_legacy_span_spool_is_removed_when_the_event_spool_starts(runner, spooling):
    """tt <= 1.0.1 spooled OpenTelemetry spans under other names. Nothing can upload
    them any more, so they are deleted — not migrated, not left to grow stale."""
    spooling.dir.mkdir(parents=True, exist_ok=True)
    for name in spool_module._LEGACY_FILES:
        (spooling.dir / name).write_text('{"resourceSpans": []}\n')
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert not any((spooling.dir / name).exists() for name in spool_module._LEGACY_FILES)
    assert spooling.stats().events == 1


# -- async delivery: the hand-off decision -------------------------------------------
@pytest.fixture
def spawns(monkeypatch):
    """Record hand-offs instead of forking a real uploader."""
    launched = []
    monkeypatch.setattr(
        session_module, "spawn_drainer", lambda spool, **kwargs: launched.append(spool)
    )
    return launched


def test_no_hand_off_below_the_threshold(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 5)
    for _ in range(4):
        runner.invoke(app, ["config", "path"])
    assert spawns == []


def test_hand_off_once_the_spool_fills(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 3)
    for _ in range(3):
        runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_hand_off_when_the_oldest_event_goes_stale(runner, spooling, spawns, monkeypatch):
    """A `tt device status`-only user never reaches the event threshold, so age has to
    be an independent trigger or their data never leaves the machine."""
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1000)
    monkeypatch.setattr(spool_module, "DRAIN_AGE_SECONDS", 0.0)
    runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_no_hand_off_while_a_drainer_holds_the_lock(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1)
    with spooling.lock() as acquired:
        assert acquired
        runner.invoke(app, ["config", "path"])
    assert spawns == []


def test_detached_uploader_releases_all_three_descriptors(monkeypatch):
    """Homebrew#29: a detached child that inherits stdout keeps the pipe open, so
    `$(tt ...)` blocks until the child exits — analytics added ~250 ms to `brew search`.
    Also start_new_session, so the child survives the parent and a Ctrl-C can't hit it
    mid-upload."""
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

    monkeypatch.setattr(session_module.subprocess, "Popen", FakePopen)
    assert session_module.spawn_drainer(Spool(get_paths())) is True

    kwargs = captured["kwargs"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert captured["argv"][1:] == ["-m", "tenstorrent", "self", "send-telemetry"]


# -- the drain: does the upload actually happen? -------------------------------------
@pytest.fixture
def collector():
    """A local stand-in for PostHog's /batch/ endpoint that records what it was sent.

    Asserting on a real request is the point: Yarn Berry's telemetry upload has been
    dead code on master since 2023 (an inverted emptiness check) with no issue filed,
    because nothing tested that the send *happened*. And PostHog itself answers 200 to
    payloads it then discards, so the assertions are on the received body, never on
    the status code alone.
    """
    received = []
    status = {"code": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
            received.append({"path": self.path, "body": body, "raw": raw, "headers": dict(self.headers)})
            payload = b'{"status": 1}' if status["code"] < 300 else b'{"status": 0, "error": "nope"}'
            self.send_response(status["code"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}/batch/",
            received=received,
            status=status,
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def live(monkeypatch, collector):
    """Async telemetry pointed at the local collector fixture."""
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", collector.url)
    monkeypatch.setenv("TT_TELEMETRY_POSTHOG_KEY", "phc_test_key")
    _opt_in()
    return collector


def _commands(request: dict) -> list[str]:
    return [event["properties"]["command"] for event in request["body"]["batch"]]


def test_drain_uploads_the_batch_and_empties_the_spool(runner, live):
    paths = get_paths()
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    spool = Spool(paths)
    assert spool.stats().events == 3

    result = drain(paths, ConfigStore(paths))
    assert result.status == "sent"
    assert result.events == 3

    assert len(live.received) == 1
    request = live.received[0]
    assert request["path"] == "/batch/"
    assert request["headers"]["Content-Type"] == "application/json"
    assert "Authorization" not in request["headers"]  # the key travels in the body
    assert request["body"]["api_key"] == "phc_test_key"
    assert _commands(request) == ["tt config path"] * 3
    # sent_at is the upload time, which PostHog uses to correct client clock skew.
    sent_at = datetime.fromisoformat(request["body"]["sent_at"])
    assert sent_at.utcoffset().total_seconds() == 0
    # The drain happened AND cleaned up: nothing left to re-send.
    assert spool.stats().events == 0
    assert not spool.sending_path.exists()


def test_the_batch_is_the_spool_verbatim(runner, live):
    """What is on disk is what is sent: no re-encoding step that could drift from the
    format PostHog was verified against (the OTLP predecessor needed a byte-identity
    test for exactly that; here the spool line *is* the wire object)."""
    paths = get_paths()
    runner.invoke(app, ["model", "info", "Llama-3.1-8B-Instruct"])
    spooled = json.loads(_spool_lines(Spool(paths))[0])
    assert drain(paths, ConfigStore(paths)).status == "sent"
    assert live.received[0]["body"]["batch"] == [spooled]


def test_one_upload_carries_every_spooled_command(runner, live):
    """Batching is the whole win: upload cost is per-request, not per-event, so many
    commands must collapse into one POST."""
    paths = get_paths()
    for argv in (["config", "path"], ["model", "info", "Llama-3.1-8B-Instruct"], ["config", "path"]):
        runner.invoke(app, argv)
    assert drain(paths, ConfigStore(paths)).status == "sent"
    assert len(live.received) == 1
    assert _commands(live.received[0]) == ["tt config path", "tt model info", "tt config path"]


def test_failed_upload_keeps_the_batch_for_the_next_attempt(runner, live):
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    live.status["code"] = 503

    result = drain(paths, ConfigStore(paths))
    assert result.status == "failed"
    # PostHog's error body is worth surfacing (it names a bad key, for instance).
    assert result.detail.startswith("HTTP 503")
    assert "nope" in result.detail
    spool = Spool(paths)
    # Held under the in-flight name, so a crash mid-upload cannot lose it either.
    assert spool.sending_path.exists()

    live.status["code"] = 200
    assert drain(paths, ConfigStore(paths)).status == "sent"
    assert _commands(live.received[-1]) == ["tt config path"]
    assert not spool.sending_path.exists()


def test_a_retried_batch_keeps_the_same_event_uuids(runner, live):
    """A POST that times out after the server accepted it would double-count without
    stable ids; PostHog deduplicates on the event uuid, so the retry must resend the
    very same ones."""
    paths = get_paths()
    for _ in range(2):
        runner.invoke(app, ["config", "path"])
    live.status["code"] = 503
    assert drain(paths, ConfigStore(paths)).status == "failed"
    live.status["code"] = 200
    assert drain(paths, ConfigStore(paths)).status == "sent"
    first = [e["uuid"] for e in live.received[0]["body"]["batch"]]
    second = [e["uuid"] for e in live.received[1]["body"]["batch"]]
    assert len(first) == 2 and first == second


def test_unreachable_endpoint_leaves_the_batch_intact(runner, monkeypatch):
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/batch/")
    monkeypatch.setenv("TT_TELEMETRY_POSTHOG_KEY", "phc_test_key")
    _opt_in()
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    result = drain(paths, ConfigStore(paths))
    assert result.status == "failed"
    assert Spool(paths).sending_path.exists()


def test_a_second_drainer_does_nothing(runner, live):
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    spool = Spool(paths)
    with spool.lock() as acquired:
        assert acquired
        assert drain(paths, ConfigStore(paths)).status == "busy"
    assert live.received == []
    assert spool.stats().events == 1  # untouched, still there to send


def test_send_telemetry_command_drains(runner, live):
    """The hidden subcommand is what the detached process runs, so it has to work as the
    entry point, not just `drain()` as a function."""
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    result = runner.invoke(app, ["self", "send-telemetry", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["status"] == "sent"
    assert report["events"] == 1
    assert len(live.received) == 1
    assert Spool(paths).stats().events == 0


def test_the_drainer_does_not_record_an_event_for_itself(runner, live):
    """`tt self send-telemetry` must be invisible to telemetry: an event for the uploader
    would refill the spool on every drain and eventually hand off to another uploader."""
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    runner.invoke(app, ["self", "send-telemetry"])
    assert Spool(paths).stats().events == 0


def test_drain_removes_legacy_span_spools(runner, live):
    paths = get_paths()
    spool = Spool(paths)
    spool.dir.mkdir(parents=True, exist_ok=True)
    for name in spool_module._LEGACY_FILES:
        (spool.dir / name).write_text('{"resourceSpans": []}\n')
    assert drain(paths, ConfigStore(paths)).status == "empty"
    assert live.received == []
    assert not any((spool.dir / name).exists() for name in spool_module._LEGACY_FILES)


# -- the drain: batch fidelity -------------------------------------------------------
def test_a_torn_line_does_not_strand_the_batch():
    """Concurrent appenders make a partial line possible in principle; one bad record
    must not cost us every good one."""
    good = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)
    batch = build_batch(['{"event": "tt_command", "uuid": "trunc', json.dumps(good)])
    assert batch == [good]


def test_records_posthog_would_silently_drop_are_not_sent():
    """PostHog answers 200 and discards an event with no name or no distinct_id; a
    record without a uuid could not be retried safely. None of them go out."""
    good = build_event(None, instance_id=INSTANCE, exit_code=ExitCode.OK)
    nameless = {**good, "event": ""}
    anonymous = {k: v for k, v in good.items() if k != "distinct_id"}
    unidentified = {**good, "uuid": None}
    lines = [json.dumps(r) for r in (nameless, "not-a-dict", anonymous, good, unidentified, 42)]
    assert build_batch(lines) == [good]


# -- opt-out, now that data can sit on disk unsent -----------------------------------
def test_opting_out_discards_unsent_events(runner, spooling):
    """Spooling creates a window where data sits on the machine unsent. Opting out has
    to delete it — uploading it later would be worse than an in-process flush, where
    opt-out was immediate and total."""
    for _ in range(2):
        runner.invoke(app, ["config", "path"])
    assert spooling.stats().events == 2

    ConfigStore(get_paths()).set("telemetry.enabled", False)
    runner.invoke(app, ["config", "path"])

    assert spooling.stats().events == 0
    assert not spooling.path.exists()
    assert not spooling.sending_path.exists()


def test_drain_rechecks_opt_out_and_deletes_rather_than_uploads(runner, live):
    """The opt-out may land after the hand-off is already in flight."""
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    ConfigStore(paths).set("telemetry.enabled", False)

    result = drain(paths, ConfigStore(paths))
    assert result.status == "discarded"
    assert live.received == []
    assert Spool(paths).stats().events == 0


def test_do_not_track_opts_out_and_discards(runner, spooling, monkeypatch):
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().events == 1
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().events == 0


def test_do_not_track_zero_is_not_opting_out(runner, spooling, monkeypatch):
    """The convention is that "0"/"false"/empty mean unset, so DO_NOT_TRACK=0 must not
    silently disable telemetry."""
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().events == 1


def test_kill_switch_keeps_the_spool(runner, spooling, monkeypatch):
    """TT_TELEMETRY_DISABLED is a per-run kill switch, not a durable opt-out: it must
    stop collection without destroying events an earlier run legitimately collected."""
    runner.invoke(app, ["config", "path"])
    monkeypatch.setenv("TT_TELEMETRY_DISABLED", "1")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().events == 1


def test_offline_keeps_the_spool(runner, spooling):
    """`--offline` means "no network this run", not "forget what I did" — deleting the
    spool for it would be data loss from an unrelated flag."""
    runner.invoke(app, ["config", "path"])
    runner.invoke(app, ["--offline", "config", "path"])
    assert spooling.stats().events == 1


def test_drain_declines_while_the_kill_switch_is_set(runner, live, monkeypatch):
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    monkeypatch.setenv("TT_TELEMETRY_DISABLED", "1")
    assert drain(paths, ConfigStore(paths)).status == "disabled"
    assert live.received == []
    assert Spool(paths).stats().events == 1  # kept, not deleted


# -- bounds --------------------------------------------------------------------------
def test_upload_is_capped_and_drops_the_oldest(runner, live, monkeypatch):
    """A permanently firewalled machine must not accumulate forever, and when we do trim,
    recent history is the part worth keeping."""
    monkeypatch.setattr(spool_module, "MAX_EVENTS", 2)
    paths = get_paths()
    for _ in range(5):
        runner.invoke(app, ["config", "path"])
    result = drain(paths, ConfigStore(paths))
    assert result.status == "sent"
    assert result.events == 2


def test_append_stops_at_the_byte_ceiling(runner, spooling, monkeypatch):
    runner.invoke(app, ["config", "path"])
    size = spooling.stats().bytes
    monkeypatch.setattr(spool_module, "MAX_SPOOL_BYTES", size)
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert spooling.stats().events == 1


# -- CI ------------------------------------------------------------------------------
def test_ci_is_recorded_rather_than_dropped(runner, collected, monkeypatch):
    """CI runs are real usage and are kept; `ci` is what lets them be filtered later."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert _props(collected)["ci"] is True


def test_ci_marker_is_false_for_a_person(runner, collected):
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert _props(collected)["ci"] is False


def test_ci_never_exports_env_var_values(runner, collected, monkeypatch):
    """Only the *names* in CI_ENV_VARS are read. Their values are repo slugs, branch
    names and build URLs, none of which may leave the machine."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CI", "tenstorrent/private-repo#4711")
    runner.invoke(app, ["config", "path"])
    blob = json.dumps(collected)
    assert "private-repo" not in blob
    assert "4711" not in blob


def test_ci_forces_synchronous_delivery(monkeypatch):
    """A detached uploader is reaped with the build's process group, and the container is
    destroyed with the spool still on it — so async on CI collects data that is
    guaranteed never to arrive."""
    config = ConfigStore(get_paths())
    assert session_module.flush_mode(config) == "async"
    monkeypatch.setenv("GITLAB_CI", "true")
    assert session_module.flush_mode(config) == "sync"


def test_ci_does_not_spool(runner, monkeypatch, collector):
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", collector.url)
    monkeypatch.setenv("TT_TELEMETRY_POSTHOG_KEY", "phc_test_key")
    monkeypatch.setenv("CI", "true")
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert not Spool(get_paths()).path.exists()
    assert len(collector.received) == 1  # delivered in-process instead
    assert _commands(collector.received[0]) == ["tt config path"]


def test_explicit_flush_mode_overrides_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "async")
    assert session_module.flush_mode(ConfigStore(get_paths())) == "async"


def test_unknown_flush_mode_falls_back_to_async(monkeypatch):
    """Async is the safe default: its failure mode is delayed data, sync's is a slow CLI
    for every user."""
    store = ConfigStore(get_paths())
    store.set("telemetry.flush_mode", "eventually")
    assert session_module.flush_mode(store) == "async"


# -- the log-file disclosure seam ----------------------------------------------------
def test_log_file_records_events_without_opting_in(runner, monkeypatch, tmp_path):
    """TT_TELEMETRY_LOG_FILE shows the user exactly what would be sent. It is local-only
    and uploads nothing, so it must work *before* consent — it exists precisely so a
    user can inspect the data while deciding whether to opt in."""
    log = tmp_path / "telemetry.jsonl"
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_LOG_FILE", str(log))
    assert runner.invoke(app, ["config", "path"]).exit_code == 0

    record = json.loads(log.read_text().splitlines()[0])
    assert record["event"] == "tt_command"
    assert record["properties"]["command"] == "tt config path"
    # Not opted in: local record only — nothing spooled for upload.
    assert not Spool(get_paths()).path.exists()


def test_log_file_alone_stays_local_even_when_opted_in(runner, monkeypatch, tmp_path):
    """Opted in but with no endpoint/key: the log file records, nothing accumulates."""
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    ConfigStore(get_paths()).set("telemetry.posthog_project_key", "")
    log = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("TT_TELEMETRY_LOG_FILE", str(log))
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert log.exists()
    assert not Spool(get_paths()).path.exists()


def test_log_file_and_spool_record_the_same_event(runner, spooling, monkeypatch, tmp_path):
    """The disclosure must be honest: the log line is the spool line."""
    log = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("TT_TELEMETRY_LOG_FILE", str(log))
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert json.loads(log.read_text()) == json.loads(_spool_lines(spooling)[0])


# -- resilience --------------------------------------------------------------------
def test_unwritable_spool_does_not_break_the_command(runner, spooling, monkeypatch):
    monkeypatch.setattr(
        spool_module.Spool,
        "_open_for_append",
        lambda self: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )
    assert runner.invoke(app, ["config", "path"]).exit_code == 0


def test_drain_never_raises(monkeypatch):
    paths = get_paths()
    monkeypatch.setattr(
        spool_module.Spool, "lock", lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert drain(paths, ConfigStore(paths)).status == "error"


def test_drain_of_an_empty_spool_sends_nothing(runner, live):
    paths = get_paths()
    assert drain(paths, ConfigStore(paths)).status == "empty"
    assert live.received == []


def _flat(text: str) -> str:
    """Collapse whitespace: Rich wraps stderr at terminal width, so a message can be
    split mid-phrase and a plain substring check fails for no real reason."""
    return " ".join(text.split())


# -- delivery survives an ambient proxy ----------------------------------------------
def test_drain_reaches_a_local_collector_with_a_proxy_configured(runner, live, monkeypatch):
    """A proxy in the environment must not break delivery to a local collector.

    httpx honours HTTP_PROXY and ALL_PROXY and does not bypass loopback on its own — so
    on a box behind a proxy every collector-backed test failed with "the collector was
    never reached", while the same tests passed on a machine with no proxy set.
    `isolated_dirs` pins NO_PROXY for loopback to make the suite independent of that;
    this test is what keeps it pinned.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://10.255.255.1:3128")
    monkeypatch.setenv("ALL_PROXY", "http://10.255.255.1:3128")
    paths = get_paths()
    assert runner.invoke(app, ["config", "path"]).exit_code == 0

    result = drain(paths, ConfigStore(paths))
    assert result.status == "sent", f"proxy leaked into the loopback POST: {result.detail}"
    assert len(live.received) == 1


# -- diagnosing silent delivery ------------------------------------------------------
def test_verbose_explains_why_no_hand_off_happened(runner, spooling, spawns, monkeypatch):
    """Below the threshold, doing nothing is correct — but indistinguishable from broken
    without a way to ask."""
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 5)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "below the hand-off threshold of 5" in _flat(result.stderr)
    assert spawns == []


def test_verbose_announces_the_hand_off(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "handing 1 event(s) to a background uploader" in _flat(result.stderr)
    assert len(spawns) == 1


def test_verbose_reports_a_failed_launch(runner, spooling, monkeypatch):
    """A hand-off that cannot start used to return False into the void."""
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1)

    def boom(*args, **kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(session_module.subprocess, "Popen", boom)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0  # still must not break the command
    assert "could not launch the uploader: OSError: fork failed" in _flat(result.stderr)


def test_verbose_flags_a_batch_stuck_awaiting_retry(runner, spooling, monkeypatch):
    """The one state that means "delivery is broken" rather than "not yet attempted"."""
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1000)
    spooling.dir.mkdir(parents=True, exist_ok=True)
    spooling.sending_path.write_text('{"event": "tt_command"}\n' * 3)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "3 event(s) from an earlier batch are awaiting retry" in _flat(result.stderr)
    assert "tt self send-telemetry" in _flat(result.stderr)


def test_delivery_diagnostics_stay_silent_without_verbose(runner, spooling, spawns, monkeypatch):
    """Telemetry has no business on a normal command's output."""
    monkeypatch.setattr(spool_module, "DRAIN_EVENT_THRESHOLD", 1)
    spooling.dir.mkdir(parents=True, exist_ok=True)
    spooling.sending_path.write_text('{"event": "tt_command"}\n')
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "telemetry" not in result.stderr.lower()
    assert len(spawns) == 1
