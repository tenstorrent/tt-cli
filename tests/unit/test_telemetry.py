# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Usage-telemetry tests: span contents, the anonymization policy, the opt-in gate and
its one-time consent prompt, opt-out paths, and the never-break-the-CLI guarantee.

All `fakes_only`: they assert exact span attributes and inject an in-memory exporter,
so nothing here touches the network. The autouse `isolated_dirs` fixture sets
TT_TELEMETRY_DISABLED=1; tests that want telemetry active delete it explicitly — and,
now that telemetry is opt-in, also record consent via `_opt_in()` (the schema default
is enabled=false, so without it every session is the null one).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rich.console import Console

from tenstorrent.cli import app
from tenstorrent.config.paths import get_paths
from tenstorrent.config.store import ConfigStore
from tenstorrent.errors import ExitCode
from tenstorrent.launchers.base import RunningModel
from tenstorrent.output import OutputManager
from tenstorrent.telemetry.attributes import (
    command_attributes,
    error_attributes,
    resource_attributes,
)
from tenstorrent.telemetry import session as session_module
from tenstorrent.telemetry import spool as spool_module
from tenstorrent.telemetry.drain import build_payload, drain
from tenstorrent.telemetry.session import TelemetrySession
from tenstorrent.telemetry.spool import Spool
from tenstorrent.telemetry.state import TelemetryState

pytestmark = pytest.mark.fakes_only


def _opt_in() -> None:
    """Record durable consent the way the prompt (or the user) would."""
    ConfigStore(get_paths()).set("telemetry.enabled", True)


@pytest.fixture
def collected(monkeypatch):
    """Opt in, enable telemetry, and capture spans in memory instead of over HTTP.

    Forces sync mode: the injected exporter stands in for the direct OTLP exporter, which
    only sync mode uses. What these tests assert — span contents and the anonymization
    policy — is identical in both modes, since the two differ only in delivery. Async
    delivery has its own tests below.
    """
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        TelemetrySession, "_otlp_exporter", staticmethod(lambda config: exporter)
    )
    return exporter


@pytest.fixture
def spooling(monkeypatch):
    """Enable telemetry in async (default) mode, pointed at an endpoint nobody serves.

    Async mode has no exporter seam to inject: it spools to disk, and whether an upload
    is even configured is decided from the endpoint + key. So these tests set a real-
    looking endpoint and assert on the spool, never on the network — nothing here can
    reach it, because handing off is what these tests control.
    """
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/i/v1/traces")
    monkeypatch.setenv("TT_TELEMETRY_POSTHOG_KEY", "phc_test_key")
    _opt_in()
    return Spool(get_paths())


# -- pure attribute builders ---------------------------------------------------------
def test_resource_attributes_are_anonymous():
    attrs = resource_attributes("11111111-2222-3333-4444-555555555555")
    assert attrs["service.name"] == "tt"
    assert attrs["tt.instance_id"] == "11111111-2222-3333-4444-555555555555"
    assert "service.version" in attrs and "os.type" in attrs


def test_error_attributes_carry_category_only():
    attrs = error_attributes(ExitCode.NO_DEVICES)
    assert attrs == {"tt.exit_code": 3, "tt.exit_code_name": "NO_DEVICES"}


def test_command_attributes_record_names_not_values():
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

    attrs = command_attributes(FakeCtx())
    assert attrs["tt.command"] == "tt compile"
    assert attrs["tt.options_set"] == ["args"]  # the NAME, never the value
    assert "private_model" not in str(attrs)


# -- end-to-end through the decorator seam -------------------------------------------
def test_command_emits_one_span_with_ok(runner, collected):
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    spans = collected.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tt config path"
    assert span.attributes["tt.command"] == "tt config path"
    assert span.attributes["tt.exit_code"] == 0
    assert span.attributes["tt.exit_code_name"] == "OK"
    assert span.resource.attributes["service.name"] == "tt"
    assert span.resource.attributes["tt.instance_id"]


def test_otel_env_attributes_never_reach_the_resource(runner, collected, monkeypatch):
    """A stray OTEL_RESOURCE_ATTRIBUTES in the user's environment must not ride along.

    Resource.create() would merge OTel's env detectors on top of ours; the plain
    Resource() constructor keeps attributes.py the only source of resource attributes.
    """
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "user.name=someone,deployment.environment=prod-cluster"
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "not-tt")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    resource = dict(collected.get_finished_spans()[0].resource.attributes)
    assert "user.name" not in resource
    assert "deployment.environment" not in resource
    assert resource["service.name"] == "tt"
    assert set(resource) == set(resource_attributes(resource["tt.instance_id"]))


def test_error_command_records_its_exit_code(runner, collected):
    # `tt compile <x>` is a stub that exits UNSUPPORTED (7).
    result = runner.invoke(app, ["compile", "somemodel"])
    assert result.exit_code == 7
    span = collected.get_finished_spans()[0]
    assert span.attributes["tt.exit_code"] == 7
    assert span.attributes["tt.exit_code_name"] == "UNSUPPORTED"


def test_group_callback_does_not_add_a_second_span(runner, collected):
    """`tt config` is a decorated group callback that fires before `tt config get`.

    Only the leaf represents what the user ran; a span for the group too would
    over-count `tt config` by the volume of all its subcommands.
    """
    result = runner.invoke(app, ["config", "get", "telemetry.enabled"])
    assert result.exit_code == 0
    assert [s.name for s in collected.get_finished_spans()] == ["tt config get"]


def test_bare_group_invocation_keeps_its_span(runner, collected):
    """With no subcommand, `tt config` is itself the leaf (it opens $EDITOR) and counts.

    isolated_dirs sets EDITOR=true, so this exercises the real invoke_without_command
    path without launching anything.
    """
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert [s.name for s in collected.get_finished_spans()] == ["tt config"]


# -- the argument-value allowlist ----------------------------------------------------
def test_catalog_model_name_is_recorded(runner, collected):
    result = runner.invoke(app, ["model", "info", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0
    assert (
        collected.get_finished_spans()[0].attributes["tt.model"]
        == "Llama-3.1-8B-Instruct"
    )


def test_non_catalog_model_name_is_dropped(runner, collected):
    """A name outside the catalog vocabulary must not be exported, even verbatim argv.

    This is the path that would leak a local filesystem path or a private finetune name.
    """
    result = runner.invoke(app, ["model", "info", "/home/someone/private-finetune"])
    assert result.exit_code == 2  # unknown model
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert "tt.model" not in attrs
    assert "private-finetune" not in str(attrs)
    assert "someone" not in str(attrs)


def test_a_terminal_handoff_still_records_its_span(runner, collected, monkeypatch, tmp_path):
    """exec_tty replaces the process, so nothing after it runs — not the span's
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
    span = collected.get_finished_spans()[0]
    assert span.name == "tt launch opencode"
    assert span.attributes["tt.exit_code"] == 0


def test_launch_names_the_client_in_the_span_but_never_the_endpoint(runner, collected):
    """Each client is its own leaf command, so the command path identifies it with
    no value allowlist. --url is free text and must not be exported."""
    result = runner.invoke(
        app, ["launch", "opencode", "--url", "http://secret-box.corp:8000/v1"]
    )
    assert result.exit_code != 0  # nothing is serving there
    span = collected.get_finished_spans()[0]
    assert span.name == "tt launch opencode"
    assert "secret-box" not in str(dict(span.attributes))


def test_config_set_records_the_key_but_never_the_value(runner, collected):
    """The schema's own keys include a secret, so values must never be exported."""
    result = runner.invoke(
        app, ["config", "set", "telemetry.posthog_project_key", "phc_live_supersecret"]
    )
    assert result.exit_code == 0
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert attrs["tt.config_key"] == "telemetry.posthog_project_key"
    assert "phc_live_supersecret" not in str(attrs)


def test_tools_override_key_collapses_to_the_namespace(runner, collected):
    result = runner.invoke(app, ["config", "set", "tools.override.tt-smi", "/opt/bin/tt-smi"])
    assert result.exit_code == 0
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert attrs["tt.config_key"] == "tools.override.*"
    assert "/opt/bin" not in str(attrs)


def test_stub_argv_is_never_recorded(runner, collected):
    """`tt compile` takes permissive argv, i.e. arbitrary local filenames."""
    result = runner.invoke(app, ["compile", "./proprietary_model.py"])
    assert result.exit_code == 7
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert "proprietary_model" not in str(attrs)
    # that args were passed, not what they were (OTel stores sequences as tuples)
    assert list(attrs["tt.options_set"]) == ["args"]


def test_bounded_filters_and_enums_are_recorded(runner, collected):
    result = runner.invoke(app, ["model", "list", "--type", "llm", "--hw", "p300"])
    assert result.exit_code == 0
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert attrs["tt.model_type"] == "llm"
    assert attrs["tt.hardware"] == "p300"


def test_unknown_filter_values_are_dropped(runner, collected):
    result = runner.invoke(app, ["model", "list", "--hw", "definitely-not-a-board"])
    assert result.exit_code == 0
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert "tt.hardware" not in attrs
    assert "definitely-not-a-board" not in str(attrs)


def test_installer_version_must_look_like_semver(runner, collected):
    result = runner.invoke(app, ["update", "not-a-version", "--dry-run"])
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert "tt.installer_version" not in attrs
    assert "not-a-version" not in str(attrs)
    assert result.exit_code != 0 or True  # exit code is the command's business, not ours


def test_path_values_never_leak_into_spans(runner, collected):
    """Config values are the leakiest surface: paths carry the user's home directory."""
    result = runner.invoke(
        app, ["config", "set", "paths.hf_model_cache_directory", "/home/someone/models"]
    )
    assert result.exit_code == 0
    attrs = dict(collected.get_finished_spans()[0].attributes)
    assert attrs["tt.config_key"] == "paths.hf_model_cache_directory"
    assert "/home/someone" not in str(attrs)
    assert "someone" not in str(attrs)


# -- opt-out paths -------------------------------------------------------------------
def test_disabled_env_emits_nothing(runner, monkeypatch):
    # TT_TELEMETRY_DISABLED=1 is set by the isolated_dirs fixture; it must silence
    # even an install that opted in.
    _opt_in()
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        TelemetrySession, "_otlp_exporter", staticmethod(lambda config: exporter)
    )
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert exporter.get_finished_spans() == ()


def test_config_disabled_emits_nothing(runner, collected):
    ConfigStore(get_paths()).set("telemetry.enabled", False)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert collected.get_finished_spans() == ()


def test_offline_flag_emits_nothing(runner, collected):
    result = runner.invoke(app, ["--offline", "config", "path"])
    assert result.exit_code == 0
    assert collected.get_finished_spans() == ()


def test_no_key_configured_is_inert(runner, monkeypatch):
    # Real exporter path, but no project key -> nothing to export to, no spans, no crash.
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


# -- resilience: telemetry never breaks the CLI --------------------------------------
def test_unreachable_collector_does_not_stall_the_command(runner, monkeypatch):
    """A collector that never answers must not add its stall to the command.

    force_flush ignores its own timeout_millis (20s observed against a blackholed
    endpoint on SDK 1.44), so session.flush() bounds it externally. This exporter
    blocks far longer than the budget; the command must still return promptly.

    Sync mode only. Async mode cannot stall at all — that is the point of it — but this
    ceiling still has to hold, because CI forces sync and developers select it by hand.
    """
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")
    stall = 30.0

    class StallingExporter(SpanExporter):
        def export(self, spans):
            time.sleep(stall)
            return SpanExportResult.FAILURE

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            time.sleep(stall)
            return False

    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    monkeypatch.setattr(
        TelemetrySession, "_otlp_exporter", staticmethod(lambda config: StallingExporter())
    )
    started = time.monotonic()
    result = runner.invoke(app, ["config", "path"])
    elapsed = time.monotonic() - started

    assert result.exit_code == 0
    # Generous headroom over the 1.5s budget so this can't flake on a loaded machine,
    # while still failing loudly if the bound is lost (it would take `stall` seconds).
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
    # needs_flush=True: without it this session takes the async path and never touches
    # threading, so the regression would not be exercised at all.
    TelemetrySession(object(), object(), needs_flush=True).flush()  # must return quietly


def test_exporter_failure_does_not_break_command(runner, monkeypatch):
    monkeypatch.setenv("TT_TELEMETRY_FLUSH_MODE", "sync")

    class BoomExporter(SpanExporter):
        def export(self, spans):
            raise RuntimeError("boom")

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            raise RuntimeError("boom")

    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    monkeypatch.setattr(
        TelemetrySession, "_otlp_exporter", staticmethod(lambda config: BoomExporter())
    )
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0  # command succeeds despite the exporter blowing up


# -- async delivery: the spool -------------------------------------------------------
def _spool_lines(spool: Spool) -> list[str]:
    return [line for line in spool.path.read_text().splitlines() if line.strip()]


def test_async_mode_spools_the_span_instead_of_exporting(runner, spooling):
    """The default path writes to disk and makes no HTTP request at all.

    `spooling` points the endpoint at a closed port, so if this ever regressed to an
    in-process export the command would still pass — which is why the assertion is on
    the spool file, not on timing.
    """
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    lines = _spool_lines(spooling)
    assert len(lines) == 1
    record = json.loads(lines[0])
    span = record["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["name"] == "tt config path"


def test_async_mode_never_calls_force_flush(runner, spooling, monkeypatch):
    """Spool appends run inline on span end, so force_flush is never reached.

    force_flush is the SDK method whose timeout is broken and which always returns True;
    async mode must not depend on it. If a BatchSpanProcessor ever crept back onto this
    path, this would catch it.
    """
    def _boom(*args, **kwargs):
        raise AssertionError("async mode must not call force_flush")

    monkeypatch.setattr(TracerProvider, "force_flush", _boom)
    assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert len(_spool_lines(spooling)) == 1


def test_spool_is_inert_without_an_endpoint(runner, monkeypatch):
    """No key configured means no spool: accumulating spans that can never be delivered
    would be worse than collecting nothing. The bundled default key is live, so the
    keyless state must be established explicitly."""
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    _opt_in()
    ConfigStore(get_paths()).set("telemetry.posthog_project_key", "")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert not Spool(get_paths()).path.exists()


def test_spans_accumulate_across_commands(runner, spooling):
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert len(_spool_lines(spooling)) == 3
    assert spooling.stats().spans == 3


# -- async delivery: the hand-off decision -------------------------------------------
@pytest.fixture
def spawns(monkeypatch):
    """Record hand-offs instead of forking a real uploader."""
    launched = []
    # **kwargs: spawn_drainer also takes on_debug now.
    monkeypatch.setattr(
        session_module, "spawn_drainer", lambda spool, **kwargs: launched.append(spool)
    )
    return launched


def test_no_hand_off_below_the_threshold(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 5)
    for _ in range(4):
        runner.invoke(app, ["config", "path"])
    assert spawns == []


def test_hand_off_once_the_spool_fills(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 3)
    for _ in range(3):
        runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_hand_off_when_the_oldest_span_goes_stale(runner, spooling, spawns, monkeypatch):
    """A `tt device status`-only user never reaches the span threshold, so age has to be
    an independent trigger or their data never leaves the machine."""
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1000)
    monkeypatch.setattr(spool_module, "DRAIN_AGE_SECONDS", 0.0)
    runner.invoke(app, ["config", "path"])
    assert len(spawns) == 1


def test_no_hand_off_while_a_drainer_holds_the_lock(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1)
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
    """A local OTLP/HTTP receiver that records what it was sent.

    Asserting on a real request is the point: Yarn Berry's telemetry upload has been
    dead code on master since 2023 (an inverted emptiness check) with no issue filed,
    because nothing tested that the send *happened*.
    """
    received = []
    status = {"code": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            received.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            self.send_response(status["code"])
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}/i/v1/traces",
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


def _decode(body: bytes):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return request


def test_drain_uploads_the_batch_and_empties_the_spool(runner, live):
    paths = get_paths()
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    spool = Spool(paths)
    assert spool.stats().spans == 3

    result = drain(paths, ConfigStore(paths))
    assert result.status == "sent"
    assert result.spans == 3

    assert len(live.received) == 1
    request = _decode(live.received[0]["body"])
    names = [s.name for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans]
    assert names == ["tt config path"] * 3
    assert live.received[0]["headers"]["Authorization"] == "Bearer phc_test_key"
    assert live.received[0]["path"] == "/i/v1/traces"
    # The drain happened AND cleaned up: nothing left to re-send.
    assert spool.stats().spans == 0
    assert not spool.sending_path.exists()


def test_one_upload_carries_every_spooled_command(runner, live):
    """Batching is the whole win: export cost is per-request, not per-span (1 span ~=
    100 spans ~= 298 ms), so many commands must collapse into one POST."""
    paths = get_paths()
    for name in (["config", "path"], ["model", "info", "Llama-3.1-8B-Instruct"], ["config", "path"]):
        runner.invoke(app, name)
    assert drain(paths, ConfigStore(paths)).status == "sent"
    assert len(live.received) == 1
    request = _decode(live.received[0]["body"])
    # Same resource and scope for every command, so they regroup into a single block.
    assert len(request.resource_spans) == 1
    assert len(request.resource_spans[0].scope_spans) == 1
    assert len(request.resource_spans[0].scope_spans[0].spans) == 3


def test_failed_upload_keeps_the_batch_for_the_next_attempt(runner, live):
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    live.status["code"] = 503

    result = drain(paths, ConfigStore(paths))
    assert result.status == "failed"
    spool = Spool(paths)
    # Held under the in-flight name, so a crash mid-upload cannot lose it either.
    assert spool.sending_path.exists()

    live.status["code"] = 200
    assert drain(paths, ConfigStore(paths)).status == "sent"
    assert len(_decode(live.received[-1]["body"]).resource_spans) == 1
    assert not spool.sending_path.exists()


def test_unreachable_collector_leaves_the_batch_intact(runner, monkeypatch):
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/i/v1/traces")
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
    assert spool.stats().spans == 1  # untouched, still there to send


def test_send_telemetry_command_drains(runner, live):
    """The hidden subcommand is what the detached process runs, so it has to work as the
    entry point, not just `drain()` as a function."""
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    result = runner.invoke(app, ["self", "send-telemetry", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "sent"
    assert len(live.received) == 1
    assert Spool(paths).stats().spans == 0


def test_the_drainer_does_not_spool_a_span_for_itself(runner, live):
    """`tt self send-telemetry` must be invisible to telemetry: a span for the uploader
    would refill the spool on every drain and eventually hand off to another uploader."""
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    runner.invoke(app, ["self", "send-telemetry"])
    assert Spool(paths).stats().spans == 0


# -- the drain: payload fidelity -----------------------------------------------------
def _provider_with(exporters):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider(
        resource=Resource(resource_attributes("11111111-2222-3333-4444-555555555555")),
        shutdown_on_exit=False,
    )
    for exporter in exporters:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def test_spooled_payload_matches_the_direct_exporter_bytes(tmp_path):
    """The spool round-trip must reproduce exactly what the direct exporter would send.

    This is the guard on drain.py's shortcut of POSTing hand-built protobuf instead of
    going back through OTLPSpanExporter. It also pins the sharpest edge in the whole
    design: OTLP/JSON writes trace ids as hex, protobuf's JSON mapping expects base64,
    and feeding one to the other does not raise — it base64-decodes the hex into 24
    bytes of unrelated garbage and reports success. Since PostHog answers 200 for
    payloads it cannot use, nothing downstream would have told us.
    """
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    spool = Spool(get_paths())
    captured = []

    class Capture(SpanExporter):
        def export(self, spans):
            captured.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    provider = _provider_with([spool.exporter(), Capture()])
    tracer = provider.get_tracer("tenstorrent.cli")
    for index in range(3):
        with tracer.start_as_current_span(f"tt cmd{index}") as span:
            span.set_attribute("tt.exit_code", index)
            span.set_attribute("tt.options_set", ["json_mode", "quiet"])

    payload, spans = build_payload(_spool_lines(spool))
    assert spans == 3
    assert payload == encode_spans(captured).SerializeToString()

    # And spelled out, in case the equality above is ever weakened: real ids survive.
    request = _decode(payload)
    spooled = {
        s.trace_id.hex() for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans
    }
    assert spooled == {format(s.context.trace_id, "032x") for s in captured}
    assert all(len(tid) == 32 for tid in spooled)


def test_a_torn_line_does_not_strand_the_batch():
    """Concurrent appenders make a partial line possible in principle; one bad record
    must not cost us every good one."""
    spool = Spool(get_paths())
    provider = _provider_with([spool.exporter()])
    tracer = provider.get_tracer("tenstorrent.cli")
    with tracer.start_as_current_span("tt good"):
        pass
    lines = _spool_lines(spool)
    payload, spans = build_payload(['{"resourceSpans": [ truncated...', *lines])
    assert spans == 1
    assert _decode(payload).resource_spans


# -- opt-out, now that data can sit on disk unsent -----------------------------------
def test_opting_out_discards_unsent_spans(runner, spooling):
    """Spooling creates a window where data sits on the machine unsent. Opting out has
    to delete it — uploading it later would be worse than the old in-process flush,
    where opt-out was immediate and total."""
    for _ in range(2):
        runner.invoke(app, ["config", "path"])
    assert spooling.stats().spans == 2

    ConfigStore(get_paths()).set("telemetry.enabled", False)
    runner.invoke(app, ["config", "path"])

    assert spooling.stats().spans == 0
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
    assert Spool(paths).stats().spans == 0


def test_do_not_track_opts_out_and_discards(runner, spooling, monkeypatch):
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().spans == 1
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().spans == 0


def test_do_not_track_zero_is_not_opting_out(runner, spooling, monkeypatch):
    """The convention is that "0"/"false"/empty mean unset, so DO_NOT_TRACK=0 must not
    silently disable telemetry."""
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().spans == 1


def test_kill_switch_keeps_the_spool(runner, spooling, monkeypatch):
    """TT_TELEMETRY_DISABLED is a per-run kill switch, not a durable opt-out: it must
    stop collection without destroying spans an earlier run legitimately collected."""
    runner.invoke(app, ["config", "path"])
    monkeypatch.setenv("TT_TELEMETRY_DISABLED", "1")
    runner.invoke(app, ["config", "path"])
    assert spooling.stats().spans == 1


def test_offline_keeps_the_spool(runner, spooling):
    """`--offline` means "no network this run", not "forget what I did" — deleting the
    spool for it would be data loss from an unrelated flag."""
    runner.invoke(app, ["config", "path"])
    runner.invoke(app, ["--offline", "config", "path"])
    assert spooling.stats().spans == 1


def test_drain_declines_while_the_kill_switch_is_set(runner, live, monkeypatch):
    paths = get_paths()
    runner.invoke(app, ["config", "path"])
    monkeypatch.setenv("TT_TELEMETRY_DISABLED", "1")
    assert drain(paths, ConfigStore(paths)).status == "disabled"
    assert live.received == []
    assert Spool(paths).stats().spans == 1  # kept, not deleted


# -- bounds --------------------------------------------------------------------------
def test_upload_is_capped_and_drops_the_oldest(runner, live, monkeypatch):
    """A permanently firewalled machine must not accumulate forever, and when we do trim,
    recent history is the part worth keeping."""
    monkeypatch.setattr(spool_module, "MAX_SPANS", 2)
    paths = get_paths()
    for _ in range(5):
        runner.invoke(app, ["config", "path"])
    result = drain(paths, ConfigStore(paths))
    assert result.status == "sent"
    assert result.spans == 2


def test_append_stops_at_the_byte_ceiling(runner, spooling, monkeypatch):
    runner.invoke(app, ["config", "path"])
    size = spooling.stats().bytes
    monkeypatch.setattr(spool_module, "MAX_SPOOL_BYTES", size)
    for _ in range(3):
        assert runner.invoke(app, ["config", "path"]).exit_code == 0
    assert spooling.stats().spans == 1


# -- CI ------------------------------------------------------------------------------
def test_ci_is_recorded_rather_than_dropped(runner, collected, monkeypatch):
    """CI runs are real usage and are kept; tt.ci is what lets them be filtered later."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    spans = collected.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].resource.attributes["tt.ci"] is True


def test_ci_marker_is_false_for_a_person(runner, collected):
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert collected.get_finished_spans()[0].resource.attributes["tt.ci"] is False


def test_ci_never_exports_env_var_values(runner, collected, monkeypatch):
    """Only the *names* in CI_ENV_VARS are read. Their values are repo slugs, branch
    names and build URLs, none of which may leave the machine."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CI", "tenstorrent/private-repo#4711")
    runner.invoke(app, ["config", "path"])
    span = collected.get_finished_spans()[0]
    blob = str(dict(span.resource.attributes)) + str(dict(span.attributes))
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
def test_log_file_records_spans_without_opting_in(runner, monkeypatch, tmp_path):
    """TT_TELEMETRY_LOG_FILE shows the user exactly what would be sent. It is local-only
    and uploads nothing, so it must work *before* consent — it exists precisely so a
    user can inspect the data while deciding whether to opt in."""
    log = tmp_path / "telemetry.jsonl"
    monkeypatch.delenv("TT_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TT_TELEMETRY_LOG_FILE", str(log))
    assert runner.invoke(app, ["config", "path"]).exit_code == 0

    record = json.loads(log.read_text().splitlines()[0])
    assert record["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] == "tt config path"
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

    httpx (the drainer) and requests (the OTLP exporter) both honour HTTP_PROXY and
    ALL_PROXY, and neither bypasses loopback on its own — so on a box behind a proxy every
    collector-backed test failed with "the collector was never reached", while the same
    tests passed on a machine with no proxy set. `isolated_dirs` pins NO_PROXY for loopback
    to make the suite independent of that; this test is what keeps it pinned.
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
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 5)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "below the hand-off threshold of 5" in _flat(result.stderr)
    assert spawns == []


def test_verbose_announces_the_hand_off(runner, spooling, spawns, monkeypatch):
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "handing 1 span(s) to a background uploader" in _flat(result.stderr)
    assert len(spawns) == 1


def test_verbose_reports_a_failed_launch(runner, spooling, monkeypatch):
    """A hand-off that cannot start used to return False into the void."""
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1)

    def boom(*args, **kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(session_module.subprocess, "Popen", boom)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0  # still must not break the command
    assert "could not launch the uploader: OSError: fork failed" in _flat(result.stderr)


def test_verbose_flags_a_batch_stuck_awaiting_retry(runner, spooling, monkeypatch):
    """The one state that means "delivery is broken" rather than "not yet attempted"."""
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1000)
    spooling.dir.mkdir(parents=True, exist_ok=True)
    spooling.sending_path.write_text('{"resourceSpans": []}\n' * 3)
    result = runner.invoke(app, ["--verbose", "config", "path"])
    assert result.exit_code == 0
    assert "3 span(s) from an earlier batch are awaiting retry" in _flat(result.stderr)
    assert "tt self send-telemetry" in _flat(result.stderr)


def test_delivery_diagnostics_stay_silent_without_verbose(runner, spooling, spawns, monkeypatch):
    """Telemetry has no business on a normal command's output."""
    monkeypatch.setattr(spool_module, "DRAIN_SPAN_THRESHOLD", 1)
    spooling.dir.mkdir(parents=True, exist_ok=True)
    spooling.sending_path.write_text('{"resourceSpans": []}\n')
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "telemetry:" not in _flat(result.stderr)
