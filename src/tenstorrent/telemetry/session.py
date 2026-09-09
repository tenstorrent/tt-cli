# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""TelemetrySession: one OpenTelemetry span per command.

Design invariants:
- **Never break the CLI.** Every operation here is guarded; any failure (missing SDK,
  bad config, unreachable collector) degrades to a silent no-op. `create()` returns the
  shared NULL_SESSION sentinel whenever telemetry is off or setup fails.
- **Never block on the network.** By default (`telemetry.flush_mode = "async"`) a span
  is appended to an on-disk spool and a detached process uploads batches later; the
  command itself makes no HTTP request. `"sync"` restores the in-process export for
  development, where seeing a span land in a collector immediately is the point.
- **Never touch stdout.** The only user-visible side effect is the one-time opt-in
  prompt, emitted on stderr via OutputManager.
- **Opt-in only.** Nothing is collected or sent unless the user consented: either by
  answering yes to the one-time first-run prompt (interactive runs only) or by setting
  `telemetry.enabled=true`. `DO_NOT_TRACK`, the root `--offline` flag, and the
  TT_TELEMETRY_DISABLED kill switch each keep an opted-in install silent too.
  The one thing that works *without* consent is TT_TELEMETRY_LOG_FILE, which is
  local-only (uploads nothing) and exists so a user can inspect exactly what would
  be sent before deciding.

The default backend is PostHog, a generic OTLP/HTTP trace receiver: we point at
`.../i/v1/traces` with an `Authorization: Bearer <project token>` header. No PostHog
SDK, so the endpoint can be repointed at any OTLP collector later.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
from typing import Any, Callable, Iterator

from .._compat import Abort, confirm, style
from ..config.paths import Paths
from ..config.store import ConfigStore
from ..errors import ExitCode
from ..output import OutputManager
from . import attributes
from .env import env_flag, is_ci
from . import spool as spool_module
from .spool import Spool
from .state import TelemetryState

_DISABLE_ENV = "TT_TELEMETRY_DISABLED"
_ENDPOINT_ENV = "TT_TELEMETRY_ENDPOINT"
_KEY_ENV = "TT_TELEMETRY_POSTHOG_KEY"
_FLUSH_MODE_ENV = "TT_TELEMETRY_FLUSH_MODE"
# Write every span here as well, in the spec'd OTLP/JSON Lines format. A second test
# seam alongside the in-memory exporter, and a far more convincing disclosure than
# documentation: the user can read exactly what would be sent (cf. Flutter's
# FLUTTER_ANALYTICS_LOG_FILE). Local-only, so it works with no endpoint configured.
_LOG_FILE_ENV = "TT_TELEMETRY_LOG_FILE"

ASYNC_MODE = "async"
SYNC_MODE = "sync"

# Only used in sync mode. Async mode never calls force_flush on the live path.
_FLUSH_TIMEOUT_MS = 1500
# Per-HTTP-attempt budget. Only bounds the abandoned flush thread (see flush()); the
# user-visible ceiling is _FLUSH_TIMEOUT_MS regardless of what this is set to.
_EXPORT_TIMEOUT_S = 2

# OTel logs export failures at ERROR, which Python's lastResort handler prints to
# stderr — four lines of connection-pool detail on every command for anyone behind a
# firewall that blocks the collector. Telemetry is best-effort and must stay invisible,
# so these are silenced unless --verbose asked for diagnostics.
_NOISY_LOGGERS = (
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.otlp.json.file.trace_exporter",
    "opentelemetry.sdk.trace.export",
)

_TELEMETRY_DOC_URL = "https://github.com/tenstorrent/tt-cli/blob/main/TELEMETRY.md"
_DISCORD_URL = "https://discord.gg/tenstorrent"
_VISION_URL = "https://openfuture.tenstorrent.com"


def _link(url: str) -> str:
    """A URL the terminal can open on click (OSC 8, via Rich's link markup) that still
    reads as a plain URL where that isn't supported — the visible text is the URL
    itself, so nothing is lost when the escape is ignored or stripped (a pipe, a log).
    Rich measures the visible text, so the wrapping of the surrounding copy is unchanged.
    """
    return f"[link={url}]{url}[/link]"


_PROMPT_INTRO = (
f"""
    Tenstorrent would like to collect anonymous, PII-free telemetry within tt-cli.
    We’re real people (come say hi at {_link(_DISCORD_URL)}),
    and this data helps us improve the Tenstorrent developer experience
    to build our vision for an open future for AI.

    See {_link(_TELEMETRY_DOC_URL)}
    for what is (and isn't) collected, and read
    {_link(_VISION_URL)} for more about our vision.
"""
)
_PROMPT_QUESTION = "    Enable anonymous usage telemetry?"
_OPTED_IN_MSG = "    Telemetry enabled. Disable any time: tt config set telemetry.enabled false"
_OPTED_OUT_MSG = "    Telemetry stays off. Enable any time: tt config set telemetry.enabled true"


def _interactive() -> bool:
    """A person is plausibly at the other end: prompt only with real TTYs on both the
    channel we ask on (stderr) and the one we read the answer from (stdin)."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except Exception:
        return False


# -- gates, shared with drain.py -----------------------------------------------------
def opted_out(config: ConfigStore) -> bool:
    """Is there no durable opt-in in force?

    True unless `telemetry.enabled=true` (opt-in is the burden of proof: the schema
    default is false, so a fresh install, a missing key, and an explicit `false` all
    land here) — and true regardless whenever the cross-tool DO_NOT_TRACK convention is
    set. Deliberately narrower than "telemetry is inactive right now": the per-run
    switches (TT_TELEMETRY_DISABLED, `--offline`) don't count, because this gate also
    triggers deleting already-spooled spans (see Spool.discard) and "skip this run" must
    not destroy data an opted-in run legitimately collected.
    """
    if not _config_bool(config, "telemetry.enabled", False):
        return True
    return env_flag("DO_NOT_TRACK")


def resolve_endpoint(config: ConfigStore) -> tuple[str, str]:
    """(endpoint, token), env overriding config. Either being empty means inert."""
    endpoint = os.environ.get(_ENDPOINT_ENV) or _config_str(config, "telemetry.endpoint")
    token = os.environ.get(_KEY_ENV) or _config_str(config, "telemetry.posthog_project_key")
    return endpoint, token


def flush_mode(config: ConfigStore, output: OutputManager | None = None) -> str:
    """"async" (spool + detached upload) or "sync" (export in-process).

    Precedence: TT_TELEMETRY_FLUSH_MODE, then CI, then config, then async.

    CI forces sync because async delivery cannot work there. A detached uploader is
    reaped along with the build's process group, and even if it survived, the container
    is destroyed with the spool still on disk — so spooling on CI means silently
    collecting data that is guaranteed never to arrive. A build can afford to wait for
    the (externally bounded) in-process export; a developer's shell cannot. The env var
    still overrides, so this stays testable and overridable.

    Anything unrecognised falls back to async: the failure mode of async is delayed
    data, while the failure mode of sync is a slow CLI for every user.
    """
    override = (os.environ.get(_FLUSH_MODE_ENV) or "").strip().lower()
    if override in (ASYNC_MODE, SYNC_MODE):
        return override
    if not override and is_ci():
        return SYNC_MODE
    raw = override or (_config_str(config, "telemetry.flush_mode") or ASYNC_MODE).strip().lower()
    if raw in (ASYNC_MODE, SYNC_MODE):
        return raw
    if output is not None and output.verbose:
        with contextlib.suppress(Exception):
            output.warn(f"Unknown telemetry.flush_mode {raw!r}; using {ASYNC_MODE!r}.")
    return ASYNC_MODE


class _NullSpanHandle:
    """No-op span handle: same surface as _SpanHandle, does nothing."""

    def set_exit_code(self, code: Any) -> None:
        pass

    def record_error(self, err: Any) -> None:
        pass


class _SpanHandle:
    """Wraps a live OTel span. The decorator stamps the exit code before the span ends;
    `_finalize` writes it (plus an ERROR status for non-zero codes) at span close."""

    def __init__(self, span: Any) -> None:
        self._span = span
        self._code = ExitCode.OK

    def set_exit_code(self, code: Any) -> None:
        try:
            self._code = ExitCode(int(code))
        except Exception:
            self._code = ExitCode.ERROR

    def record_error(self, err: Any) -> None:
        self.set_exit_code(getattr(err, "exit_code", ExitCode.ERROR))

    def _finalize(self) -> None:
        try:
            for key, value in attributes.error_attributes(self._code).items():
                self._span.set_attribute(key, value)
            if self._code != ExitCode.OK:
                from opentelemetry.trace import Status, StatusCode

                # Status without a description: the code category is enough, and the
                # message text could leak paths/argv.
                self._span.set_status(Status(StatusCode.ERROR))
        except Exception:
            pass


class _NullSession:
    """The disabled session: every hook is a no-op."""

    @contextlib.contextmanager
    def command_span(self, click_ctx: Any) -> Iterator[Any]:
        yield _NullSpanHandle()

    def flush(self) -> None:
        pass


class TelemetrySession(_NullSession):
    def __init__(
        self,
        provider: Any,
        tracer: Any,
        *,
        spool: Spool | None = None,
        needs_flush: bool = False,
        output: OutputManager | None = None,
    ) -> None:
        self._provider = provider
        self._tracer = tracer
        self._spool = spool
        self._needs_flush = needs_flush
        # Only ever written to via _debug(), i.e. only under --verbose. Telemetry has no
        # business on a normal command's output.
        self._output = output

    def _debug(self, message: str) -> None:
        """Surface a delivery decision under --verbose, and only there.

        Delivery is otherwise completely silent by design, which is right for users and
        miserable for diagnosis: a hand-off that never fires, an uploader that fails to
        launch, and a healthy spool waiting for its threshold are indistinguishable from
        the outside. `-v` is the seam that tells them apart. (Yarn Berry's telemetry
        upload was dead code for two years partly because nothing could report on it.)
        """
        try:
            if self._output is not None:
                self._output.debug(message)
        except Exception:
            pass

    # -- construction -----------------------------------------------------------
    @classmethod
    def create(
        cls,
        paths: Paths,
        config: ConfigStore,
        *,
        offline: bool,
        output: OutputManager | None,
        exporter: Any = None,
    ) -> "_NullSession":
        """Build a session, or return NULL_SESSION if telemetry is off / setup fails."""
        try:
            # opted_out() is checked before the kill switch on purpose: it is the first
            # config read on many commands, and that read is what surfaces the stray-key
            # warning (ConfigStore.unknown_keys) even under TT_TELEMETRY_DISABLED.
            if opted_out(config):
                # No consent in force (the default) or an explicit opt-out: nothing may
                # be collected or kept. The discard matters after a *revoked* opt-in —
                # data that has not left the machine must not survive the decision to
                # stop sending; for a never-opted-in install it is a no-op.
                with contextlib.suppress(Exception):
                    Spool(paths).discard()
                if os.environ.get(_DISABLE_ENV):
                    # Per-run kill switch: no prompt, no log file, nothing.
                    return NULL_SESSION
                if not cls._maybe_prompt_opt_in(paths, config, output, offline=offline):
                    # Still no consent. export=False: only the local, upload-nothing
                    # TT_TELEMETRY_LOG_FILE seam may record spans (usually NULL_SESSION).
                    return cls._build(paths, config, output, exporter=None, export=False)
                # The user just opted in; the prompt already excluded offline.
                return cls._build(paths, config, output, exporter)
            if os.environ.get(_DISABLE_ENV) or offline:
                return NULL_SESSION
            return cls._build(paths, config, output, exporter)
        except Exception:
            # Telemetry must never break the CLI.
            return NULL_SESSION

    @classmethod
    def _build(
        cls,
        paths: Paths,
        config: ConfigStore,
        output: OutputManager | None,
        exporter: Any,
        *,
        export: bool = True,
    ) -> "_NullSession":
        mode = flush_mode(config, output)
        spool: Spool | None = None
        direct: Any = None

        # export=False is the no-consent path: nothing may leave the machine (or even
        # accumulate on disk waiting to), so neither an exporter nor the spool is wired
        # up — only the local TT_TELEMETRY_LOG_FILE below can record anything.
        if not export:
            pass
        elif exporter is not None:
            direct = exporter
        elif mode == SYNC_MODE:
            direct = cls._otlp_exporter(config)
        else:
            endpoint, token = resolve_endpoint(config)
            if endpoint and token:
                # Spool only when there is somewhere for the batch to go; otherwise we
                # would accumulate spans on disk that can never be delivered.
                spool = Spool(paths)

        log_path = os.environ.get(_LOG_FILE_ENV)
        if direct is None and spool is None and not log_path:
            # Nothing to export to, nothing to disclose. Returning before the SDK
            # imports also keeps ~12 ms off every command while telemetry ships dark.
            return NULL_SESSION

        _quiet_exporter_logs(output)

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        state = TelemetryState(paths)
        # Resource(), NOT Resource.create(): create() merges OpenTelemetry's environment
        # detectors, so anything in the user's OTEL_RESOURCE_ATTRIBUTES (a deployment
        # name, a user.name) would ride along and defeat attributes.py as the single
        # anonymization chokepoint. The plain constructor sends only what we hand it.
        resource = Resource(attributes.resource_attributes(state.instance_id()))
        # shutdown_on_exit=False: the SDK's atexit hook would block process exit draining
        # the queue against an unreachable collector, outside any budget of ours. The
        # worker is a daemon thread, so dropping the hook just abandons pending spans.
        provider = TracerProvider(resource=resource, shutdown_on_exit=False)

        if direct is not None:
            # Batched + externally bounded: this is the only processor that can block on
            # a socket, so it is the only one that needs flush()'s escape hatch.
            provider.add_span_processor(BatchSpanProcessor(direct))
        for local in (spool.exporter() if spool else None, _log_file_exporter(log_path)):
            if local is not None:
                # Local file appends cost ~0.01 ms, so they run inline on span end:
                # no worker thread to start and no force_flush to bound.
                provider.add_span_processor(SimpleSpanProcessor(local))

        return cls(
            provider,
            provider.get_tracer("tenstorrent.cli"),
            spool=spool,
            needs_flush=direct is not None,
            output=output,
        )

    @staticmethod
    def _otlp_exporter(config: ConfigStore) -> Any:
        endpoint, token = resolve_endpoint(config)
        if not endpoint or not token:
            return None
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Pass the full traces path as `endpoint` so the exporter does not append
        # /v1/traces (PostHog serves traces at /i/v1/traces).
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {token}"},
            timeout=_EXPORT_TIMEOUT_S,
        )

    @classmethod
    def _maybe_prompt_opt_in(
        cls,
        paths: Paths,
        config: ConfigStore,
        output: OutputManager | None,
        *,
        offline: bool,
    ) -> bool:
        """One-time interactive consent prompt. True only if the user opts in right now.

        Asked at most once per install (TelemetryState remembers that it was answered,
        whichever way it went), and only when the answer could matter and a person is
        actually there: never on CI, never under --quiet/--json (pipelines stay clean),
        never without real TTYs, never when a per-run switch (--offline, DO_NOT_TRACK)
        already says no this run, and never with no endpoint/key configured — nothing
        could be sent, so there is nothing to consent to. A "yes" is persisted as
        `telemetry.enabled=true` in config.toml, so it is visible, commented, and
        revocable exactly like a hand-made opt-in.
        """
        try:
            if offline or env_flag("DO_NOT_TRACK") or is_ci():
                return False
            if output is None or output.quiet or output.json_mode:
                return False
            if not _interactive():
                return False
            state = TelemetryState(paths)
            if state.prompt_answered():
                return False
            endpoint, token = resolve_endpoint(config)
            if not endpoint or not token:
                return False

            output.status(_PROMPT_INTRO)
            try:
                # default=None: consent must be an explicit y/n — a bare Enter re-asks
                # instead of silently picking an answer for the user.
                answer = bool(confirm(style(_PROMPT_QUESTION, bold=True), default=None, err=True))
            except Abort:
                # EOF / Ctrl-C at the prompt is "not now", not an answer: stay off and
                # ask again on a later interactive run.
                return False
            state.mark_prompt_answered()
            if answer:
                config.set("telemetry.enabled", True)
            output.status(_OPTED_IN_MSG if answer else _OPTED_OUT_MSG, style="dim")
            return answer
        except Exception:
            # Telemetry must never break the CLI; an unanswerable prompt means "off".
            return False

    # -- per-command span -------------------------------------------------------
    @contextlib.contextmanager
    def command_span(self, click_ctx: Any) -> Iterator[Any]:
        # A group callback runs on the way through to its subcommand (`tt config` before
        # `tt config get`) and is decorated too, so it would open a second span for the
        # same invocation and over-count the group as a command in its own right. The
        # leaf's span is the one that represents what the user ran. When the group is
        # invoked bare (invoke_without_command, no subcommand) it *is* the leaf and keeps
        # its span.
        if getattr(click_ctx, "invoked_subcommand", None) is not None:
            yield _NullSpanHandle()
            return
        span_cm = None
        handle: Any = _NullSpanHandle()
        try:
            command = getattr(click_ctx, "command_path", None) if click_ctx else None
            span_cm = self._tracer.start_as_current_span(command or "tt")
            span = span_cm.__enter__()
            try:
                for key, value in attributes.command_attributes(click_ctx).items():
                    span.set_attribute(key, value)
            except Exception:
                pass
            handle = _SpanHandle(span)
        except Exception:
            span_cm = None
            handle = _NullSpanHandle()
        try:
            yield handle
        finally:
            if isinstance(handle, _SpanHandle):
                handle._finalize()
            if span_cm is not None:
                # Pass no exception info: we record the outcome via the exit code,
                # never a stack/message that could carry PII.
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass

    def flush(self) -> None:
        """End-of-command hook. Called from @handle_tt_errors' `finally`, so it runs on
        all four exit paths (OK / TTError / typer.Exit / unexpected) and must be cheap
        and silent on every one of them."""
        try:
            if self._needs_flush:
                self._flush_direct()
            if self._spool is not None:
                self._hand_off()
        except Exception:
            # A slow/unreachable collector must never delay or fail process exit.
            pass

    def _flush_direct(self) -> None:
        """Sync mode: export pending spans, giving up after _FLUSH_TIMEOUT_MS.

        `force_flush(timeout_millis=...)` does NOT honour its own timeout and always
        returns True: against a collector that drops or rejects packets it blocks for
        the exporter's full retry sequence (measured at 20s on SDK 1.44). So the flush
        runs on a daemon thread we simply stop waiting on — the ceiling is ours, and an
        abandoned thread dies with the process without delaying exit.
        """
        done = threading.Event()

        def _run() -> None:
            try:
                self._provider.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS)
            except Exception:
                pass
            finally:
                done.set()

        threading.Thread(target=_run, name="tt-telemetry-flush", daemon=True).start()
        done.wait(_FLUSH_TIMEOUT_MS / 1000)

    def _hand_off(self) -> None:
        """Async mode: spawn a detached uploader, but only when it is worth it.

        The span is already on disk by now (SimpleSpanProcessor wrote it on span end),
        so doing nothing here is always a valid outcome — the next command that crosses
        a threshold hands off instead. At ~50 commands/day that is ~2 spawns.
        """
        assert self._spool is not None
        stats = self._spool.stats()
        if self._output is not None and self._output.verbose:
            # A batch left under the in-flight name means a previous upload failed and is
            # waiting to be retried — the one symptom that says "delivery is broken"
            # rather than "delivery hasn't happened yet", and otherwise invisible.
            self._report_pending_retry()
        if not stats.ready_to_drain:
            self._debug(
                f"telemetry: {stats.spans} span(s) spooled, below the hand-off "
                f"threshold of {spool_module.DRAIN_SPAN_THRESHOLD}; nothing to do"
            )
            return
        # Optimisation only: two commands finishing together can both get past this and
        # spawn, and the drainer's own flock is what makes that safe.
        if self._spool.drain_in_progress():
            self._debug("telemetry: an uploader is already running; leaving it to finish")
            return
        self._debug(f"telemetry: handing {stats.spans} span(s) to a background uploader")
        spawn_drainer(self._spool, on_debug=self._debug)

    def _report_pending_retry(self) -> None:
        assert self._spool is not None
        try:
            if not self._spool.sending_path.exists():
                return
            waiting = self._spool.sending_path.read_bytes().count(b"\n")
        except OSError:
            return
        self._debug(
            f"telemetry: {waiting} span(s) from an earlier batch are awaiting retry "
            f"({self._spool.sending_path}) — the last upload did not succeed. "
            "Run `tt self send-telemetry` to see why."
        )


def spawn_drainer(spool: Spool, *, on_debug: Callable[[str], None] | None = None) -> bool:
    """Start the detached uploader. True if it was launched.

    Failures are reported through `on_debug` rather than raised: a hand-off that cannot
    start must not break the command, but silently returning False is how a CLI ends up
    with telemetry that has quietly not worked for a year.
    """

    def debug(message: str) -> None:
        if on_debug is not None:
            on_debug(message)

    if not sys.executable:
        debug("telemetry: no interpreter available to launch the uploader")
        return False
    try:
        subprocess.Popen(
            # `python -m tenstorrent` rather than argv[0]: sys.executable is the
            # interpreter tt is installed into, so the package is importable from it
            # under uv-tool, pipx and venv layouts alike, whereas argv[0] may be a
            # relative path into a directory the child no longer sits in.
            [sys.executable, "-m", "tenstorrent", "self", "send-telemetry"],
            # All three descriptors must go to devnull. A detached child that inherits
            # stdout keeps the pipe open, so `$(tt ...)` blocks until the child exits —
            # this is Homebrew#29, where analytics added ~250 ms to `brew search`.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # New session: survives the parent exiting, and a Ctrl-C in the parent's
            # process group never reaches it mid-upload.
            start_new_session=True,
            close_fds=True,
            # Somewhere guaranteed to exist and to stay put; the child inherits the
            # environment (including TT_*_DIR and any endpoint override) unchanged, so
            # it reads the same config and the same spool.
            cwd=str(spool.dir),
        )
        return True
    except Exception as exc:
        debug(f"telemetry: could not launch the uploader: {type(exc).__name__}: {exc}")
        return False


def _log_file_exporter(path: str | None) -> Any:
    """OTLP/JSON-Lines exporter for TT_TELEMETRY_LOG_FILE, or None."""
    if not path:
        return None
    try:
        from opentelemetry.exporter.otlp.json.file.trace_exporter import FileSpanExporter

        return FileSpanExporter(path)
    except Exception:
        return None


def _quiet_exporter_logs(output: OutputManager | None) -> None:
    """Keep OTel's export failures off stderr unless the user asked for diagnostics."""
    try:
        if output is not None and output.verbose:
            return
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.CRITICAL)
    except Exception:
        pass


def _config_bool(config: ConfigStore, key: str, default: bool) -> bool:
    try:
        value = config.get(key)
    except Exception:
        return default
    return value if isinstance(value, bool) else default


def _config_str(config: ConfigStore, key: str) -> str:
    try:
        value = config.get(key)
    except Exception:
        return ""
    return str(value) if value else ""


NULL_SESSION = _NullSession()
