# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""TelemetrySession: one PostHog event per command.

Design invariants:
- **Never break the CLI.** Every operation here is guarded; any failure (bad config,
  unwritable spool, unreachable endpoint) degrades to a silent no-op. `create()` returns
  the shared NULL_SESSION sentinel whenever telemetry is off or setup fails.
- **Never block on the network.** By default (`telemetry.flush_mode = "async"`) an
  event is appended to an on-disk spool and a detached process uploads batches later;
  the command itself makes no HTTP request. `"sync"` posts in-process (bounded by a
  hard ceiling) for development, where seeing an event land immediately is the point.
- **Never touch stdout.** The only user-visible side effect is the one-time opt-in
  prompt, emitted on stderr via OutputManager.
- **Opt-in only.** Nothing is collected or sent unless the user consented: either by
  answering yes to the one-time first-run prompt (interactive runs only) or by setting
  `telemetry.enabled=true`. `DO_NOT_TRACK`, the root `--offline` flag, and the
  TT_TELEMETRY_DISABLED kill switch each keep an opted-in install silent too.
  The one thing that works *without* consent is TT_TELEMETRY_LOG_FILE, which is
  local-only (uploads nothing) and exists so a user can inspect exactly what would
  be sent before deciding.

The backend is PostHog's batch capture API (`.../batch/`, the project key travels in
the body). What an event contains is decided entirely in attributes.py; this module
decides only whether and how it leaves.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
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
# Write every event here as well, one JSON object per line — byte-for-byte what would
# go into the upload's `batch` array. A test seam, and a far more convincing disclosure
# than documentation: the user can read exactly what would be sent (cf. Flutter's
# FLUTTER_ANALYTICS_LOG_FILE). Local-only, so it works with no endpoint configured.
_LOG_FILE_ENV = "TT_TELEMETRY_LOG_FILE"

ASYNC_MODE = "async"
SYNC_MODE = "sync"

# Only used in sync mode: the most a command may wait for its in-process POST. Async
# mode never opens a socket on the live path.
_FLUSH_TIMEOUT_MS = 1500

# tt <= 1.0.1 sent OpenTelemetry spans to PostHog's OTLP traces endpoint, and a
# config.toml materialized by those releases pins that URL as `telemetry.endpoint`.
# Events posted there would be accepted and discarded, so the old default is mapped to
# the new one. Any *other* endpoint is the user's (self-hosted PostHog, a local sink)
# and passes through untouched.
_LEGACY_ENDPOINTS = {
    "https://us.i.posthog.com/i/v1/traces": "https://us.i.posthog.com/batch/",
    "https://eu.i.posthog.com/i/v1/traces": "https://eu.i.posthog.com/batch/",
}

_TELEMETRY_DOC_URL = "https://github.com/tenstorrent/tt-cli/blob/main/TELEMETRY.md"
_DISCORD_URL = "https://discord.gg/tenstorrent"
_VISION_URL = "https://openfuture.tenstorrent.com"

# What the session hands events to. A sink records locally and must be cheap (spool
# append, log-file append); the transport is the in-process POST used only in sync mode.
Sink = Callable[[dict[str, Any]], Any]
Transport = Callable[[list[dict[str, Any]]], Any]


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
    triggers deleting already-spooled events (see Spool.discard) and "skip this run" must
    not destroy data an opted-in run legitimately collected.
    """
    if not _config_bool(config, "telemetry.enabled", False):
        return True
    return env_flag("DO_NOT_TRACK")


def resolve_endpoint(config: ConfigStore) -> tuple[str, str]:
    """(endpoint, token), env overriding config. Either being empty means inert."""
    endpoint = os.environ.get(_ENDPOINT_ENV) or _config_str(config, "telemetry.endpoint")
    token = os.environ.get(_KEY_ENV) or _config_str(config, "telemetry.posthog_project_key")
    endpoint = endpoint.strip()
    endpoint = _LEGACY_ENDPOINTS.get(endpoint.rstrip("/"), endpoint)
    return endpoint, token


def flush_mode(config: ConfigStore, output: OutputManager | None = None) -> str:
    """"async" (spool + detached upload) or "sync" (post in-process).

    Precedence: TT_TELEMETRY_FLUSH_MODE, then CI, then config, then async.

    CI forces sync because async delivery cannot work there. A detached uploader is
    reaped along with the build's process group, and even if it survived, the container
    is destroyed with the spool still on disk — so spooling on CI means silently
    collecting data that is guaranteed never to arrive. A build can afford to wait for
    the (bounded) in-process post; a developer's shell cannot. The env var still
    overrides, so this stays testable and overridable.

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


class _NullEventHandle:
    """No-op handle: same surface as _EventHandle, does nothing."""

    def set_exit_code(self, code: Any) -> None:
        pass

    def record_error(self, err: Any) -> None:
        pass

    def record_exception(self, exc: Any) -> None:
        pass


class _EventHandle:
    """The in-flight command. @handle_tt_errors stamps the outcome on it; the session
    turns it into an event when the command's context closes."""

    def __init__(self, click_ctx: Any) -> None:
        self.click_ctx = click_ctx
        self.code = ExitCode.OK
        # The TTError or the crash, for attributes.error_properties — which reads a
        # `reason` slug or a class name off it and nothing else.
        self.error: Any = None
        self.finished = False
        self._started = time.perf_counter()

    def set_exit_code(self, code: Any) -> None:
        try:
            self.code = ExitCode(int(code))
        except Exception:
            self.code = ExitCode.ERROR

    def record_error(self, err: Any) -> None:
        """A TTError: the documented failure path."""
        self.error = err
        self.set_exit_code(getattr(err, "exit_code", ExitCode.ERROR))

    def record_exception(self, exc: Any) -> None:
        """Anything else that escaped the command: a crash."""
        self.error = exc
        self.set_exit_code(ExitCode.ERROR)

    def duration_ms(self) -> int:
        return int(round((time.perf_counter() - self._started) * 1000))


class _NullSession:
    """The disabled session: every hook is a no-op."""

    @contextlib.contextmanager
    def command_event(self, click_ctx: Any) -> Iterator[Any]:
        yield _NullEventHandle()

    def flush(self) -> None:
        pass


class TelemetrySession(_NullSession):
    def __init__(
        self,
        *,
        instance_id: str,
        sinks: list[Sink] | None = None,
        transport: Transport | None = None,
        spool: Spool | None = None,
        output: OutputManager | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._sinks = list(sinks or [])
        self._transport = transport
        self._spool = spool
        # Events recorded this process and not yet posted. Only ever non-empty in sync
        # mode; async mode's sinks have already written them to disk.
        self._pending: list[dict[str, Any]] = []
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
        transport: Transport | None = None,
    ) -> "_NullSession":
        """Build a session, or return NULL_SESSION if telemetry is off / setup fails.

        `transport` injects the in-process sender (tests); it implies sync delivery.
        """
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
                    # TT_TELEMETRY_LOG_FILE seam may record events (usually NULL_SESSION).
                    return cls._build(paths, config, output, transport=None, export=False)
                # The user just opted in; the prompt already excluded offline.
                return cls._build(paths, config, output, transport)
            if os.environ.get(_DISABLE_ENV) or offline:
                return NULL_SESSION
            return cls._build(paths, config, output, transport)
        except Exception:
            # Telemetry must never break the CLI.
            return NULL_SESSION

    @classmethod
    def _build(
        cls,
        paths: Paths,
        config: ConfigStore,
        output: OutputManager | None,
        transport: Transport | None,
        *,
        export: bool = True,
    ) -> "_NullSession":
        mode = flush_mode(config, output)
        spool: Spool | None = None
        send: Transport | None = None

        # export=False is the no-consent path: nothing may leave the machine (or even
        # accumulate on disk waiting to), so neither a transport nor the spool is wired
        # up — only the local TT_TELEMETRY_LOG_FILE below can record anything.
        if not export:
            pass
        elif transport is not None:
            send = transport
        elif mode == SYNC_MODE:
            send = cls._transport(config)
        else:
            endpoint, token = resolve_endpoint(config)
            if endpoint and token:
                # Spool only when there is somewhere for the batch to go; otherwise we
                # would accumulate events on disk that can never be delivered.
                spool = Spool(paths)

        log_path = os.environ.get(_LOG_FILE_ENV)
        if send is None and spool is None and not log_path:
            # Nothing to export to, nothing to disclose.
            return NULL_SESSION

        sinks: list[Sink] = []
        if spool is not None:
            sinks.append(spool.append)
        if log_path:
            sinks.append(_log_file_sink(log_path))

        return cls(
            instance_id=TelemetryState(paths).instance_id(),
            sinks=sinks,
            transport=send,
            spool=spool,
            output=output,
        )

    @staticmethod
    def _transport(config: ConfigStore) -> Transport | None:
        """The in-process sender for sync mode, or None when nothing is configured.
        One seam: tests replace this to capture events instead of posting them."""
        endpoint, token = resolve_endpoint(config)
        if not endpoint or not token:
            return None
        from .drain import post_batch

        return lambda events: post_batch(endpoint, token, events)

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

    # -- per-command event ------------------------------------------------------
    @contextlib.contextmanager
    def command_event(self, click_ctx: Any) -> Iterator[Any]:
        # A group callback runs on the way through to its subcommand (`tt config` before
        # `tt config get`) and is decorated too, so it would record a second event for
        # the same invocation and over-count the group as a command in its own right.
        # The leaf's event is the one that represents what the user ran. When the group
        # is invoked bare (invoke_without_command, no subcommand) it *is* the leaf and
        # keeps its event.
        if getattr(click_ctx, "invoked_subcommand", None) is not None:
            yield _NullEventHandle()
            return
        handle: Any
        try:
            handle = _EventHandle(click_ctx)
        except Exception:
            handle = _NullEventHandle()
        try:
            yield handle
        finally:
            if isinstance(handle, _EventHandle):
                self._record(handle)

    def _record(self, handle: _EventHandle) -> None:
        """Turn the finished command into an event and hand it to every sink.

        Idempotent: a command that hands the terminal over (exec_tty) closes its
        context early via AppContext.before_exec, and nothing may record twice if the
        normal exit path then runs after all (as it does under CliRunner).
        """
        try:
            if handle.finished:
                return
            handle.finished = True
            event = attributes.build_event(
                handle.click_ctx,
                instance_id=self._instance_id,
                exit_code=handle.code,
                error=handle.error,
                duration_ms=handle.duration_ms(),
            )
        except Exception:
            return
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:
                pass
        if self._transport is not None:
            self._pending.append(event)

    def flush(self) -> None:
        """End-of-command hook. Called from @handle_tt_errors' `finally`, so it runs on
        all four exit paths (OK / TTError / typer.Exit / unexpected) and must be cheap
        and silent on every one of them."""
        try:
            if self._transport is not None:
                self._flush_direct()
            if self._spool is not None:
                self._hand_off()
        except Exception:
            # A slow/unreachable endpoint must never delay or fail process exit.
            pass

    def _flush_direct(self) -> None:
        """Sync mode: post pending events, giving up after _FLUSH_TIMEOUT_MS.

        The POST runs on a daemon thread we simply stop waiting on — the ceiling is
        ours, not the HTTP client's, and an abandoned thread dies with the process
        without delaying exit.
        """
        pending, self._pending = self._pending, []
        if not pending or self._transport is None:
            return
        done = threading.Event()
        send = self._transport

        def _run() -> None:
            try:
                send(pending)
            except Exception:
                pass
            finally:
                done.set()

        threading.Thread(target=_run, name="tt-telemetry-flush", daemon=True).start()
        done.wait(_FLUSH_TIMEOUT_MS / 1000)

    def _hand_off(self) -> None:
        """Async mode: spawn a detached uploader, but only when it is worth it.

        The event is already on disk by now (the spool sink wrote it as the command's
        context closed), so doing nothing here is always a valid outcome — the next
        command that crosses a threshold hands off instead. At ~50 commands/day that is
        ~2 spawns.
        """
        assert self._spool is not None
        stats = self._spool.stats()
        # The probe is an optimisation only: two commands finishing together can both
        # see "not in progress" and spawn, and the drainer's own flock is what makes
        # that safe.
        in_progress = self._spool.drain_in_progress()
        if self._output is not None and self._output.verbose and not in_progress:
            # A batch left under the in-flight name with no drainer holding the lock
            # means a previous upload failed and is waiting to be retried — the one
            # symptom that says "delivery is broken" rather than "delivery hasn't
            # happened yet", and otherwise invisible.
            self._report_pending_retry()
        if not stats.ready_to_drain:
            self._debug(
                f"telemetry: {stats.events} event(s) spooled, below the hand-off "
                f"threshold of {spool_module.DRAIN_EVENT_THRESHOLD}; nothing to do"
            )
            return
        if in_progress:
            self._debug("telemetry: an uploader is already running; leaving it to finish")
            return
        self._debug(f"telemetry: handing {stats.events} event(s) to a background uploader")
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
            f"telemetry: {waiting} event(s) from an earlier batch are awaiting retry "
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


def _log_file_sink(path: str) -> Sink:
    """Append each event to TT_TELEMETRY_LOG_FILE as one JSON line, opening the file
    per event so a long command never holds a descriptor on it."""

    def append(event: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    return append


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
