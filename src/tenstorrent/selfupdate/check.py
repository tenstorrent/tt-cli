# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Is there a newer tt? Once a day, in the background, never in the command's path.

Shape borrowed from gh and Node's update-notifier: the command that notices the state
file is stale spawns a detached `tt self check-update` and moves on; the *next*
interactive command prints a one-line notice. The version source is PyPI's JSON API,
filtered the way pip filters its own self-check — final releases only, nothing yanked,
nothing this interpreter cannot install (`requires_python`).

Two contracts, one module. The *passive* path (`after_command`, run after every leaf)
never opens a socket in-process: it reads the state file and at most spawns the
detached check. The *explicit* network commands — `tt update`, `tt self update` — may
look up the release inline, `tt update` only when the cached state is stale (see
update.py `_latest_for_update`).

This is not telemetry: the request carries nothing but tt's version in the User-Agent,
and it has its own switch (`update.check` / TT_NO_UPDATE_CHECK). It is still a network
request the user did not ask for, so it is disclosed in docs/DEVELOPERS.md and is off under
--offline, in CI, and wherever nobody could read the notice.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import tomlkit

from .. import __version__
from ..config.paths import Paths
from ..config.store import ConfigStore
from ..telemetry.env import env_flag, is_ci
from .layout import PACKAGE, InstallLayout, detect_layout

DEFAULT_SOURCE = f"https://pypi.org/pypi/{PACKAGE}/json"
# A URL (fetched) or a local path (read) that serves a PyPI-shaped JSON document.
# Tests point it at a file; an air-gapped site could point it at a mirror.
SOURCE_ENV = "TT_UPDATE_CHECK_URL"
DISABLE_ENV = "TT_NO_UPDATE_CHECK"
CONFIG_KEY = "update.check"
CHECK_INTERVAL_S = 24 * 60 * 60
NOTICE_INTERVAL_S = 24 * 60 * 60
FETCH_TIMEOUT_S = 10
# Leaves that manage the check themselves, or must not trigger one (a background
# `check-update` spawning another check would be the telemetry drainer's feedback
# loop all over again).
_EXEMPT_COMMANDS = frozenset({"tt update", "tt self update", "tt self check-update"})


def check_enabled(config: ConfigStore, *, offline: bool = False) -> bool:
    """Every switch that turns the background check off, in one place."""
    if offline or env_flag(DISABLE_ENV) or is_ci():
        return False
    try:
        return bool(config.get(CONFIG_KEY))
    except Exception:
        return False


def _interactive() -> bool:
    """A notice on stderr is only worth printing when a person will see it."""
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


# -- version selection ---------------------------------------------------------------
def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def latest_release(doc: dict[str, Any], *, python_version: str | None = None) -> str | None:
    """Newest final release this interpreter can install, or None.

    Reads PyPI's project JSON (`releases` → version → list of files). A release counts
    if it is not a pre/dev release, has at least one file that is not yanked, and that
    file's `requires_python` admits the running interpreter."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    python_version = python_version or _python_version()
    best: Version | None = None
    for raw, files in (doc.get("releases") or {}).items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if not isinstance(files, list) or not files:
            continue
        installable = False
        for file in files:
            if not isinstance(file, dict) or file.get("yanked"):
                continue
            spec = file.get("requires_python")
            if spec:
                try:
                    if python_version not in SpecifierSet(spec, prereleases=True):
                        continue
                except InvalidSpecifier:
                    continue
            installable = True
            break
        if installable and (best is None or version > best):
            best = version
    return str(best) if best is not None else None


def is_newer(candidate: str | None, current: str | None) -> bool:
    from packaging.version import InvalidVersion, Version

    if not candidate or not current:
        return False
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return False


# -- source ------------------------------------------------------------------------
def source_location() -> str:
    return os.environ.get(SOURCE_ENV) or DEFAULT_SOURCE


def fetch_document(source: str | None = None, *, timeout: float = FETCH_TIMEOUT_S) -> dict:
    """Load the PyPI-shaped document from a URL or a local file. Raises on failure;
    callers decide whether that is silent (background) or an error (`tt self update`)."""
    import json

    source = source or source_location()
    if source.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(
            source,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": f"tt-cli/{__version__}", "Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    return json.loads(Path(source).read_text())


# -- state file --------------------------------------------------------------------
class UpdateState:
    """`$TT_DATA_DIR/self-update.toml`: when we last looked, what we found, and which
    tt version was running then — a cached `latest` is only meaningful next to the
    version it was compared against, so an upgrade invalidates it by construction."""

    def __init__(self, paths: Paths) -> None:
        self._file = paths.self_update_file
        self._lock_path = paths.self_update_lock

    @contextlib.contextmanager
    def lock(self) -> Iterator[bool]:
        """Hold the check lock; yields False if another check already has it.

        Same shape as telemetry's Spool.lock. While a slow lookup is in flight the state
        file is still stale, so without this every command run in the meantime would
        spawn another check. The *child* taking the lock is what actually prevents the
        pile-up; the parent's `check_in_progress()` probe is only an optimisation.
        Degrades to unlocked where fcntl is unavailable — the worst case there is a
        duplicate lookup, not a wrong answer."""
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield True
            return
        handle = None
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(self._lock_path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    handle.close()

    def check_in_progress(self) -> bool:
        """Best-effort probe: is a background check holding the lock right now?"""
        with self.lock() as acquired:
            return not acquired

    def load(self) -> dict:
        try:
            if not self._file.exists():
                return {}
            return dict(tomlkit.parse(self._file.read_text()).unwrap().get("check", {}))
        except Exception:
            return {}

    def save(self, data: dict) -> None:
        doc = tomlkit.document()
        doc["check"] = data
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(tomlkit.dumps(doc))

    def is_stale(self, now: float | None = None) -> bool:
        checked_at = self.load().get("checked_at")
        if not isinstance(checked_at, (int, float)):
            return True
        return (now if now is not None else time.time()) - checked_at >= CHECK_INTERVAL_S

    def record(self, *, latest: str | None, current: str, now: float | None = None) -> None:
        data = self.load()
        data["checked_at"] = now if now is not None else time.time()
        data["current"] = current
        if latest is None:
            data.pop("latest", None)
        else:
            data["latest"] = latest
        self.save(data)

    def notice_due(self, latest: str, now: float | None = None) -> bool:
        """Show the notice for `latest` at most once a day — a user who has chosen not
        to upgrade should not be told again on every command. A *different* newer
        version is news, and is announced at once."""
        data = self.load()
        if data.get("notified_latest") != latest:
            return True
        notified_at = data.get("notified_at")
        if not isinstance(notified_at, (int, float)):
            return True
        return (now if now is not None else time.time()) - notified_at >= NOTICE_INTERVAL_S

    def mark_notified(self, latest: str, now: float | None = None) -> None:
        data = self.load()
        data["notified_latest"] = latest
        data["notified_at"] = now if now is not None else time.time()
        self.save(data)

    def pending(self, current: str) -> str | None:
        """A newer version recorded against *this* running version, else None."""
        data = self.load()
        if data.get("current") != current:
            return None
        latest = data.get("latest")
        return str(latest) if is_newer(str(latest) if latest else None, current) else None


@dataclass(frozen=True)
class CheckResult:
    current: str
    latest: str | None
    source: str
    error: str | None = None
    busy: bool = False  # another check held the lock; nothing was fetched or recorded

    @property
    def newer(self) -> bool:
        return is_newer(self.latest, self.current)


def run_check(
    paths: Paths,
    *,
    current: str | None = None,
    timeout: float = FETCH_TIMEOUT_S,
    skip_if_busy: bool = False,
) -> CheckResult:
    """Fetch, select, record. Never raises: a failed lookup records the attempt (so the
    next command does not immediately spawn another) and reports the error.

    The background `check-update` passes `skip_if_busy` and steps aside if another
    check holds the lock. Explicit commands (`tt self update`, `tt update`) wait for
    the lock instead: the user asked, so they get an answer, and a concurrent
    background run finishing first simply leaves a fresh state file behind."""
    current = current or __version__
    source = source_location()
    state = UpdateState(paths)
    with state.lock() as acquired:
        if not acquired:
            if skip_if_busy:
                return CheckResult(current, None, source, error="another check is running", busy=True)
            return _run_check_blocking(state, source, current, timeout)
        return _fetch_and_record(state, source, current, timeout)


def _run_check_blocking(state: UpdateState, source: str, current: str, timeout: float) -> CheckResult:
    """Wait for the background check to release the lock, then look up ourselves (its
    result may be for a different source or already consumed; a lookup is cheap)."""
    try:
        import fcntl

        with open(state._lock_path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return _fetch_and_record(state, source, current, timeout)
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX
        return _fetch_and_record(state, source, current, timeout)


def _fetch_and_record(state: UpdateState, source: str, current: str, timeout: float) -> CheckResult:
    try:
        latest = latest_release(fetch_document(source, timeout=timeout))
    except Exception as exc:
        try:
            state.record(latest=None, current=current)
        except Exception:
            pass
        return CheckResult(current, None, source, error=f"{type(exc).__name__}: {exc}")
    try:
        state.record(latest=latest, current=current)
    except Exception as exc:
        return CheckResult(current, latest, source, error=f"could not save state: {exc}")
    return CheckResult(current, latest, source)


# -- detached refresh ----------------------------------------------------------------
def spawn_check(paths: Paths, *, on_debug: Callable[[str], None] | None = None) -> bool:
    """Refresh the state file from a detached process. Same shape as the telemetry
    drainer's spawn (see spawn_drainer for the reasoning behind every argument): all
    three descriptors to devnull so `$(tt ...)` never blocks, a new session so it
    outlives the parent, `python -m tenstorrent` so the package resolves from the
    interpreter tt is actually installed into."""
    if not sys.executable:
        return False
    try:
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [sys.executable, "-m", "tenstorrent", "self", "check-update"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            cwd=str(paths.data_dir),
        )
        return True
    except Exception as exc:
        if on_debug is not None:
            on_debug(f"update check: could not launch the background check: {exc}")
        return False


# -- the end-of-command hook -----------------------------------------------------------
def notice_text(latest: str, current: str, layout: InstallLayout) -> str:
    action = "tt self update" if layout.isolated else layout.manual_hint
    # Two lines: the command must never be wrapped mid-word by a narrow terminal.
    return f"A new release of tt is available: {current} → {latest}\n  To upgrade: {action}"


def after_command(appctx: Any, click_ctx: Any) -> None:
    """Called by @handle_tt_errors after every decorated command. Prints a pending
    notice if a person is watching (at most once a day per version), and kicks off a
    background refresh if the state is stale. Every path is guarded: this must never
    break, slow, or noisy-up a command."""
    try:
        # A decorated group callback (`tt config` on the way to `tt config get`) would
        # run this twice per invocation; only the leaf counts — same rule as the span.
        if getattr(click_ctx, "invoked_subcommand", None) is not None:
            return
        if getattr(click_ctx, "command_path", None) in _EXEMPT_COMMANDS:
            return
        if not check_enabled(appctx.config, offline=appctx.offline):
            return
        output = appctx.output
        # --quiet / --json are how scripts call tt: neither should start a background
        # process that goes to the network, and neither could show the notice anyway
        # (same rule as gh: no check where nobody could read the result).
        if output.quiet or output.json_mode:
            return
        state = UpdateState(appctx.paths)
        current = __version__
        pending = state.pending(current)
        stale = state.is_stale()
        if not pending and not stale:
            return
        # Layout detection costs a metadata scan, so it only runs on the two paths
        # that need it: a dev checkout neither shows notices nor spawns checks, and
        # a system install belongs to its package manager.
        layout = detect_layout()
        if not layout.notice_applies:
            return
        if pending and _interactive() and state.notice_due(pending):
            output.status(notice_text(pending, current, layout), style="yellow")
            state.mark_notified(pending)
        if stale:
            # A slow lookup leaves the state stale until it finishes; without this probe
            # every command run meanwhile would spawn another check.
            if state.check_in_progress():
                output.debug("update check: a background check is already running")
                return
            spawn_check(appctx.paths, on_debug=output.debug)
    except Exception:
        pass
