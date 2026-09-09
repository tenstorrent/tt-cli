# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt self update`: upgrade tt through the tool that owns its environment.

Only isolated layouts are acted on (see layout.py). The version is pinned to exactly
what the check found, so the install does what the notice said and prereleases are
excluded by construction. Everything else exits UNSUPPORTED with the command to run by
hand — pip and uv resolve only the requested package, and a shared venv's other
tenants are the user's to weigh, not ours.

Order of operations matters: the running process has tt's *old* modules loaded and
still imports lazily, so the installer runs last and nothing after it imports anything
new. The post-install version is read by a subprocess for the same reason.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from .. import __version__
from ..errors import ExitCode, TTError
from . import check as check_module
from .layout import (
    KIND_PIPX,
    KIND_UV_TOOL,
    KIND_VENV,
    PACKAGE,
    InstallLayout,
    detect_layout,
    tt_on_path_matches,
)


def _find_uv_for(layout: InstallLayout) -> str:
    """For a *user's* uv tool venv prefer the user's uv (its receipts, its defaults);
    the wheel bundled with tt is the fallback, not the first choice as it is for the
    tool venvs tt itself manages."""
    override = os.environ.get("TT_UV_BIN")
    if override:
        return override
    found = shutil.which("uv")
    if found:
        return found
    from ..tools.installers import find_uv_bin

    return find_uv_bin()


def _has_pip() -> bool:
    import importlib.util

    return importlib.util.find_spec("pip") is not None


def upgrade_command(layout: InstallLayout, version: str) -> tuple[list[str], dict[str, str]]:
    """argv + environment that upgrades *this* install to `version`. Raises TTError for
    layouts tt must not act on."""
    env = dict(os.environ)
    requirement = f"{PACKAGE}=={version}"
    if layout.kind == KIND_UV_TOOL:
        # Point uv back at the dirs this tool venv lives in; with uv's defaults a user
        # who installed under a custom UV_TOOL_DIR would get a second copy instead.
        env["UV_TOOL_DIR"] = layout.extra.get("tool_dir", str(layout.prefix.parent))
        if layout.extra.get("bin_dir"):
            env["UV_TOOL_BIN_DIR"] = layout.extra["bin_dir"]
        # `uv tool upgrade` honours the specifier recorded at install time, so a pinned
        # install (`tenstorrent==0.1.0`) would be a silent no-op. `install --force`
        # re-pins to the version we announced.
        return [_find_uv_for(layout), "tool", "install", "--force", requirement], env
    if layout.kind == KIND_PIPX:
        pipx = shutil.which("pipx")
        if not pipx:
            raise TTError(
                "tt was installed with pipx, but `pipx` is not on PATH.",
                next_step=f"Run `pipx upgrade {PACKAGE}` from a shell where pipx is available.",
                exit_code=ExitCode.TOOL_MISSING,
            )
        # pipx's venvs live under PIPX_HOME/venvs/<name>; the running prefix is that
        # venv, so PIPX_HOME is two levels up. `upgrade` takes no version — the venv is
        # isolated, so what it resolves is what the check saw; verified afterwards.
        env.setdefault("PIPX_HOME", str(layout.prefix.parent.parent))
        return [pipx, "upgrade", PACKAGE], env
    if layout.kind == KIND_VENV:
        # Same installer that wrote the venv: a uv-made venv has no pip module, and a
        # pip-made one has pip's own config (mirrors, index pins) that uv would bypass.
        if layout.installer == "pip" and _has_pip():
            return [layout.python, "-m", "pip", "install", requirement], env
        return [_find_uv_for(layout), "pip", "install", "--python", layout.python, requirement], env
    raise TTError(
        "tt cannot upgrade itself in this environment.",
        why=layout.detail,
        next_step=layout.manual_hint,
        exit_code=ExitCode.UNSUPPORTED,
        details={"layout": layout.to_dict()},
    )


def installed_version(layout: InstallLayout) -> str | None:
    """Read the version now on disk from a fresh interpreter — never by importing into
    this process, which still holds the old modules."""
    import subprocess

    try:
        proc = subprocess.run(
            [
                layout.python,
                "-c",
                f"import importlib.metadata as m; print(m.version({PACKAGE!r}))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _fetch_latest(appctx: Any, *, offline: bool) -> check_module.CheckResult:
    if offline:
        raise TTError(
            "Checking for a newer tt needs the network.",
            why="--offline is set (or configured), and a self-update is inherently a download.",
            next_step="Re-run without --offline.",
            exit_code=ExitCode.OFFLINE,
        )
    result = check_module.run_check(appctx.paths, current=__version__)
    if result.error and result.latest is None:
        raise TTError(
            "Could not determine the latest tt release.",
            why=result.error,
            next_step=f"Check your network, or point {check_module.SOURCE_ENV} at a reachable mirror.",
            exit_code=ExitCode.ERROR,
            details={"source": result.source},
        )
    return result


def _confirm(appctx: Any, question: str, *, yes: bool, interactive: bool) -> bool:
    if yes:
        return True
    if appctx.output.json_mode or not interactive:
        raise TTError(
            "Upgrading tt needs confirmation.",
            next_step="Re-run with --yes to confirm.",
            exit_code=ExitCode.USAGE,
        )
    import typer

    return bool(typer.confirm(question, default=True))


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except Exception:
        return False


def perform_upgrade(appctx: Any, layout: InstallLayout, latest: str) -> dict:
    """Run the owning tool. Returns the JSON-shaped record of what happened."""
    argv, env = upgrade_command(layout, latest)
    on_path = shutil.which("tt")
    if not tt_on_path_matches(layout.prefix, on_path):
        raise TTError(
            "The `tt` on your PATH is not the one running.",
            why=f"PATH resolves to {on_path}; this process runs from {layout.prefix}.",
            next_step="Remove one of the two installs (or fix PATH), then retry.",
            exit_code=ExitCode.UNSUPPORTED,
            details={"on_path": on_path, "prefix": str(layout.prefix)},
        )
    appctx.output.status(f"Upgrading tt {__version__} → {latest} via {os.path.basename(argv[0])}…")
    appctx.output.debug("$ " + " ".join(argv))
    # Nothing below this line may import a module that is not already loaded.
    appctx.runner.stream(argv, env=env, tool=f"{os.path.basename(argv[0])} (upgrading tt)")
    now_installed = installed_version(layout)
    record = {
        "status": "upgraded",
        "from": __version__,
        "to": now_installed or latest,
        "command": argv,
    }
    if now_installed and now_installed != latest:
        appctx.output.warn(
            f"Expected tt {latest} but {now_installed} is installed — the installer "
            "picked a different version (a lagging mirror or index pin?)."
        )
        record["status"] = "mismatch"
    try:
        check_module.UpdateState(appctx.paths).record(latest=latest, current=now_installed or latest)
    except Exception:
        pass
    return record


def run_self_update(appctx: Any, *, check_only: bool, yes: bool) -> dict:
    """The body of `tt self update`."""
    layout = detect_layout()
    current = __version__
    if check_only and appctx.offline:
        cached = check_module.UpdateState(appctx.paths).load()
        latest = cached.get("latest") if cached.get("current") == current else None
        if latest is None:
            raise TTError(
                "No cached update check is available for this version.",
                why="--offline forbids the lookup and nothing was recorded earlier.",
                exit_code=ExitCode.OFFLINE,
            )
        result = check_module.CheckResult(current, str(latest), "cache")
    else:
        result = _fetch_latest(appctx, offline=appctx.offline)
    base = {
        "current": current,
        "latest": result.latest,
        "layout": layout.to_dict(),
        "hint": "tt self update" if layout.isolated else layout.manual_hint,
    }
    if not result.newer:
        return {**base, "status": "up-to-date"}
    if check_only:
        return {**base, "status": "available"}
    if not layout.isolated:
        raise TTError(
            "tt cannot upgrade itself in this environment.",
            why=layout.detail,
            next_step=layout.manual_hint,
            exit_code=ExitCode.UNSUPPORTED,
            details={"layout": layout.to_dict(), "latest": result.latest},
        )
    assert result.latest is not None
    if not _confirm(
        appctx,
        f"Upgrade tt {current} → {result.latest}?",
        yes=yes,
        interactive=_stdin_is_interactive(),
    ):
        return {**base, "status": "declined"}
    return {**base, **perform_upgrade(appctx, layout, result.latest)}


def _latest_for_update(appctx: Any, *, dry_run: bool) -> str | None:
    """The newest release `tt update` should mention, at the cost of at most one
    network round-trip a day. A fresh cache answers outright; a stale one is refreshed
    inline (3 s ceiling — `tt update` is a network command anyway, and this is the one
    place where knowing matters most, because a newer tt carries newer pins). A
    --dry-run preview never waits on the network: it uses whatever the cache says."""
    state = check_module.UpdateState(appctx.paths)
    if not state.is_stale():
        return state.pending(__version__)
    if dry_run:
        return None
    result = check_module.run_check(appctx.paths, current=__version__, timeout=3)
    return result.latest if result.newer else None


def offer_before_update(appctx: Any, *, offline: bool, dry_run: bool = False) -> bool:
    """Head of `tt update`: if a newer tt exists and this install can take it, ask a
    live person whether to upgrade tt first. True means tt was upgraded and the
    caller should stop — the *new* tt (with its newer pins) should run the update.

    Scripts are never surprised: without a TTY this only prints the notice, `--yes`
    notwithstanding, because that flag means "I accept the firmware reset", not
    "replace the program mid-run"."""
    output = appctx.output
    # --quiet suppresses the notice text, so a prompt after it would appear without
    # its context; --json must stay a single document. Neither mode gets the offer.
    if (
        offline
        or output.json_mode
        or output.quiet
        or not check_module.check_enabled(appctx.config)
    ):
        return False
    try:
        layout = detect_layout()
        if not layout.notice_applies:
            return False
        latest = _latest_for_update(appctx, dry_run=dry_run)
    except Exception:
        return False
    if latest is None:
        return False
    notice = check_module.notice_text(latest, __version__, layout)
    try:
        # Counts as today's notice: the next command should not repeat it.
        check_module.UpdateState(appctx.paths).mark_notified(latest)
    except Exception:
        pass
    if not layout.isolated or not _stdin_is_interactive():
        output.warn(notice)
        return False
    import typer

    output.status(notice, style="yellow")
    if not typer.confirm("Upgrade tt before updating the system?", default=True):
        return False
    perform_upgrade(appctx, layout, latest)
    output.status(
        f"tt is now {latest}. Re-run `tt update` to continue with the new version.",
        style="green",
    )
    return True
