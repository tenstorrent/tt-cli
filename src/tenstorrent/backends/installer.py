# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Installer backend for `tt update`: plan (diff installed vs goldens) and apply.

Two layers converge:
- uv-managed tools (tt-smi, tt-flash): re-pinned idempotently in per-tool venvs.
- the system stack (kmd, sfpi, tenstorrent-tools, firmware): delegated to
  tt-installer via `--versions=release`, which makes install.sh fetch the golden
  `.ttis` for the running distro itself (the flow tt-sw-manifest's CI validates).
  The interpreter for the installer's own venv is pinned from the manifest too, so
  ~/.tenstorrent-venv doesn't inherit whatever python the distro happens to ship.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config.paths import Paths
from ..errors import ExitCode, TTError
from ..output import OutputManager
from ..tools.installers import fetch_https
from ..tools.manifest import (
    GOLDEN_PATH_ENV,
    ToolSpec,
    golden_cache_read,
    golden_cache_write,
    parse_golden,
)
from ..tools.registry import ToolRegistry
from ..tools.runner import Runner

INSTALLER_TOOL = "tt-installer"

# The tt-sw-manifest release tag install.sh converges the system stack to
# (`readonly TTIS_GOLDEN_VERSIONS_TAG="vX"` in the script, present since 3.5.x).
_TTIS_TAG_RE = re.compile(r'^readonly TTIS_GOLDEN_VERSIONS_TAG="([^"]+)"', re.MULTILINE)


@dataclass(frozen=True)
class PlanItem:
    name: str
    kind: str  # a manifest tool kind ("uv-tool", "git-venv", …) or "system"
    current: str | None
    target: str
    # A display string: install | upgrade | up-to-date | converge (the system row
    # may add detail, e.g. "converge (force downgrades)") — plus two that mean "not
    # touched": external (an env/config override owns it) and optional (a lazy tool
    # that is not installed; `--include-lazy` installs it).
    action: str


@dataclass(frozen=True)
class UpdatePlan:
    items: list[PlanItem]
    manifest_origin: str
    firmware: str | None
    installer_version: str | None = None  # None → the golden pin; else a user-requested release
    force: bool = False  # allow the installer to downgrade the system stack to the goldens
    installer_python: str | None = None  # interpreter pinned for ~/.tenstorrent-venv


@dataclass(frozen=True)
class UpdateResult:
    updated: list[str] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # lazy, not installed
    failed: list[dict] = field(default_factory=list)  # {"tool", "error"}
    installer_ran: bool = False


# Oldest tt-installer release `tt update <semver>` can drive. Every flag we pass
# unconditionally must exist at and above this floor, verified against the released
# scripts (2026-09-02):
#   --mode-non-interactive, --reboot-option   all releases
#   --no-install-inference-server             >= 2.0.0
#   --no-install-studio                       >= 2.1.0
#   --versions                                >= 3.0.0   <- the binding constraint
# argbash exits non-zero on an unrecognized option, so an older tag dies with a raw
# usage dump partway through a sudo'd run. Refuse it up front instead, before the
# script is even fetched. (--python-version is newer still; `_python_pin` drops it
# for every non-golden version rather than raising, since it is an optimization.)
_MIN_INSTALLER_VERSION = (3, 0, 0)


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """(3, 5, 4) from "3.5.4", or None if it is not plain dotted numbers."""
    core = version.split("-", 1)[0]
    parts = core.split(".")
    if not all(part.isdigit() for part in parts) or not parts:
        return None
    return tuple(int(part) for part in parts)


def _check_installer_floor(normalized: str) -> None:
    """Refuse a requested installer release too old for the flags tt passes."""
    parsed = _version_tuple(normalized)
    if parsed is None:
        return  # unparseable: let the fetch 404 say so, rather than guessing
    if parsed >= _MIN_INSTALLER_VERSION:
        return
    floor = ".".join(str(part) for part in _MIN_INSTALLER_VERSION)
    raise TTError(
        f"tt cannot drive tt-installer {normalized}.",
        why=f"tt runs install.sh with --versions, which only exists in {floor} "
        "and later. Older releases abort on the unrecognized option.",
        next_step=f"Pick {floor} or newer, or run that install.sh yourself.",
        exit_code=ExitCode.USAGE,
    )


class InstallerBackend:
    def __init__(
        self, registry: ToolRegistry, runner: Runner, paths: Paths, output: OutputManager
    ) -> None:
        self.registry = registry
        self.runner = runner
        self.paths = paths
        self.output = output

    # -- golden versions --------------------------------------------------------------
    def refresh_goldens(self, *, offline: bool = False) -> None:
        """Make the golden.json cache current for the pinned tt-sw-manifest tag.

        A valid cache (tag matches the supplement pin) is left alone — the pin names
        an immutable release, so there is nothing to re-fetch. Otherwise the asset is
        downloaded at the pinned tag, verified against the recorded sha256, and
        cached; --offline with no valid cache is an error (the pins are unknowable).
        A TT_GOLDEN_PATH override is the user's business and skips all of this."""
        manifest = self.registry.manifest
        if os.environ.get(GOLDEN_PATH_ENV):
            return
        tag, url = manifest.golden_tag, manifest.golden_url
        if not tag or not url:
            return  # a custom supplement without a [golden] pin manages its own truth
        if golden_cache_read(self.paths, tag) is not None:
            return
        if offline:
            raise TTError(
                f"Golden versions for {tag} are not cached locally.",
                why="--offline forbids fetching golden.json.",
                next_step="Run `tt update` once with network access, or point "
                f"{GOLDEN_PATH_ENV} at a local golden.json.",
                exit_code=ExitCode.OFFLINE,
            )
        self.output.status(f"Fetching golden versions ({tag}) …")
        blob = fetch_https(url)
        if manifest.golden_sha256:
            digest = hashlib.sha256(blob).hexdigest()
            if digest != manifest.golden_sha256:
                raise TTError(
                    f"Checksum mismatch for golden.json downloaded from {url}.",
                    why=f"expected sha256 {manifest.golden_sha256}, got {digest}",
                    next_step="Re-run `tt update`; if it persists, report it.",
                    exit_code=ExitCode.TOOL_FAILED,
                )
        golden = parse_golden(blob.decode("utf-8", errors="replace"), url)
        golden_cache_write(self.paths, tag, golden)
        self.registry.reload_manifest()  # make the fresh pins visible to plan()

    # -- planning -------------------------------------------------------------------
    def plan(
        self,
        *,
        version: str | None = None,
        force: bool = False,
        include_lazy: bool = False,
    ) -> UpdatePlan:
        manifest = self.registry.manifest
        # Reject an undrivable installer release before the plan is rendered or a
        # confirmation is asked for: --dry-run should surface it too, and nobody
        # should be prompted to approve a run that cannot start.
        if version is not None:
            normalized = version[1:] if version.startswith("v") else version
            if normalized != manifest.spec(INSTALLER_TOOL).golden_version:
                _check_installer_floor(normalized)
        items: list[PlanItem] = []
        for name, spec in sorted(manifest.tools.items()):
            if name == INSTALLER_TOOL:
                continue  # it is part of the system stack below, not a row of its own
            resolved = self.registry._resolve_or_none(name)
            if resolved is not None and resolved[1] in ("env", "override"):
                items.append(
                    PlanItem(name, spec.kind, None, spec.golden_version, "external")
                )
                continue
            installed = self.registry.state.get(name)
            # installed.toml records an install that may since have been deleted.
            # `_resolve_or_none` (and so `ensure()`) treats a recorded tool whose
            # path is gone as not installed; plan() must agree, or `tt update`
            # reports "up-to-date" for a tool that is missing and apply() skips it.
            if installed is not None and not installed.path.exists():
                installed = None
            if installed is None:
                # A lazy tool is optional: `tt update` keeps it current once you
                # have it, but does not download it for someone who never asked.
                action, current = ("install", None)
                if spec.lazy and not include_lazy:
                    action = "optional"
            elif installed.version != spec.golden_version:
                action, current = "upgrade", installed.version
            else:
                action, current = "up-to-date", installed.version
            items.append(PlanItem(name, spec.kind, current, spec.golden_version, action))
        system_target = ", ".join(f"{k} {v}" for k, v in sorted(manifest.system.items()))
        if manifest.firmware:
            system_target += f", firmware {manifest.firmware}"
        installer_shown = version or manifest.spec(INSTALLER_TOOL).golden_version
        python_pin = self._python_pin(version)
        system_target += f" (via install.sh {installer_shown}"
        system_target += f", python {python_pin})" if python_pin else ")"
        items.append(
            PlanItem(
                name="system-stack",
                kind="system",
                current=None,  # converged by tt-installer; it inspects the live system
                target=system_target,
                action="converge (force downgrades)" if force else "converge",
            )
        )
        return UpdatePlan(
            items=items,
            manifest_origin=manifest.origin,
            firmware=manifest.firmware,
            installer_version=version,
            force=force,
            installer_python=python_pin,
        )

    # -- applying -------------------------------------------------------------------
    def _python_pin(self, version: str | None) -> str | None:
        """The interpreter to build the installer's venv with, or None to leave the
        choice to install.sh. Skipped for a user-requested release: `--python-version`
        is newer than some installer tags, and argbash exits 1 on an unknown flag."""
        spec = self.registry.spec(INSTALLER_TOOL)
        if not spec.python:
            return None
        normalized = version[1:] if version and version.startswith("v") else version
        if normalized is not None and normalized != spec.golden_version:
            return None
        return spec.python

    def _installer_script(self, *, version: str | None, offline: bool) -> Path:
        """The install.sh to run: the golden pin (ensure, sha-verified) unless a
        specific release was requested, which is fetched unpinned at that tag."""
        if version is None:
            return self.registry.ensure(INSTALLER_TOOL, offline=offline)
        spec = self.registry.spec(INSTALLER_TOOL)
        normalized = version[1:] if version.startswith("v") else version
        if normalized == spec.golden_version:
            return self.registry.ensure(INSTALLER_TOOL, offline=offline)
        _check_installer_floor(normalized)
        if not spec.url_template:
            raise TTError(
                f"Cannot fetch a specific {INSTALLER_TOOL} version.",
                why="the manifest entry has no url_template with a {version} slot.",
                exit_code=ExitCode.CONFIG,
            )
        custom = dataclasses.replace(
            spec,
            golden_version=normalized,
            url=spec.url_template.format(version=normalized),
            sha256=None,  # only the golden release is checksum-pinned
        )
        self.output.warn(
            f"Fetching tt-installer {normalized} unpinned (no bundled sha256); "
            "only the golden release is checksum-verified."
        )
        return self.registry.install(custom, offline=offline).path

    def _check_golden_tag(self, script: Path) -> None:
        """Refuse to run a golden install.sh whose TTIS_GOLDEN_VERSIONS_TAG differs
        from the tt-sw-manifest release we pin ([golden] in the supplement): the
        installer would converge the system stack to versions tt neither displays
        nor pins. Only the managed (sha-verified) script is checked — an env/config
        override is the user's business, and `tt update <semver>` legitimately
        diverges."""
        expected = self.registry.manifest.golden_tag
        if expected is None:
            return
        resolved = self.registry._resolve_or_none(INSTALLER_TOOL)
        if resolved is not None and resolved[1] in ("env", "override"):
            return
        match = _TTIS_TAG_RE.search(script.read_text(errors="replace"))
        actual = match.group(1) if match else None
        if actual != expected:
            golden = self.registry.spec(INSTALLER_TOOL).golden_version
            raise TTError(
                f"install.sh {golden} pins the system stack to tt-sw-manifest "
                f"{actual or 'an undeclared release'}, but tt pins {expected}.",
                why="Running it would install versions tt does not display or pin. "
                "This is a tt packaging bug: the installer pin and the [golden] "
                "table in the supplement must be bumped together.",
                next_step="Update tt, or point TT_MANIFEST_PATH at a matching "
                "supplement.",
                exit_code=ExitCode.CONFIG,
            )

    def _run_system_installer(
        self, *, offline: bool, version: str | None = None, force: bool = False
    ) -> bool:
        if offline:
            self.output.status(
                "Skipping the system stack: tt-installer needs the network to fetch "
                "golden versions and distro packages. Re-run without --offline."
            )
            return False
        script = self._installer_script(version=version, offline=offline)
        normalized = version[1:] if version and version.startswith("v") else version
        if normalized in (None, self.registry.spec(INSTALLER_TOOL).golden_version):
            self._check_golden_tag(script)
        self.output.status("Running tt-installer (it will ask for sudo itself) …")
        # --versions=release makes install.sh download the golden .ttis for the
        # running distro itself and pin every component to it. Flag spelling
        # verified against tt-installer 3.5.4 (2026-08-18); the --import-schema
        # flag from earlier designs no longer exists.
        argv = [
            str(script),
            "--mode-non-interactive",
            "--versions=release",
            "--reboot-option=never",
            "--no-install-inference-server",  # tt-cli owns these two at its own pinned versions.
            "--no-install-studio",
        ]
        python_pin = self._python_pin(version)
        if python_pin:
            # install.sh warns and ignores --python-version unless --use-uv is on:
            # uv is what provisions the interpreter and creates the venv. With it,
            # ~/.tenstorrent-venv gets a known-good python instead of the distro's.
            argv += ["--use-uv", f"--python-version={python_pin}"]
        elif version is not None and self.registry.spec(INSTALLER_TOOL).python:
            self.output.warn(
                f"Not pinning python for install.sh {version}: --python-version is "
                "only passed to the golden release (older tags reject the flag)."
            )
        if force:
            # Let the installer move firmware to the golden version even when that
            # is a downgrade. Since 3.5.1 the installer also passes
            # --allow-downgrades on apt installs, so downgrading apt-delivered
            # components (kmd, sfpi, tools) no longer dies on its own -y.
            argv.append("--update-firmware=force")
        else:
            # install.sh 3.5.4 defaults this option to "force", which would reflash
            # every device even when its firmware is already current or newer.
            # "on" delegates the per-device version check to tt-flash: it flashes
            # only devices whose running/SPI firmware is older than the bundle.
            argv.append("--update-firmware=on")

        # Run from a scratch dir, not the user's CWD: install.sh reads nothing from
        # its working directory, but releases before 3.5.4 can drop a `wget-log`
        # there (fixed upstream; kept because `tt update <semver>` runs old scripts).
        workdir = self.paths.installer_work_dir
        workdir.mkdir(parents=True, exist_ok=True)
        self.runner.stream(argv, tool=INSTALLER_TOOL, cwd=str(workdir))
        return True

    # Hardcoded in install.sh (install_inference_server/install_studio), not derived
    # from our paths — that is exactly why they are invisible to the registry.
    _APP_CLONES = ("tt-inference-server", "tt-studio")

    def warn_unmanaged_app_clones(self) -> None:
        """Point out clones an earlier install.sh left in ~/.local/lib.

        Those copies are frozen: install.sh skips the clone when the directory
        exists and only rewrites the wrapper, so they never see an update. tt does
        not delete what it did not install — it just says where the space is."""
        home = Path.home()
        found = [name for name in self._APP_CLONES if (home / ".local/lib" / name).is_dir()]
        if not found:
            return
        self.output.warn(
            "An earlier tt-installer run left "
            + " and ".join(f"~/.local/lib/{name}" for name in found)
            + ". tt does not use or update those copies — it manages its own at the "
            "pinned versions. To reclaim the space:\n"
            + "\n".join(
                f"  rm -rf ~/.local/lib/{name} ~/.local/bin/{name}" for name in found
            )
        )

    def apply(
        self,
        plan: UpdatePlan,
        *,
        offline: bool = False,
        version: str | None = None,
        force: bool = False,
    ) -> UpdateResult:
        updated: list[str] = []
        up_to_date: list[str] = []
        external: list[str] = []
        skipped: list[str] = []
        failed: list[dict] = []
        for item in plan.items:
            if item.kind == "system":
                continue  # the installer run below
            if item.action == "external":
                external.append(item.name)
                self.output.status(
                    f"[dim]{item.name}: managed outside tt (override in effect), skipping[/dim]"
                )
                continue
            if item.action == "optional":
                skipped.append(item.name)
                self.output.status(
                    f"[dim]{item.name}: optional, not installed "
                    f"(`tt update --include-lazy` installs it)[/dim]"
                )
                continue
            if item.action == "up-to-date":
                up_to_date.append(item.name)
                continue
            spec = self.registry.spec(item.name)
            self.output.status(f"Installing {item.name} {item.target} …")
            try:
                self.registry.install(spec, offline=offline)
            except TTError as err:
                # One tool must not sink the whole update: a git fetch can fail for
                # reasons that have nothing to do with the system stack, which is
                # the part a user most needs converged. Report and carry on; the
                # command exits non-zero at the end.
                failed.append({"tool": item.name, "error": err.what})
                self.output.warn(f"{item.name} was not updated: {err.what}")
                continue
            updated.append(item.name)
        installer_ran = self._run_system_installer(
            offline=offline, version=version, force=force
        )
        return UpdateResult(
            updated=updated,
            up_to_date=up_to_date,
            external=external,
            skipped=skipped,
            failed=failed,
            installer_ran=installer_ran,
        )
