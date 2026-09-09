# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""How was *this* tt installed? Decided from evidence, never guessed.

Everything is read from the running interpreter's prefix and tt's own dist-info:

* ``direct_url.json`` (PEP 610) with ``dir_info.editable`` — a development checkout.
* ``<prefix>/uv-receipt.toml`` — a ``uv tool`` venv; ``<prefix>/pipx_metadata.json`` —
  a pipx venv. Both isolated by construction.
* A plain venv is *isolated* only if every distribution in it is reachable from tt's
  own requirements (plus venv furniture such as pip itself). Anything else present is
  another tenant whose pins an upgrade could break, so the venv counts as shared.
* Outside a venv: the user site (``pip install --user``) or the system site-packages.
  The latter is a distro package or a ``--break-system-packages`` install — either way
  the package manager owns it and tt stays quiet.
* ``INSTALLER`` (PEP 376) says which installer wrote the files: ``pip`` and ``uv``
  behave differently (a uv-made venv has no pip module), so a venv upgrade must go
  through the same tool that installed tt in the first place.
"""

from __future__ import annotations

import json
import re
import site
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

PACKAGE = "tenstorrent"

# Seeded into a venv by `python -m venv` / virtualenv; never a tenant in its own right.
_VENV_FURNITURE = frozenset({"pip", "setuptools", "wheel"})

KIND_EDITABLE = "editable"
KIND_UV_TOOL = "uv-tool"
KIND_PIPX = "pipx"
KIND_VENV = "venv"  # sole tenant
KIND_SHARED_VENV = "shared-venv"
KIND_USER_SITE = "user-site"
KIND_SYSTEM = "system"

_ISOLATED_KINDS = frozenset({KIND_UV_TOOL, KIND_PIPX, KIND_VENV})
# Layouts where telling the user about a newer release is actionable. A development
# checkout is updated with git; a system install belongs to the package manager, whose
# repository — not PyPI — decides what version exists.
_NOTICE_KINDS = _ISOLATED_KINDS | {KIND_SHARED_VENV, KIND_USER_SITE}


@dataclass(frozen=True)
class InstallLayout:
    kind: str
    prefix: Path
    installer: str | None  # contents of INSTALLER, normalized; None if absent
    version: str | None  # the installed distribution's version (None: not installed)
    python: str  # interpreter tt runs under (the venv's, for venv layouts)
    tenants: tuple[str, ...] = ()  # other top-level distributions in a shared venv
    detail: str = ""  # one human sentence on what was found
    extra: dict = field(default_factory=dict)  # receipt facts the upgrade needs

    @property
    def isolated(self) -> bool:
        return self.kind in _ISOLATED_KINDS

    @property
    def notice_applies(self) -> bool:
        return self.kind in _NOTICE_KINDS

    @property
    def manual_hint(self) -> str:
        """The command a person runs by hand. Unpinned on purpose: it goes through the
        user's own installer configuration (mirrors, index pins) and `--upgrade` is the
        familiar spelling; tt's own action pins the exact version instead."""
        if self.kind == KIND_EDITABLE:
            return "git pull in the checkout"
        if self.kind == KIND_UV_TOOL:
            return "uv tool upgrade tenstorrent"
        if self.kind == KIND_PIPX:
            return "pipx upgrade tenstorrent"
        if self.kind == KIND_USER_SITE:
            return f"{self.python} -m pip install --user --upgrade tenstorrent"
        if self.kind == KIND_SYSTEM:
            return "upgrade the tenstorrent package with your system package manager"
        if self.installer == "uv":
            return f"uv pip install --python {self.python} --upgrade tenstorrent"
        return f"{self.python} -m pip install --upgrade tenstorrent"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "prefix": str(self.prefix),
            "installer": self.installer,
            "isolated": self.isolated,
            "version": self.version,
            "tenants": list(self.tenants),
            "detail": self.detail,
        }


def normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _read(dist: metadata.Distribution, name: str) -> str | None:
    try:
        return dist.read_text(name)
    except Exception:
        return None


def _is_editable(dist: metadata.Distribution) -> bool:
    raw = _read(dist, "direct_url.json")
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except (ValueError, AttributeError):
        return False


def _installer(dist: metadata.Distribution) -> str | None:
    raw = _read(dist, "INSTALLER")
    if not raw:
        return None
    return raw.strip().lower() or None


def _marker_allows(req, extras: frozenset[str]) -> bool:
    if req.marker is None:
        return True
    environments = [{"extra": e} for e in sorted(extras)] or [{"extra": ""}]
    for env in environments:
        try:
            if req.marker.evaluate(env):
                return True
        except Exception:
            # An unevaluable marker is treated as "not required": a distribution only
            # counts as tt's own when we can prove tt asked for it.
            continue
    return False


def requirement_closure(
    dists: dict[str, metadata.Distribution], root: str = PACKAGE
) -> set[str]:
    """Normalized names of every distribution `root` (transitively) requires, judged
    against the distributions actually present. Extras propagate along the edge that
    asked for them, so an optional dependency is in the closure only if requested."""
    from packaging.requirements import InvalidRequirement, Requirement

    closure: set[str] = set()
    seen: set[tuple[str, frozenset[str]]] = set()
    stack: list[tuple[str, frozenset[str]]] = [(normalize(root), frozenset())]
    while stack:
        name, extras = stack.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        closure.add(name)
        dist = dists.get(name)
        if dist is None:
            continue
        for raw in dist.requires or ():
            try:
                req = Requirement(raw)
            except InvalidRequirement:
                continue
            if _marker_allows(req, extras):
                stack.append((normalize(req.name), frozenset(req.extras)))
    return closure


def _distributions_in(site_packages: Path) -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions(path=[str(site_packages)]):
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            found.setdefault(normalize(name), dist)
    return found


def other_tenants(site_packages: Path, root: str = PACKAGE) -> tuple[str, ...]:
    """Distributions in `site_packages` that are neither `root`, its requirements,
    nor venv furniture — the packages a careless upgrade could break."""
    dists = _distributions_in(site_packages)
    keep = requirement_closure(dists, root) | _VENV_FURNITURE
    return tuple(sorted(name for name in dists if name not in keep))


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def detect_layout(
    *,
    dist: metadata.Distribution | None = None,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
    python: str | None = None,
    user_site: str | None = None,
) -> InstallLayout:
    """Classify the running install. Every input defaults to the live interpreter; tests
    pass a synthetic dist-info tree and prefixes instead."""
    prefix = Path(prefix if prefix is not None else sys.prefix)
    base_prefix = Path(base_prefix if base_prefix is not None else sys.base_prefix)
    python = python or sys.executable
    if dist is None:
        try:
            dist = metadata.distribution(PACKAGE)
        except metadata.PackageNotFoundError:
            dist = None

    version = dist.version if dist is not None else None
    installer = _installer(dist) if dist is not None else None
    # Where tt's files live: the site-packages dir holding its dist-info. This is what
    # the tenant scan walks, so it is derived from the dist rather than assumed.
    site_packages = Path(str(dist.locate_file(""))) if dist is not None else None

    if dist is not None and _is_editable(dist):
        return InstallLayout(
            KIND_EDITABLE, prefix, installer, version, python,
            detail="tt is an editable install of a source checkout.",
        )
    receipt = prefix / "uv-receipt.toml"
    if receipt.is_file():
        return InstallLayout(
            KIND_UV_TOOL, prefix, installer, version, python,
            detail=f"tt is a uv tool ({receipt}).",
            extra=_uv_receipt_facts(receipt),
        )
    pipx_meta = prefix / "pipx_metadata.json"
    if pipx_meta.is_file():
        return InstallLayout(
            KIND_PIPX, prefix, installer, version, python,
            detail=f"tt is a pipx application ({pipx_meta}).",
        )
    in_venv = not _same_path(prefix, base_prefix)
    if in_venv:
        tenants = other_tenants(site_packages) if site_packages is not None else ()
        if tenants:
            shown = ", ".join(tenants[:5]) + (", …" if len(tenants) > 5 else "")
            return InstallLayout(
                KIND_SHARED_VENV, prefix, installer, version, python, tenants,
                detail=f"tt shares {prefix} with other packages ({shown}).",
            )
        return InstallLayout(
            KIND_VENV, prefix, installer, version, python,
            detail=f"tt is the only package in {prefix}.",
        )
    if user_site is None:
        try:
            user_site = site.getusersitepackages()
        except Exception:
            user_site = None
    if user_site and site_packages is not None and _under(site_packages, Path(user_site)):
        return InstallLayout(
            KIND_USER_SITE, prefix, installer, version, python,
            detail=f"tt is a per-user install in {site_packages}.",
        )
    return InstallLayout(
        KIND_SYSTEM, prefix, installer, version, python,
        detail=f"tt is installed system-wide in {site_packages or prefix}.",
    )


def _uv_receipt_facts(receipt: Path) -> dict:
    """The two locations `uv tool install` must be pointed back at so it updates *this*
    tool venv rather than creating a second one under uv's default dirs: the tool dir
    is the receipt's parent's parent, the bin dir is where the receipt says the entry
    point was linked."""
    facts: dict = {"tool_dir": str(receipt.parent.parent)}
    try:
        import tomlkit

        doc = tomlkit.parse(receipt.read_text()).unwrap()
        for entry in doc.get("tool", {}).get("entrypoints", []):
            path = entry.get("install-path")
            if path:
                facts["bin_dir"] = str(Path(path).parent)
                break
        requirements = doc.get("tool", {}).get("requirements", [])
        if requirements:
            facts["specifier"] = requirements[0].get("specifier", "")
    except Exception:
        pass
    return facts


def tt_on_path_matches(prefix: Path, which: str | None) -> bool:
    """Is the `tt` a shell would run (`shutil.which("tt")`) the one in `prefix`? None
    (nothing on PATH) counts as a match: the user is invoking this install by path.
    Mirrors uv's receipt-vs-current_exe guard — upgrading one copy while the shell runs
    another is the likeliest way for a self-update to appear to do nothing."""
    if which is None:
        return True
    return _under(Path(which), prefix)
