# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Install-layout detection: every verdict comes from on-disk evidence, so each case
here builds the evidence (dist-info trees, receipts) in a temp dir and nothing else."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

from tenstorrent.selfupdate import layout as L

RECEIPT = """\
[tool]
requirements = [{{ name = "tenstorrent", specifier = "==0.1.0" }}]
entrypoints = [
    {{ name = "tt", install-path = "{bin}/tt", from = "tenstorrent" }},
]
"""


def make_dist(site: Path, name: str, version: str, *, requires=(), installer=None, direct_url=None):
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    meta += "".join(f"Requires-Dist: {req}\n" for req in requires)
    (info / "METADATA").write_text(meta)
    if installer:
        (info / "INSTALLER").write_text(installer + "\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    return metadata.PathDistribution(info)


@pytest.fixture
def venv(tmp_path):
    """A venv-shaped prefix with tt and its (tiny) closure installed by pip."""
    prefix = tmp_path / "venv"
    site = prefix / "lib" / "python3.12" / "site-packages"
    tt = make_dist(site, "tenstorrent", "0.1.0", requires=["typer>=0.27", "rich>=13"], installer="pip")
    make_dist(site, "typer", "0.27.2", requires=["click>=8"])
    make_dist(site, "click", "8.2.0")
    make_dist(site, "rich", "14.0.0", requires=["ipywidgets; extra == 'jupyter'"])
    make_dist(site, "pip", "25.0")  # venv furniture, never a tenant
    return prefix, site, tt


def detect(prefix, tt, **kw):
    return L.detect_layout(
        dist=tt,
        prefix=prefix,
        base_prefix=kw.pop("base_prefix", Path("/usr")),
        python=str(prefix / "bin" / "python"),
        user_site=kw.pop("user_site", "/nonexistent/user-site"),
        **kw,
    )


def test_sole_tenant_venv_is_isolated(venv):
    prefix, _, tt = venv
    layout = detect(prefix, tt)
    assert layout.kind == L.KIND_VENV
    assert layout.isolated and layout.notice_applies
    assert layout.installer == "pip"
    assert layout.version == "0.1.0"
    assert layout.tenants == ()
    assert "-m pip install --upgrade tenstorrent" in layout.manual_hint


def test_another_package_makes_the_venv_shared(venv):
    prefix, site, tt = venv
    make_dist(site, "requests", "2.32.0", requires=["urllib3"])
    make_dist(site, "urllib3", "2.3.0")
    layout = detect(prefix, tt)
    assert layout.kind == L.KIND_SHARED_VENV
    assert not layout.isolated and layout.notice_applies
    # Both the tenant and *its* dependency are foreign; neither is in tt's closure.
    assert layout.tenants == ("requests", "urllib3")
    assert "requests" in layout.detail


def test_optional_dependency_is_only_ours_if_we_asked_for_the_extra(venv, tmp_path):
    prefix, site, tt = venv
    make_dist(site, "ipywidgets", "8.0.0")  # rich's `jupyter` extra, which tt did not request
    assert detect(prefix, tt).kind == L.KIND_SHARED_VENV
    # The same site, but tt asks for rich[jupyter]: ipywidgets is now in the closure.
    site2 = tmp_path / "site2"
    tt2 = make_dist(site2, "tenstorrent", "0.1.0", requires=["typer>=0.27", "rich[jupyter]>=13"], installer="pip")
    make_dist(site2, "typer", "0.27.2", requires=["click>=8"])
    make_dist(site2, "click", "8.2.0")
    make_dist(site2, "rich", "14.0.0", requires=["ipywidgets; extra == 'jupyter'"])
    make_dist(site2, "ipywidgets", "8.0.0")
    assert L.other_tenants(site2) == ()
    assert detect(prefix, tt2).kind == L.KIND_VENV


def test_uv_installer_is_recorded_and_changes_the_hint(venv):
    prefix, site, _ = venv
    tt = make_dist(site / "uv", "tenstorrent", "0.1.0", installer="uv")
    layout = detect(prefix, tt)
    assert layout.installer == "uv"
    assert layout.manual_hint.startswith("uv pip install --python ")


def test_editable_checkout_is_never_touched_or_nagged(venv):
    prefix, site, _ = venv
    tt = make_dist(
        site / "dev", "tenstorrent", "0.1.0.dev0", installer="uv",
        direct_url={"url": "file:///home/x/tt-cli", "dir_info": {"editable": True}},
    )
    layout = detect(prefix, tt)
    assert layout.kind == L.KIND_EDITABLE
    assert not layout.isolated and not layout.notice_applies
    assert "git pull" in layout.manual_hint


def test_uv_tool_receipt_wins_and_carries_the_dirs(venv, tmp_path):
    prefix, _, tt = venv
    bin_dir = tmp_path / "home" / ".local" / "bin"
    (prefix / "uv-receipt.toml").write_text(RECEIPT.format(bin=bin_dir))
    layout = detect(prefix, tt)
    assert layout.kind == L.KIND_UV_TOOL and layout.isolated
    assert layout.extra["tool_dir"] == str(prefix.parent)
    assert layout.extra["bin_dir"] == str(bin_dir)
    assert layout.extra["specifier"] == "==0.1.0"
    assert layout.manual_hint == "uv tool upgrade tenstorrent"


def test_pipx_metadata_is_recognized(venv):
    prefix, _, tt = venv
    (prefix / "pipx_metadata.json").write_text('{"main_package": {"package": "tenstorrent"}}')
    layout = detect(prefix, tt)
    assert layout.kind == L.KIND_PIPX and layout.isolated
    assert layout.manual_hint == "pipx upgrade tenstorrent"


def test_user_site_install_gets_a_notice_but_no_action(tmp_path):
    user_site = tmp_path / "home" / ".local" / "lib" / "python3.12" / "site-packages"
    tt = make_dist(user_site, "tenstorrent", "0.1.0", installer="pip")
    layout = L.detect_layout(
        dist=tt, prefix=Path("/usr"), base_prefix=Path("/usr"),
        python="/usr/bin/python3", user_site=str(user_site),
    )
    assert layout.kind == L.KIND_USER_SITE
    assert not layout.isolated and layout.notice_applies
    assert "--user --upgrade" in layout.manual_hint


def test_system_site_packages_belong_to_the_package_manager(tmp_path):
    site = tmp_path / "usr" / "lib" / "python3" / "dist-packages"
    tt = make_dist(site, "tenstorrent", "0.1.0")
    layout = L.detect_layout(
        dist=tt, prefix=Path("/usr"), base_prefix=Path("/usr"),
        python="/usr/bin/python3", user_site=str(tmp_path / "elsewhere"),
    )
    assert layout.kind == L.KIND_SYSTEM
    assert not layout.isolated and not layout.notice_applies
    assert layout.installer is None
    assert "package manager" in layout.manual_hint


def test_symlinked_prefix_still_counts_as_a_venv(venv, tmp_path):
    prefix, _, tt = venv
    link = tmp_path / "link"
    link.symlink_to(prefix)
    assert detect(link, tt).kind == L.KIND_VENV
    # ...and a prefix that IS the base prefix (through a symlink) is not a venv.
    assert detect(link, tt, base_prefix=prefix).kind == L.KIND_SYSTEM


def test_live_detection_of_this_test_process_is_a_dev_checkout_or_a_venv():
    """Whatever runs the suite (editable checkout in uv's venv, a CI install...) must be
    classifiable without raising, and must never be one of the isolated kinds tt would
    act on, since the suite runs no installer."""
    layout = L.detect_layout()
    assert layout.kind in {L.KIND_EDITABLE, L.KIND_SHARED_VENV, L.KIND_VENV, L.KIND_SYSTEM, L.KIND_USER_SITE}
    assert layout.version


@pytest.mark.parametrize(
    "which, expected",
    [(None, True), ("{prefix}/bin/tt", True), ("/usr/local/bin/tt", False)],
)
def test_tt_on_path_must_be_this_install(tmp_path, which, expected):
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "tt").write_text("")
    resolved = which.format(prefix=prefix) if which else None
    assert L.tt_on_path_matches(prefix, resolved) is expected


def test_installer_file_is_normalized_and_optional(venv):
    prefix, site, _ = venv
    assert detect(prefix, make_dist(site / "a", "tenstorrent", "0.1.0", installer="  UV \n")).installer == "uv"
    assert detect(prefix, make_dist(site / "b", "tenstorrent", "0.1.0", installer="")).installer is None
    assert detect(prefix, make_dist(site / "c", "tenstorrent", "0.1.0")).installer is None


def test_name_normalization_matches_pep_503():
    assert L.normalize("Typer_Slim.Extra") == "typer-slim-extra"
