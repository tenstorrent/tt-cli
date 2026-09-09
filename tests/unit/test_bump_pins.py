# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""scripts/bump_pins.py against a fake GitHub — the weekly pin bump runs unattended."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import tomlkit

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bump_pins.py"
_spec = importlib.util.spec_from_file_location("bump_pins", _SCRIPT)
bump_pins = importlib.util.module_from_spec(_spec)
sys.modules["bump_pins"] = bump_pins  # dataclasses resolve postponed annotations via sys.modules
_spec.loader.exec_module(bump_pins)

SUPPLEMENT = """\
schema_version = 1

[golden]
# The tt-sw-manifest release we pin.
tag = "v1.0.0"
sha256 = "{golden_sha}"
url_template = "https://example.test/tt-sw-manifest/{{tag}}/golden.json"

[tools.tt-installer]
kind = "script"
# Flag contract re-verified against the v3.5.4 script.
golden_version = "3.5.4"
python = "3.12"
url = "https://example.test/tt-installer/v3.5.4/install.sh"
url_template = "https://example.test/tt-installer/v{{version}}/install.sh"
sha256 = "{installer_sha}"
needs_sudo = true # install.sh sudo's internally

[tools.tt-inference-server]
kind = "git-venv"
golden_version = "v0.18.0"
repo = "https://github.com/tenstorrent/tt-inference-server"

[tools.tt-model]
kind = "uv-tool"
package = "tt-model"
repo = "https://github.com/tenstorrent/tt-model-manager"
golden_version = "0000000000000000000000000000000000000000"
"""

TESTS_YML = "      - uses: tenstorrent/tt-installer@v3.5.4\n"

GOLDEN_V1 = json.dumps({"smi": "6.1.0", "flash": "3.10.0", "kmd": "2.10.0", "firmware": "19.13.1"}).encode()
GOLDEN_V2 = json.dumps({"smi": "6.2.0", "flash": "3.11.0", "kmd": "2.11.0", "firmware": "19.14.0"}).encode()


def _script(tag: str, drop_flag: str | None = None) -> bytes:
    flags = [f for f in bump_pins.INSTALLER_FLAGS if f != drop_flag]
    return f'#!/bin/bash\nreadonly TTIS_GOLDEN_VERSIONS_TAG="{tag}"\n# {" ".join(flags)}\n'.encode()


def _spec_json(version: str, models: int = 2) -> bytes:
    specs = {
        f"org/model-{i}": {"P300": {"vLLM": {"impl": {"model_name": f"model-{i}", "status": "supported"}}}}
        for i in range(models)
    }
    return json.dumps({"schema_version": "0.1.0", "release_version": version, "model_specs": specs}).encode()


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


class FakeGitHub:
    """Releases + downloadable blobs; every URL not registered is a test bug."""

    def __init__(self):
        self.releases: dict[str, dict[str, bump_pins.Release]] = {}  # repo → tag → Release
        self.latest: dict[str, str] = {}
        self.blobs: dict[str, bytes] = {}
        self.head = ("main", "0000000000000000000000000000000000000000", "2026-01-01")
        self.tag_list: list[str] = []

    def add_release(self, repo: str, tag: str, assets: dict[str, bytes], *, latest: bool = False, digest: bool = True):
        rel = bump_pins.Release(tag=tag)
        for name, blob in assets.items():
            url = f"https://example.test/{repo.split('/')[1]}/{tag}/{name}"
            rel.assets[name] = (url, f"sha256:{_sha(blob)}" if digest else None)
            self.blobs[url] = blob
        self.releases.setdefault(repo, {})[tag] = rel
        if latest:
            self.latest[repo] = tag

    def latest_release(self, repo):
        return self.releases[repo][self.latest[repo]]

    def release(self, repo, tag):
        return self.releases[repo][tag]

    def default_branch_head(self, repo):
        return self.head

    def tags(self, repo):
        return self.tag_list

    def download(self, url):
        return self.blobs[url]


@pytest.fixture
def gh(tmp_path, monkeypatch) -> FakeGitHub:
    """Everything current: installer 3.5.4 → golden v1.0.0, server v0.18.0, model head pinned."""
    fake = FakeGitHub()
    fake.add_release("tenstorrent/tt-installer", "v3.5.4", {"install.sh": _script("v1.0.0")}, latest=True)
    fake.add_release("tenstorrent/tt-sw-manifest", "v1.0.0", {"golden.json": GOLDEN_V1}, latest=True)
    fake.add_release("tenstorrent/tt-inference-server", "v0.18.0", {}, latest=True)
    # The bundled spec, and the support list generated from it — the script
    # rebuilds the second whenever it re-bundles the first, and compares model
    # counts against it.
    bundled = tmp_path / "release_model_spec.json"
    bundled.write_bytes(_spec_json("0.18.0", models=2))
    monkeypatch.setattr(bump_pins, "MODEL_SPEC", bundled)
    support = tmp_path / "model_support.json"
    document, _ = bump_pins.support_build.build_document(
        json.loads(bundled.read_text()),
        bump_pins.support_build.load_overrides(bump_pins.support_build.OVERRIDES_PATH),
    )
    support.write_text(bump_pins.support_build.render(document))
    monkeypatch.setattr(bump_pins, "MODEL_SUPPORT", support)
    return fake


def _supplement(gh: FakeGitHub) -> str:
    return SUPPLEMENT.format(golden_sha=_sha(GOLDEN_V1), installer_sha=_sha(_script("v1.0.0")))


def _plan(gh: FakeGitHub, supplement: str | None = None) -> bump_pins.Plan:
    return bump_pins.build_plan(supplement or _supplement(gh), TESTS_YML, gh)


def test_nothing_to_do_when_every_pin_is_current(gh):
    plan = _plan(gh)
    assert not plan.changed
    assert plan.files == {}
    assert plan.notes == []
    assert "Nothing to change" in bump_pins.render_summary(plan)


def test_installer_bump_pulls_golden_along_and_keeps_comments(gh):
    gh.add_release("tenstorrent/tt-installer", "v3.6.0", {"install.sh": _script("v1.1.0")}, latest=True)
    gh.add_release("tenstorrent/tt-sw-manifest", "v1.1.0", {"golden.json": GOLDEN_V2}, latest=True)

    plan = _plan(gh)

    assert [(c.component, c.old, c.new) for c in plan.changes] == [
        ("tt-installer", "3.5.4", "3.6.0"),
        ("tt-sw-manifest (golden)", "v1.0.0", "v1.1.0"),
    ]
    doc = tomlkit.parse(plan.files[bump_pins.SUPPLEMENT].decode())
    tool = doc["tools"]["tt-installer"]
    assert tool["golden_version"] == "3.6.0"
    assert tool["url"] == "https://example.test/tt-installer/v3.6.0/install.sh"
    assert tool["sha256"] == _sha(_script("v1.1.0"))
    assert doc["golden"]["tag"] == "v1.1.0"
    assert doc["golden"]["sha256"] == _sha(GOLDEN_V2)
    # tomlkit rewrite: comments and untouched keys survive verbatim.
    text = plan.files[bump_pins.SUPPLEMENT].decode()
    assert "# Flag contract re-verified against the v3.5.4 script." in text
    assert "needs_sudo = true # install.sh sudo's internally" in text
    assert plan.files[bump_pins.TESTS_WORKFLOW] == b"      - uses: tenstorrent/tt-installer@v3.6.0\n"
    summary = bump_pins.render_summary(plan)
    assert "| tt-installer | `3.5.4` | `3.6.0` |" in summary
    assert "smi 6.2.0" in summary
    assert "- [ ] Re-read the new install.sh" in summary


def test_golden_follows_the_installer_and_never_leads_it(gh):
    # A newer manifest release exists, but the pinned installer still converges to v1.0.0.
    gh.add_release("tenstorrent/tt-sw-manifest", "v1.1.0", {"golden.json": GOLDEN_V2}, latest=True)

    plan = _plan(gh)

    assert not plan.changed
    assert any("v1.1.0 is released, but tt-installer v3.5.4 still converges to v1.0.0" in n for n in plan.notes)


@pytest.mark.parametrize(
    ("script", "reason"),
    [
        (b"#!/bin/bash\n# " + " ".join(bump_pins.INSTALLER_FLAGS).encode() + b"\n", "no TTIS_GOLDEN_VERSIONS_TAG"),
        (_script("v1.1.0", drop_flag="--use-uv"), "dropped --use-uv"),
    ],
)
def test_installer_bump_is_refused_when_the_script_breaks_our_contract(gh, script, reason):
    gh.add_release("tenstorrent/tt-installer", "v3.6.0", {"install.sh": script}, latest=True)

    plan = _plan(gh)

    assert not plan.changed
    assert any(reason in n and "left at 3.5.4" in n for n in plan.notes)


def test_a_changed_asset_under_the_current_pin_is_reported_not_repinned(gh):
    gh.add_release("tenstorrent/tt-installer", "v3.5.4", {"install.sh": _script("v1.0.0") + b"# tampered\n"}, latest=True)

    plan = _plan(gh)

    assert not plan.changed
    assert any("no longer matches the pinned sha256" in n for n in plan.notes)


def test_a_download_that_disagrees_with_githubs_digest_aborts(gh):
    gh.add_release("tenstorrent/tt-installer", "v3.6.0", {"install.sh": _script("v1.0.0")}, latest=True)
    url, _ = gh.releases["tenstorrent/tt-installer"]["v3.6.0"].assets["install.sh"]
    gh.blobs[url] = b"something else entirely"

    with pytest.raises(RuntimeError, match="does not match GitHub's"):
        _plan(gh)


def test_inference_server_bump_rebundles_the_spec(gh):
    gh.add_release("tenstorrent/tt-inference-server", "v0.21.0", {}, latest=True)
    spec = _spec_json("0.21.0", models=3)
    gh.blobs["https://raw.githubusercontent.com/tenstorrent/tt-inference-server/v0.21.0/release_model_spec.json"] = spec

    plan = _plan(gh)

    (change,) = plan.changes
    assert (change.component, change.old, change.new) == ("tt-inference-server", "v0.18.0", "v0.21.0")
    assert change.notes[0] == (
        "release_model_spec.json re-bundled verbatim, and model_support.json rebuilt "
        "from it: 3 models (was 2)."
    )
    assert plan.files[bump_pins.MODEL_SPEC] == spec
    # the generated list has to move with the spec, or the CLI reads a support
    # list describing the previous release
    rebuilt = json.loads(plan.files[bump_pins.MODEL_SUPPORT])
    assert rebuilt["release_version"] == "0.21.0"
    assert len(rebuilt["models"]) == 3
    # and the CLI has to be able to read what the bump writes: a support list the
    # runtime parser rejects would only surface after the PR merged
    from tenstorrent.modelhub.catalog import parse_model_support

    version, entries = parse_model_support(
        plan.files[bump_pins.MODEL_SUPPORT].decode(), "bumped"
    )
    assert version == "0.21.0"
    assert len(entries) == 3
    doc = tomlkit.parse(plan.files[bump_pins.SUPPLEMENT].decode())
    assert doc["tools"]["tt-inference-server"]["golden_version"] == "v0.21.0"
    assert "BOARD_TYPE_COUNT_TO_DEVICE" in bump_pins.render_summary(plan)


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (json.dumps({"schema_version": "0.2.0", "release_version": "0.21.0", "model_specs": {}}).encode(), "not usable"),
        (_spec_json("0.20.0"), "says release_version 0.20.0"),
        (b"not json", "not usable"),
    ],
)
def test_inference_server_bump_is_refused_when_the_spec_cannot_be_bundled(gh, spec, reason):
    gh.add_release("tenstorrent/tt-inference-server", "v0.21.0", {}, latest=True)
    gh.blobs["https://raw.githubusercontent.com/tenstorrent/tt-inference-server/v0.21.0/release_model_spec.json"] = spec

    plan = _plan(gh)

    assert not plan.changed
    assert any(reason in n and "left at v0.18.0" in n for n in plan.notes)


def test_model_manager_tracks_the_default_branch_head(gh):
    gh.head = ("main", "be04da02d8ed9ef35c670aea9918d0c01c68b6e6", "2026-09-01")

    plan = _plan(gh)

    (change,) = plan.changes
    assert (change.component, change.new) == ("tt-model (tt-model-manager)", "be04da02d8ed")
    doc = tomlkit.parse(plan.files[bump_pins.SUPPLEMENT].decode())
    assert doc["tools"]["tt-model"]["golden_version"] == "be04da02d8ed9ef35c670aea9918d0c01c68b6e6"


def test_model_manager_tags_are_pointed_out_once_upstream_publishes_them(gh):
    gh.tag_list = ["v0.1.0"]

    plan = _plan(gh)

    assert not plan.changed
    assert any("now publishes tags (v0.1.0)" in n for n in plan.notes)
