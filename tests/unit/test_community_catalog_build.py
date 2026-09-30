# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""scripts/build_community_catalog.py — the community_catalog.json generator.

Loaded from its path, as a maintainer tool rather than part of the package. Its
output ships in the wheel and is published with each release, so the committed
artifact is checked here too: a stale one would ship.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_community_catalog.py"

_spec = importlib.util.spec_from_file_location("build_community_catalog", SCRIPT)
build = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build
_spec.loader.exec_module(build)


def _entry(**overrides):
    return {"repo": "ns/model", "hardware": ["p150"], "validation": "sweep", **overrides}


def _build(*entries):
    return build.build_document({"schema_version": 1, "bundle": list(entries)})


def test_an_entry_gains_its_arch_and_nulls_for_unknown_fields():
    (bundle,) = _build(_entry(hardware=["P300x2", "p150"]))["bundles"]
    assert bundle == {
        "repo": "ns/model",
        "kind": None,
        "engine": None,
        "arch": "blackhole",
        "hardware": ["p150", "p300x2"],
        "validation": "sweep",
        "validated_on": None,
    }


def test_bundles_are_sorted_case_insensitively():
    document = _build(_entry(repo="zz/b"), _entry(repo="AA/c"), _entry(repo="mm/a"))
    assert [b["repo"] for b in document["bundles"]] == ["AA/c", "mm/a", "zz/b"]


def test_validated_on_is_normalised_from_a_toml_date():
    import datetime

    (bundle,) = _build(_entry(validated_on=datetime.date(2026, 9, 25)))["bundles"]
    assert bundle["validated_on"] == "2026-09-25"


@pytest.mark.parametrize(
    "entry, message",
    [
        (_entry(repo="no-namespace"), "not a namespace/name"),
        (_entry(hardware=[]), "hardware is empty"),
        (_entry(hardware=["p250"]), "unknown hardware p250"),
        (_entry(hardware=["p150", "n300"]), "spans architectures"),
        (_entry(kind="docker"), "kind 'docker'"),
        (_entry(engine="sglang"), "engine 'sglang'"),
        (_entry(validation="vibes"), "validation 'vibes'"),
        (_entry(validated_on="Sep 25"), "not a YYYY-MM-DD date"),
    ],
    ids=["repo", "no-hardware", "unknown-hardware", "mixed-arch", "kind", "engine",
         "validation", "date"],
)
def test_an_invalid_entry_is_refused(entry, message):
    with pytest.raises(build.BuildError, match=message):
        _build(entry)


def test_a_duplicate_repo_is_refused_regardless_of_case():
    with pytest.raises(build.BuildError, match="more than once"):
        _build(_entry(repo="ns/Model"), _entry(repo="NS/model"))


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(build.BuildError, match="schema_version"):
        build.build_document({"schema_version": 2})


# -- the committed artifact -----------------------------------------------------------
def test_the_committed_artifact_matches_its_input():
    """What `--check` enforces: a hand edit to the JSON, or a TOML change without a
    rebuild, ships a catalog that disagrees with its reviewed input."""
    document = build.build_document(build.load_input(build.INPUT_PATH))
    assert build.OUTPUT_PATH.read_text() == build.render(document)


def test_the_committed_artifact_is_what_tt_reads():
    from tenstorrent.modelhub import bundles

    committed = json.loads(build.OUTPUT_PATH.read_text())
    assert bundles.curated_ids() == {b["repo"].lower() for b in committed["bundles"]}
