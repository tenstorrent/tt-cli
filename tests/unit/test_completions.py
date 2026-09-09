# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Shell completion for model-name arguments.

Everything here must hold without network: completion reads only the bundled
release spec, tt-model's install index, and the community cache written by
`tt model list --community`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tenstorrent.modelhub import bundles, completions


def _write_installed_index(entries: dict) -> None:
    root = Path(os.environ["XDG_CACHE_HOME"]) / "tt-model"
    root.mkdir(parents=True, exist_ok=True)
    (root / "installed.json").write_text(json.dumps(entries))


def test_catalog_completion_offers_spec_names_filtered_by_prefix():
    got = completions.complete_catalog_model("Llama-3.1")
    assert "Llama-3.1-8B-Instruct" in got
    assert all(name.lower().startswith("llama-3.1") for name in got)


def test_complete_model_unions_spec_installed_and_community_cache():
    _write_installed_index({"ns/local-bundle": {"repo_id": "ns/local-bundle"}})
    bundles.save_community_cache(["tt-hous/qwen3.6-27b-p150x4"])
    got = completions.complete_model("")
    assert "Llama-3.1-8B-Instruct" in got
    assert "ns/local-bundle" in got
    assert "tt-hous/qwen3.6-27b-p150x4" in got
    # prefix filtering applies to bundle ids too
    assert completions.complete_model("tt-hous/") == ["tt-hous/qwen3.6-27b-p150x4"]


def test_complete_local_model_ignores_the_community_cache():
    _write_installed_index({"ns/local-bundle": {"repo_id": "ns/local-bundle"}})
    bundles.save_community_cache(["ns/never-pulled"])
    got = completions.complete_local_model("ns/")
    assert "ns/local-bundle" in got
    assert "ns/never-pulled" not in got


def test_no_community_cache_still_offers_spec_names():
    assert "Llama-3.1-8B-Instruct" in completions.complete_model("Llama-3.1")


def test_corrupt_caches_never_break_completion():
    cache_file = Path(os.environ["TT_CACHE_DIR"]) / "community-bundles.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{not json")
    _write_installed_index({})
    (Path(os.environ["XDG_CACHE_HOME"]) / "tt-model" / "installed.json").write_text("also not json")
    got = completions.complete_model("Llama-3.1")
    assert "Llama-3.1-8B-Instruct" in got


def test_community_cache_roundtrip():
    bundles.save_community_cache(["ns/b", "ns/a"])
    assert bundles.cached_community_names() == ["ns/a", "ns/b"]
