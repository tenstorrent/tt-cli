# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Shell-completion sources for model-name arguments.

These run inside the tab-completion subprocess (`_TT_COMPLETE=...`), where there
is no AppContext and no budget for slow work. Every source is a local file: the
bundled release spec (what `tt model list` shows), tt-model's install index, and
the community-catalog cache written by the last `tt model list --community` —
never the network. Before the first `--community` run the cache simply doesn't
exist and bundle ids aren't offered. A broken source means fewer suggestions,
never a broken tab key, so everything is swallowed.
"""

from __future__ import annotations

from . import bundles


def _catalog_names() -> list[str]:
    from .catalog import ModelCatalog

    # cached_sizes={} skips list()'s HF-cache disk scan — completion doesn't
    # render the cached column, and scanning ~100 GB of blobs on tab would stall.
    return [m.name for m in ModelCatalog().list(cached_sizes={})]


def _installed_ids() -> list[str]:
    return [
        str(entry.get("repo_id") or key)
        for key, entry in bundles.installed_bundles().items()
    ]


def _matches(incomplete: str, names: list[str]) -> list[str]:
    prefix = incomplete.lower()
    return sorted({n for n in names if n.lower().startswith(prefix)}, key=str.lower)


def complete_catalog_model(incomplete: str) -> list[str]:
    """Released-spec names only — for `tt model info`, which knows nothing else."""
    try:
        return _matches(incomplete, _catalog_names())
    except Exception:
        return []


def complete_model(incomplete: str) -> list[str]:
    """Anything serveable or pullable: spec names, installed bundles, and the
    cached community listing."""
    try:
        return _matches(
            incomplete,
            [*_catalog_names(), *_installed_ids(), *bundles.cached_community_names()],
        )
    except Exception:
        return []


def complete_local_model(incomplete: str) -> list[str]:
    """Spec names + installed bundles — stop/rm act on what's on this machine."""
    try:
        return _matches(incomplete, [*_catalog_names(), *_installed_ids()])
    except Exception:
        return []
