# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Model downloads via huggingface_hub, honoring tt config and --offline."""

from __future__ import annotations

import os
from pathlib import Path

from ..config.store import ConfigStore
from ..errors import ExitCode, TTError
from ..models.model import ModelInfo


# Model types whose containers fetch their own weights and ignore the host HF
# cache: STT (audio) and TTS. Their hf_model_repo is a real Hub repo, so the
# cache can be populated — it just is not what the server reads.
_CONTAINER_FETCHED_TYPES = frozenset({"audio", "text_to_speech"})


def has_hub_weights(model: ModelInfo) -> bool:
    """Whether the model has weights on the Hub at all.

    Forge CNNs carry a bare label as their hf_model_repo ("resnet-50", "vit")
    instead of an org/name repo: their weights ship inside the container image,
    so there is nothing to download and no cache to miss.
    """
    return "/" in model.hf_repo


def uses_host_weight_cache(model: ModelInfo) -> bool:
    """Whether the inference server reads this model's weights from the host
    HuggingFace cache — i.e. whether populating that cache helps at all."""
    return has_hub_weights(model) and model.model_type not in _CONTAINER_FETCHED_TYPES


def hf_home_dir(config: ConfigStore) -> Path:
    """The HF cache root, with HF_HOME semantics (hub cache lives at <root>/hub):
    config override → HF_HOME env → ~/.cache/huggingface.

    This single resolution is shared by `tt model pull` (write side) and
    `tt serve` (run.py --host-hf-cache mount side) so both always agree on one
    cache and the inference server never re-downloads pulled weights.
    Limitation: an HF_HUB_CACHE env pointing outside HF_HOME is honored by
    huggingface_hub but cannot be expressed to run.py, which only takes a root."""
    configured = str(config.get("paths.hf_model_cache_directory") or "").strip()
    if configured:
        return Path(configured).expanduser()
    env_home = os.environ.get("HF_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "huggingface"


def hf_cache_dir(config: ConfigStore) -> str | None:
    """snapshot_download cache_dir for the resolved root; None defers to HF's
    own defaults/env when nothing is configured (identical outcome, but keeps
    HF_HUB_CACHE-style env overrides working for pull)."""
    configured = str(config.get("paths.hf_model_cache_directory") or "").strip()
    if not configured:
        return None
    return str(hf_home_dir(config) / "hub")


def cached_weights(hf_repo: str, config: ConfigStore) -> tuple[list[str], int]:
    """Revision hashes and total on-disk bytes for `hf_repo` in the resolved cache.

    ([], 0) when the repo is not cached, the cache does not exist, or it cannot be
    scanned — reporting "nothing to reclaim" is always safe for a caller that is
    about to delete, and `tt model rm` must not fail because of a cache hiccup."""
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound
    except ImportError:  # pragma: no cover - hard dependency
        return [], 0
    try:
        info = scan_cache_dir(cache_dir=hf_cache_dir(config))
    except (CacheNotFound, OSError, ValueError):
        return [], 0
    wanted = hf_repo.lower()
    for repo in info.repos:
        if repo.repo_type == "model" and repo.repo_id.lower() == wanted:
            return [rev.commit_hash for rev in repo.revisions], repo.size_on_disk
    return [], 0


def cached_sizes(config: ConfigStore) -> dict[str, int]:
    """{lowercased repo id: bytes on disk} for every model in the resolved cache.

    One scan for a whole listing, mirroring catalog.scan_hf_cache(): calling
    cached_weights() per model would re-walk the cache each time (~18 ms per scan
    here, so O(N) on a bundle listing). Empty when the cache is missing or
    unreadable — "nothing cached" is always a safe answer."""
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound
    except ImportError:  # pragma: no cover - hard dependency
        return {}
    try:
        info = scan_cache_dir(cache_dir=hf_cache_dir(config))
    except (CacheNotFound, OSError, ValueError):
        return {}
    return {
        repo.repo_id.lower(): repo.size_on_disk
        for repo in info.repos
        if repo.repo_type == "model"
    }


def delete_cached_weights(hf_repo: str, config: ConfigStore) -> int:
    """Delete every cached revision of `hf_repo`; returns the bytes reclaimed.

    Goes through huggingface_hub's own delete_revisions strategy rather than
    rmtree: the cache is content-addressed with blobs shared between revisions,
    so only the library knows what is actually safe to unlink."""
    revisions, _ = cached_weights(hf_repo, config)
    if not revisions:
        return 0
    from huggingface_hub import scan_cache_dir

    strategy = scan_cache_dir(cache_dir=hf_cache_dir(config)).delete_revisions(*revisions)
    freed = int(strategy.expected_freed_size)
    try:
        strategy.execute()
    except OSError as exc:
        raise TTError(
            f"Could not delete cached weights for {hf_repo}.",
            why=str(exc),
            next_step="Check permissions on the HF cache, or delete the repo "
            "directory by hand.",
            exit_code=ExitCode.NEEDS_SUDO if isinstance(exc, PermissionError) else ExitCode.ERROR,
        ) from exc
    return freed


def _snapshot_download_with_xet_fallback(kwargs: dict, output) -> str:
    """snapshot_download, retried once over plain HTTP if the Xet chunk backend
    fails. Seen on hardware (2026-07-20, hf-xet 1.5.2, meta-llama/Llama-3.1-8B):
    'Task error: Unable to parse string as hex hash value' at ~9MB in, while
    HF_HUB_DISABLE_XET=1 downloads the same repo fine."""
    import importlib.util

    import huggingface_hub
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    def xet_active() -> bool:  # mirrors hub's is_xet_available (not public API)
        if getattr(huggingface_hub.constants, "HF_HUB_DISABLE_XET", False):
            return False
        return importlib.util.find_spec("hf_xet") is not None

    try:
        return snapshot_download(**kwargs)
    except (GatedRepoError, LocalEntryNotFoundError):
        raise  # real answers, not transport trouble — no point retrying
    except Exception as exc:
        if kwargs.get("local_files_only") or not xet_active():
            raise
        if output is not None:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            output.warn(
                f"download via the Xet backend failed ({reason}); "
                "retrying over plain HTTP (HF_HUB_DISABLE_XET=1) …"
            )
        # is_xet_available() reads this constant at call time; the env var only
        # matters to child processes but is set for consistency.
        huggingface_hub.constants.HF_HUB_DISABLE_XET = True
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        return snapshot_download(**kwargs)


def _honor_output_mode(output) -> None:
    """huggingface_hub draws its own tqdm bars straight to stderr, outside
    OutputManager — so `--quiet` and `--json` did not silence them. Suppress them
    in those modes (a `tt model pull --json` must emit one JSON document and
    nothing else). Only ever disables: re-enabling would clobber a user's own
    HF_HUB_DISABLE_PROGRESS_BARS, and the interactive default is left untouched
    because a live download bar genuinely earns its space.
    """
    if output is None or not (output.quiet or output.json_mode):
        return
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover - never fail a pull over cosmetics
        pass


def download_weights(
    model: ModelInfo, config: ConfigStore, *, offline: bool = False, output=None
) -> Path:
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    _honor_output_mode(output)
    kwargs = dict(
        repo_id=model.hf_repo,
        cache_dir=hf_cache_dir(config),
        local_files_only=offline,
        # skip the torch-format duplicates of the weights, matching the
        # inference server's own `hf download --exclude original/**` — for
        # Llama-class repos this halves the download.
        ignore_patterns=["original/*"],
    )
    try:
        path = _snapshot_download_with_xet_fallback(kwargs, output)
    except LocalEntryNotFoundError as exc:
        raise TTError(
            f"Model {model.name} ({model.hf_repo}) is not in the local cache.",
            why="--offline forbids downloading it."
            if offline
            else "Download failed and no cached copy exists.",
            next_step=f"Run `tt model pull {model.name}` on a connected machine first.",
            exit_code=ExitCode.OFFLINE,
        ) from exc
    except GatedRepoError as exc:
        raise TTError(
            f"Model {model.name} is gated on HuggingFace.",
            why=str(exc).splitlines()[0] if str(exc) else None,
            next_step=f"Accept the license at https://huggingface.co/{model.hf_repo} "
            "and log in with `hf auth login`.",
            exit_code=ExitCode.ERROR,
        ) from exc
    except Exception as exc:  # huggingface_hub raises a zoo of network errors
        raise TTError(
            f"Failed to download {model.name} ({model.hf_repo}).",
            why=str(exc).splitlines()[0] if str(exc) else None,
            next_step="Check connectivity and `hf auth whoami`, then retry.",
            exit_code=ExitCode.ERROR,
        ) from exc
    return Path(path)
