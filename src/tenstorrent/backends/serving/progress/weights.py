# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""How far a weights download has got, measured from the cache rather than told.

Neither backend can say, and both for the same sort of reason.
tt-inference-server shells out to `hf download`, which prints no progress at all
when its output is not a terminal; tt-model hands huggingface_hub a writer that
routes the byte counts to a row documented as "TTY only; a no-op when piped".
So on the longest step of a cold serve, the only honest source left is the cache 
directory the bytes are landing in.

The total comes from the Hub, once per repo and off the watch loop's thread: it
is a network call, it is allowed to fail, and a size that never arrives just
means the row reports what has landed without a percentage.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

#: Downloaded by neither backend: run.py passes `--exclude original/**` and tt's
#: own pull sets the matching ignore_patterns, so counting them would leave the
#: bar stuck short of a download that had in fact finished.
_NOT_FETCHED = ("original/",)


def repo_cache_dir(cache_root: Path, repo: str) -> Path:
    """Where huggingface_hub puts `repo` under an HF_HOME-shaped root."""
    return cache_root / "hub" / ("models--" + repo.replace("/", "--"))


def directory_size(path: Path) -> int:
    """Bytes on disk under `path`, including partial `.incomplete` blobs.

    Counts what is there right now, so a resumed download starts from what an
    earlier attempt left rather than from zero. Symlinks are *followed*: the Xet
    backend keeps one content-addressed store per cache and links each repo's
    files into it, so not following them reports almost nothing. Each file is
    then counted once by identity, because the cache reaches the same bytes
    through both `blobs/` and `snapshots/` and would otherwise read double.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            stat = entry.stat()  # follows into the store
                            if (stat.st_dev, stat.st_ino) in seen:
                                continue
                            seen.add((stat.st_dev, stat.st_ino))
                            total += stat.st_size
                    except OSError:  # vanished, or a dangling link, mid-download
                        continue
        except OSError:
            continue
    return total


class WeightsProgress:
    """Byte progress for whichever repo is being fetched, if any."""

    def __init__(self, cache_root: Path, *, token: str | None = None) -> None:
        self._cache_root = cache_root
        self._token = token
        self._repo: str | None = None
        self._totals: dict[str, int] = {}
        self._asked: set[str] = set()
        self._lock = threading.Lock()

    def track(self, repo: str | None) -> None:
        """Watch `repo` (or nothing). Cheap enough to call on every pass."""
        if repo == self._repo:
            return
        self._repo = repo
        if repo and repo not in self._asked:
            self._asked.add(repo)
            # Off this thread: the watch loop must keep reading the container's
            # log and answering the health probe while the Hub takes its time.
            threading.Thread(target=self._fetch_total, args=(repo,), daemon=True).start()

    def sample(self) -> tuple[int, int] | None:
        """`(bytes on disk, bytes expected)`, or None when nothing is tracked.

        An expected size of 0 means the Hub has not answered (yet, or at all) —
        the caller should report the figure it has rather than a percentage.
        """
        if self._repo is None:
            return None
        with self._lock:
            total = self._totals.get(self._repo, 0)
        return directory_size(repo_cache_dir(self._cache_root, self._repo)), total

    def _fetch_total(self, repo: str) -> None:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self._token).model_info(repo, files_metadata=True)
            total = sum(
                sibling.size or 0
                for sibling in (info.siblings or ())
                if not str(sibling.rfilename).startswith(_NOT_FETCHED)
            )
        except Exception:  # noqa: BLE001 — a missing total costs a percentage, nothing more
            return
        with self._lock:
            self._totals[repo] = total
