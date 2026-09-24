# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Measuring a weights download that nothing reports."""

from tenstorrent.backends.serving.progress.weights import WeightsProgress, directory_size, repo_cache_dir


def _blob(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def test_the_cache_path_matches_huggingfaces_own_layout(tmp_path):
    assert repo_cache_dir(tmp_path, "Qwen/Qwen3.6-35B-A3B") == (
        tmp_path / "hub" / "models--Qwen--Qwen3.6-35B-A3B"
    )


def test_a_file_reachable_by_two_names_is_counted_once(tmp_path):
    """The cache reaches the same bytes through blobs/ and snapshots/, so a
    naive walk reports exactly double the size of the download."""
    repo = tmp_path / "models--org--model"
    _blob(repo / "blobs" / "abc123", 4096)
    (repo / "snapshots" / "rev").mkdir(parents=True)
    (repo / "snapshots" / "rev" / "model.safetensors").symlink_to(repo / "blobs" / "abc123")
    assert directory_size(repo) == 4096


def test_symlinks_into_the_shared_store_are_followed(tmp_path):
    """The Xet backend keeps one content-addressed store per cache and links
    each repo's files into it; not following them reports almost nothing."""
    store = tmp_path / "hub" / "blobs" / "cf"
    _blob(store / "cf4945", 8192)
    repo = tmp_path / "hub" / "models--org--model"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "07e2").symlink_to(store / "cf4945")
    assert directory_size(repo) == 8192


def test_a_partial_download_counts_what_has_landed(tmp_path):
    repo = tmp_path / "models--org--model"
    _blob(repo / "blobs" / "abc.incomplete", 1500)
    assert directory_size(repo) == 1500


def test_a_dangling_link_and_a_missing_directory_are_not_fatal(tmp_path):
    """Both are ordinary mid-download states, not errors."""
    repo = tmp_path / "models--org--model"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "gone").symlink_to(tmp_path / "never-existed")
    assert directory_size(repo) == 0
    assert directory_size(tmp_path / "not-a-directory") == 0


def test_nothing_is_reported_until_a_repo_is_tracked(tmp_path):
    weights = WeightsProgress(tmp_path)
    assert weights.sample() is None
    weights.track("org/model")
    assert weights.sample() == (0, 0)  # nothing on disk, no total yet


def test_a_total_the_hub_never_gives_leaves_the_bytes_reported_anyway(tmp_path, monkeypatch):
    """A percentage needs the Hub; the figure that has landed does not, and a
    failed lookup must not take a serve down with it."""
    _blob(repo_cache_dir(tmp_path, "org/model") / "blobs" / "a", 2048)
    weights = WeightsProgress(tmp_path)
    monkeypatch.setattr(weights, "_fetch_total", lambda repo: None)
    weights.track("org/model")
    assert weights.sample() == (2048, 0)


def test_switching_repos_does_not_reuse_the_previous_total(tmp_path):
    weights = WeightsProgress(tmp_path)
    weights._totals["org/first"] = 999
    weights.track("org/first")
    assert weights.sample()[1] == 999
    weights.track("org/second")
    assert weights.sample()[1] == 0
