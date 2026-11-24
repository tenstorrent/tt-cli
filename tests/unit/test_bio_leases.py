# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""chips.leased(): tt-bio's per-card leases, read the way tt-bio enforces them.

Real lease files and real flocks held by a real second process — the property
under test is the kernel's, so a mock would only test the mock.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from tenstorrent.backends.serving import chips

IDS = [0, 1, 2, 3]


def _lease_file(card: int, host: str | None = None) -> Path:
    directory = Path(os.environ["TT_BIO_LEASE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{host or socket.gethostname()}-card{card}.json"
    path.write_text('{"card": "%d", "released": null}\n' % card)
    return path


def test_no_lease_directory_means_nothing_is_leased():
    assert chips.leased(IDS) == set()


def test_a_held_lease_is_a_claim(tt_bio_lease):
    tt_bio_lease(2)
    assert chips.leased(IDS) == {2}


def test_a_stale_lease_file_is_not_a_claim():
    """A crashed run leaves its file behind; only a held flock is a live claim."""
    path = _lease_file(1)
    assert chips.leased(IDS) == set()
    assert path.exists(), "the probe must never delete tt-bio's files"


def test_a_lease_is_free_once_its_holder_dies(tt_bio_lease):
    holder = tt_bio_lease(3)
    assert chips.leased(IDS) == {3}
    holder.kill()
    holder.wait(timeout=10)
    assert chips.leased(IDS) == set()


def test_probing_does_not_keep_the_lock(tt_bio_lease):
    """tt checks with a non-blocking lock and lets go at once, so tt-bio can still
    take the card straight after (the holder only reports once it has the lock)."""
    _lease_file(0)
    assert chips.leased(IDS) == set()
    tt_bio_lease(0)
    assert chips.leased(IDS) == {0}


def test_another_hosts_lease_is_ignored(tt_bio_lease):
    """A fleet can share one lease directory across hosts; their cards are not ours."""
    tt_bio_lease(2, host="some-other-host")
    assert chips.leased(IDS) == set()


def test_the_lease_host_override_is_honoured(tt_bio_lease, monkeypatch):
    monkeypatch.setenv("TT_BIO_LEASE_HOST", "qb1")
    tt_bio_lease(1, host="qb1")
    assert chips.leased(IDS) == {1}


def test_only_the_asked_for_chips_are_checked(tt_bio_lease):
    tt_bio_lease(3)
    assert chips.leased([0, 1]) == set()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open any file")
def test_a_lease_that_cannot_be_checked_counts_as_held():
    """Never report a chip free that tt failed to check."""
    path = _lease_file(2)
    path.chmod(0)
    try:
        assert chips.leased(IDS) == {2}
    finally:
        path.chmod(0o644)


def test_lease_dir_resolves_like_tt_bio(monkeypatch, tmp_path):
    home = tmp_path / "somebody"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    monkeypatch.setenv("TT_BIO_LEASE_DIR", str(tmp_path / "explicit"))
    assert chips.lease_dir() == tmp_path / "explicit"

    monkeypatch.delenv("TT_BIO_LEASE_DIR")
    assert chips.lease_dir() == Path("/tmp/tt-bio-device-leases")

    (home / ".coworker").mkdir()
    assert chips.lease_dir() == home / ".coworker" / "state" / "leases"
