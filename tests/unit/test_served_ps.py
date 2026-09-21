# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The pure helpers behind `tt model ps`: inspect-record parsing and the
uptime cell."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tenstorrent.backends.serving.ps import (
    _inspect_all,
    _published_port,
    _started_at,
    human_duration,
)
from tenstorrent.tools.runner import CaptureResult


def _state(started):
    return {"State": {"StartedAt": started}}


def test_started_at_parses_docker_nanoseconds_and_z():
    got = _started_at(_state("2026-09-08T15:37:59.62109313Z"))
    assert got == datetime(2026, 9, 8, 15, 37, 59, tzinfo=timezone.utc)


def test_started_at_parses_a_podman_offset_and_normalises_to_utc():
    got = _started_at(_state("2026-09-08T11:37:59.123456789-04:00"))
    assert got == datetime(2026, 9, 8, 15, 37, 59, tzinfo=timezone.utc)


def test_started_at_zero_time_means_never_started():
    assert _started_at(_state("0001-01-01T00:00:00Z")) is None


@pytest.mark.parametrize("raw", ["", None, "yesterday", "2026-13-45T00:00:00Z"])
def test_started_at_garbage_is_none(raw):
    assert _started_at(_state(raw)) is None
    assert _started_at({}) is None


def _bindings(**ports):
    return {"HostConfig": {"PortBindings": ports}}


def test_published_port_collapses_ipv4_and_ipv6_bindings():
    entry = _bindings(**{
        "8000/tcp": [
            {"HostIp": "0.0.0.0", "HostPort": "8000"},
            {"HostIp": "::", "HostPort": "8000"},
        ]
    })
    assert _published_port(entry) == 8000


def test_published_port_picks_the_lowest_of_several():
    entry = _bindings(**{
        "9090/tcp": [{"HostIp": "", "HostPort": "9090"}],
        "7000/tcp": [{"HostIp": "", "HostPort": "7000"}],
    })
    assert _published_port(entry) == 7000


def test_published_port_none_when_nothing_is_published():
    assert _published_port(_bindings()) is None
    assert _published_port({"HostConfig": {"PortBindings": None}}) is None
    assert _published_port({}) is None
    # a binding without a HostPort (docker picks one at start) is skipped
    assert _published_port(_bindings(**{"8000/tcp": [{"HostIp": "", "HostPort": ""}]})) is None


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, ""),
        (0, "0s"),
        (59, "59s"),
        (60, "1m"),
        (1080, "18m"),
        (11520, "3h 12m"),
        (93600, "1d 2h"),
        (-5, "0s"),
    ],
)
def test_human_duration(seconds, text):
    assert human_duration(seconds) == text


class _Docker:
    """A runner whose `inspect` prints one pretty-printed JSON array, as podman does."""

    def __init__(self, inspect_stdout: str):
        self.inspect_stdout = inspect_stdout
        self.argv: list[list[str]] = []

    def capture(self, argv, **kwargs):
        self.argv.append(list(argv))
        if argv[1] == "ps":
            return CaptureResult(returncode=0, stdout="abc123def456\n", stderr="")
        return CaptureResult(returncode=0, stdout=self.inspect_stdout, stderr="")


def test_inspect_all_accepts_a_pretty_printed_array():
    runner = _Docker('[\n  {\n    "Id": "abc123def456",\n    "Name": "/x"\n  }\n]\n')
    entries = _inspect_all(runner, "podman", include_stopped=True)
    assert [e["Id"] for e in entries] == ["abc123def456"]
    assert runner.argv[0] == ["podman", "ps", "--all", "--format", "{{.ID}}"]
    assert runner.argv[1] == ["podman", "inspect", "--format", "{{json .}}", "abc123def456"]


def test_inspect_all_skips_unparseable_lines():
    runner = _Docker('{"Id": "abc123def456"}\nnot json\n')
    assert [e["Id"] for e in _inspect_all(runner, "docker", include_stopped=False)] == [
        "abc123def456"
    ]
    assert runner.argv[0] == ["docker", "ps", "--format", "{{.ID}}"]
