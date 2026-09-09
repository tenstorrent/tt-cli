# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure-function tests for the issue URL/body builders."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from tenstorrent.commands.report import (
    _MAX_BODY_CHARS,
    REPO_TARGETS,
    build_issue_url,
)


def test_build_issue_url_round_trips_through_encoding():
    target = REPO_TARGETS["tt-smi"]
    title = "crash & burn: 100% repro"
    body = "line one\nline two\n<details>with & special = chars?</details>"
    url = build_issue_url(target, title=title, body=body)
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/tenstorrent/tt-smi/issues/new"
    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert query["title"] == title
    assert query["body"] == body
    assert query["labels"] == "bug"


def test_body_truncation_cap(monkeypatch):
    # Route an oversized environment through build_issue_body's truncation.
    import tenstorrent.commands.report as report

    monkeypatch.setattr(
        report, "_environment_lines", lambda appctx: ["x" * 10_000]
    )
    body = report.build_issue_body(appctx=None)
    assert body.endswith("(environment truncated)")
    assert len(body) <= _MAX_BODY_CHARS + len("\n\n(environment truncated)")
