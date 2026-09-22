# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure helpers behind the `tt report bundle` support email: subject, body, mailto,
the .eml with the bundle attached, and the desktop launcher fallbacks."""

from __future__ import annotations

import datetime
import email
import re
import subprocess
from email import policy
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tenstorrent import __version__
from tenstorrent.commands import report_email as re_mod
from tenstorrent.commands.report_email import (
    ROTATION,
    SUPPORT_EMAIL,
    assignee_for_date,
    build_body,
    build_eml,
    build_mailto_url,
    build_subject,
    clean_title,
    eml_path_for,
    make_ref,
    open_file,
    open_mailto,
)


def test_make_ref_matches_the_studio_shape():
    assert re.fullmatch(r"ttbr-[0-9a-f]{12}", make_ref())
    assert make_ref() != make_ref()


def test_assignee_rotates_weekly_by_iso_week():
    # 2026-09-22 is ISO week 39 -> 39 % 3 == 0 -> first entry.
    assert assignee_for_date(datetime.date(2026, 9, 22)) == ROTATION[0]
    assert assignee_for_date(datetime.date(2026, 9, 29)) == ROTATION[1]
    assert assignee_for_date(datetime.date(2026, 10, 6)) == ROTATION[2]
    assert assignee_for_date(datetime.date(2026, 10, 13)) == ROTATION[0]
    assert assignee_for_date() in ROTATION


def test_subject_prefix_default_and_clipping():
    assert build_subject(None, "ttbr-abc") == "[TT-CLI] Bug report [ttbr-abc]"
    assert build_subject("   ", "ttbr-abc") == "[TT-CLI] Bug report [ttbr-abc]"
    assert build_subject(" serve hangs ", "ttbr-abc") == "[TT-CLI] serve hangs [ttbr-abc]"
    long = "x" * 150
    assert clean_title(long) == "x" * 99 + "…"
    assert len(clean_title(long)) == 100


def test_body_leads_with_the_machine_readable_lines():
    body = build_body(
        ref="ttbr-abc",
        assignee=("Jashan", "jashansingh@tenstorrent.com"),
        title="serve hangs",
        environment_lines=["- tt CLI: 1.0", "- OS: Linux x86_64"],
        archive_name="tt-report-ttbr-abc.tar.gz",
    )
    lines = body.splitlines()
    assert lines[:3] == [
        "Assignee: Jashan <jashansingh@tenstorrent.com>",
        "Reference: ttbr-abc",
        "Product: tt-cli",
    ]
    assert "## Summary\nserve hangs\n" in body
    assert body.count("_fill in_") == 4
    assert "- tt CLI: 1.0\n- OS: Linux x86_64\n" in body
    assert "attached tt-report-ttbr-abc.tar.gz" in body
    assert f"tt-cli {__version__}" in body


def test_body_without_environment_says_unknown():
    body = build_body(
        ref="r", assignee=ROTATION[0], title=None, environment_lines=[], archive_name="a"
    )
    assert "## Environment\n_unknown_\n" in body


def test_mailto_url_quotes_safely_and_reminds_to_attach():
    url = build_mailto_url("[TT-CLI] a+b [r]", "line 1\nline 2\n", "tt-report-r.tar.gz")
    assert url.startswith(f"mailto:{SUPPORT_EMAIL}?subject=")
    assert "+" not in url.split("?", 1)[1]  # quote(safe=""), never quote_plus
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    assert query["subject"] == ["[TT-CLI] a+b [r]"]
    assert query["body"][0].startswith("line 1\nline 2\n")
    assert "IMPORTANT: attach tt-report-r.tar.gz to this email before sending." in query["body"][0]


def test_mailto_body_is_clipped_but_keeps_the_reminder():
    url = build_mailto_url("s", "y" * 5000, "bundle.tar.gz")
    body = parse_qs(urlsplit(url).query)["body"][0]
    assert len(body) <= re_mod.MAX_MAILTO_BODY
    assert "[truncated — full details in the attached bundle]" in body
    assert body.rstrip().endswith("IMPORTANT: attach bundle.tar.gz to this email before sending.")


def test_eml_is_an_unsent_draft_with_the_archive_attached(tmp_path):
    archive = tmp_path / "tt-report-ttbr-abc.tar.gz"
    archive.write_bytes(b"\x1f\x8b" + bytes(range(256)) * 40)
    raw = build_eml(
        subject="[TT-CLI] serve hangs [ttbr-abc]",
        body="Assignee: A <a@x>\nReference: ttbr-abc\n\nhello — ünïcode\n",
        archive=archive,
        ref="ttbr-abc",
        now=datetime.datetime(2026, 9, 22, 12, 0, tzinfo=datetime.timezone.utc),
    )
    msg = email.message_from_bytes(raw, policy=policy.default)
    assert msg["X-Unsent"] == "1"
    assert msg["To"] == SUPPORT_EMAIL
    assert msg["From"] is None  # the mail client fills in the sender
    assert msg["Subject"] == "[TT-CLI] serve hangs [ttbr-abc]"
    assert msg["X-TT-Reference"] == "ttbr-abc"
    assert msg["X-Mailer"] == f"tt-cli/{__version__}"
    assert msg["Date"] == "Tue, 22 Sep 2026 12:00:00 +0000"
    assert msg["Message-ID"].endswith("@tt-cli.local>")
    assert msg.get_body().get_content().replace("\r\n", "\n") == (
        "Assignee: A <a@x>\nReference: ttbr-abc\n\nhello — ünïcode\n"
    )
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    (part,) = attachments
    assert part.get_content_type() == "application/gzip"
    assert part.get_filename() == "tt-report-ttbr-abc.tar.gz"
    assert part.get_payload(decode=True) == archive.read_bytes()
    assert raw.count(b"\r\n") > 10  # SMTP policy: CRLF line endings throughout


def test_eml_path_sits_beside_the_archive():
    assert eml_path_for(Path("/x/tt-report-ttbr-abc.tar.gz")) == Path("/x/tt-report-ttbr-abc.eml")
    assert eml_path_for(Path("/x/bundle.tgz")) == Path("/x/bundle.eml")
    assert eml_path_for(Path("/x/odd.bin")) == Path("/x/odd.eml")


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(re_mod.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def test_open_file_is_false_without_a_display(monkeypatch, linux, tmp_path):
    monkeypatch.delenv("DISPLAY")
    calls = []
    monkeypatch.setattr(re_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    assert open_file(tmp_path / "x.eml") is False
    assert calls == []


def test_open_file_is_false_without_xdg_open(monkeypatch, linux, tmp_path):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: None)
    assert open_file(tmp_path / "x.eml") is False


@pytest.mark.parametrize("returncode, expected", [(0, True), (3, False), (4, False)])
def test_open_file_trusts_the_launcher_exit_code(monkeypatch, linux, tmp_path, returncode, expected):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: "/usr/bin/xdg-open")
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(re_mod.subprocess, "run", fake_run)
    assert open_file(tmp_path / "x.eml") is expected
    assert seen["argv"] == ["/usr/bin/xdg-open", str(tmp_path / "x.eml")]
    assert seen["kwargs"]["check"] is False and seen["kwargs"]["timeout"]


def test_open_file_swallows_launcher_errors(monkeypatch, linux, tmp_path):
    monkeypatch.setattr(re_mod.shutil, "which", lambda name: "/usr/bin/xdg-open")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(a, 15)

    monkeypatch.setattr(re_mod.subprocess, "run", boom)
    assert open_file(tmp_path / "x.eml") is False


def test_open_file_on_macos_uses_open(monkeypatch, tmp_path):
    monkeypatch.setattr(re_mod.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("DISPLAY", raising=False)  # no X needed on macOS
    seen = {}
    monkeypatch.setattr(
        re_mod.subprocess,
        "run",
        lambda argv, **k: seen.setdefault("argv", argv) and subprocess.CompletedProcess(argv, 0),
    )
    assert open_file(tmp_path / "x.eml") is True
    assert seen["argv"] == ["open", str(tmp_path / "x.eml")]


def test_open_mailto_is_false_on_headless_linux(monkeypatch, linux):
    monkeypatch.delenv("DISPLAY")
    calls = []
    monkeypatch.setattr(re_mod.webbrowser, "open", lambda url: calls.append(url) or True)
    assert open_mailto("mailto:x") is False
    assert calls == []


def test_open_mailto_reports_the_browser_answer(monkeypatch, linux):
    monkeypatch.setattr(re_mod.webbrowser, "open", lambda url: True)
    assert open_mailto("mailto:x") is True
    monkeypatch.setattr(re_mod.webbrowser, "open", lambda url: False)
    assert open_mailto("mailto:x") is False

    def boom(url):
        raise OSError("no browser")

    monkeypatch.setattr(re_mod.webbrowser, "open", boom)
    assert open_mailto("mailto:x") is False
