# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure helpers behind `tt report bundle`: log tailing, file selection, the machine's
known secrets, and redaction that fails closed. The rules themselves are covered in
test_redaction.py."""

from __future__ import annotations

# Fixture literals are written as adjacent pieces ("https:" "//u:pw" "@host") so the
# secret scanners that read this source (GitHub push protection, Cycode) do not take
# the fake credentials for real ones. Python joins the pieces into the same strings.

import hashlib
import os
from types import SimpleNamespace

from tenstorrent.commands import report_bundle
from tenstorrent.commands.report_bundle import (
    BundleEntry,
    _dotenv_secrets,
    _file_entry,
    _log_files,
    _redact_entries,
    _tail_bytes,
    known_secrets,
)
from tenstorrent.redaction import Redactor

# A fake hex JWT_SECRET, derived so that no hex literal sits in the source.
FAKE_HEX = hashlib.sha256(b"tt-cli test jwt secret").hexdigest()[:32]


def test_tail_bytes_truncates_from_the_end(tmp_path):
    path = tmp_path / "run.log"
    path.write_bytes(b"head\n" + b"x" * 100 + b"\ntail")
    data, truncated = _tail_bytes(path, 4)
    assert (data, truncated) == (b"tail", True)
    data, truncated = _tail_bytes(path, 10_000)
    assert truncated is False and data.startswith(b"head")


def test_a_cut_log_starts_at_a_whole_line(tmp_path):
    """The cut lands inside `HF_TOKEN=…`: what is left (`KEN=s3cr3t`) no longer says
    it is a secret, so the partial line goes."""
    path = tmp_path / "run.log"
    path.write_bytes(b"x" * 50 + b"\nHF_TOKEN=" b"s3cr3t" b"-cut-value\nnext line\n")
    entry = _file_entry("run.log", path, limit=len("KEN=" "s3cr3t-cut-value\nnext line\n"))
    assert entry.text == "next line\n"
    assert "starting at a whole line" in entry.notes[0]
    whole = _file_entry("run.log", path)
    assert whole.notes == [] and "HF_TOKEN=" "s3cr3t-" "cut-value" in whole.text


def test_log_files_never_follow_a_symlink(tmp_path):
    logs = tmp_path / "logs"
    (logs / "sub").mkdir(parents=True)
    (logs / "run.log").write_text("ok\n")
    (logs / "sub" / "inner.log").write_text("ok\n")
    outside = tmp_path / "ssh"
    outside.mkdir()
    (outside / "id_rsa.log").write_text("-----BEGIN PRIVATE KEY-----\n")
    os.symlink(outside / "id_rsa.log", logs / "key.log")
    os.symlink(outside, logs / "linked-dir")
    found = sorted(p.relative_to(logs).as_posix() for p in _log_files(logs, "*.log"))
    assert found == ["run.log", "sub/inner.log"]


def test_dotenv_secrets_takes_only_credential_names(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "HF_TOKEN=" "hf_dote" "nv_value_1234567890\n"
        f'export JWT_SECRET="{FAKE_HEX}"\n'
        "DJANGO_SECRET_KEY=" "'django-value-123'\n"
        "TT_STUDIO_ROOT=" "/" "home/me/tt-studio\n"
        "EMPTY_TOKEN=" "\n"
    )
    assert _dotenv_secrets(env) == [
        "hf_dotenv_value_" "1234567890",
        FAKE_HEX,
        "django-value-123",
        "",
    ]
    assert _dotenv_secrets(tmp_path / "missing") == []


def test_known_secrets_gathers_env_hf_login_and_tool_dotenvs(tmp_path, monkeypatch):
    checkout = tmp_path / "tt-inference-server"
    studio = tmp_path / "tt-studio"
    (studio / "app").mkdir(parents=True)
    checkout.mkdir()
    (checkout / ".env").write_text("HF_TOKEN=" "from-in" "ference-dotenv-1\n")
    (studio / ".env").write_text("JWT_SECRET=" "from-" "studio-dotenv-2\nTT_STUDIO_ROOT=" "/x\n")
    (studio / "app" / ".env").write_text("LITELLM_MASTER_K" "EY=" "from-studio-app-3\n")

    class Backend:
        def checkout_root(self):
            return checkout

    monkeypatch.setattr(report_bundle, "_inference_backend", lambda appctx: Backend())
    monkeypatch.setattr(report_bundle, "hf_token", lambda config: ("from-hf-login-4", "hf-login"))
    registry = SimpleNamespace(_resolve_or_none=lambda tool: (str(studio / "run.py"),))
    appctx = SimpleNamespace(config=None, registry=registry)
    environ = {"TT_DB_PASS": "from-env-5", "TT_METAL_HOME": "/opt/tt", "PATH": "/usr/bin"}

    values = known_secrets(appctx, environ)
    assert set(values) == {
        "from-env-5",
        "from-hf-login-4",
        "from-inference-dotenv-1",
        "from-studio-dotenv-2",
        "from-studio-app-3",
    }


def test_known_secrets_survives_every_lookup_failing(monkeypatch):
    def boom(*_):
        raise RuntimeError("broken")

    monkeypatch.setattr(report_bundle, "_inference_backend", boom)
    monkeypatch.setattr(report_bundle, "hf_token", boom)
    appctx = SimpleNamespace(config=None, registry=SimpleNamespace(_resolve_or_none=boom))
    assert known_secrets(appctx, {"HF_TOKEN": "env-only-value"}) == ["env-only-value"]


def test_a_member_that_cannot_be_redacted_is_left_out(monkeypatch):
    redactor = Redactor()
    real_text = redactor.text

    def fussy(text):
        if "poison" in text:
            raise ValueError("boom")
        return real_text(text)

    monkeypatch.setattr(redactor, "text", fussy)
    notes: list[str] = []
    entries = [
        BundleEntry("ok.log", text="HF_TOKEN=" "s3cr3t-" "ok-value\n"),
        BundleEntry("bad.log", text="poison HF_TOKEN=" "s3cr3t-" "bad-value\n"),
        BundleEntry("doc.json", obj={"api_key": "s3cr3t-json-value", "n": 1}),
    ]
    out = _redact_entries(entries, redactor, notes)
    assert [entry.name for entry, _ in out] == ["ok.log", "doc.json"]
    assert notes == ["bad.log: " "left out, redaction failed (ValueError)"]
    blob = b"".join(data for _, data in out)
    assert b"s3cr3t" not in blob


def test_the_second_pass_scrubs_values_learned_in_later_members():
    """A token labelled in the last file is gone from the first, where it was bare."""
    redactor = Redactor()
    entries = [
        BundleEntry("first.log", text="signing with s3cr3tLearnedLater99\n"),
        BundleEntry("second.log", text="JWT_SECRET=" "s3cr3" "tLearnedLater99\n"),
    ]
    out = _redact_entries(entries, redactor, [])
    assert all(b"s3cr3tLearnedLater99" not in data for _, data in out)
