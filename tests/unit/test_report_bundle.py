# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure helpers behind `tt report bundle`: redaction and log tailing."""

from __future__ import annotations

from tenstorrent.commands.report_bundle import _tail_bytes, redact

HF = "hf_" + "A" * 30
PHC = "phc_" + "b" * 30


def test_redact_named_keys_and_bare_tokens():
    text = "\n".join(
        [
            "JWT_SECRET=deadbeef",
            f'export HF_TOKEN="{HF}"',
            f"posthog_project_key = \"{PHC}\"",
            "Authorization: Bearer eyJ.abc.def",
            "GET /v1/models?token=abc123&x=1",
            f"stray {HF} in a log line",
            "hf_abc is too short to be a token",
        ]
    )
    out = redact(text)
    assert out.splitlines() == [
        "JWT_SECRET=<redacted>",
        'export HF_TOKEN="<redacted>"',
        'posthog_project_key = "<redacted>"',
        "Authorization: Bearer <redacted>",
        "GET /v1/models?token=<redacted>&x=1",
        "stray <redacted> in a log line",
        "hf_abc is too short to be a token",
    ]
    assert "deadbeef" not in out and HF not in out and PHC not in out


def test_redact_leaves_clean_text_and_placeholders_alone():
    clean = "HF_TOKEN=<set>\nJWT_SECRET=<unset>\nTT_DATA_DIR=/home/me/.local/share\n"
    assert redact(clean) == clean
    assert redact("") == ""


def test_tail_bytes_truncates_from_the_end(tmp_path):
    path = tmp_path / "run.log"
    path.write_bytes(b"head\n" + b"x" * 100 + b"\ntail")
    data, truncated = _tail_bytes(path, 4)
    assert (data, truncated) == (b"tail", True)
    data, truncated = _tail_bytes(path, 10_000)
    assert truncated is False and data.startswith(b"head")
