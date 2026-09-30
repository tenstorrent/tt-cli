# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Pure helpers behind `tt report bundle`: redaction and log tailing."""

from __future__ import annotations

from tenstorrent.commands.report_bundle import _tail_bytes, redact, scrub_container_record

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


def test_redact_any_secret_named_env_var():
    """Every *_TOKEN / *_KEY / *SECRET / *PASSWORD name, in env, JSON and log shapes."""
    text = "\n".join(
        [
            '"HUGGING_FACE_HUB_TOKEN=fake-hub-value",',
            '"VLLM_API_KEY=fake-vllm-value",',
            '"OPENAI_API_KEY": "sk-fakefakefakefakefakefake",',
            "DB_PASSWORD: hunter2",
            "hf_token = fake-lower-value",
            "vllm serve m --api-key fake-flag-value --port 8000",
            f"bare sk-{'z' * 24} key",
        ]
    )
    out = redact(text)
    assert out.splitlines() == [
        '"HUGGING_FACE_HUB_TOKEN=<redacted>",',
        '"VLLM_API_KEY=<redacted>",',
        '"OPENAI_API_KEY": "<redacted>",',
        "DB_PASSWORD: <redacted>",
        "hf_token = <redacted>",
        "vllm serve m --api-key <redacted> --port 8000",
        "bare <redacted> key",
    ]
    assert "fake" not in out and "hunter2" not in out


def test_redact_keeps_non_secret_token_counts():
    text = "MAX_NUM_BATCHED_TOKENS=8192\nmax_tokens: 512\nTOKENIZER_PATH=/models/tok\n"
    assert redact(text) == text


def test_scrub_container_record_env_and_args():
    record = {
        "Id": "abc",
        "Args": ["serve", "--api-key", "fake-arg-value", "--port", "8000"],
        "Config": {
            "Env": ["HUGGING_FACE_HUB_TOKEN=fake-hub", "MODEL=Qwen", "VLLM_API_KEY=fake-k"],
            "Cmd": ["--hf-token", "fake-cmd-value"],
        },
    }
    out = scrub_container_record(record)
    assert out["Config"]["Env"] == [
        "HUGGING_FACE_HUB_TOKEN=<redacted>",
        "MODEL=Qwen",
        "VLLM_API_KEY=<redacted>",
    ]
    assert out["Args"] == ["serve", "--api-key", "<redacted>", "--port", "8000"]
    assert out["Config"]["Cmd"] == ["--hf-token", "<redacted>"]
    assert record["Config"]["Env"][0] == "HUGGING_FACE_HUB_TOKEN=fake-hub"  # input untouched
