# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Secret redaction (tenstorrent.redaction), as exhaustively as a table allows.

- Every format from the PR #38 retest (108 planted secrets, R1-R6 and F2), by id.
- A grid: credential names x the syntaxes they are written in x value shapes x
  the prefixes log lines carry. One test per syntax, so a failure names the shape.
- Bare tokens with a published prefix, PEM keys (whole, cut by a log tail, escaped
  inside JSON), unlabelled random strings.
- Structure: inspect records, argv and env lists, label maps.
- Known values: a value learned in one place is gone from every other, in every
  encoding it is likely to be written in.
- Controls that must come through untouched, many of them real log lines, and the
  invariants (line count, idempotence, placeholders, speed on huge single lines).
"""

from __future__ import annotations

import itertools
import json
import re
import time
from urllib.parse import quote

import pytest

from tenstorrent.redaction import REDACTED, Redactor, is_secret_name, redact


def _scrub(text: str, known: tuple[str, ...] = ()) -> str:
    redactor = Redactor(known)
    return redactor.final(redactor.text(text))


def _leaks(text: str, secrets: list[str], known: tuple[str, ...] = ()) -> list[str]:
    out = _scrub(text, known)
    return [secret for secret in secrets if secret in out]


# Fake provider tokens are assembled at import time: written out whole, GitHub's push
# protection takes them for real ones and refuses the push.
SLACK_BOT = "xo" + "xb-1234567890-0987654321-SlackBotPPPPPPPPPP"
AWS_KEY_ID = "AK" + "IAIOSFODNN7EXAMPLQ"


# -- the retest, case by case -------------------------------------------------------------
# (id, text, the values that must be gone). Ids match the retest report's table.
RETEST = [
    ("env-hf", "HF_TOKEN=s3cr3t-env-hf-001", ["s3cr3t-env-hf-001"]),
    ("env-spaces", "HF_TOKEN = s3cr3t-env-spaces-002", ["s3cr3t-env-spaces-002"]),
    ("hdr-bearer", "Authorization: Bearer s3cr3t-hdr-bearer-003", ["s3cr3t-hdr-bearer-003"]),
    ("hdr-bearer-nospace", "Authorization:Bearer s3cr3t-hdr-nospace-004", ["s3cr3t-hdr-nospace-004"]),
    ("hdr-bearer-lower", "authorization: bearer s3cr3t-hdr-lower-004b", ["s3cr3t-hdr-lower-004b"]),
    ("json-bearer", '{"headers": {"Authorization": "Bearer s3cr3t-json-bearer-005"}}', ["s3cr3t-json-bearer-005"]),
    ("pyrepr-bearer", "{'Authorization': 'Bearer s3cr3t-pyrepr-bearer-006'}", ["s3cr3t-pyrepr-bearer-006"]),
    ("hdr-basic", "Authorization: Basic Y2ktdXNlcjpzM2NyM3QtYmFzaWMtcHc=", ["Y2ktdXNlcjpzM2NyM3QtYmFzaWMtcHc="]),
    ("hdr-token", "Authorization: Token s3cr3t-hdr-token-008", ["s3cr3t-hdr-token-008"]),
    ("hdr-x-api-key", "X-Api-Key: s3cr3t-hdr-x-api-key-009", ["s3cr3t-hdr-x-api-key-009"]),
    ("hdr-x-auth-token", "X-Auth-Token: s3cr3t-hdr-x-auth-010", ["s3cr3t-hdr-x-auth-010"]),
    ("hdr-cookie", "Cookie: session=s3cr3t-hdr-cookie-011; theme=dark", ["s3cr3t-hdr-cookie-011"]),
    (
        "hdr-set-cookie",
        "Set-Cookie: sessionid=s3cr3t-hdr-set-cookie-012; Path=/; HttpOnly",
        ["s3cr3t-hdr-set-cookie-012"],
    ),
    ("flag-api-key", "vllm serve m --api-key s3cr3t-flag-api-key-013", ["s3cr3t-flag-api-key-013"]),
    ("flag-api-key-eq", "vllm serve m --api-key=s3cr3t-flag-eq-014", ["s3cr3t-flag-eq-014"]),
    ("flag-hf-token-eq", "download --hf-token=s3cr3t-flag-hf-015", ["s3cr3t-flag-hf-015"]),
    ("flag-hf_token", "download --hf_token s3cr3t-flag-hf_token-016", ["s3cr3t-flag-hf_token-016"]),
    ("flag-token", "hf download --token s3cr3t-flag-token-017", ["s3cr3t-flag-token-017"]),
    ("flag-password-eq", "tool --password=s3cr3t-flag-pw-018", ["s3cr3t-flag-pw-018"]),
    ("flag-passwd", "tool login --passwd s3cr3t-flag-passwd-019", ["s3cr3t-flag-passwd-019"]),
    ("flag-auth-token", "tool --auth-token s3cr3t-flag-auth-token-020", ["s3cr3t-flag-auth-token-020"]),
    (
        "flag-docker-login-p",
        "docker login ghcr.io -u bot -p s3cr3t-flag-docker-login-p-021",
        ["s3cr3t-flag-docker-login-p-021"],
    ),
    ("yaml-password", "password: s3cr3t-yaml-password-022", ["s3cr3t-yaml-password-022"]),
    ("yaml-token", "token: s3cr3t-yaml-token-023", ["s3cr3t-yaml-token-023"]),
    ("yaml-api-key-dash", "api-key: s3cr3t-yaml-dash-024", ["s3cr3t-yaml-dash-024"]),
    ("yaml-client-secret", "client_secret: s3cr3t-yaml-client-secret-025", ["s3cr3t-yaml-client-secret-025"]),
    ("env-db-password", "DB_PASSWORD=s3cr3t-env-db-026", ["s3cr3t-env-db-026"]),
    ("env-pgpassword", "PGPASSWORD=s3cr3t-env-pg-027", ["s3cr3t-env-pg-027"]),
    ("env-vllm", "VLLM_API_KEY=s3cr3t-env-vllm-028", ["s3cr3t-env-vllm-028"]),
    ("env-openai", "OPENAI_API_KEY=s3cr3t-env-openai-029", ["s3cr3t-env-openai-029"]),
    ("env-hf-hub", "HUGGING_FACE_HUB_TOKEN=s3cr3t-env-hub-030", ["s3cr3t-env-hub-030"]),
    ("env-jwt", "export JWT_SECRET=s3cr3t-env-jwt-031", ["s3cr3t-env-jwt-031"]),
    ("env-aws-secret", "AWS_SECRET_ACCESS_KEY=s3cr3t-env-aws-032", ["s3cr3t-env-aws-032"]),
    ("env-aws-session", "AWS_SESSION_TOKEN=s3cr3t-env-sess-033", ["s3cr3t-env-sess-033"]),
    ("env-lower", "hf_token=s3cr3t-env-lower-034", ["s3cr3t-env-lower-034"]),
    (
        "env-comma-tail",
        "password=s3cr3t-env-comma-head,s3cr3t-env-comma-tail-035",
        ["s3cr3t-env-comma-head", "s3cr3t-env-comma-tail-035"],
    ),
    ("json-api-key", '{"api_key": "s3cr3t-json-api-036"}', ["s3cr3t-json-api-036"]),
    ("json-nospace-pw", '{"password":"s3cr3t-json-pw-037"}', ["s3cr3t-json-pw-037"]),
    ("json-client-secret", '{"client_secret": "s3cr3t-json-client-secret-038"}', ["s3cr3t-json-client-secret-038"]),
    ("json-refresh", '{"refresh_token": "s3cr3t-json-refresh-039"}', ["s3cr3t-json-refresh-039"]),
    ("json-access", '{"access_token": "s3cr3t-json-access-040"}', ["s3cr3t-json-access-040"]),
    ("pyrepr-hf", "{'hf_token': 's3cr3t-pyrepr-hf-041'}", ["s3cr3t-pyrepr-hf-041"]),
    ("pykw-token", "login(token='s3cr3t-pykw-token-042')", ["s3cr3t-pykw-token-042"]),
    ("url-token", "GET /v1/models?token=s3cr3t-url-token-043&x=1", ["s3cr3t-url-token-043"]),
    ("url-access-token", "GET /cb?access_token=s3cr3t-url-access-044", ["s3cr3t-url-access-044"]),
    ("url-api-key", "GET /q?api_key=s3cr3t-url-api-045&y=2", ["s3cr3t-url-api-045"]),
    ("url-key", "GET https://maps.example.com/q?key=s3cr3t-url-key-046", ["s3cr3t-url-key-046"]),
    (
        "url-sig",
        "GET https://bucket.s3.amazonaws.com/obj?X-Amz-Signature=s3cr3t-url-sig-047",
        ["s3cr3t-url-sig-047"],
    ),
    (
        "url-userinfo",
        "pip install --index-url https://ci-user:s3cr3t-url-userinfo-048@pypi.internal.example/simple foo",
        ["s3cr3t-url-userinfo-048"],
    ),
    ("url-postgres", "DATABASE_URL=postgres://app:s3cr3t-url-postgres-049@db:5432/app", ["s3cr3t-url-postgres-049"]),
    ("url-redis", "connecting to redis://:s3cr3t-url-redis-050@cache:6379/0", ["s3cr3t-url-redis-050"]),
    ("url-git", "git clone https://oauth2:s3cr3t-url-git-051@gitlab.example.com/x.git", ["s3cr3t-url-git-051"]),
    ("netrc", "machine pypi.internal.example login ci password s3cr3t-netrc-052", ["s3cr3t-netrc-052"]),
    ("bare-hf", "using hf_" + "B" * 34, ["hf_" + "B" * 34]),
    ("bare-sk-ant", "key sk-ant-api03-" + "C" * 40, ["sk-ant-api03-" + "C" * 40]),
    ("bare-sk-proj", "key sk-proj-" + "D" * 40, ["sk-proj-" + "D" * 40]),
    ("bare-ghp", "remote uses ghp_GitHubPATLLLLLLLLLLLLLLLLLLLLLLLLLLLL", ["ghp_GitHubPATLLLLLLLLLLLLLLLLLLLLLLLLLLLL"]),
    (
        "bare-github-pat",
        "remote uses github_pat_11AAAAAAA0FineGrainedMMMMMMMMMMMMMMMM",
        ["github_pat_11AAAAAAA0FineGrainedMMMMMMMMMMMMMMMM"],
    ),
    ("bare-gho", "oauth gho_GitHubOAuthNNNNNNNNNNNNNNNNNNNNNNNNNN", ["gho_GitHubOAuthNNNNNNNNNNNNNNNNNNNNNNNNNN"]),
    ("bare-glpat", "remote uses glpat-GitLabTokenOOOOOOOO", ["glpat-GitLabTokenOOOOOOOO"]),
    (
        "bare-slack",
        "notify via " + SLACK_BOT,
        [SLACK_BOT],
    ),
    ("bare-aws-akid", "aws key id " + AWS_KEY_ID, [AWS_KEY_ID]),
    (
        "bare-jwt",
        "session eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiYXJlIn0.BareJwtSigQQQQQQQQ",
        ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiYXJlIn0.BareJwtSigQQQQQQQQ"],
    ),
    ("bare-google", "gcp key AIzaSyFakeGoogleKeyRRRRRRRRRRRRRRRRRRR", ["AIzaSyFakeGoogleKeyRRRRRRRRRRRRRRRRRRR"]),
    (
        "pem-body",
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAAPEMBODYSSSSSSSSSSSSSSSSSSSS\n"
        "-----END OPENSSH PRIVATE KEY-----",
        ["b3BlbnNzaC1rZXktdjEAAAAAPEMBODYSSSSSSSSSSSSSSSSSSSS"],
    ),
    (
        "crash-kwarg",
        "TypeError: f() got an unexpected keyword argument: token='s3cr3t-crash-kwarg-067'",
        ["s3cr3t-crash-kwarg-067"],
    ),
    (
        "crash-url",
        "requests.exceptions.HTTPError: 401 for url: https://hf.co/api?token=s3cr3t-crash-url-068",
        ["s3cr3t-crash-url-068"],
    ),
    ("cfg-phc", 'posthog_project_key = "phc_' + "k" * 30 + '"', ["phc_" + "k" * 30]),
    ("cfg-hf", 'hf_token = "s3cr3t-cfg-hf-069"', ["s3cr3t-cfg-hf-069"]),
    ("cfg-api-key", 'api_key = "s3cr3t-cfg-api-069b"', ["s3cr3t-cfg-api-069b"]),
    ("cfg-proxy-pw", 'url = "http://proxyuser:s3cr3t-cfg-proxy-pw-070@proxy.example:3128"', ["s3cr3t-cfg-proxy-pw-070"]),
    (
        "state-url-token",
        'source = "https://x-access-token:s3cr3t-state-071@github.com/tenstorrent/x.git"',
        ["s3cr3t-state-071"],
    ),
    ("selfupdate-gh", 'github_token = "s3cr3t-selfupdate-gh-072"', ["s3cr3t-selfupdate-gh-072"]),
    ("wf-hf", "2026-09-30 12:00:01 - run - INFO - HF_TOKEN=s3cr3t-wf-hf-073", ["s3cr3t-wf-hf-073"]),
    ("wf-jwt", "docker run -e JWT_SECRET=s3cr3t-wf-jwt-074 img", ["s3cr3t-wf-jwt-074"]),
    (
        "wf-pyrepr-bearer",
        "POST /v1/completions headers={'Authorization': 'Bearer s3cr3t-wf-pyrepr-bearer-075'}",
        ["s3cr3t-wf-pyrepr-bearer-075"],
    ),
    ("wf-docker-e", 'docker run -e "VLLM_API_KEY=s3cr3t-wf-docker-076" img', ["s3cr3t-wf-docker-076"]),
    ("wf-flag", "run.py --hf-token s3cr3t-wf-flag-077 --model m", ["s3cr3t-wf-flag-077"]),
    ("shell-tt-pass", "TT_DB_PASS=s3cr3t-shell-tt-pass-103", ["s3cr3t-shell-tt-pass-103"]),
    ("shell-tt-creds", "TT_CREDENTIALS=s3cr3t-shell-tt-creds-104", ["s3cr3t-shell-tt-creds-104"]),
    ("shell-hf-auth", "HF_AUTH=s3cr3t-shell-hf-auth-105", ["s3cr3t-shell-hf-auth-105"]),
    ("shell-tt-proxy", "TT_PROXY=http://u:s3cr3t-shell-tt-proxy-106@proxy:3128", ["s3cr3t-shell-tt-proxy-106"]),
    (
        "shell-hf-endpoint",
        "HF_ENDPOINT=https://mirror-user:s3cr3t-shell-hf-endpoint-107@hf-mirror.example",
        ["s3cr3t-shell-hf-endpoint-107"],
    ),
    ("ctr-env-pass", "API_PASS=s3cr3t-ctr-env-pass-085", ["s3cr3t-ctr-env-pass-085"]),
    ("ctr-env-creds", "OPENAI_CREDENTIALS=s3cr3t-ctr-env-creds-087", ["s3cr3t-ctr-env-creds-087"]),
    (
        "ctr-label-token",
        '"Labels": {"org.example.deploy-token": "s3cr3t-ctr-label-token-093", "a": "b"}',
        ["s3cr3t-ctr-label-token-093"],
    ),
    # The gaps the first pass of this rewrite left open.
    ("yaml-next-line", "token:\n  s3cr3t-yaml-next-line-201\nport: 80", ["s3cr3t-yaml-next-line-201"]),
    ("yaml-block", "password: |\n  s3cr3t-block-202\n  s3cr3t-block-203\nport: 80", ["s3cr3t-block-202", "s3cr3t-block-203"]),
    ("ini-spaces", "password = s3cr3t two words", ["s3cr3t two words"]),
    ("toml-section", '[auth]\nvalue = "s3cr3t-section-204"\n[server]\nport = 80', ["s3cr3t-section-204"]),
    (
        "url-token-user",
        "https://ghp1234567890abcdefghij:x-oauth-basic@github.com/r.git",
        ["ghp1234567890abcdefghij"],
    ),
    ("subscript", 'os.environ["HF_TOKEN"] = "s3cr3t-subscript-205"', ["s3cr3t-subscript-205"]),
    ("dockerfile-env", "ENV HF_TOKEN s3cr3t-dockerfile-206", ["s3cr3t-dockerfile-206"]),
    ("xml", "<password>s3cr3t-xml-207</password>", ["s3cr3t-xml-207"]),
    ("tab-table", "HF_TOKEN\ts3cr3t-tab-208", ["s3cr3t-tab-208"]),
    ("mysql-p", "mysql -u root -ps3cr3t-mysql-209 db", ["s3cr3t-mysql-209"]),
    ("curl-u", "curl -u alice:s3cr3t-curl-210 https://x", ["s3cr3t-curl-210"]),
    ("redis-cli", "redis-cli -h h -a s3cr3t-redis-211 ping", ["s3cr3t-redis-211"]),
    ("sshpass", "sshpass -p s3cr3t-sshpass-212 ssh host", ["s3cr3t-sshpass-212"]),
    ("proxy-no-scheme", "--proxy user:s3cr3t-proxy-213@host:3128", ["s3cr3t-proxy-213"]),
    (
        "slack-webhook",
        "https://hooks.slack.com/services/T000/B000/s3cr3tWebhook214",
        ["s3cr3tWebhook214"],
    ),
    ("escaped-json", '{\\"api_key\\": \\"s3cr3t-escaped-215\\"}', ["s3cr3t-escaped-215"]),
    ("argv-list-repr", "['vllm', '--api-key', 's3cr3t-argv-repr-216']", ["s3cr3t-argv-repr-216"]),
    ("camel", '{"apiKey": "s3cr3t-camel-217", "clientSecret": "s3cr3t-camel-218"}', ["s3cr3t-camel-217", "s3cr3t-camel-218"]),
    ("random-unlabelled", "id aB3dE5gH7jK9mN1pQ3rS5tU7vW9xY1zA2cD4 seen", ["aB3dE5gH7jK9mN1pQ3rS5tU7vW9xY1zA2cD4"]),
]


@pytest.mark.parametrize(("text", "secrets"), [c[1:] for c in RETEST], ids=[c[0] for c in RETEST])
def test_retest_format_is_redacted(text, secrets):
    assert _leaks(text, secrets) == [], _scrub(text)


def test_retest_formats_keep_their_surroundings():
    """F2: the value goes, the quote, host and scheme around it stay."""
    cases = {
        'curl -H "Authorization: Bearer s3cr3tA1b2c3" http://localhost:8000/v1/models':
            'curl -H "Authorization: Bearer <redacted>" http://localhost:8000/v1/models',
        "echo 'Authorization: Bearer s3cr3tA1b2c3'; echo done":
            "echo 'Authorization: Bearer <redacted>'; echo done",
        'source = "https://x-access-token:s3cr3t@github.com/tt/x.git"':
            'source = "https://x-access-token:<redacted>@github.com/tt/x.git"',
        "Cookie: session=s3cr3t; theme=dark": "Cookie: <redacted>",
        "postgres://app:s3cr3t@db:5432/app": "postgres://app:<redacted>@db:5432/app",
        '{"api_key": "s3cr3t", "port": 8000}': '{"api_key": "<redacted>", "port": 8000}',
        "--api-key s3cr3t --port 8000": "--api-key <redacted> --port 8000",
        "HF_TOKEN=s3cr3t TT_METAL_HOME=/opt/tt": "HF_TOKEN=<redacted> TT_METAL_HOME=/opt/tt",
        "f(token='s3cr3t', n=1)": "f(token='<redacted>', n=1)",
        '"CLOUD_TOKEN=",': '"CLOUD_TOKEN=",',
    }
    for text, expected in cases.items():
        assert redact(text) == expected, text


# -- the grid -----------------------------------------------------------------------------
GRID_NAMES = """HF_TOKEN API_KEY VLLM_API_KEY OPENAI_API_KEY JWT_SECRET DB_PASSWORD PGPASSWORD
TT_DB_PASS HF_AUTH TT_CREDENTIALS OPENAI_CREDENTIALS client_secret refresh_token github_token
token apiKey authToken x-api-key X-Auth-Token AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
DJANGO_SECRET_KEY LITELLM_MASTER_KEY VITE_LOGIN_PASSWORD secret password passwd access_token
private_key HF_TOKEN_2 API_KEY_PROD HUGGING_FACE_HUB_TOKEN MYSQL_PWD DOCKER_CONTROL_JWT_SECRET
CLOUD_CHAT_UI_AUTH_TOKEN secretKey clientSecret Password api-key auth_token session_token
id_token bearer_token SLACK_WEBHOOK_SECRET ANTHROPIC_API_KEY WANDB_API_KEY GH_TOKEN""".split()
GRID_VALUES = [
    "s3cr3tGridValue01",
    "p@ss:w0rd/with,comma;semi",
    "abc+/def==",
    "Zm9vYmFyYmF6cXV4",
    "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12",  # a JWT_SECRET's hex
    "hunter22",
    "x-y_z.1-2_3",
    "1234567890",
]
GRID_PREFIXES = ["", "2026-10-01 12:00:00 INFO ", "app-1  | ", "    "]
GRID_SYNTAXES = [
    "{n}={v}",
    "export {n}={v}",
    '{n}="{v}"',
    "{n}='{v}'",
    "{n}: {v}",
    '"{n}": "{v}"',
    "'{n}': '{v}'",
    '{n} = "{v}"',
    "{n} = {v}",
    "{n}:={v}",
    '{n} => "{v}"',
    "set {n}={v}",
    "-e {n}={v}",
    "--env {n}={v}",
    "GET /x?{n}={v}&y=1",
    '{{"env": {{"{n}": "{v}"}}}}',
    'os.environ["{n}"] = "{v}"',
    "ENV {n} {v}",
    "<{n}>{v}</{n}>",
    '{n}: "{v}"',
    "f({n}=\"{v}\")",
    "{n}={v}, other=1",
    '"{n}":"{v}"',
    '\\"{n}\\": \\"{v}\\"',
    "{n}\t{v}",
    "{{'{n}': '{v}'}}",
    'docker run -e "{n}={v}" img',
    "[{n}]={v}",
    '"{n}={v}"',
    "{n}=`{v}`",
    'export {n}="{v}" && run',
]


@pytest.mark.parametrize("syntax", GRID_SYNTAXES)
def test_grid_every_name_value_and_prefix(syntax):
    failures = []
    for name, value, prefix in itertools.product(GRID_NAMES, GRID_VALUES, GRID_PREFIXES):
        text = prefix + syntax.format(n=name, v=value)
        if value == "1234567890" and syntax.startswith("ENV"):
            continue  # a bare number after a plain ENV word is a count, by design
        out = _scrub(text)
        if value in out:
            failures.append(f"{text!r} -> {out!r}")
    assert failures == [], f"{len(failures)} leaks, e.g.\n" + "\n".join(failures[:10])


def test_grid_yaml_value_on_the_next_line():
    for name, value in itertools.product(GRID_NAMES, GRID_VALUES):
        for text in (f"{name}:\n  {value}\nnext: 1", f"  {name}:\n    {value}\n  next: 1"):
            assert value not in _scrub(text), text
        assert value not in _scrub(f"{name}: >-\n  {value}\n  more\nnext: 1")


def test_identifier_names_with_a_space_before_the_value():
    """`NAME value`: env-style and camelCase names count; prose words only when the
    value looks like a credential."""
    for name in ("HF_TOKEN", "PGPASSWORD", "apiKey", "X-Auth-Token", "auth_token"):
        assert "s3cr3t-spaced-1" not in _scrub(f"{name} s3cr3t-spaced-1"), name
    for name in ("token", "password", "secret"):
        assert "hunter22" not in _scrub(f"{name} hunter22"), name


# -- bare tokens and keys -----------------------------------------------------------------
BARE_TOKENS = [
    "hf_" + "a1B2" * 9,
    "api_org_" + "Q" * 30,
    "phc_" + "x" * 40,
    "sk-" + "z" * 48,
    "sk-ant-admin01-" + "Y" * 60,
    "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "rk_" + "test_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "whsec_" + "a" * 32,
    "ghp_" + "A" * 36,
    "gho_" + "B" * 36,
    "ghu_" + "C" * 36,
    "ghs_" + "D" * 36,
    "ghr_" + "E" * 36,
    "github_pat_11ABCDEFG0" + "h" * 60,
    "glpat-" + "F" * 20,
    "gldt-" + "G" * 20,
    "xo" + "xb-1234567890-0987654321-" + "H" * 24,
    "xo" + "xp-1234567890-0987654321-" + "I" * 24,
    "xa" + "pp-1-A0000000000-1234567890-" + "j" * 32,
    "AK" + "IA" + "IOSFODNN7EXAMPLE",
    "AS" + "IA" + "IOSFODNN7EXAMPLE",
    "AIza" + "SyD" + "k" * 32,
    "ya29." + "a0" * 30,
    "GOCSPX-" + "l" * 28,
    "eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJ4In0." + "m" * 40,
    "npm_" + "n" * 36,
    "pypi-AgEIcHlwaS5vcmc" + "o" * 60,
    "dckr_pat_" + "p" * 27,
    "hvs." + "q" * 24,
    "glsa_" + "r" * 32 + "_abcdef12",
    "sntrys_" + "s" * 40,
    "dop_v1_" + "ab" * 32,
    "ATATT3xFfGF0" + "t" * 40,
    "gsk_" + "u" * 52,
    "r8_" + "v" * 37,
    "nvapi-" + "w" * 64,
    "xai-" + "x" * 80,
    "pplx-" + "y" * 48,
    "tvly-" + "z" * 32,
    "lsv2_pt_" + "a" * 32 + "_" + "b" * 10,
    "lin_api_" + "c" * 40,
    "ntn_" + "d" * 46,
    "SG." + "e" * 22 + "." + "f" * 43,
    "shpat_" + "0123456789abcdef" * 2,
    "AGE-SECRET-KEY-1" + "QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L" + "QPZRY9X8GF2TVDW0S3JN54KHCE",
]


@pytest.mark.parametrize("token", BARE_TOKENS, ids=[t[:10] for t in BARE_TOKENS])
@pytest.mark.parametrize("around", ["{t}", "using {t} now", "'{t}'", "[{t}]", "app-1  | {t}", "x={t}y"])
def test_bare_token_with_a_known_prefix(token, around):
    text = around.format(t=token)
    if around == "x={t}y":  # glued to a following char: the prefix still marks it
        text = "x=" + token
    assert token not in _scrub(text), _scrub(text)


PEM_BODY = [
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj",
    "MjMwZDcxZGU3NTQ5NjRlNmQ5ZTYwMzJhMDhiNTVmZjAyZjQ2ZmZkNGJiYmU1Zm",
    "Q2==",
]


@pytest.mark.parametrize(
    "shape",
    [
        "-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n\n{body}\n-----END RSA PRIVATE KEY-----",
        "{body}\n-----END EC PRIVATE KEY-----",  # the log tail cut off the BEGIN line
        "app-1  | -----BEGIN OPENSSH PRIVATE KEY-----\n{prefixed}\napp-1  | -----END OPENSSH PRIVATE KEY-----",
        '{{"key": "-----BEGIN PRIVATE KEY-----\\n{escaped}\\n-----END PRIVATE KEY-----\\n"}}',
    ],
)
def test_private_key_bodies(shape):
    text = shape.format(
        body="\n".join(PEM_BODY),
        prefixed="\n".join(f"app-1  | {line}" for line in PEM_BODY),
        escaped="\\n".join(PEM_BODY),
    )
    out = _scrub(text)
    assert not any(line in out for line in PEM_BODY[:2]), out
    assert "PRIVATE KEY-----" in out  # support still sees that a key was there


# -- structure ----------------------------------------------------------------------------
def test_inspect_record_is_scrubbed_field_by_field():
    record = {
        "Id": "aaaaaaaaaaaa11",
        "Args": ["serve", "--api-key", "s3cr3t-a1", "--auth-token", "s3cr3t-a2", "--port", "8000"],
        "Config": {
            "Env": [
                "HUGGING_FACE_HUB_TOKEN=s3cr3t-e1",
                "MODEL=Qwen",
                "API_PASS=s3cr3t-e2",
                "HF_AUTH=s3cr3t-e3",
                "OPENAI_CREDENTIALS=s3cr3t-e4",
                "DATABASE_URL=postgres://app:s3cr3t-e5@db/app",
                "HTTPS_PROXY=http://u:s3cr3t-e6@proxy:3128",
                "EMPTY_TOKEN=",
            ],
            "Cmd": ["--hf-token", "s3cr3t-c1", "--password=s3cr3t-c2"],
            "Entrypoint": ["sh", "-c", "export JWT_SECRET=s3cr3t-c3 && exec run"],
            "Labels": {
                "org.example.deploy-token": "s3cr3t-l1",
                "registry.auth": "s3cr3t-l2",
                "org.opencontainers.image.source": "https://ci:s3cr3t-l3@git.example/x",
                "org.tenstorrent.tt-model": "1",
            },
            "Healthcheck": {"Test": ["CMD-SHELL", "curl -H 'Authorization: Bearer s3cr3t-h1' localhost"]},
        },
        "HostConfig": {"Binds": ["/home/me/.cache/huggingface:/hf"]},
        "State": {"Error": "login failed: password=s3cr3t-s1"},
        "Secrets": {"db": {"value": "s3cr3t-n1"}},
        "credentials": ["s3cr3t-n2", "s3cr3t-n3"],
    }
    redactor = Redactor()
    out = redactor.final(json.dumps(redactor.obj(record), indent=2))
    planted = re.findall(r"s3cr3t-[a-z]\d", json.dumps(record))
    assert len(planted) == 19
    assert [secret for secret in planted if secret in out] == [], out
    scrubbed = json.loads(out)
    assert scrubbed["Config"]["Env"][1] == "MODEL=Qwen"
    assert scrubbed["Config"]["Env"][-1] == "EMPTY_TOKEN="
    assert scrubbed["Args"][-2:] == ["--port", "8000"]
    assert scrubbed["Config"]["Labels"]["org.tenstorrent.tt-model"] == "1"
    assert scrubbed["HostConfig"]["Binds"] == ["/home/me/.cache/huggingface:/hf"]
    assert record["Config"]["Env"][0] == "HUGGING_FACE_HUB_TOKEN=s3cr3t-e1"  # input untouched


def test_argv_after_login_and_secret_flags():
    redactor = Redactor()
    assert redactor.argv(["docker", "login", "-u", "bot", "-p", "s3cr3t"]) == [
        "docker", "login", "-u", "bot", "-p", REDACTED
    ]
    assert redactor.argv(["docker", "run", "-p", "8000:8000", "img"]) == [
        "docker", "run", "-p", "8000:8000", "img"
    ]
    assert redactor.argv(["tool", "--token", "--verbose"]) == ["tool", "--token", "--verbose"]


# -- known values -------------------------------------------------------------------------
def test_a_value_learned_once_is_gone_everywhere():
    """The JWT hex labelled in one file and printed bare in another."""
    hexed = "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12"
    redactor = Redactor()
    first = redactor.text(f"JWT_SECRET={hexed}")
    second = redactor.text(f"signing with {hexed} now")
    assert hexed in second  # no rule can tell it from a hash on its own
    assert hexed not in redactor.final(first) and hexed not in redactor.final(second)


def test_known_values_in_every_encoding():
    secret = "p@ss w0rd/+=&x"
    texts = [
        secret,
        quote(secret, safe=""),
        quote(secret).replace("%20", "+"),
        json.dumps({"v": secret}),
        f"prefix{secret}suffix",
    ]
    redactor = Redactor([secret])
    for text in texts:
        assert redactor.final(text).count("w0rd") == 0, text


def test_known_values_skip_what_would_wreck_ordinary_text():
    redactor = Redactor(["true", "1", "8000", "/home/me", "short", "https://x.example", "<set>"])
    text = "true 1 8000 /home/me short https://x.example <set>"
    assert redactor.final(text) == text


# -- what must survive --------------------------------------------------------------------
CONTROLS = [
    "MAX_NUM_BATCHED_TOKENS=8192",
    "TOKENIZER_PATH=/models/tok",
    "max_tokens: 512",
    "num_tokens=5",
    "passwordless sudo enabled",
    "TT_METAL_HOME=/home/me/tt-metal",
    "--max-model-len 8192",
    "bos_token: <s>",
    "vllm serve --no-auth",
    "docker login --password-stdin ghcr.io",
    "docker run -p 8000:8000 img",
    "commit 5de5962a1b2c3d4e5f60718293a4b5c6d7e8f901",
    "tokens=1234 max_tokens=512",
    "token_embd.weight shape [4096, 128256]",
    "HF_TOKEN_PATH=/home/x/.hf/token",
    "key_cache allocated",
    "sha256:3f2a9c0e8d7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0",
    "Bearer authentication required",
    "for first_pass in x: PWD=/home/j",
    "meta-llama/Llama-3.1-8B-Instruct",
    "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc47-e2e0002ac7dc",
    "HF_TOKEN=<set>",
    "JWT_SECRET=<unset>",
    "HF_TOKEN is not set",
    "Neither VLLM_API_KEY nor JWT_SECRET environment variables are set.",
    "Time to first token: 0.62s",
    "time_to_first_token:4000 output_token_throughput_per_user:45",
    "| **ITL** | Inter-Token Latency - same as TPOT | ms |",
    "average inter-token latency",
    "token budget exceeded",
    "password reset link sent",
    "Loading tokenizer_config.json",
    "use_auth_token=True",
    "auth_type: basic",
    "uuid=550e8400-e29b-41d4-a716-446655440000",
    "Added request cmpl-8f3e2a1b4c5d6e7f8091a2b3c4d5e6f7-0 prompt_tokens=512",
    '  File "/home/me/.venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 1234, in step',
    "libtt_metal.so(_ZN2tt8tt_metal12MetalContext16destroy_aB3dE5gH7jK9mN1pQ3rS5tU7vW9xY1zA2cD4_12Ctx+0x90)",
    '"integrity": "sha512-hsBTNUqQTDwkWtcdYI2i06Y/nwbe8TgrRkZ6JOqOq6zfYbRz2Gk2pjDy5HBmMjRCJ2gq+g==",',
    '"object-keys": "1.1.1"',
    "[INFO] started",
    "[tokenizer]\npath = /x",
    "https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json",
    "git@github.com:tenstorrent/tt-metal.git",
    "ssh://git@github.com/tenstorrent/tt-metal.git",
    "redis://cache:6379/0",
    "Assignee: Jashan <jashansingh@tenstorrent.com>",
    "Reference: ttbr-0123456789ab",
    "device 0: p300c, firmware 18.10.0, eth 7.0.0",
    "KV cache blocks: 4096",
    "Setting TT_METAL_CACHE=/home/me/.cache/tt-metal-cache",
    "",
]


@pytest.mark.parametrize("text", CONTROLS)
def test_control_survives_untouched(text):
    assert _scrub(text) == text


def test_line_count_idempotence_and_placeholders():
    text = "\n".join(c[1] for c in RETEST) + "\r\nA=1\r\n\n"
    once = _scrub(text)
    assert once.count("\n") == text.count("\n")
    assert _scrub(once) == once
    assert "<redacted><redacted>" not in once


def test_huge_single_line_json_is_linear():
    """A one-line JSON document with thousands of secret keys must not go quadratic."""
    doc = json.dumps(
        [{"name": f"m{i}", "api_key": f"abc{i:08d}xyz", "path": "/a"} for i in range(30000)]
    )
    start = time.monotonic()
    out = redact(doc)
    assert time.monotonic() - start < 10
    assert "abc00000001xyz" not in out and '"path": "/a"' in out


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("HF_TOKEN", True), ("hf_token", True), ("apiKey", True), ("X-Amz-Security-Token", True),
        ("DJANGO_SECRET_KEY", True), ("PASSWORD_HASH", True), ("aws_credentials_v2", True),
        ("GITHUB_TOKEN_READONLY", True), ("HF_TOKEN_2", True), ("Set-Cookie", True),
        ("api_keys", True), ("MYSQL_PWD", True), ("GITHUB_PAT", True),
        ("MAX_NUM_BATCHED_TOKENS", False), ("max_tokens", False), ("bos_token", False),
        ("HF_TOKEN_PATH", False), ("token_count", False), ("auth_type", False),
        ("TOKENIZER_PATH", False), ("key", False), ("PWD", False), ("first_pass", False),
        ("object-keys", False), ("time_to_first_token", False), ("no-auth", False),
        ("use_auth_token", False), ("SECRET_NAME", False),
    ],
)
def test_is_secret_name(name, secret):
    assert is_secret_name(name) is secret
