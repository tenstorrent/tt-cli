# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Seeded fuzzing of tenstorrent.redaction, on top of the tables in test_redaction.py.

Each test draws thousands of cases from a fixed seed, so a failure reproduces
exactly: random credential names (every case style, prefixes, qualifiers, glued
digits), random values with the symbols each syntax allows, random log prefixes,
ANSI colour and CRLF around them. A case fails if the value, or any 8-character
piece of it, survives. Fake values are built at runtime, so nothing here looks
like a real credential to a secret scanner reading the source.
"""

from __future__ import annotations

import base64
import json
import random
import re
import string
from urllib.parse import quote, quote_plus

import pytest

from tenstorrent.redaction import Redactor


def _scrub(text: str, known: tuple[str, ...] = ()) -> str:
    redactor = Redactor(known)
    return redactor.final(redactor.text(text))


_PREFIXES = ["", "HF", "VLLM", "OPENAI", "TT", "DB", "AWS", "GITHUB", "SLACK", "LITELLM",
             "DOCKER_CONTROL", "CLOUD_CHAT_UI", "MY_APP", "X", "PROD"]
_CORES = ["TOKEN", "API_KEY", "SECRET", "PASSWORD", "PASS", "PASSWD", "AUTH_TOKEN", "ACCESS_TOKEN",
          "REFRESH_TOKEN", "CLIENT_SECRET", "SECRET_KEY", "PRIVATE_KEY", "CREDENTIALS", "AUTH",
          "JWT_SECRET", "MASTER_KEY", "SESSION_TOKEN", "BEARER_TOKEN", "ID_TOKEN", "WEBHOOK_SECRET",
          "SIGNING_KEY", "ENCRYPTION_KEY", "PWD", "COOKIE"]
_SUFFIXES = ["", "", "", "_2", "_PROD", "_V2", "_READONLY", "_VALUE", "_B64", "2"]
_SAFE = string.ascii_letters + string.digits
_SYMBOLS = "-_.+/=@%!*~^$"
_QUOTED_SYMBOLS = _SYMBOLS + ",;:()[]{}<>?&#| "
# (syntax, value is quoted)
_SYNTAXES = [
    ("{n}={v}", False), ("export {n}={v}", False), ('{n}="{v}"', True), ("{n}='{v}'", True),
    ("{n}: {v}", False), ('"{n}": "{v}"', True), ("'{n}': '{v}'", True), ('{n} = "{v}"', True),
    ("{n}:={v}", False), ('{n} => "{v}"', True), ("-e {n}={v}", False), ("--env {n}={v}", False),
    ("GET /x?{n}={v}&y=1", False), ('{{"env": {{"{n}": "{v}"}}}}', True),
    ('os.environ["{n}"] = "{v}"', True), ('{n}: "{v}"', True), ("f({n}='{v}', x=1)", True),
    ('"{n}":"{v}"', True), ('\\"{n}\\": \\"{v}\\"', True), ("{n}\t{v}", False),
    ("{{'{n}': '{v}'}}", True), ('docker run -e "{n}={v}" img', True), ('"{n}={v}"', True),
    ("--{n} {v}", False), ("--{n}={v}", False), ('["--{n}", "{v}"]', True),
    ("{n}:\n  {v}\nnext: 1", False), ("<{n}>{v}</{n}>", False), ('{n} = "{v}"  # comment', True),
    ("Namespace({n}='{v}')", True), ("set {n}={v}", False), ('export {n}="{v}" && run', True),
    ("{n}={v}; other", False),
]
_CONTEXTS = ["", "2026-10-01 12:00:00,123 INFO ", "app-1  | ", "[2026-10-01T12:00:00Z] ",
             "\x1b[32mINFO\x1b[0m ", "> ", "DEBUG:root:"]
_ENDINGS = ["", " ", " done", "\r", " \x1b[0m", ", next=1"]


def _name(rng: random.Random) -> str:
    raw = "_".join(w for w in (rng.choice(_PREFIXES), rng.choice(_CORES)) if w)
    raw += rng.choice(_SUFFIXES)
    parts = raw.lower().split("_")
    style = rng.choice(["upper", "lower", "camel", "pascal", "kebab", "header"])
    if style == "upper":
        return raw
    if style == "lower":
        return raw.lower()
    if style == "camel":
        return parts[0] + "".join(p.title() for p in parts[1:])
    if style == "pascal":
        return "".join(p.title() for p in parts)
    if style == "kebab":
        return "-".join(parts)
    return "-".join(p.title() for p in parts)


def _value(rng: random.Random, quoted: bool) -> str:
    alphabet = _SAFE * 3 + (_QUOTED_SYMBOLS if quoted else _SYMBOLS)
    middle = "".join(rng.choice(alphabet) for _ in range(rng.randint(6, 46)))
    return rng.choice(_SAFE) + middle + rng.choice(_SAFE)


def _cases(seed: int, count: int):
    rng = random.Random(seed)
    while count:
        syntax, quoted = rng.choice(_SYNTAXES)
        name, value = _name(rng), _value(rng, quoted)
        if quoted:
            quote_mark = '"' if '"{v}"' in syntax else "'"
            value = value.replace(quote_mark, "")
        if syntax.startswith("<"):
            value = value.replace("<", "")
        if "?" in syntax:
            value = value.replace("&", "").replace("#", "")
        if re.fullmatch(r"(?i)pass|pwd", re.sub(r"[^A-Za-z]", "", name)):
            continue  # a lone PASS / PWD is the shell's or a test's, by design
        if syntax.startswith(("--{n}", '["--')) and not re.fullmatch(r"[A-Za-z][\w.-]*", name):
            continue
        context = rng.choice(_CONTEXTS)
        if "\n" in syntax and context.strip() == "" and context:
            continue  # a key indented past its own value is not YAML
        yield context + syntax.format(n=name, v=value) + rng.choice(_ENDINGS), value
        count -= 1


def _survives(value: str, out: str) -> bool:
    pieces = {value[i : i + 8] for i in range(max(1, len(value) - 7))}
    return value in out or any(p in out for p in pieces if p.strip() and not p.isdigit())


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_fuzz_names_syntaxes_values_and_contexts(seed):
    failures = []
    for text, value in _cases(seed, 5000):
        out = _scrub(text)
        if _survives(value, out):
            failures.append(f"{text!r} -> {out!r}")
    assert failures == [], f"{len(failures)} leaks, e.g.\n" + "\n".join(failures[:10])


def test_fuzz_known_values_in_every_encoding():
    """A value found on the machine, however odd its characters, goes in every form."""
    rng = random.Random(7)
    alphabet = _SAFE + "-_.+/=@%!*~^$&:;,?#[]{}()<>| "
    failures = []
    for _ in range(3000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(10, 40))).strip()
        if len(value) < 10 or value.isalpha():
            continue
        forms = [
            value, quote(value, safe=""), quote(value), quote_plus(value), json.dumps(value)[1:-1],
            base64.b64encode(value.encode()).decode(), base64.urlsafe_b64encode(value.encode()).decode(),
            value.encode().hex(),
        ]
        out = _scrub(" | ".join(f"got {form} here" for form in forms), (value,))
        if any(form in out for form in forms):
            failures.append(f"{value!r} -> {out!r}")
    assert failures == [], f"{len(failures)} leaks, e.g.\n" + "\n".join(failures[:10])


_JSON_NAMES = ["api_key", "password", "token", "HF_TOKEN", "clientSecret", "auth", "credentials",
               "privateKey", "x-api-key", "Authorization", "refresh_token", "DB_PASS"]


def _document(rng: random.Random, depth: int, planted: list[str]):
    def fresh(kind: str) -> str:
        value = f"fz{kind}{rng.randrange(10**12)}q"
        planted.append(value)
        return value

    if depth == 0 or rng.random() < 0.25:
        return rng.choice([f"plain{rng.randrange(99)}", rng.randrange(10**6), True, None])
    if rng.random() < 0.5:
        out: dict = {}
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.4:
                value = fresh("Key")
                out[rng.choice(_JSON_NAMES)] = rng.choice(
                    [value, [value], {"value": value}, f"Bearer {value}"]
                )
            else:
                out[f"field{rng.randrange(99)}"] = _document(rng, depth - 1, planted)
        return out
    items = [_document(rng, depth - 1, planted) for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.4:
        items.append(f"{rng.choice(['HF_TOKEN', 'API_KEY', 'DB_PASSWORD'])}={fresh('Env')}")
    if rng.random() < 0.4:
        items += [rng.choice(["--api-key", "--hf-token", "--password"]), fresh("Arg")]
    if rng.random() < 0.2:
        items.append("curl -H 'Authori" f"zation: Bearer {fresh('Hdr')}' x")
    return items


def test_fuzz_nested_json_documents():
    """Secrets at any depth of an inspect-record-like document: under a credential
    key, in env lists, after argv flags, in header strings, in mixed lists."""
    rng = random.Random(11)
    failures = []
    for _ in range(2000):
        planted: list[str] = []
        doc = _document(rng, 5, planted)
        redactor = Redactor()
        out = redactor.final(json.dumps(redactor.obj(doc)))
        leaked = [value for value in planted if value in out]
        if leaked:
            failures.append(f"{leaked} in {out[:300]}")
    assert failures == [], f"{len(failures)} leaks, e.g.\n" + "\n".join(failures[:5])


# -- shapes the fuzzers do not generate ----------------------------------------------------
def _v() -> str:
    return "fz" + "Probe" + "Value" + "77x"


@pytest.mark.parametrize(
    "build",
    [
        lambda v: f"\x1b[32mINFO\x1b[0mHF_TOKEN={v}",
        lambda v: f"\x1b[36mapi_key\x1b[0m: {v}",
        lambda v: f"\x1b]0;title\x07HF_TOKEN={v}",
        lambda v: f"HF_TOKEN=\x1b[1m{v}\x1b[0m",
        lambda v: json.dumps(json.dumps(json.dumps({"pass" + "word": v}))),
        lambda v: json.dumps(json.dumps(json.dumps(json.dumps({"api_key": v})))),
        lambda v: '{"api_key": "' + "".join(f"\\u{ord(c):04x}" for c in v) + '"}',
        lambda v: f'"msg": "start\\nHF_TOKEN={v}\\nend"',
        lambda v: "send: b'GET / HTTP/1.1\\r\\nAuthori" + f"zation: Bearer {v}\\r\\n\\r\\n'",
        lambda v: json.dumps({"log": f"export HF_TOKEN={v}\n", "stream": "stdout"}),
        lambda v: "db:5432:app:bob:" + v,
        lambda v: "Authori" + f"zation: AWS4-HMAC-SHA256 Credential=AKID/1/us/s3/aws4_request, Signature={v}",
        lambda v: "Proxy-Authori" + f"zation: Basic {v}",
        lambda v: f"Namespace(model='m', api_key='{v}', port=8000)",
        lambda v: f"environ({{'PATH': '/usr/bin', 'HF_TOKEN': '{v}'}})",
        lambda v: f"environment:\n  - HF_TOKEN={v}\n  - PORT=1",
        lambda v: f"https://app/#access_token={v}&type=bearer",
        lambda v: f"protocol=https\nhost=github.com\nusername=bot\npass" + f"word={v}",
        lambda v: f'token: "{v}\n  continued"',
        lambda v: f"auth:\n\ttoken: {v}",
        lambda v: f"HF_TOKEN={v}\x00tail",
        lambda v: f"loading\rHF_TOKEN={v}\r",
    ],
)
def test_shape_is_redacted(build):
    text = build(_v())
    assert _v() not in _scrub(text), _scrub(text)


def test_known_value_with_path_like_start_is_still_learned():
    value = "/T@C0$M&rAU?8"
    assert value not in _scrub(f"echo {value}", (value,))
    assert _scrub("cd /home/me/x", ("/home/me/x",)) == "cd /home/me/x"  # a real path stays


def test_escaped_line_breaks_end_a_value():
    v = _v()
    assert _scrub(f'"msg": "start\\nHF_TOKEN={v}\\nend"') == '"msg": "start\\nHF_TOKEN=<redacted>\\nend"'
    assert _scrub(f"\x1b[32mINFO\x1b[0m model loaded port=8000") == "INFO model loaded port=8000"
