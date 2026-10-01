# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Secret redaction for text tt hands to someone else (the `tt report bundle` archive).

Three layers, because no single one is enough:

- Rules over text: a credential-named assignment in any common shape (`HF_TOKEN=…`,
  `"api_key": "…"`, `password: …`, `login(token='…')`, `?key=…`), a secret flag
  (`--api-key …`, `"--auth-token", "…"`), credentials inside URLs, Authorization and
  Cookie headers, tokens with a well-known prefix (hf_, ghp_, sk-, AKIA …), JWTs,
  PEM private keys, and long random strings that carry no label at all.
- Structure: JSON documents (container inspect records above all) are scrubbed field
  by field before they are serialized, so an env list, an argv list or a label map
  is judged by its names rather than by how it happens to print.
- Known values: every value a rule redacted, plus every credential the caller found
  on the machine, is replaced wherever else it appears (`Redactor.final`). A token
  echoed bare into a log line that no rule can parse still goes, as long as it was
  labelled somewhere.

When unsure it hides: a false positive costs a support engineer one value, a false
negative costs the user a credential.
"""

from __future__ import annotations

import base64
import bisect
import functools
import json
import math
import os
import re
from collections import Counter
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote, quote_plus

REDACTED = "<redacted>"
# What tt writes in place of a value; never redacted, or learned, again.
PLACEHOLDERS = (REDACTED, "<set>", "<unset>")

Span = tuple[int, int]


# -- which names hold a credential ------------------------------------------------------
# Words that make a name a credential wherever they sit in it: SECRET_KEY_BASE,
# PASSWORD_HASH, aws_credentials_v2, PGPASSWORD.
_STRONG_ENDINGS = (
    "secret",
    "secrets",
    "password",
    "passwords",
    "passwd",
    "passphrase",
    "credential",
    "credentials",
)
_STRONG_WORDS = frozenset({"cred", "creds"})
# Words that only count as the last one, so ML and tokenizer names such as
# token_embd.weight or key_cache keep their values: HF_TOKEN, HF_AUTH, Set-Cookie.
_LAST_ENDINGS = ("token", "auth", "cookie", "cookies")
_LAST_WORDS = frozenset({"authorization", "auths", "jwt", "bearer"})
# Trailing words that qualify a credential rather than describe it: HF_TOKEN_2,
# API_KEY_PROD, GITHUB_TOKEN_READONLY, TOKEN_VALUE.
_QUALIFIERS = frozenset(
    "prod production dev development staging stage test testing ci local old new "
    "backup primary secondary alt read write ro rw readonly readwrite admin default "
    "live sandbox internal external personal org team bot value raw b64 base64 enc "
    "encrypted hex str string plain plaintext data".split()
)
# A last word that makes the value a fact about a credential rather than the
# credential: HF_TOKEN_PATH, SECRET_NAME, auth_type, token_count, PASSWORD_MIN_LENGTH.
_BENIGN_LAST = frozenset(
    "path paths file files filename filepath dir dirs directory folder url urls uri "
    "endpoint host hostname port server realm domain name names id ids label type "
    "types kind size length len count counts total limit max min budget rate usage "
    "stats ttl expiry expires expiration at lifetime timeout interval seconds secs "
    "sec ms minutes mins hours days enabled enable disabled disable required "
    "optional mode header headers prefix suffix format env var vars source provider "
    "method algorithm alg scope scopes version user username helper process command "
    "cmd location field fields pattern hint policy rotation cache level stdin schema "
    "store storage manager backend class module plugin handler text index idx offset "
    "position pos logprobs error errors status state valid verified check reset "
    "secure".split()
)
# A first word that makes the name a count, a tokenizer's special token or a switch:
# max_tokens, bos_token, --no-auth, use_auth_token=True.
_BENIGN_FIRST = frozenset(
    "max min num total count bos eos pad unk cls sep mask eot eom sos image video "
    "audio vision start end stop special prompt completion input output decoder "
    "encoder no disable enable skip use allow require with without ignore time first "
    "last next per inter avg mean median forward backward second third multi single "
    "one two render shader compile build test".split()
)
_VERSION_WORD = re.compile(r"v\d+")
_KEY_KINDS = frozenset(
    "api access secret private ssh auth signing encryption master service client "
    "deploy license account admin user".split()
)
# Split on any separator and on camelCase humps: apiKey, HFToken, X-Amz-Security-Token.
_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _words(name: str) -> list[str]:
    return [word.lower() for word in _WORD_SPLIT.split(name) if word]


def _unglue(word: str) -> str:
    """secret2 -> secret; v2, b64 and plain numbers stay as they are."""
    stem = word.rstrip("0123456789")
    if len(stem) < 3 or stem == word or word in _QUALIFIERS or _VERSION_WORD.fullmatch(word):
        return word
    return stem


@functools.lru_cache(maxsize=8192)  # a log repeats the same few hundred names
def is_secret_name(name: str, *, flag: bool = False, query: bool = False) -> bool:
    """Whether a variable, key, header, label or flag name (without its dashes) holds
    a credential. `flag` admits --pass and --pwd; `query` admits URL parameters that
    only mean a credential there (?key=, &sig=, X-Amz-Signature=)."""
    # A digit glued to a word qualifies it like a separate one: clientSecret2, TOKEN1.
    words = [_unglue(word) for word in _words(name)]
    compound = len(words) > 1  # before qualifiers go: pass_prod is not a lone PASS
    while len(words) > 1 and (
        words[-1] in _QUALIFIERS or words[-1].isdigit() or _VERSION_WORD.fullmatch(words[-1])
    ):
        words.pop()
    if not words or words[-1] in _BENIGN_LAST:
        return False
    if len(words) > 1 and words[0] in _BENIGN_FIRST:
        return False
    if any(word.endswith(_STRONG_ENDINGS) or word in _STRONG_WORDS for word in words):
        return True
    last = words[-1]
    if last.endswith(_LAST_ENDINGS) or last in _LAST_WORDS:
        return True
    if last.endswith("key"):
        # A bare `key` is a key into anything (a dict, a cache); api_key is not.
        return last != "key" or len(words) > 1 or query
    if last.endswith("keys"):
        # api_keys, SSH_KEYS; not object-keys or result_keys
        return last != "keys" or any(word in _KEY_KINDS for word in words[:-1])
    if last in ("pass", "pwd", "pw"):
        # DB_PASS, MYSQL_PWD, slack_pass, litellmPwd. Never on its own, so the shell's
        # PWD and a test's PASS keep their values; forward_pass is a first-word case.
        return query or flag or compound
    if last == "pat":
        return query or (name.isupper() and len(words) > 1)  # GITHUB_PAT
    if last.endswith("pass"):
        return name.isupper()  # PGPASS
    return query and last in ("sig", "signature")


# -- the text rules -----------------------------------------------------------------------
_EOL = re.compile(r"\r\n|[\r\n]")
# NAME and its separator: HF_TOKEN=, "api_key": , password: , token=', apiKey := …
# The lookahead/backreference pair takes the whole name at once (an atomic group,
# which `re` only has from 3.11), so a name is never re-tried from its tail.
_ASSIGNMENT = re.compile(
    r"(?<![\w.-])(?=(?P<name>[A-Za-z_][\w.-]*))(?P=name)"
    r"(?P<keyquote>\\*[\"'])?\]?[ \t]*(?P<sep>=>|:=|=|:)[ \t]*"
)
# `ENV HF_TOKEN value` in a Dockerfile, and an env-style NAME, a space or a tab and
# its value, as `env | column` or a settings table prints it. Not when the next word
# is plain English: "HF_TOKEN is not set" keeps its words.
_SPACED = re.compile(
    r"(?<![\w.$-])(?P<name>[A-Za-z_][\w-]*)"
    r"(?=(?P<sep>[ \t]+)(?P<v>[^\s\"'`=:<>()\[\]{},;|&][^\s\"'`]*?)[,;.)\]}]*(?:\s|$))"
)
# A name in that position must look like an identifier, not a word of prose
# ("token budget", "Password reset"): it has a `_` or `-`, a camelCase hump, or is
# an all-caps word. After a tab, or after ENV, any credential name counts.
_IDENTIFIER = re.compile(r".*[_-].*|.*[a-z][A-Z].*|[A-Z0-9]{6,}")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z][A-Z0-9]{5,}")
# …or, after a plain word (`password hunter22`), a value that is no word itself:
# eight or more characters with a letter and a digit or symbol in them.
_CREDENTIAL_WORD = re.compile(r"(?=.*[A-Za-z])(?=.*[0-9@#%+/=!?*~^$_.-])\S{8,}")
_PROSE = frozenset(
    "is isn't not was wasn't were missing unset set required must should has have had "
    "and or to for from in of on at by the a an found invalid expired empty provided "
    "env environment variable var value values given loaded using used will would "
    "can cannot could did does doesn't if when with without as be been being needs "
    "need also only already now then nor".split()
)
# <password>…</password>, <ApiKey attr="x">…</ApiKey>
_XML = re.compile(r"<(?P<name>[A-Za-z_][\w.:-]*)(?:[ \t][^<>]*)?>(?P<v>[^<\r\n]+)</(?P=name)>")
# --api-key value, --hf_token=value, -token value, and the list forms an argv takes
# when it is logged: "--api-key", "value" or '--api-key', 'value'.
_FLAG = re.compile(
    r"(?<![\w.-])--?(?=(?P<name>[A-Za-z][\w.-]*))(?P=name)"
    r"(?P<sep>=|[ \t]+|\\*[\"'][ \t]*,[ \t]*(?P<quote>\\*[\"']))"
)
# An auth scheme is kept so the redacted header still says what it was.
_SCHEME = re.compile(
    r"(?i)(?:Bearer|Basic|Token|Digest|Negotiate|NTLM|Bot|Api-?Key|Key|HMAC|OAuth|"
    r"Signature|SharedAccessSignature|AWS4-HMAC-SHA256)[ \t]+(?=[^\s\"'])"
)
_LITERALS = frozenset({"null", "none", "true", "false", "nil", "undefined"})
# A value ends at whitespace, written or escaped: `\n` inside a JSON string or a
# bytes repr ends the line it is on.
_WORD_VALUE = re.compile(r"(?:[^\s\"'`\\]|\\(?![nrt]))+")
_LINE_VALUE = re.compile(r"(?:[^\"'`\r\n\\]|\\(?![nrt]))*")
_QUERY_VALUE = re.compile(r"[^&#\s\"'`<>]*")
_CLOSERS = {"{": "}", "[": "]", "(": ")"}
_OPENER = re.compile(r"\\*[\"']|`")
# Terminal colour and cursor codes: `\x1b[0mHF_TOKEN=…` must still read as a name.
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

# user:password@ in any URL. The password may hold an unescaped `@`: the greedy group
# backs off to the last one.
_URL_USERINFO = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://(?P<user>[^\s:/?#@\"'`<>]*)"
    r"(?::(?P<pw>[^\s/?#\"'`<>]*))?@"
)
_TOKEN_USER = re.compile(r"[A-Za-z0-9_-]{16,}")  # https://<token>@github.com/…
_BEARER = re.compile(r"(?i)\bbearer[ \t]+(?P<v>[A-Za-z0-9._~+/=-]{8,})")

# Credentials with a published prefix, wherever they turn up bare (cf. gitleaks).
_TOKEN_SHAPES = (
    r"(?:hf|api_org|phc|phx)_[A-Za-z0-9]{20,}",  # Hugging Face, PostHog
    r"sk-[A-Za-z0-9_-]{20,}",  # OpenAI, Anthropic, DeepSeek, OpenRouter …
    r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}",  # Stripe
    r"whsec_[A-Za-z0-9]{16,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",  # GitHub
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"gl(?:pat|dt|rt|ptt|cbt|imt|oas|ft|soat|agent|wt)-[A-Za-z0-9_-]{16,}",  # GitLab
    r"xox[abeoprs]-[A-Za-z0-9-]{10,}",  # Slack
    r"xapp-[A-Za-z0-9-]{10,}",
    r"(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}(?![A-Z0-9])",  # AWS key id
    r"AIza[A-Za-z0-9_-]{30,}",  # Google API key
    r"ya29\.[A-Za-z0-9_-]{20,}",  # Google OAuth access token
    r"GOCSPX-[A-Za-z0-9_-]{20,}",  # Google OAuth client secret
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*",  # JWT
    r"npm_[A-Za-z0-9]{30,}",
    r"pypi-[A-Za-z0-9_-]{40,}",
    r"dckr_(?:pat|oat)_[A-Za-z0-9_-]{20,}",  # Docker Hub
    r"hv[sbr]\.[A-Za-z0-9_-]{20,}",  # HashiCorp Vault
    r"(?:glc|glsa)_[A-Za-z0-9+/=_-]{30,}",  # Grafana
    r"sntry[su]_[A-Za-z0-9+/=_-]{30,}",  # Sentry
    r"do[opr]_v1_[a-f0-9]{64}",  # DigitalOcean
    r"ATATT3[A-Za-z0-9_=-]{30,}",  # Atlassian
    r"(?:gsk|r8|csk)[_-][A-Za-z0-9]{30,}",  # Groq, Replicate, Cerebras
    r"(?:nvapi|xai|pplx|tvly)-[A-Za-z0-9_-]{20,}",  # NVIDIA, xAI, Perplexity, Tavily
    r"lsv2_(?:pt|sk)_[A-Za-z0-9_]{30,}",  # LangSmith
    r"lin_api_[A-Za-z0-9]{30,}",  # Linear
    r"(?:secret|ntn)_[A-Za-z0-9]{40,}",  # Notion
    r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}",  # SendGrid
    r"shp(?:at|ss|ca|pa)_[a-fA-F0-9]{32}",  # Shopify
    r"AGE-SECRET-KEY-1[0-9A-Z]{58}",  # age
)
_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(_TOKEN_SHAPES) + ")")
# Long alphanumeric runs: judged for randomness in Python (_looks_random).
_RANDOM_RUN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{32,}")

_PEM_BEGIN = re.compile(r"-----BEGIN[A-Z0-9 ]* PRIVATE KEY(?: BLOCK)?-----")
_PEM_END = re.compile(r"-----END[A-Z0-9 ]* PRIVATE KEY(?: BLOCK)?-----")
# A key line: base64 on its own, or ending a prefixed log line ("app-1  | MIIEvQ…").
_PEM_BODY = re.compile(r"(?:.*[ \t|])?(?P<v>[A-Za-z0-9+/=]{16,})[ \t]*")
_PEM_SHORT = re.compile(r"[ \t]*(?P<v>[A-Za-z0-9+/=]{1,15})[ \t]*")  # a block's last line
_PEM_HEADER = re.compile(r"[ \t]*(?:[A-Za-z-]+:.*)?")  # Proc-Type:, DEK-Info:, blank

# Credentials that only a command's own syntax gives away.
_CONTEXT_RULES = (
    # docker/podman/helm … login -p, sshpass -p: there -p is the password
    re.compile(r"\b(?:login|sshpass)\b[^\r\n]*?[ \t]-p(?:[ \t]+|=)?(?P<v>[^\s\"'`-][^\s\"'`]*)"),
    # mysql -pSECRET
    re.compile(r"\bmysql(?:dump|admin|import|sh)?\b[^\r\n]*?[ \t]-p(?P<v>[^\s\"'`-][^\s\"'`]*)"),
    # curl -u user:password
    re.compile(
        r"\bcurl\b[^\r\n]*?[ \t](?:-u|--user)(?:[ \t]+|=)[\"']?[^\s:\"']*:(?P<v>[^\s\"'`]+)"
    ),
    # redis-cli -a password
    re.compile(r"\bredis-cli\b[^\r\n]*?[ \t]-a[ \t]+(?P<v>[^\s\"'`]+)"),
    # .netrc: machine … login … password …, or a `password …` line of its own
    re.compile(
        r"\b(?:machine|login|account|default)[ \t]+\S+[ \t]+"
        r"(?:password|passwd)[ \t]+(?P<v>[^\s\"'`=:][^\s\"'`]*)"
    ),
    # …or a `password …` line of its own, as in a multi-line .netrc entry
    re.compile(r"(?m)^[ \t]*(?:password|passwd)[ \t]+(?P<v>[^\s\"'`=:][^\s\"'`]*)[ \t]*$"),
    # .pgpass: host:port:database:user:password
    re.compile(r"(?m)^[^\s:#]+:(?:\d+|\*):[^\s:]+:[^\s:]+:(?P<v>\S+)[ \t]*$"),
    # proxy credentials with no scheme: --proxy user:pass@host, HTTPS_PROXY=user:pass@host
    re.compile(
        r"(?i)proxy[\w-]*(?:[ \t]*[=:][ \t]*|[ \t]+)[\"']?[^\s:@/\"']+:(?P<v>[^\s@/\"']+)@"
    ),
    # Slack and Discord webhooks: the path is the credential
    re.compile(
        r"https://(?:hooks\.slack\.com/(?:services|workflows|triggers)|"
        r"(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks)/(?P<v>[A-Za-z0-9/_-]+)"
    ),
)


class _LineEnds:
    """Where each line of a text ends, found once: looking it up per value would
    rescan a long single-line JSON document from every secret key in it."""

    def __init__(self, text: str) -> None:
        breaks = [found.span() for found in _EOL.finditer(text)]
        self._ends = [start for start, _ in breaks]
        self._starts = [0] + [end for _, end in breaks]
        self._size = len(text)

    def after(self, pos: int) -> int:
        """The end of the line `pos` is on."""
        index = bisect.bisect_left(self._ends, pos)
        return self._ends[index] if index < len(self._ends) else self._size

    def lines_after(self, pos: int) -> Iterator[Span]:
        """(start, end) of every line below the one `pos` is on."""
        index = bisect.bisect_right(self._starts, pos)
        for start in self._starts[index:]:
            yield start, self.after(start)

    def start_of(self, pos: int) -> int:
        return self._starts[bisect.bisect_right(self._starts, pos) - 1]


def _line_spans(text: str) -> Iterator[Span]:
    pos = 0
    for found in _EOL.finditer(text):
        yield pos, found.start()
        pos = found.end()
    yield pos, len(text)


def _closing_quote(text: str, start: int, eol: int, quote_mark: str) -> int:
    """Where the value opened just before `start` closes, or the line end."""
    pos = start
    while True:
        found = text.find(quote_mark, pos, eol)
        if found == -1:
            return eol
        backslashes = 0
        while found - backslashes - 1 >= start and text[found - backslashes - 1] == "\\":
            backslashes += 1
        if len(quote_mark) == 2 or backslashes % 2 == 0:
            return found
        pos = found + 1


def _closing_bracket(text: str, start: int, eol: int) -> int:
    """The end of a bracketed value ({…}, […], (…)) on this line, or the line end."""
    expected: list[str] = []
    for pos in range(start, eol):
        char = text[pos]
        if char in _CLOSERS:
            expected.append(_CLOSERS[char])
        elif expected and char == expected[-1]:
            expected.pop()
            if not expected:
                return pos + 1
    return eol


def _value_span(
    text: str, eols: _LineEnds, pos: int, *, sep: str, query: bool = False
) -> Span | None:
    """The value that starts at `pos`, right after a credential's name or flag."""
    eol = eols.after(pos)
    if pos >= eol:
        return None
    quoted = _OPENER.match(text, pos, eol)
    opener = quoted.group() if quoted else text[pos]
    if quoted:  # "…", '…', `…`, and \"…\" at any depth of JSON-in-JSON escaping
        start = pos + len(opener)
        end = _closing_quote(text, start, eol, opener)
    elif opener in _CLOSERS:
        start, end = pos, _closing_bracket(text, pos, eol)
    else:
        start = pos
        if query:
            end = _QUERY_VALUE.match(text, pos, eol).end()
        elif sep in ("=", ":="):
            word = _WORD_VALUE.match(text, pos, eol)
            end = word.end() if word else pos
        else:  # `name: value`, as in a header or YAML: the rest of the line
            end = _LINE_VALUE.match(text, pos, eol).end()
        while end > start and text[end - 1] in " \t,;)]}":
            end -= 1
    scheme = _SCHEME.match(text, start, end)
    if scheme:
        start = scheme.end()
    value = text[start:end]
    if not value.strip() or value.startswith(PLACEHOLDERS) or value.lower() in _LITERALS:
        return None
    return start, end


# A YAML block scalar indicator: the value is on the lines below (`token: |`).
_BLOCK_SCALAR = re.compile(r"[|>][+-]?[ \t]*")
_INDENT = re.compile(r"[ \t]*")


def _block_spans(text: str, eols: _LineEnds, key_pos: int) -> Iterator[Span]:
    """A YAML value written below its key (`token:` then an indented line, a block
    scalar, or a list): every line indented past the key, or a `- item` at its
    level, up to the first one that is not."""
    line_start = eols.start_of(key_pos)
    indent = _INDENT.match(text, line_start).end() - line_start
    for start, end in eols.lines_after(key_pos):
        body = _INDENT.match(text, start, end).end()
        if body == end:  # a blank line does not end the block
            continue
        depth = body - start
        if depth > indent or (depth == indent and text.startswith("- ", body)):
            yield body, end
        else:
            return


def _assignment_spans(text: str, eols: _LineEnds) -> Iterator[Span]:
    for found in _ASSIGNMENT.finditer(text):
        name, sep = found.group("name"), found.group("sep")
        query = found.start() > 0 and text[found.start() - 1] in "?&;"
        if not is_secret_name(name, query=query):
            continue
        if text.startswith(sep[-1], found.end("sep")):  # `token == x`, C++ `Token::get`
            continue
        if text.endswith("://", 0, found.start()):  # URL userinfo: _url_spans has it
            continue
        pos, eol = found.end(), eols.after(found.end())
        if (
            not found.group("keyquote")
            and found.start() > 0
            and pos < eol
            and text[pos] in "\"'"
            and text[found.start() - 1] == text[pos]
        ):
            continue  # "NAME=" inside a quoted string: the quote ends it, empty

        if sep == ":" and not found.group("keyquote") and (
            pos == eol or _BLOCK_SCALAR.fullmatch(text, pos, eol)
        ):
            yield from _block_spans(text, eols, pos)
            continue
        outer = text[found.start() - 1] if found.start() > 0 else ""
        if (
            sep == "="
            and outer in "\"'"
            and not found.group("keyquote")
            and not text.startswith(outer, pos)
        ):
            # "NAME=a value" as one quoted item (docker -e "…", an env list): the
            # value runs to the quote that opened before the name.
            end = _closing_quote(text, pos, eol, outer)
            if end > pos and not text.startswith(PLACEHOLDERS, pos):
                yield pos, end
            continue
        if sep == "=" and text[found.start("sep") - 1] in " \t" and pos > found.end("sep"):
            sep = ":"  # INI and TOML style `password = two words`: the rest of the line
        span = _value_span(text, eols, pos, sep=sep, query=query)
        if span is not None and not _is_measure(name, text[span[0] : span[1]]):
            yield span


# A measurement, not a credential: "Time to first token: 0.62s", "key: 3".
_MEASURE = re.compile(
    r"[ \t]*-?\d+(?:\.\d+)?(?:e-?\d+)?[ \t]*"
    r"(?:(?:ns|us|µs|ms|s|sec|secs|seconds|min|mins|minutes|h|hours|%|x|"
    r"[KMGT]i?B|B|tok|toks|tokens|t)(?:/s)?)?[ \t]*"
)


def _is_measure(name: str, value: str) -> bool:
    """A decimal or a number with a unit, after a name that is not a password's.
    A bare integer still goes: HF_TOKEN=1234567890 is as likely a value as a count."""
    found = _MEASURE.fullmatch(value)
    if found is None or _strong_name(name):
        return False
    return "." in value or not value.strip().lstrip("-").isdigit()


def _strong_name(name: str) -> bool:
    """A name that holds a password or secret, where even 123456 is the value."""
    return any(word.endswith(_STRONG_ENDINGS) or word in _STRONG_WORDS for word in _words(name))


# `[auth]`, `[secrets.github]`, `[[credentials]]`: every value in the section is one.
_SECTION = re.compile(r"[ \t]*\[\[?(?P<name>[^\[\]\r\n]+)\]\]?[ \t]*(?:[#;].*)?")
_SECTION_ITEM = re.compile(
    r"[ \t]*[\w.\"'-]+[ \t]*[=:][ \t]*(?P<q>[\"']?)(?P<v>.*?)(?P=q)[ \t]*,?"
)


def _section_spans(text: str) -> Iterator[Span]:
    secret = False
    for start, end in _line_spans(text):
        header = _SECTION.fullmatch(text, start, end)
        if header:
            secret = any(
                is_secret_name(part.strip().strip("\"'"))
                for part in header.group("name").split(".")
            )
            continue
        if secret:
            item = _SECTION_ITEM.fullmatch(text, start, end)
            if item and item.group("v") and not item.group("v").startswith(PLACEHOLDERS):
                yield item.span("v")


def _flag_spans(text: str, eols: _LineEnds) -> Iterator[Span]:
    for found in _FLAG.finditer(text):
        if not is_secret_name(found.group("name"), flag=True):
            continue
        pos = found.end()
        quote_mark = found.group("quote")
        if quote_mark:  # "--api-key", "value": the value runs to its closing quote
            end = _closing_quote(text, pos, eols.after(pos), quote_mark)
            if end > pos and not text.startswith(PLACEHOLDERS, pos):
                yield pos, end
        elif found.group("sep") == "=" or not text.startswith("-", pos):
            span = _value_span(text, eols, pos, sep="=")
            if span is not None:
                yield span


def _url_spans(text: str) -> Iterator[Span]:
    """The password in user:password@host goes; the user and host stay, since support
    needs to know which proxy or registry it was. A user that is itself a token
    (https://<token>@github.com/…) goes as well."""
    for found in _URL_USERINFO.finditer(text):
        user = found.group("user")
        if found.group("pw"):
            yield found.span("pw")
            # https://<token>:x-oauth-basic@github.com: the "user" is the credential
            if _TOKEN_USER.fullmatch(user) and any(char.isdigit() for char in user):
                yield found.span("user")
        elif _TOKEN_USER.fullmatch(user):
            yield found.span("user")


def _looks_random(run: str) -> bool:
    """A credential with no label and no known prefix: long, mixed-case, with digits,
    and close to uniformly random. Hex (hashes, container ids) is single-case, and
    identifiers, paths and model names are split by `_`, `-`, `.` and `/` well
    before they get this long."""
    counts = Counter(run)
    kinds = (str.isdigit, str.isupper, str.islower)
    if any(sum(n for char, n in counts.items() if kind(char)) < 2 for kind in kinds):
        return False
    size = len(run)
    entropy = -sum(n / size * math.log2(n / size) for n in counts.values())
    return entropy >= 4.2


_MANGLED = re.compile(r"_Z[A-Za-z0-9_]*\Z")  # a C++ symbol: _ZN2tt8tt_metal12Metal…
# A digest written as `sha512-<base64>` (npm and pip integrity) or `sha256:<base64>`.
_DIGEST = re.compile(r"\b(?:sha(?:1|224|256|384|512)|md5)[-:][A-Za-z0-9+/=]*\Z")


def _random_spans(text: str) -> Iterator[Span]:
    for found in _RANDOM_RUN.finditer(text):
        if not _looks_random(found.group()):
            continue
        window = max(0, found.start() - 256)
        if _MANGLED.search(text, window, found.start()) or _DIGEST.search(
            text, window, found.start()
        ):
            continue
        yield found.span()


def _pem_spans(text: str) -> Iterator[Span]:
    """Private key bodies. The BEGIN and END lines stay, so support can see a key was
    there; a block whose top the log tail cut off (END without BEGIN) still goes."""
    inside = False
    pending: list[Span] = []  # key-shaped lines outside a block, in case an END follows
    for start, end in _line_spans(text):
        line = text[start:end]
        begin, close = _PEM_BEGIN.search(line), _PEM_END.search(line)
        if begin and close and close.start() > begin.end():  # one line, \n-escaped
            yield start + begin.end(), start + close.start()
            inside, pending = False, []
        elif begin:
            if end > start + begin.end():
                yield start + begin.end(), end
            inside, pending = True, []
        elif close:
            if not inside:
                yield from pending
            inside, pending = False, []
        elif inside:
            body = _PEM_BODY.fullmatch(line) or _PEM_SHORT.fullmatch(line)
            if body:
                yield start + body.start("v"), start + body.end("v")
            elif not _PEM_HEADER.fullmatch(line):
                inside = False  # not a key after all (a BEGIN quoted in prose): stop
        else:
            body = _PEM_BODY.fullmatch(line)
            if body:
                pending.append((start + body.start("v"), start + body.end("v")))
                if len(pending) > 400:
                    del pending[:-200]
            else:
                pending = []


def _match_spans(
    pattern: re.Pattern[str], text: str, keep: Callable[[str], bool] | None = None
) -> Iterator[Span]:
    group = "v" if "v" in pattern.groupindex else 0
    for found in pattern.finditer(text):
        if keep is None or keep(found.group(group)):
            yield found.span(group)


def _bearer_token(value: str) -> bool:
    """A token after "Bearer", not a word ("Bearer authentication required")."""
    return len(value) >= 20 or any(char.isdigit() for char in value)


def _spans(text: str) -> Iterator[Span]:
    eols = _LineEnds(text)
    yield from _assignment_spans(text, eols)
    yield from _flag_spans(text, eols)
    if "://" in text:
        yield from _url_spans(text)
    yield from _match_spans(_BEARER, text, _bearer_token)
    yield from _match_spans(_TOKEN, text)
    yield from _random_spans(text)
    if "PRIVATE KEY" in text:
        yield from _pem_spans(text)
    if "[" in text:
        yield from _section_spans(text)
    for found in _SPACED.finditer(text):
        name = found.group("name")
        loose = "\t" in found.group("sep") or text.endswith("ENV ", 0, found.start())
        value = found.group("v")
        if (
            (
                loose
                or _ENV_NAME.fullmatch(name)
                or (_IDENTIFIER.fullmatch(name) and not value.isalpha())
                or _CREDENTIAL_WORD.fullmatch(value)
            )
            and is_secret_name(name)
            and value.lower() not in _PROSE
        ):
            yield found.span("v")
    if "</" in text:
        for found in _XML.finditer(text):
            if is_secret_name(found.group("name").rpartition(":")[2]):
                yield found.span("v")
    for rule in _CONTEXT_RULES:
        yield from _match_spans(rule, text)


def _merge(spans: Iterable[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# -- known values -------------------------------------------------------------------------
def _names_a_file(value: str) -> bool:
    """A credential variable that points at a key file (GOOGLE_APPLICATION_CREDENTIALS
    =/home/me/key.json) names the secret rather than holding it; the path itself
    stays readable everywhere else. Anything else that merely looks like a path, a
    password starting with `/`, is learned like any other value."""
    if not value.startswith(("/", "~", "./", "../")):
        return False
    try:
        return os.path.exists(os.path.expanduser(value))
    except (OSError, ValueError):
        return False


def _learnable(value: str) -> bool:
    """Specific enough to scrub from everywhere once seen. Never a short word, a bare
    number or the path of an existing file: replacing those everywhere would take
    ordinary text down with them."""
    if len(value) < 8 or "\n" in value:
        return False
    if any(value in mark or mark in value for mark in PLACEHOLDERS):
        return False
    if value.isdigit() or _names_a_file(value):
        return False
    return len(value) >= 16 or not value.isalpha()


def _forms(value: str) -> list[str]:
    """A value as it may be written: raw, percent-encoded, or escaped inside JSON."""
    forms = {
        value,
        quote(value, safe=""),
        quote(value),  # a URL path keeps its `/`
        quote_plus(value),
        quote_plus(value, safe="/"),
        json.dumps(value)[1:-1],
        value.encode().hex(),
    }
    raw = value.encode()
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        forms.add(encoded.decode())
        forms.add(encoded.decode().rstrip("="))
    return sorted((form for form in forms if len(form) >= 8), key=len, reverse=True)


def _has_content(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    return not (isinstance(value, (str, list, dict)) and not value)


_ENV_ITEM = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)=")


class Redactor:
    """Rules, structure and known values. Every value it redacts is remembered, so
    `final` can take it out of every other text as well: use one per bundle."""

    def __init__(self, known: Iterable[str] = ()) -> None:
        self.known: set[str] = set()
        for value in known:
            self.learn(value)

    def learn(self, value: object) -> None:
        for line in str(value).splitlines():
            line = line.strip().strip("\"'")
            if _learnable(line):
                self.known.add(line)

    def _learn_leaves(self, value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                self._learn_leaves(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._learn_leaves(item)
        elif value is not None:
            self.learn(value)

    def text(self, text: str) -> str:
        """Redact by rule. Line breaks are never touched, so line counts survive.
        Terminal escape codes are dropped first: they would hide a name from the
        rules, and a support engineer reads the bundle without a terminal anyway."""
        if "\x1b" in text:
            text = _ANSI.sub("", text)
        spans = _merge(_spans(text))
        if not spans:
            return text
        out: list[str] = []
        pos = 0
        for start, end in spans:
            self.learn(text[start:end])
            out.append(text[pos:start])
            out.append(REDACTED)
            pos = end
        out.append(text[pos:])
        return "".join(out)

    def obj(self, value: Any) -> Any:
        """A JSON-shaped document, field by field: the value under a credential-named
        key goes whole (a dict of them too), lists of strings are read as argv or
        env, and every other string goes through the text rules. Returns a copy."""
        if isinstance(value, dict):
            out: dict = {}
            for key, item in value.items():
                name = self.text(str(key))
                if name in out:  # two keys that differed only by a secret
                    name = f"{name} ({len(out)})"
                if is_secret_name(str(key)) and _has_content(item):
                    self._learn_leaves(item)
                    out[name] = REDACTED
                else:
                    out[name] = self.obj(item)
            return out
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(item, str) for item in value):
                return self.argv(list(value))
            # Mixed: the strings still read as one argv ([{…}, "--api-key", "v"]).
            out_list = [None if isinstance(item, str) else self.obj(item) for item in value]
            strings = [i for i, item in enumerate(value) if isinstance(item, str)]
            for i, item in zip(strings, self.argv([value[i] for i in strings])):
                out_list[i] = item
            return out_list
        if isinstance(value, str):
            return self.text(value)
        return value

    def argv(self, argv: list[str]) -> list[str]:
        """An argv- or env-shaped list (Cmd, Args, Entrypoint, Env, Healthcheck.Test,
        podman's CreateCommand). A value is judged by the flag or name in front of it,
        which the text rules cannot see once each item is its own string."""
        out: list[str] = []
        secret_next = login = False
        for item in argv:
            if secret_next and not item.startswith("-"):
                self.learn(item)
                out.append(REDACTED)
                secret_next = False
                continue
            secret_next = False
            login = login or item in ("login", "sshpass")
            flag, eq, value = item.partition("=")
            if flag.startswith("-") and (
                (login and flag == "-p") or is_secret_name(flag.lstrip("-"), flag=True)
            ):
                if not eq:
                    secret_next = True
                    out.append(item)
                elif value:
                    self.learn(value)
                    out.append(f"{flag}={REDACTED}")
                else:
                    out.append(item)
                continue
            env = _ENV_ITEM.match(item)
            if env and is_secret_name(env.group("name")) and env.end() < len(item):
                self.learn(item[env.end() :])
                out.append(item[: env.end()] + REDACTED)
                continue
            out.append(self.text(item))
        return out

    def final(self, text: str) -> str:
        """Replace every known value, in the encodings it is likely to be written in.
        Run it after every text has been through `text`/`obj`, so a value redacted in
        one place is gone from all of them."""
        for value in sorted(self.known, key=len, reverse=True):
            for form in _forms(value):
                if form in text:
                    text = text.replace(form, REDACTED)
        return text


def redact(text: str) -> str:
    """The text rules alone, with nothing known in advance."""
    return Redactor().text(text)
