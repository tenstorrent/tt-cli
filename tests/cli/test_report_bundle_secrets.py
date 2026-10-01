# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report bundle` end to end: plant fake secrets in every source the command
reads, then search everything a user could hand to someone else.

Sources: tt's logs (a run log carrying every redaction format, a crash log, a log
whose tail cut lands inside a secret, a symlink to a file outside the logs dir, a
file named after a token), config.toml / installed.toml / self-update.toml, the
inference-server workflow logs and checkout .env, the tt-studio checkout .env, the
Hugging Face login, the shell environment, container inspect records (env, argv,
labels, healthcheck, state), container stdout and stderr, and the email title.

Outputs: every archive member's bytes and name, the .eml as raw bytes and decoded
(subject, body, the attachment unpacked again), stdout in human and --json mode,
and the decoded mailto: link. A value counts as leaked if any of its encodings
(raw, percent-encoded, JSON-escaped, base64) appears anywhere."""

from __future__ import annotations

# Fixture literals are written as adjacent pieces ("https:" "//u:pw" "@host") so the
# secret scanners that read this source (GitHub push protection, Cycode) do not take
# the fake credentials for real ones. Python joins the pieces into the same strings.

import base64
import email
import importlib.util
import io
import json
import os
import re
import shutil
import tarfile
from email import policy
from pathlib import Path
from urllib.parse import quote, quote_plus, unquote

import pytest

from tenstorrent.cli import app

pytestmark = pytest.mark.fakes_only

FAKES_DIR = Path(__file__).parent.parent / "fakes"
CORPUS = Path(__file__).parent.parent / "unit" / "test_redaction.py"


@pytest.fixture(autouse=True)
def in_scratch_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)


class _Openers:
    def __init__(self) -> None:
        self.mailtos: list[str] = []
        self.file_ok = True

    def open_file(self, path: Path) -> bool:
        return self.file_ok

    def open_mailto(self, url: str) -> bool:
        self.mailtos.append(url)
        return True


@pytest.fixture(autouse=True)
def openers(monkeypatch):
    """Never launch a real mail client or browser from the suite."""
    rec = _Openers()
    monkeypatch.setattr("tenstorrent.commands.report_email.open_file", rec.open_file)
    monkeypatch.setattr("tenstorrent.commands.report_email.open_mailto", rec.open_mailto)
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: False)
    return rec


def _paths(result) -> tuple[Path, Path]:
    ref = re.search(r"ttbr-[0-9a-f]{12}", result.stdout).group(0)
    return Path.cwd() / f"tt-cli-logs-{ref}.tar.gz", Path.cwd() / f"tt-cli-bug-report-{ref}.eml"


def _retest_lines() -> list[tuple[str, list[str]]]:
    """The unit corpus: every retest format with the values it must lose."""
    spec = importlib.util.spec_from_file_location("redaction_corpus", CORPUS)
    corpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus)
    return [(text, secrets) for _, text, secrets in corpus.RETEST]


# Values that only the machine-wide "known secrets" pass can catch: no rule can tell
# a bare hex JWT_SECRET from a hash, so each is labelled in one place and printed
# bare in another.
JWT_HEX = "9f8e7d6c5b4a3928" "1706f5e4d3c2b1a0" "9f8e7d6c5b4a3928" "1706f5e4d3c2b1a0"
DJANGO = "django-insecure-" "s3cr3tDjangoKey77"
HF_LOGIN = "hf_" + "LoginStoreToken" + "Z" * 20
SHELL_PASS = "s3cr3t-shell-pas" "s-bare-91"
LABEL_ONLY = "s3cr3t-label-onl" "y-bare-92"
TITLE_TOKEN = "hf_" + "TitlePastedToken" + "Y" * 20
CUT = "s3cr3t-cut-at-th" "e-tail-93"


def _plant(tmp_path: Path, monkeypatch) -> tuple[list[str], list[str]]:
    """Plant everything; return (secrets that must vanish, controls that must stay)."""
    secrets: list[str] = []
    controls = [
        "MAX_NUM_BATCHED_" "TOKENS=" "8192",
        "TOKENIZER_PATH=" "/" "models/tok",
        "Time to first token: " "0.62s",
        "passwordless sudo enabled",
        "--max-model-len 8192",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    corpus = _retest_lines()
    for _, values in corpus:
        secrets.extend(values)
    corpus_text = "\n".join(text for text, _ in corpus) + "\n" + "\n".join(controls) + "\n"

    # tt's own logs
    logs = Path(os.environ["TT_DATA_DIR"]) / "logs"
    (logs / "run").mkdir(parents=True)
    (logs / "crash").mkdir()
    (logs / "run" / "serve-001.log").write_text(
        corpus_text + f"bare known values: " f"{JWT_HEX} {DJANGO} {HF_LOGIN} {SHELL_PASS} {LABEL_ONLY}\n"
    )
    (logs / "crash" / "crash-002.log").write_text(
        "Traceback (most recent call last):" "\n"
        '  File "x.py", line 1, in <module>\n'
        "    login(token=" "'s3cr3t-crash-tb-94', password=" "\"s3cr3t-crash-tb-95\")\n"
        "requests.exceptions.HTTPError: " "401 for url: " "https:" "//u:" "s3cr3t-crash-tb-96" "@hf.co/api\n"
    )
    secrets += ["s3cr3t-crash-tb-94", "s3cr3t-crash-tb-95", "s3cr3t-crash-tb-96"]
    # a log over the 2 MB cap whose cut lands inside `HF_TOKEN=<CUT>`
    big = logs / "run" / "big.log"
    kept = f"KEN=" f"{CUT}\n".encode()
    filler = 2 * 1024 * 1024 - len(kept) - len(b"\nlast line\n")
    big.write_bytes(b"x" * 100 + b"\nHF_TO" + kept + b"z" * filler + b"\nlast line\n")
    secrets.append(CUT)
    # a symlink to a key outside the logs dir, and a file named after a token
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\ns3cr3t-outside-" "symlink-97\n")
    os.symlink(outside / "id_rsa", logs / "run" / "linked.log")
    os.symlink(outside, logs / "linked-dir")
    secrets.append("s3cr3t-outside-s" "ymlink-97")
    (logs / "run" / f"{HF_LOGIN}.log").write_text("named after a token\n")

    # tt's state files
    config_dir = Path(os.environ["TT_CONFIG_DIR"])
    config_dir.mkdir(parents=True, exist_ok=True)
    phc = "phc_" + "q" * 40
    (config_dir / "config.toml").write_text(
        f'[telemetry]\nposthog_project_key =' f' "{phc}"\n'
        '[network]\nproxy =' ' "http:' '//proxyuser:' 's3cr3t-cfg-proxy-98' '@proxy.example:' '3128"\n'
        '[auth]\nvalue =' ' "s3cr3t-cfg-section-99"\n'
    )
    data_dir = Path(os.environ["TT_DATA_DIR"])
    (data_dir / "installed.toml").write_text(
        '[tools.x]\nsource =' ' "https:' '//x-access-token:' 's3cr3t-installed-100' '@github.com/tt/x.git"\n'
    )
    secrets += [phc, "s3cr3t-cfg-proxy-98", "s3cr3t-cfg-section-99", "s3cr3t-installed-100"]
    (data_dir / "self-update.toml").write_text('github_token =' ' "s3cr3t-selfupdate-101"\n')
    secrets.append("s3cr3t-selfupdate-101")

    # the inference-server checkout: workflow log + .env
    repo = tmp_path / "inference-repo"
    (repo / "workflow_logs" / "run_logs").mkdir(parents=True)
    shutil.copy(FAKES_DIR / "inference-repo" / "run.py", repo / "run.py")
    monkeypatch.setenv("TT_TOOL_BIN_TT_I" "NFERENCE_SERVER", str(repo / "run.py"))
    (repo / ".env").write_text(f"HF_TOKEN=" f"s3cr3t-" f"inference-dotenv-102\nJWT_SECRET=" f"{JWT_HEX}\n")
    (repo / "workflow_logs" / "run_logs" / "run.log").write_text(
        corpus_text + "using inference token s3cr3t-inference" "-dotenv-102 bare\n"
    )
    secrets += ["s3cr3t-inference" "-dotenv-102", JWT_HEX]

    # the tt-studio checkout's .env
    studio = tmp_path / "studio-repo"
    studio.mkdir()
    shutil.copy(FAKES_DIR / "studio-repo" / "run.py", studio / "run.py")
    monkeypatch.setenv("TT_TOOL_BIN_TT_STUDIO", str(studio / "run.py"))
    (studio / ".env").write_text(f"DJANGO_SECRET_KEY=" f"{DJANGO}\nTT_STUDIO_ROOT=" f"{studio}\n")
    secrets.append(DJANGO)

    # the Hugging Face login store, and the shell
    hf_home = Path(os.environ["HF_HOME"])
    hf_home.mkdir(parents=True, exist_ok=True)
    (hf_home / "token").write_text(HF_LOGIN + "\n")
    secrets.append(HF_LOGIN)
    shell = {
        "HF_TOKEN": "s3cr3t-shell-hf-103",
        "JWT_SECRET": "s3cr3t-shell-jwt-104",
        "TT_API_KEY": "s3cr3t-shell-apikey-105",
        "TT_DB_PASS": SHELL_PASS,
        "TT_CREDENTIALS": "s3cr3t-shell-creds-106",
        "HF_AUTH": "s3cr3t-shell-hfauth-107",
        "TT_PROXY": "http:" "//u:" "s3cr3t-shell-proxy-108" "@proxy:" "3128",
        "HF_ENDPOINT": "https:" "//mirror:" "s3cr3t-shell-end" "point-109" "@hf-mirror.example",
        "TT_SOME_SETTING": "visible-value",
    }
    for name, value in shell.items():
        monkeypatch.setenv(name, value)
    secrets += [
        "s3cr3t-shell-hf-103", "s3cr3t-shell-jwt-104", "s3cr3t-shell-apikey-105", SHELL_PASS,
        "s3cr3t-shell-creds-106", "s3cr3t-shell-hfauth-107", "s3cr3t-shell-proxy-108",
        "s3cr3t-shell-end" "point-109",
    ]
    controls.append("TT_SOME_SETTING=" "visible-value")

    # containers
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which",
        lambda name: str(FAKES_DIR / "bin" / "docker") if name == "docker" else None,
    )
    record = {
        "Id": "aaaaaaaaaaaa11",
        "Name": "/tt-inference-se" "rver-prtest",
        "Args": ["--api-key", "s3cr3t-ctr-arg-110", "--auth-token", "s3cr3t-ctr-arg-111"],
        "Config": {
            "Image": "ghcr.io/tt/vllm:" "1",
            "Env": [
                "HF_TOKEN=" "s3cr3t-" "ctr-env-112",
                "VLLM_API_KEY=" "s3c" "r3t-ctr-env-113",
                f"JWT_SECRET=" f"{JWT_HEX}",
                "DATABASE_URL=" "postgres:" "//app:" "s3cr3t-ctr-env-114" "@db/app",
                "HTTPS_PROXY=" "http:" "//u:" "s3cr3t-ctr-env-115" "@proxy:" "3128",
                "API_PASS=" "s3cr3t-" "ctr-env-116",
                "HF_AUTH=" "s3cr3t-c" "tr-env-117",
                "OPENAI_CREDENTIA" "LS=" "s3cr3t-ctr-en" "v-118",
                "MODEL=" "Qwen/Qwen3-8B",
            ],
            "Cmd": ["--hf-token", "s3cr3t-ctr-cmd-119", "--password=" "s3cr3" "t-ctr-cmd-120"],
            "Entrypoint": ["sh", "-c", "export TT_KEY=" "s3cr3t-ct" "r-entry-121 && exec serve"],
            "Labels": {
                "org.example.deploy-token": LABEL_ONLY,
                "org.opencontainers.image.source": "https:" "//ci:" "s3cr3t-ctr-label-122" "@git.example/x",
            },
            "Healthcheck": {"Test": ["CMD-SHELL", "curl -H 'Authorization" ": " "Bearer " "s3cr3t-ctr-hc-123' localhost"]},
        },
        "State": {"Error": "auth failed: " "password=" "s3cr3t-" "ctr-state-124"},
        "Mounts": [],
    }
    secrets += [f"s3cr3t-ctr-{k}" for k in (
        "arg-110", "arg-111", "env-112", "env-113", "env-114", "env-115", "env-116",
        "env-117", "env-118", "cmd-119", "cmd-120", "entry-121", "label-122", "hc-123",
        "state-124",
    )] + [LABEL_ONLY]
    monkeypatch.setenv("FAKE_DOCKER_CONTAINERS", json.dumps([record]))
    monkeypatch.setenv(
        "FAKE_DOCKER_LOGS",
        json.dumps(
            {
                "aaaaaaaaaaaa": {
                    "stdout": corpus_text + f"jwt in use {JWT_HEX}\n",
                    "stderr": "ERROR: " "login(token=" "'s3cr3t-ctr-stderr-125') failed\n"
                    f"studio secret printed bare: " f"{DJANGO}\n",
                }
            }
        ),
    )
    secrets.append("s3cr3t-ctr-stderr-125")
    controls.append("MODEL=" "Qwen/Qwen3-8B")
    return sorted(set(secrets)), controls


def _forms(value: str) -> set[str]:
    """Encodings a leak could hide in."""
    forms = {value, quote(value, safe=""), quote(value), quote_plus(value), json.dumps(value)[1:-1]}
    if len(value) >= 12:
        forms.add(base64.b64encode(value.encode()).decode().rstrip("="))
    return {form for form in forms if len(form) >= 8}


def _haystacks(archive: Path, eml: Path, stdout: str) -> dict[str, str]:
    """Every text a user could hand over, decoded, keyed by where it came from."""
    out: dict[str, str] = {"stdout": stdout, "stdout (url-decoded)": unquote(stdout)}

    def unpack(blob: bytes, label: str) -> None:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:" "gz") as tar:
            for info in tar.getmembers():
                out[f"{label} member name {info.name}"] = info.name
                data = tar.extractfile(info).read().decode("utf-8", "replace")
                out[f"{label} {info.name}"] = data

    unpack(archive.read_bytes(), "archive")
    raw = eml.read_bytes()
    out["eml raw"] = raw.decode("utf-8", "replace")
    msg = email.message_from_bytes(raw, policy=policy.default)
    out["eml subject"] = str(msg["Subject"])
    out["eml headers"] = "\n".join(f"{k}: " f"{v}" for k, v in msg.items())
    out["eml body"] = msg.get_body().get_content()
    (part,) = msg.iter_attachments()
    unpack(part.get_payload(decode=True), "eml attachment")
    return out


def _assert_nothing_leaks(secrets, controls, haystacks):
    leaks = []
    for secret in secrets:
        for form in _forms(secret):
            for where, text in haystacks.items():
                if form in text:
                    leaks.append(f"{secret!r} (as {form!r}) in {where}")
    assert leaks == [], f"{len(leaks)} leaks:" f"\n" + "\n".join(leaks[:40])
    everything = "\n".join(haystacks.values())
    for control in controls:
        assert control in everything, f"control {control!r} was redacted"


TITLE = f"serve fails: " f"HF_TOKEN=" f"{TITLE_TOKEN} and {SHELL_PASS}"


def test_no_planted_secret_reaches_the_bundle_the_email_or_stdout(runner, tmp_path, monkeypatch, openers):
    secrets, controls = _plant(tmp_path, monkeypatch)
    secrets.append(TITLE_TOKEN)
    assert len(secrets) > 140

    result = runner.invoke(app, ["report", "bundle", "--title", TITLE, "--no-open"])
    assert result.exit_code == 0, result.output
    archive, eml = _paths(result)
    haystacks = _haystacks(archive, eml, result.output)
    _assert_nothing_leaks(secrets, controls, haystacks)

    members = {
        k.removeprefix("archive "): v
        for k, v in haystacks.items()
        if k.startswith("archive ") and not k.startswith("archive member name")
    }
    # The redaction kept what support needs.
    names = list(members)
    assert any(n.endswith("tt-logs/run/serve-001.log") for n in names)
    assert not any("linked" in n for n in names), "a symlink was followed"
    assert any(n.endswith("tt-logs/run/<redacted>.log") for n in names)
    big = next(v for k, v in members.items() if k.endswith("tt-logs/run/big.log"))
    assert big.startswith("z") and big.endswith("\nlast line\n") and "KEN" not in big
    env_txt = next(v for k, v in members.items() if k.endswith("/env.txt"))
    for name in ("HF_TOKEN", "JWT_SECRET", "TT_API_KEY", "TT_DB_PASS", "TT_CREDENTIALS", "HF_AUTH"):
        assert f"{name}=" f"<set>" in env_txt.splitlines()
    assert "TT_PROXY=" "http:" "//u:" "<redacted>" "@proxy:" "3128" in env_txt
    ps = json.loads(next(v for k, v in members.items() if k.endswith("containers/ps.json")))
    assert "MODEL=" "Qwen/Qwen3-8B" in ps[0]["Config"]["Env"]
    assert "HF_TOKEN=" "<redacted>" in ps[0]["Config"]["Env"]
    assert "Subject" in haystacks["eml headers"] and "serve fails: " "HF_TOKEN=" "<redacted>" in haystacks["eml subject"]


def test_json_and_mailto_outputs_are_redacted_too(runner, tmp_path, monkeypatch, openers):
    secrets, controls = _plant(tmp_path, monkeypatch)
    secrets.append(TITLE_TOKEN)
    out = tmp_path / "b.tar.gz"
    result = runner.invoke(app, ["report", "bundle", "--title", TITLE, "--json", "--output", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    haystacks = _haystacks(out, Path(payload["eml_path"]), result.output)
    haystacks["mailto (decoded)"] = unquote(payload["mailto"])
    haystacks["json notes"] = json.dumps(payload["notes"])
    _assert_nothing_leaks(secrets, [], haystacks)

    openers.file_ok = False  # force the mailto: fallback
    result = runner.invoke(app, ["report", "bundle", "--title", TITLE, "--output", str(tmp_path / "c.tar.gz")])
    assert result.exit_code == 0, result.output
    (url,) = openers.mailtos
    for secret in (TITLE_TOKEN, SHELL_PASS):
        assert secret not in unquote(url)
    assert "HF_TOKEN%3D%3Credacted%3E" in url


def test_collector_errors_are_redacted_in_the_manifest_and_on_stdout(runner, monkeypatch, tmp_path):
    from tenstorrent.commands import report_bundle

    def broken(appctx, note):
        raise RuntimeError("cannot reach https:" "//bot:" "s3cr3t-error-126" "@registry.example (HF_TOKEN=" "s3cr3t-" "error-127)")

    monkeypatch.setattr(report_bundle, "COLLECTORS", (*report_bundle.COLLECTORS, broken))
    out = tmp_path / "e.tar.gz"
    result = runner.invoke(app, ["report", "bundle", "--json", "--output", str(out), "--title", "x"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    haystacks = _haystacks(out, Path(payload["eml_path"]), result.output)
    _assert_nothing_leaks(["s3cr3t-error-126", "s3cr3t-error-127"], [], haystacks)
    assert any("broken: " "failed (RuntimeError" in note for note in payload["notes"])
