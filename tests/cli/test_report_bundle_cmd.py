# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report bundle`: archive layout, degradation, redaction, log selection, and
the support-email draft (.eml first, mailto: fallback, print-by-hand last).

Everything runs against the fake tools and an isolated cwd; nothing here needs
hardware, a container runtime, a mail client, or the network."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode

pytestmark = pytest.mark.fakes_only

FAKES_DIR = Path(__file__).parent.parent / "fakes"
HF = "hf_" + "x" * 30
PHC = "phc_" + "k" * 30


@pytest.fixture(autouse=True)
def in_scratch_cwd(tmp_path, monkeypatch):
    """The default archive lands in the cwd; never in the repo."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


class _Openers:
    """Records what the command tried to open; each opener's answer is settable."""

    def __init__(self) -> None:
        self.files: list[Path] = []
        self.mailtos: list[str] = []
        self.file_ok = True
        self.mailto_ok = True

    def open_file(self, path: Path) -> bool:
        self.files.append(path)
        return self.file_ok

    def open_mailto(self, url: str) -> bool:
        self.mailtos.append(url)
        return self.mailto_ok


@pytest.fixture(autouse=True)
def openers(monkeypatch):
    """Never launch a real mail client or browser from the suite."""
    rec = _Openers()
    monkeypatch.setattr("tenstorrent.commands.report_email.open_file", rec.open_file)
    monkeypatch.setattr("tenstorrent.commands.report_email.open_mailto", rec.open_mailto)
    return rec


@pytest.fixture(autouse=True)
def not_a_tty(monkeypatch):
    """CliRunner's stdin is a pipe in reality too, but pin it: the prompt tests flip it."""
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: False)


def _members(archive: Path) -> dict[str, bytes]:
    """{path-inside-bundle: bytes}, with the single top-level directory stripped."""
    out: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for info in tar.getmembers():
            top, _, rest = info.name.partition("/")
            assert top == archive.name.removesuffix(".tar.gz")
            out[rest] = tar.extractfile(info).read()
    return out


def _run(runner, *args):
    result = runner.invoke(app, ["report", "bundle", *args])
    assert result.exit_code == 0, result.output
    return result


def _archive_path(result) -> Path:
    """The archive path is the first data line on stdout (a title prompt may echo
    ahead of it). `stdout`, not `output`: CliRunner folds stderr into the latter."""
    for line in result.stdout.splitlines():
        if line.strip().endswith(".tar.gz"):
            return Path(line.strip())
    raise AssertionError(f"no archive path on stdout:\n{result.output}")


def test_bundle_default_path_and_manifest(runner, in_scratch_cwd):
    result = _run(runner)
    path = _archive_path(result)
    assert re.fullmatch(r"tt-report-ttbr-[0-9a-f]{12}\.tar\.gz", path.name)
    assert path.parent == in_scratch_cwd
    members = _members(path)
    assert {"environment.json", "env.txt", "manifest.json"} <= set(members)
    manifest = json.loads(members["manifest.json"])
    assert manifest["ref"] == path.name.removeprefix("tt-report-").removesuffix(".tar.gz")
    assert {f["name"] for f in manifest["files"]} == set(members) - {"manifest.json"}
    # isolated_dirs strips TT_TOOL_BIN_*, so tt-smi is missing: a note, not a failure
    assert any(n.startswith("tt-smi.json: unavailable") for n in manifest["notes"])
    assert "tt-smi.json" not in members


def test_bundle_output_json_and_quiet(runner, tmp_path, openers):
    target = tmp_path / "out" / "bundle.tar.gz"
    target.parent.mkdir()
    result = _run(runner, "--output", str(target), "--json", "--title", "serve hangs")
    payload = json.loads(result.output)
    assert payload["path"] == str(target)
    assert payload["size_bytes"] == target.stat().st_size
    assert set(payload["files"]) == set(_members(target))
    assert payload["eml"] == str(tmp_path / "out" / "bundle.eml")
    assert payload["to"] == "support@tenstorrent.com"
    assert payload["subject"] == f"[TT-CLI] serve hangs [{payload['ref']}]"
    assert payload["assignee"]["email"].endswith("@tenstorrent.com")
    assert payload["mailto"].startswith("mailto:support@tenstorrent.com?subject=")
    # --json is for scripts: never launches anything.
    assert openers.files == [] and openers.mailtos == []

    quiet = _run(runner, "--output", str(tmp_path / "q.tar.gz"), "--quiet")
    assert quiet.output == ""
    assert (tmp_path / "q.tar.gz").is_file() and (tmp_path / "q.eml").is_file()


def test_bundle_unwritable_output_is_the_only_hard_failure(runner):
    result = runner.invoke(
        app, ["report", "bundle", "--output", "/nonexistent-dir/tt-report.tar.gz"]
    )
    assert result.exit_code == ExitCode.ERROR
    assert "Cannot write support bundle" in result.output


def test_bundle_collects_smi_and_redacts_config(runner, smi_bin, tmp_path):
    config_dir = Path(os.environ["TT_CONFIG_DIR"])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        f'[telemetry]\nposthog_project_key = "{PHC}"\n'
    )
    data_dir = Path(os.environ["TT_DATA_DIR"])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "golden.json").write_text(json.dumps({"tag": "v1.2.3", "data": {"big": 1}}))
    (data_dir / "telemetry.toml").write_text('install_id = "do-not-ship"\n')

    members = _members(_archive_path(_run(runner)))
    assert json.loads(members["tt-smi.json"])  # the raw tt-smi document, verbatim
    env = json.loads(members["environment.json"])
    assert env["devices"], "parsed devices come from the same snapshot"
    assert any(row["name"] == "tt-smi" for row in env["tools"])

    config = members["config/config.toml"].decode()
    assert PHC not in config and 'posthog_project_key = "<redacted>"' in config
    assert json.loads(members["config/golden.json"]) == {"tag": "v1.2.3"}
    assert not any("telemetry" in name for name in members)


def test_bundle_tails_tt_logs_and_redacts_them(runner):
    logs_dir = Path(os.environ["TT_DATA_DIR"]) / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "big.log").write_bytes(b"x" * (3 * 1024 * 1024) + f"\nHF_TOKEN={HF}\n".encode())
    (logs_dir / "sub" ).mkdir()
    (logs_dir / "sub" / "small.log").write_text("fine\n")

    members = _members(_archive_path(_run(runner)))
    big = members["tt-logs/big.log"]
    assert len(big) <= 2 * 1024 * 1024 + len("<redacted>")
    assert HF.encode() not in big and b"HF_TOKEN=<redacted>" in big
    assert members["tt-logs/sub/small.log"] == b"fine\n"
    manifest = json.loads(members["manifest.json"])
    notes = {f["name"]: f["notes"] for f in manifest["files"]}
    assert notes["tt-logs/big.log"] and "truncated" in notes["tt-logs/big.log"][0]
    assert notes["tt-logs/sub/small.log"] == []


def test_bundle_keeps_the_newest_ten_workflow_logs(runner, tmp_path, monkeypatch):
    # A stand-in checkout: checkout_root() is the parent of the resolved run.py.
    repo = tmp_path / "repo"
    (repo / "workflow_logs" / "run_logs").mkdir(parents=True)
    shutil.copy(FAKES_DIR / "inference-repo" / "run.py", repo / "run.py")
    monkeypatch.setenv("TT_TOOL_BIN_TT_INFERENCE_SERVER", str(repo / "run.py"))
    for i in range(12):
        path = repo / "workflow_logs" / "run_logs" / f"run_{i:02d}.log"
        path.write_text(f"log {i}\n")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))
    (repo / "workflow_logs" / "notes.txt").write_text("not a log\n")

    members = _members(_archive_path(_run(runner)))
    logs = sorted(n for n in members if n.startswith("inference-server/"))
    assert logs == [
        f"inference-server/workflow_logs/run_logs/run_{i:02d}.log" for i in range(2, 12)
    ]
    assert members["inference-server/workflow_logs/run_logs/run_11.log"] == b"log 11\n"
    manifest = json.loads(members["manifest.json"])
    assert "inference-server: 2 older logs omitted" in manifest["notes"]


def _container(cid, name, image, labels=None):
    return {
        "Id": cid,
        "Name": f"/{name}",
        "Config": {"Image": image, "Labels": labels or {}, "Env": [f"HF_TOKEN={HF}"]},
        "Mounts": [],
    }


def test_bundle_collects_tt_container_logs(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which",
        lambda name: str(FAKES_DIR / "bin" / "docker") if name == "docker" else None,
    )
    argv_log = tmp_path / "docker-logs-argv.jsonl"
    monkeypatch.setenv("FAKE_DOCKER_LOGS_LOG", str(argv_log))
    monkeypatch.setenv(
        "FAKE_DOCKER_CONTAINERS",
        json.dumps(
            [
                _container("aaaaaaaaaaaa11", "tt-inference-server-aaaa", "ghcr.io/tt/vllm:1"),
                _container(
                    "bbbbbbbbbbbb22",
                    "tt-model-llama-n150",
                    "tt-model/llama:1",
                    labels={"org.tenstorrent.tt-model": "1"},
                ),
                _container("cccccccccccc33", "nginx", "nginx:1"),
            ]
        ),
    )

    members = _members(_archive_path(_run(runner)))
    records = json.loads(members["containers/ps.json"])
    assert [r["Name"] for r in records] == ["/tt-inference-server-aaaa", "/tt-model-llama-n150"]
    assert records[0]["Config"]["Env"] == ["HF_TOKEN=<redacted>"]
    assert {n for n in members if n.startswith("containers/")} == {
        "containers/ps.json",
        "containers/tt-inference-server-aaaa.log",
        "containers/tt-model-llama-n150.log",
    }
    log = members["containers/tt-inference-server-aaaa.log"].decode()
    assert "aaaaaaaaaaaa log line" in log
    assert HF not in log and "HF_TOKEN=<redacted>" in log
    calls = [json.loads(line) for line in argv_log.read_text().splitlines()]
    assert calls == [
        ["logs", "--tail", "5000", "aaaaaaaaaaaa"],
        ["logs", "--tail", "5000", "bbbbbbbbbbbb"],
    ]


def test_bundle_without_container_runtime_is_a_note(runner, monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )
    members = _members(_archive_path(_run(runner)))
    assert not any(n.startswith("containers/") for n in members)
    manifest = json.loads(members["manifest.json"])
    assert "containers: unavailable (TOOL_MISSING)" in manifest["notes"]


def test_bundle_env_txt_never_carries_values_of_secrets(runner, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", HF)
    monkeypatch.setenv("JWT_SECRET", "deadbeef")
    monkeypatch.delenv("SERVICE_PORT", raising=False)
    path = _archive_path(_run(runner))
    members = _members(path)
    lines = members["env.txt"].decode().splitlines()
    assert "HF_TOKEN=<set>" in lines
    assert "JWT_SECRET=<set>" in lines
    assert "SERVICE_PORT=<unset>" in lines
    assert f"TT_DATA_DIR={os.environ['TT_DATA_DIR']}" in lines
    everything = b"".join(members.values())
    assert HF.encode() not in everything and b"deadbeef" not in everything


# -- the support email ------------------------------------------------------------------
def _eml(path: Path):
    import email
    from email import policy

    return email.message_from_bytes(path.read_bytes(), policy=policy.default)


def test_bundle_writes_an_eml_draft_and_opens_it(runner, in_scratch_cwd, openers):
    result = _run(runner, "--title", "serve hangs on n150")
    archive = _archive_path(result)
    eml = archive.with_suffix("").with_suffix(".eml")
    assert eml.is_file()
    assert openers.files == [eml] and openers.mailtos == []

    msg = _eml(eml)
    ref = archive.name.removeprefix("tt-report-").removesuffix(".tar.gz")
    assert msg["X-Unsent"] == "1"
    assert msg["To"] == "support@tenstorrent.com"
    assert msg["Subject"] == f"[TT-CLI] serve hangs on n150 [{ref}]"
    body = msg.get_body().get_content().replace("\r\n", "\n")  # SMTP policy: CRLF
    assert body.splitlines()[1] == f"Reference: {ref}"
    assert body.splitlines()[2] == "Product: tt-cli"
    assert "## Summary\nserve hangs on n150\n" in body
    assert "- tt CLI:" in body  # the tt report issue environment block, reused
    (part,) = msg.iter_attachments()
    assert part.get_filename() == archive.name
    assert part.get_payload(decode=True) == archive.read_bytes()

    assert result.stdout.splitlines()[0] == str(archive)
    out = result.output
    assert f"email:     {eml}" in out
    assert "to:        support@tenstorrent.com" in out
    assert f"subject:   [TT-CLI] serve hangs on n150 [{ref}]" in out
    assert "this week's DX triage" in out
    assert "Opening the draft in your mail client" in out


def test_bundle_no_open_writes_both_files_and_launches_nothing(runner, openers):
    result = _run(runner, "--no-open")
    archive = _archive_path(result)
    assert archive.with_suffix("").with_suffix(".eml").is_file()
    assert openers.files == [] and openers.mailtos == []
    assert "Opening" not in result.output


def test_bundle_falls_back_to_mailto_when_no_eml_handler(runner, openers):
    openers.file_ok = False
    result = _run(runner, "--title", "x")
    archive = _archive_path(result)
    assert len(openers.files) == 1
    (url,) = openers.mailtos
    assert url.startswith("mailto:support@tenstorrent.com?subject=%5BTT-CLI%5D%20x%20%5B")
    # Rich may fold the status line at 80 columns, so match the head of it only.
    assert f"Opening a mailto: draft … attach {archive.name}" in result.output
    assert "warning" not in result.output


def test_bundle_mailto_flag_skips_the_eml_launch(runner, openers):
    _run(runner, "--mailto")
    assert openers.files == [] and len(openers.mailtos) == 1


def test_bundle_prints_the_email_when_nothing_can_open(runner, openers):
    openers.file_ok = False
    openers.mailto_ok = False
    result = _run(runner, "--title", "[brackets] stay literal")
    archive = _archive_path(result)
    ref = archive.name.removeprefix("tt-report-").removesuffix(".tar.gz")
    assert "warning: could not open a mail client" in result.output
    assert "To:      support@tenstorrent.com" in result.output
    assert f"Subject: [TT-CLI] [brackets] stay literal [{ref}]" in result.output
    assert "Assignee: " in result.output and f"Reference: {ref}" in result.output


def test_bundle_prompts_for_the_title_only_on_a_tty(runner, monkeypatch, openers):
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: True)
    result = runner.invoke(app, ["report", "bundle"], input="typed at the prompt\n")
    assert result.exit_code == 0, result.output
    assert "Subject for the support email [Bug report]:" in result.output
    eml = _archive_path(result).with_suffix("").with_suffix(".eml")
    assert "[TT-CLI] typed at the prompt [" in _eml(eml)["Subject"]

    result = runner.invoke(app, ["report", "bundle"], input="\n")
    assert "[TT-CLI] Bug report [" in _eml(_archive_path(result).with_suffix("").with_suffix(".eml"))["Subject"]

    # --title given, or not a TTY: no prompt at all.
    result = runner.invoke(app, ["report", "bundle", "--title", "given"])
    assert "Subject for the support email" not in result.output
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: False)
    result = runner.invoke(app, ["report", "bundle"])
    assert "Subject for the support email" not in result.output
    assert "[TT-CLI] Bug report [" in result.output


def test_bundle_warns_when_the_attachment_is_too_big(runner, monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.report_email.ATTACHMENT_WARN_BYTES", 10)
    result = _run(runner, "--no-open")
    assert "warning: the bundle is" in result.output and "25 MB" in result.output


def test_bundle_unwritable_eml_keeps_the_archive(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.commands.report_email.eml_path_for",
        lambda archive: Path("/nonexistent-dir") / "x.eml",
    )
    target = tmp_path / "b.tar.gz"
    result = runner.invoke(app, ["report", "bundle", "--output", str(target)])
    assert result.exit_code == ExitCode.ERROR
    assert "Cannot write the support email" in result.output
    assert target.is_file()
