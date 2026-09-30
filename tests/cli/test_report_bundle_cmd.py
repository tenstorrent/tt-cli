# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report bundle`: archive layout, degradation, redaction, log selection, and
the support-email draft (.eml first, mailto: fallback) and the closing panel.

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


def _ref(result) -> str:
    """The reference from the panel's first row. `stdout`, not `output`: CliRunner
    folds stderr into the latter. The short Reference row never wraps; the paths may."""
    m = re.search(r"Reference\s+→\s+(ttbr-[0-9a-f]{12})", result.stdout)
    assert m, f"no reference on stdout:\n{result.output}"
    return m.group(1)


def _archive_path(result) -> Path:
    """Default archive path, rebuilt from the reference (the panel may wrap it)."""
    return Path.cwd() / f"tt-cli-logs-{_ref(result)}.tar.gz"


def _eml_path(result) -> Path:
    return Path.cwd() / f"tt-cli-bug-report-{_ref(result)}.eml"


def test_bundle_default_path_and_manifest(runner, in_scratch_cwd):
    result = _run(runner)
    path = _archive_path(result)
    assert path.is_file()
    assert path.parent == in_scratch_cwd
    members = _members(path)
    assert {"environment.json", "env.txt", "manifest.json"} <= set(members)
    manifest = json.loads(members["manifest.json"])
    assert manifest["reference"] == _ref(result)
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
    ref = payload["reference"]
    assert re.fullmatch(r"ttbr-[0-9a-f]{12}", ref)
    assert payload["eml_path"] == str(tmp_path / "out" / f"tt-cli-bug-report-{ref}.eml")
    assert Path(payload["eml_path"]).is_file()
    assert payload["to"] == "support@tenstorrent.com"
    assert payload["subject"] == f"[TT-CLI] serve hangs [{ref}]"
    assert payload["assignee"]["email"].endswith("@tenstorrent.com")
    assert payload["mailto"].startswith("mailto:support@tenstorrent.com?subject=")
    # --json is for scripts: never launches anything.
    assert openers.files == [] and openers.mailtos == []

    quiet = _run(runner, "--output", str(tmp_path / "q.tar.gz"), "--quiet")
    assert quiet.output == ""
    assert (tmp_path / "q.tar.gz").is_file()
    assert len(list(tmp_path.glob("tt-cli-bug-report-ttbr-*.eml"))) == 1


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
    monkeypatch.setenv("FAKE_DOCKER_ARGV_LOG", str(argv_log))
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
    assert "fake docker logs aaaaaaaaaaaa" in log
    assert HF not in log and "HF_TOKEN=<redacted>" in log
    calls = [json.loads(line)["argv"] for line in argv_log.read_text().splitlines()]
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
    archive, eml, ref = _archive_path(result), _eml_path(result), _ref(result)
    assert eml.is_file()
    assert openers.files == [eml] and openers.mailtos == []

    msg = _eml(eml)
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

    out = result.stdout
    assert "🐞 Bug report ready" in out
    assert f"Bundle     →  ./{archive.name}" in out
    assert f"Email file →  ./{eml.name}" in out
    assert "Email      →  support@tenstorrent.com" in out
    assert "Assignee   →  " in out and "(this week's triage)" in out
    assert "1. Open the .eml above in your mail client. The bundle is attached." in out
    assert "on macOS: hit Forward" in out
    assert "No mail client opened" not in out
    assert "Opening the draft in your mail client" in result.output


def test_bundle_panel_shows_both_paths(runner, tmp_path, openers, monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")  # keep each panel row on one line
    target = tmp_path / "b.tar.gz"
    result = _run(runner, "--output", str(target), "--no-open")
    ref = _ref(result)
    assert f"Bundle     →  {target}" in result.stdout
    assert f"Email file →  {tmp_path / f'tt-cli-bug-report-{ref}.eml'}" in result.stdout


def test_bundle_no_open_writes_both_files_and_launches_nothing(runner, openers):
    result = _run(runner, "--no-open")
    assert _eml_path(result).is_file()
    assert openers.files == [] and openers.mailtos == []
    assert "Opening" not in result.output
    # --no-open is a choice, not a failure: no headless hint.
    assert "No mail client opened" not in result.output


def test_bundle_falls_back_to_mailto_when_no_eml_handler(runner, openers):
    openers.file_ok = False
    result = _run(runner, "--title", "x")
    assert len(openers.files) == 1
    (url,) = openers.mailtos
    assert url.startswith("mailto:support@tenstorrent.com?subject=%5BTT-CLI%5D%20x%20%5B")
    assert "Opened a mailto: draft instead" in result.stdout
    assert "No mail client opened" not in result.stdout


def test_bundle_mailto_flag_skips_the_eml_launch(runner, openers):
    _run(runner, "--mailto")
    assert openers.files == [] and len(openers.mailtos) == 1


def test_bundle_headless_says_to_copy_the_eml(runner, openers):
    """No desktop (an SSH session on a QB2): both openers fail."""
    openers.file_ok = False
    openers.mailto_ok = False
    result = _run(runner, "--title", "[brackets] stay literal")
    assert "🐞 Bug report ready" in result.stdout
    assert "No mail client opened. Copy the .eml to your own machine" in result.stdout
    assert "[TT-CLI] [brackets] stay literal [" in _eml(_eml_path(result))["Subject"]


def test_bundle_prompts_for_the_title_only_on_a_tty(runner, monkeypatch, openers):
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: True)
    result = runner.invoke(app, ["report", "bundle"], input="typed at the prompt\n")
    assert result.exit_code == 0, result.output
    assert "Subject for the support email [Bug report]:" in result.output
    assert "[TT-CLI] typed at the prompt [" in _eml(_eml_path(result))["Subject"]

    result = runner.invoke(app, ["report", "bundle"], input="\n")
    assert "[TT-CLI] Bug report [" in _eml(_eml_path(result))["Subject"]

    # --title given, or not a TTY: no prompt at all.
    result = runner.invoke(app, ["report", "bundle", "--title", "given"])
    assert "Subject for the support email" not in result.output
    monkeypatch.setattr("tenstorrent.commands.report._stdin_isatty", lambda: False)
    result = runner.invoke(app, ["report", "bundle"])
    assert "Subject for the support email" not in result.output
    assert "[TT-CLI] Bug report [" in _eml(_eml_path(result))["Subject"]


def test_bundle_warns_when_the_attachment_is_too_big(runner, monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.report_email.ATTACHMENT_WARN_BYTES", 10)
    result = _run(runner, "--no-open")
    assert "warning: the bundle is" in result.output and "25 MB" in result.output


def test_bundle_unwritable_eml_keeps_the_archive(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.commands.report_email.eml_path_for",
        lambda archive, ref: Path("/nonexistent-dir") / "x.eml",
    )
    target = tmp_path / "b.tar.gz"
    result = runner.invoke(app, ["report", "bundle", "--output", str(target)])
    assert result.exit_code == ExitCode.ERROR
    assert "Cannot write the support email" in result.output
    assert target.is_file()
