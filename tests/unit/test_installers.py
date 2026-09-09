# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import hashlib
import json

import pytest

from tenstorrent.config.paths import get_paths
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.installers import ScriptInstaller, UvToolInstaller
from tenstorrent.tools.manifest import ToolSpec
from tenstorrent.tools.runner import Runner


@pytest.fixture
def paths(isolated_dirs):
    return get_paths()


@pytest.fixture
def fake_uv(fake_bin, tmp_path, monkeypatch):
    log = tmp_path / "uv-argv.jsonl"
    monkeypatch.setenv("TT_UV_BIN", str(fake_bin / "uv"))
    monkeypatch.setenv("FAKE_UV_LOG", str(log))
    return log


SMI_SPEC = ToolSpec(
    name="tt-smi", kind="uv-tool", golden_version="5.3.0", package="tt-smi", python="3.10"
)


def test_uv_tool_installer_invokes_uv_exactly(paths, fake_uv):
    installer = UvToolInstaller(paths, Runner(sudo_command=""))
    result = installer.install(SMI_SPEC)
    argv = json.loads(fake_uv.read_text().splitlines()[0])
    assert argv == ["tool", "install", "tt-smi==5.3.0", "--force", "--python", "3.10"]
    assert result.path == paths.tool_bin_dir / "tt-smi"
    assert result.path.exists()
    assert result.version == "5.3.0"


GIT_SPEC = ToolSpec(
    name="tt-model",
    kind="uv-tool",
    golden_version="019abcb143d369450436c2778b9b40267d1e6903",
    package="tt-model",
    repo="https://github.com/tenstorrent/tt-model-manager",
    python="3.10",
)


def test_uv_tool_installer_installs_a_git_source(paths, fake_uv):
    """A uv-tool with a `repo` installs from a PEP 508 direct reference at the ref,
    for tools not yet on PyPI. The recorded version stays the ref, so bumping the
    pin still reinstalls through registry.ensure()."""
    installer = UvToolInstaller(paths, Runner(sudo_command=""))
    result = installer.install(GIT_SPEC)
    argv = json.loads(fake_uv.read_text().splitlines()[0])
    assert argv == [
        "tool",
        "install",
        "tt-model @ git+https://github.com/tenstorrent/tt-model-manager"
        "@019abcb143d369450436c2778b9b40267d1e6903",
        "--force",
        "--python",
        "3.10",
    ]
    assert result.path == paths.tool_bin_dir / "tt-model"
    assert result.path.exists()
    assert result.version == "019abcb143d369450436c2778b9b40267d1e6903"


def test_uv_tool_installer_offline_flag(paths, fake_uv):
    installer = UvToolInstaller(paths, Runner(sudo_command=""))
    installer.install(SMI_SPEC, offline=True)
    argv = json.loads(fake_uv.read_text().splitlines()[0])
    assert "--offline" in argv


def test_uv_tool_installer_failure_is_tool_failed(paths, fake_uv, monkeypatch):
    monkeypatch.setenv("FAKE_UV_FAIL", "1")
    installer = UvToolInstaller(paths, Runner(sudo_command=""))
    with pytest.raises(TTError) as exc:
        installer.install(SMI_SPEC)
    assert exc.value.exit_code == ExitCode.TOOL_FAILED


def _script_spec(blob: bytes, sha_ok: bool = True) -> ToolSpec:
    digest = hashlib.sha256(blob).hexdigest() if sha_ok else "0" * 64
    return ToolSpec(
        name="tt-installer",
        kind="script",
        golden_version="3.2.0",
        url="https://example.invalid/install.sh",
        sha256=digest,
    )


def test_script_installer_downloads_verifies_and_marks_executable(paths):
    blob = b"#!/bin/sh\necho install\n"
    installer = ScriptInstaller(paths, fetch_fn=lambda url: blob)
    result = installer.install(_script_spec(blob))
    assert result.path.read_bytes() == blob
    assert result.path.stat().st_mode & 0o100  # owner-executable


def test_script_installer_checksum_mismatch_refuses(paths):
    blob = b"#!/bin/sh\nevil\n"
    installer = ScriptInstaller(paths, fetch_fn=lambda url: blob)
    with pytest.raises(TTError) as exc:
        installer.install(_script_spec(blob, sha_ok=False))
    assert exc.value.exit_code == ExitCode.TOOL_FAILED
    assert "Checksum mismatch" in exc.value.what
    assert not installer.script_path(_script_spec(blob, sha_ok=False)).exists()


def test_script_installer_offline_without_cache(paths):
    installer = ScriptInstaller(paths, fetch_fn=lambda url: b"x")
    with pytest.raises(TTError) as exc:
        installer.install(_script_spec(b"x"), offline=True)
    assert exc.value.exit_code == ExitCode.OFFLINE
    assert "Pre-seed" in exc.value.next_step


def test_script_installer_uses_cached_copy(paths):
    calls = []

    def fetch(url):
        calls.append(url)
        return b"#!/bin/sh\n"

    installer = ScriptInstaller(paths, fetch_fn=fetch)
    spec = ToolSpec(
        name="tt-installer", kind="script", golden_version="3.2.0",
        url="https://example.invalid/install.sh",
    )
    installer.install(spec)
    installer.install(spec)  # second call: cache hit
    installer.install(spec, offline=True)  # offline works once cached
    assert len(calls) == 1
