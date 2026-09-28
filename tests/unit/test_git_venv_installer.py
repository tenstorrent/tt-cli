# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

from pathlib import Path

import pytest

from tenstorrent.config.paths import get_paths
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.tools.installers import GitVenvInstaller
from tenstorrent.tools.manifest import ToolSpec
from tenstorrent.tools.runner import CaptureResult, StreamResult

SPEC = ToolSpec(
    name="tt-inference-server",
    kind="git-venv",
    golden_version="v0.6.0",
    repo="https://github.com/tenstorrent/tt-inference-server",
    entry="run.py",
    python="3.10",
)


class StubRunner:
    """Records argv; emulates git clone / uv venv side effects.

    The installer streams git and `uv pip` (so they can show progress) and still
    captures `uv venv`, so the stub models both entry points and records them in
    one ordered list.
    """

    def __init__(self):
        self.calls = []

    def _side_effects(self, argv):
        self.calls.append(list(argv))
        if argv[0] == "git" and argv[1] == "clone":
            dest = Path(argv[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "run.py").write_text("print('hi')\n")
            (dest / "requirements.txt").write_text("fastapi\n")
        if argv[1:2] == ["venv"]:
            (Path(argv[2]) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(argv[2]) / "bin" / "python").write_text("")

    def capture(self, argv, *, env=None, timeout=None, check=True, tool=None):
        self._side_effects(argv)
        return CaptureResult(0, "", "")

    def stream_parsed(self, argv, *, on_line=None, **kwargs):
        self._side_effects(argv)
        return StreamResult(0, "", None, False)


@pytest.fixture
def paths(isolated_dirs):
    return get_paths()


def test_git_venv_installer_clones_at_pinned_ref_and_builds_venv(paths):
    runner = StubRunner()
    installer = GitVenvInstaller(paths, runner, uv_bin="uv-stub")
    result = installer.install(SPEC)
    git_call = runner.calls[0]
    # --progress because git stays quiet when its output is a pipe, and a pipe is
    # exactly what we now read to drive the activity row.
    assert git_call[:7] == [
        "git", "clone", "--progress", "--depth", "1", "--branch", "v0.6.0",
    ]
    assert git_call[7] == SPEC.repo
    venv_call = runner.calls[1]
    assert venv_call[:2] == ["uv-stub", "venv"]
    assert "--python" in venv_call and "3.10" in venv_call
    pip_call = runner.calls[2]
    assert pip_call[:3] == ["uv-stub", "pip", "install"]
    assert result.path.name == "run.py"
    assert result.path.exists()
    assert result.version == "v0.6.0"


def test_git_venv_installer_installs_manifest_deps(paths):
    # repos without a root requirements.txt (tt-inference-server) declare their
    # entry script's bootstrap deps in the manifest instead.
    spec = ToolSpec(
        name="tt-inference-server",
        kind="git-venv",
        golden_version="v0.18.0",
        repo=SPEC.repo,
        entry="run.py",
        python="3.10",
        deps=("pyyaml", "packaging"),
    )
    runner = StubRunner()
    GitVenvInstaller(paths, runner, uv_bin="uv-stub").install(spec)
    dep_calls = [c for c in runner.calls if c[:3] == ["uv-stub", "pip", "install"]]
    assert any(c[-2:] == ["pyyaml", "packaging"] for c in dep_calls)


def test_git_venv_installer_reuses_existing_checkout(paths):
    runner = StubRunner()
    installer = GitVenvInstaller(paths, runner, uv_bin="uv-stub")
    installer.install(SPEC)
    calls_after_first = len(runner.calls)
    installer.install(SPEC)  # cache hit: same ref already present
    assert len(runner.calls) == calls_after_first
    # and offline works once seeded
    installer.install(SPEC, offline=True)
    assert len(runner.calls) == calls_after_first


def test_git_venv_installer_offline_without_checkout(paths):
    installer = GitVenvInstaller(paths, StubRunner(), uv_bin="uv-stub")
    with pytest.raises(TTError) as exc:
        installer.install(SPEC, offline=True)
    assert exc.value.exit_code == ExitCode.OFFLINE
    assert "Pre-seed" in exc.value.next_step
