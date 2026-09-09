# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Shared fixtures: isolated TT_* dirs, tool wiring, and a CliRunner.

Two run modes, selected on the pytest command line:

* **fake (default)** — every tool is a stand-in under ``tests/fakes/``, wired in by
  absolute path through the ``TT_TOOL_BIN_*`` / ``TT_UV_BIN`` seams. No test touches
  the real ~/.config, ~/.local/share, PATH tools, or the network.
* **hardware (``--hardware``)** — assumes the real Tenstorrent stack is installed on
  this machine. Tool fixtures point ``tt`` at the *real* binaries (tt-smi &c.) instead
  of the fakes. Config/data/cache/HF are still redirected to a temp dir so a test run
  never clobbers the user's real state — the real hardware is reached only through the
  tools themselves. See ``pytest_collection_modifyitems`` for how the markers gate which
  tests run in each mode.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tenstorrent.telemetry.env import CI_ENV_VARS

FAKES_DIR = Path(__file__).parent / "fakes"
FAKE_BIN = FAKES_DIR / "bin"

# Where tt-installer drops the real per-tool entry points on a managed machine.
INSTALLER_VENV_BIN = Path.home() / ".tenstorrent-venv" / "bin"


# -- command-line options / markers --------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run against the real Tenstorrent stack installed on this machine "
        "instead of the fakes in tests/fakes/.",
    )
    parser.addoption(
        "--run-destructive",
        action="store_true",
        default=False,
        help="With --hardware, also run tests that mutate the machine "
        "(e.g. a real `tt device reset`). No effect in the default fake mode.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    hardware = config.getoption("--hardware")
    run_destructive = config.getoption("--run-destructive")
    skip_hardware = pytest.mark.skip(reason="hardware-only test; pass --hardware to run")
    skip_fake = pytest.mark.skip(
        reason="asserts fake-tool internals; not meaningful under --hardware"
    )
    skip_destructive = pytest.mark.skip(
        reason="mutates real hardware; pass --run-destructive to run"
    )
    for item in items:
        if not hardware and "hardware" in item.keywords:
            item.add_marker(skip_hardware)
        if hardware and "fakes_only" in item.keywords:
            item.add_marker(skip_fake)
        # "destructive" is only dangerous when tools are real: gate it on --hardware.
        if hardware and "destructive" in item.keywords and not run_destructive:
            item.add_marker(skip_destructive)


# -- mode + real-tool discovery -------------------------------------------------------
@pytest.fixture
def hardware_mode(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--hardware"))


def _resolve_real_tool(binary: str) -> Path | None:
    """Where is the real ``binary`` on this machine? PATH first, then the
    tt-installer venv. None if it isn't installed."""
    found = shutil.which(binary)
    if found:
        return Path(found)
    candidate = INSTALLER_VENV_BIN / binary
    return candidate if candidate.exists() else None


@pytest.fixture(scope="session")
def real_tools() -> dict[str, Path | None]:
    """Real binary paths for the tools tests may drive on hardware, resolved once."""
    return {name: _resolve_real_tool(name) for name in ("tt-smi", "tt-flash", "uv")}


def _require_real(real_tools: dict[str, Path | None], name: str) -> Path:
    path = real_tools.get(name)
    if path is None:
        pytest.skip(f"real {name} not found on this machine (needed for --hardware)")
    return path


# -- mode-aware tool wiring -----------------------------------------------------------
# Each fixture returns the argv log path in fake mode (tests assert against it) and
# None in hardware mode, where the real tool is driven and there is nothing to record.
@pytest.fixture
def smi_bin(isolated_dirs, hardware_mode, real_tools, monkeypatch, tmp_path) -> Path | None:
    if hardware_mode:
        monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(_require_real(real_tools, "tt-smi")))
        return None
    log = tmp_path / "smi-argv.jsonl"
    monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(FAKE_BIN / "tt-smi"))
    monkeypatch.setenv("FAKE_SMI_LOG", str(log))
    return log


@pytest.fixture
def uv_bin(isolated_dirs, hardware_mode, real_tools, monkeypatch, tmp_path) -> Path | None:
    if hardware_mode:
        monkeypatch.setenv("TT_UV_BIN", str(_require_real(real_tools, "uv")))
        return None
    log = tmp_path / "uv-argv.jsonl"
    monkeypatch.setenv("TT_UV_BIN", str(FAKE_BIN / "uv"))
    monkeypatch.setenv("FAKE_UV_LOG", str(log))
    return log


@pytest.fixture
def installer_bin(isolated_dirs, hardware_mode, monkeypatch, tmp_path) -> Path | None:
    if hardware_mode:
        # There is no persistent real install.sh to point at; tests that need the
        # real installer are destructive and gated separately.
        pytest.skip("real tt-installer is fetched on demand; no binary to wire")
    log = tmp_path / "installer-argv.log"
    monkeypatch.setenv("TT_TOOL_BIN_TT_INSTALLER", str(FAKES_DIR / "install.sh"))
    monkeypatch.setenv("FAKE_INSTALLER_LOG", str(log))
    return log


@pytest.fixture
def inference_bin(isolated_dirs, hardware_mode, monkeypatch, tmp_path) -> Path | None:
    if hardware_mode:
        # The real tt-inference-server is a managed git checkout; serving is exercised
        # by the hardware smoke suite, not by the fake-argv tests that use this seam.
        pytest.skip("real tt-inference-server is a managed checkout; no fake to wire")
    log = tmp_path / "inference-argv.jsonl"
    monkeypatch.setenv(
        "TT_TOOL_BIN_TT_INFERENCE_SERVER", str(FAKES_DIR / "inference-repo" / "run.py")
    )
    monkeypatch.setenv("FAKE_INFERENCE_LOG", str(log))
    return log


@pytest.fixture
def model_manager_bin(isolated_dirs, hardware_mode, real_tools, monkeypatch, tmp_path) -> Path | None:
    if hardware_mode:
        monkeypatch.setenv("TT_TOOL_BIN_TT_MODEL", str(_require_real(real_tools, "tt-model")))
        return None
    log = tmp_path / "tt-model-argv.jsonl"
    monkeypatch.setenv("TT_TOOL_BIN_TT_MODEL", str(FAKE_BIN / "tt-model"))
    monkeypatch.setenv("FAKE_TT_MODEL_LOG", str(log))
    return log


# -- isolation + basics ---------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_dirs(request, tmp_path, monkeypatch):
    """Point every tt filesystem location at a per-test temp dir.

    This runs in both modes: even on hardware we never write to the user's real
    config/data/cache or HF cache. Real *tools* are reached through the tool
    fixtures above (which set TT_TOOL_BIN_*), not through these dirs.
    """
    for var, sub in (
        ("TT_CONFIG_DIR", "config"),
        ("TT_DATA_DIR", "data"),
        ("TT_CACHE_DIR", "cache"),
        ("HF_HOME", "hf"),  # keep hf_home_dir() off the real ~/.cache/huggingface
        # tt-model keeps its install index at $XDG_CACHE_HOME/tt-model, so without
        # this a test would read (and report on) the developer's own pulled bundles.
        ("XDG_CACHE_HOME", "xdg-cache"),
        # `tt launch` writes third-party client configs under $XDG_CONFIG_HOME, so
        # without this a test would edit the developer's own opencode.json.
        ("XDG_CONFIG_HOME", "xdg-config"),
    ):
        monkeypatch.setenv(var, str(tmp_path / sub))
    # Never inherit manifest/golden overrides from the outer shell.
    for var in ("TT_MANIFEST_PATH", "TT_GOLDEN_PATH"):
        monkeypatch.delenv(var, raising=False)
    # Golden versions come from tt-sw-manifest's golden.json, which `tt update`
    # fetches at the pinned tag — a network call the fake suite must never make.
    # Point TT_GOLDEN_PATH at a verbatim captured copy (v1.0.0) so versions are
    # known everywhere; tests exercising the fetch/cache/unknown paths delete it.
    # Under --hardware the override stays unset so the real fetch is exercised.
    if not request.config.getoption("--hardware"):
        monkeypatch.setenv("TT_GOLDEN_PATH", str(FAKES_DIR / "data" / "golden.json"))
        # Redirect HOME too: tt reports on paths install.sh hardcodes under
        # ~/.local/lib, so without this the suite would describe the developer's own
        # machine. Only in fake mode — under --hardware the real tools need the real
        # HOME to find their own state.
        home = tmp_path / "home"
        (home / ".local" / "lib").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
    # A no-op editor so `tt config` never blocks a test run.
    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setenv("VISUAL", "true")
    # Never inherit fake-tool pointers, scenario knobs, or telemetry config from the
    # outer shell; the tool fixtures set exactly the ones each test needs.
    for var in list(os.environ):
        if var.startswith(("TT_TOOL_BIN_", "FAKE_", "TT_TELEMETRY_", "TT_UPDATE_CHECK_")):
            monkeypatch.delenv(var, raising=False)
    # The background version check is a real request to PyPI (and a real detached
    # process). Off for the suite; the self-update tests re-enable it against a local
    # file and a patched spawn.
    monkeypatch.setenv("TT_NO_UPDATE_CHECK", "1")
    # Telemetry is off by default under test: the suite must never touch the network.
    # Tests that exercise telemetry delete this and inject an in-memory exporter.
    monkeypatch.setenv("TT_TELEMETRY_DISABLED", "1")
    # DO_NOT_TRACK opts telemetry out, and CI markers force synchronous delivery and
    # set tt.ci on the span. Both are read from the ambient environment, so leaving them
    # in place would make the telemetry tests pass locally and behave differently in
    # GitHub Actions (which sets CI=true). Strip them: the suite tests the code, not the
    # machine it runs on. Tests that care set them back explicitly.
    for var in ("DO_NOT_TRACK", *CI_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    # Exempt loopback from any ambient proxy. Telemetry delivery goes through httpx
    # (drain) and requests (the OTLP exporter), and BOTH honour HTTP_PROXY/ALL_PROXY
    # with no implicit localhost bypass — so on a box behind a proxy, every test that
    # asserts against a local collector fails with the collector simply never being
    # reached. That is not hypothetical: it is what broke
    # test_detached_drainer_delivers_through_a_real_process on the QuietBox while the
    # same test passed on a machine with no proxy configured.
    #
    # NO_PROXY rather than unsetting the proxy vars: honouring a proxy is correct
    # production behaviour (a lab box may only reach PostHog through one), and a test
    # marked `network` may genuinely need it for real outbound traffic. This narrows the
    # exemption to the addresses our fake collectors bind to.
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost,::1")
    # Point TT_UV_BIN at nothing by default. find_uv_bin() otherwise falls through to
    # the uv wheel that ships with the package, so a test that drives an install path
    # without requesting the `uv_bin` fixture silently runs a REAL `uv tool install`
    # against PyPI — it passes wherever the network and wheels cooperate and fails
    # somewhere else (a py3.14 container had no matching tt-umd wheel). A missing
    # binary makes that mistake a fast TOOL_MISSING instead. `uv_bin` overrides this
    # in both modes (fake uv, or the real one under --hardware).
    monkeypatch.setenv("TT_UV_BIN", str(tmp_path / "no-uv-bin-fixture-requested"))
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fakes_dir():
    return FAKES_DIR


@pytest.fixture
def fake_bin():
    return FAKE_BIN
