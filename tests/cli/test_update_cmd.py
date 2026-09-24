# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
import shutil
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.config.paths import get_paths
from tenstorrent.commands.update import FirmwareSemVer, is_any_fw_semver_higher
from tenstorrent.errors import ExitCode, TTError
from tenstorrent.models.device import DeviceSnapshot, SystemSnapshot
from tenstorrent.tools.state import ToolState


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("19.13.1", FirmwareSemVer(19, 13, 1)),
        ("v19.13.1.2", FirmwareSemVer(19, 13, 1, 2)),
        ("80.18.3.1", FirmwareSemVer(18, 3, 1)),
        ("N/A", None),
        (None, None),
    ],
)
def test_firmware_semver_parse(value, expected):
    assert FirmwareSemVer.parse(value) == expected


def test_is_any_fw_semver_higher():
    target = FirmwareSemVer.parse("19.13.1")
    assert is_any_fw_semver_higher(
        target,
        [FirmwareSemVer.parse("19.13.1.0"), FirmwareSemVer.parse("19.11.0.0")],
    )
    assert not is_any_fw_semver_higher(
        target,
        [FirmwareSemVer.parse("19.13.1.0"), FirmwareSemVer.parse("19.14.0.0")],
    )
    assert not is_any_fw_semver_higher(None, [FirmwareSemVer.parse("19.11.0.0")])


def snapshot_with_firmware(version: str) -> SystemSnapshot:
    return SystemSnapshot(
        devices=[DeviceSnapshot(index=0, firmware={"fw_bundle_version": version})]
    )


@pytest.fixture(autouse=True)
def firmware_snapshot(monkeypatch):
    """Keep update tests independent of a real/preinstalled tt-smi.

    The default is already at the golden firmware; confirmation-specific tests
    replace the list entry with an older snapshot.
    """
    snapshots = [snapshot_with_firmware("19.13.1.0")]

    class FakeDeviceBackend:
        def snapshot(self):
            return snapshots[0]

    monkeypatch.setattr(
        "tenstorrent.commands.update.get_device_backend",
        lambda appctx: FakeDeviceBackend(),
    )
    return snapshots


@pytest.fixture
def fake_uv(uv_bin):
    """Recorded-argv log for the fake uv (fake mode); real uv under --hardware."""
    return uv_bin


@pytest.fixture
def fake_installer(installer_bin):
    """Recorded-argv log for the fake install.sh (fake mode only)."""
    return installer_bin


def _json(result):
    """Parse tt's JSON document out of a CliRunner result.

    Warnings go to stderr, which CliRunner merges into `output`, so the document
    does not always start at char 0 — under --hardware the real HOME may hold the
    stale clones `tt update` warns about."""
    return json.loads(result.output[result.output.index("{"):])


def test_update_dry_run_plans_installs(runner):
    result = runner.invoke(app, ["update", "--dry-run", "--json"])
    assert result.exit_code == 0
    plan = _json(result)
    assert plan["dry_run"] is True
    by_name = {i["name"]: i for i in plan["items"]}
    assert by_name["tt-smi"]["action"] == "install"  # nothing installed yet
    assert by_name["tt-smi"]["target"]  # golden pin from .ttis
    assert by_name["system-stack"]["action"] == "converge"
    assert "kmd" in by_name["system-stack"]["target"]
    assert "firmware" in by_name["system-stack"]["target"]


def test_update_plan_lists_absent_lazy_tools_as_optional(runner):
    """A lazy tool (tt-inference-server) is listed so you can see it exists, but an
    absent one is not downloaded — `tt update` keeps it current once present and
    never pays for one you have not used. tt-model is a base package: it installs
    eagerly like tt-smi."""
    result = runner.invoke(app, ["update", "--dry-run", "--json"])
    assert result.exit_code == 0
    actions = {i["name"]: i["action"] for i in _json(result)["items"]}
    assert actions["tt-smi"] == "install"  # eager: nothing installed yet
    assert actions["tt-model"] == "install"
    assert actions["tt-inference-server"] == "optional"
    assert "tt-installer" not in actions  # it *is* the system-stack row


def test_update_include_lazy_plans_optional_tools(runner):
    result = runner.invoke(app, ["update", "--dry-run", "--include-lazy", "--json"])
    assert result.exit_code == 0
    actions = {i["name"]: i["action"] for i in _json(result)["items"]}
    assert actions["tt-inference-server"] == "install"


def test_update_installs_lazy_tools_only_with_the_flag(runner, fake_uv, fake_installer):
    """The plan's "optional" rows must not be acted on without --include-lazy."""
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json(result)["result"]
    assert "tt-inference-server" in payload["skipped"]
    assert "tt-inference-server" not in payload["updated"]


def test_update_keeps_going_when_one_tool_fails(runner, fake_uv, fake_installer, monkeypatch):
    """A tool that cannot be installed must not stop the system stack — that is the
    part a user most needs converged — but the command still exits non-zero."""
    from tenstorrent.tools.registry import ToolRegistry

    real_install = ToolRegistry.install

    def flaky(self, spec, *, offline=False):
        if spec.name == "tt-flash":
            raise TTError("git fetch failed", exit_code=ExitCode.TOOL_FAILED)
        return real_install(self, spec, offline=offline)

    monkeypatch.setattr(ToolRegistry, "install", flaky)
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    # the per-tool warning lands on stderr, which CliRunner merges into output
    payload = _json(result)["result"]
    assert [f["tool"] for f in payload["failed"]] == ["tt-flash"]
    assert payload["installer_ran"] is True  # the system stack still converged


def test_update_dry_run_human(runner):
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0
    assert "Update plan" in result.output
    assert "tt-smi" in result.output


@pytest.mark.fakes_only
def test_update_applies_uv_pins_and_runs_installer(
    runner, fake_uv, fake_installer, isolated_dirs
):
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    # uv re-pinned both tools at the golden versions, idempotently (--force)
    uv_calls = [json.loads(line) for line in fake_uv.read_text().splitlines()]
    installs = [c for c in uv_calls if c[:2] == ["tool", "install"]]
    specs = {c[2] for c in installs}
    assert any(s.startswith("tt-smi==") for s in specs)
    assert any(s.startswith("tt-flash==") for s in specs)
    assert all("--force" in c for c in installs)
    # install.sh ran non-interactively on its golden release channel (it fetches
    # the distro-correct .ttis itself; tt-installer has no --import-schema)
    installer_argv = fake_installer.read_text().strip()
    assert "--mode-non-interactive" in installer_argv
    assert "--versions=release" in installer_argv
    assert "--reboot-option=never" in installer_argv
    assert "--update-firmware=on" in installer_argv
    assert "Continue with tt update?" not in result.output


@pytest.mark.fakes_only
def test_update_older_firmware_declined_changes_nothing(
    runner, fake_uv, fake_installer, firmware_snapshot, monkeypatch
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    monkeypatch.setattr(
        "tenstorrent.commands.update._stdin_is_interactive", lambda: True
    )
    result = runner.invoke(app, ["update"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Firmware 19.13.1 is newer" in result.output
    assert "running AI model" in result.output
    assert "Continue with tt update?" in result.output
    assert not fake_uv.exists()
    assert not fake_installer.exists()


@pytest.mark.fakes_only
def test_update_older_firmware_accepts_before_applying(
    runner, fake_uv, fake_installer, firmware_snapshot, monkeypatch
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    monkeypatch.setattr(
        "tenstorrent.commands.update._stdin_is_interactive", lambda: True
    )
    result = runner.invoke(app, ["update"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Continue with tt update?" in result.output
    assert fake_uv.exists()
    assert fake_installer.exists()


@pytest.mark.fakes_only
def test_update_older_firmware_noninteractive_requires_yes(
    runner, fake_uv, fake_installer, firmware_snapshot
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == ExitCode.USAGE
    assert "Firmware update needs confirmation" in result.output
    assert "--yes" in result.output
    assert not fake_uv.exists()
    assert not fake_installer.exists()


@pytest.mark.fakes_only
def test_update_yes_confirms_older_firmware_without_prompt(
    runner, fake_uv, fake_installer, firmware_snapshot
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    result = runner.invoke(app, ["update", "--yes"])
    assert result.exit_code == 0, result.output
    assert "running AI model" in result.output
    assert "Continue with tt update?" not in result.output
    assert fake_uv.exists()
    assert fake_installer.exists()


@pytest.mark.fakes_only
def test_update_older_firmware_json_requires_yes(
    runner, fake_uv, fake_installer, firmware_snapshot
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == ExitCode.USAGE
    payload = _json(result)
    assert payload["error"]["what"] == "Firmware update needs confirmation."
    assert "--yes" in payload["error"]["next_step"]
    assert not fake_uv.exists()
    assert not fake_installer.exists()


@pytest.mark.fakes_only
def test_update_force_declined_changes_nothing(
    runner, fake_uv, fake_installer, monkeypatch
):
    monkeypatch.setattr(
        "tenstorrent.commands.update._stdin_is_interactive", lambda: True
    )
    result = runner.invoke(app, ["update", "--force"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "forced update will reflash firmware and reset TT devices" in result.output
    assert "Continue with tt update?" in result.output
    assert not fake_uv.exists()
    assert not fake_installer.exists()


@pytest.mark.fakes_only
def test_update_quiet_confirmation_still_explains_reset(
    runner, fake_uv, fake_installer, firmware_snapshot, monkeypatch
):
    firmware_snapshot[0] = snapshot_with_firmware("19.11.0.0")
    monkeypatch.setattr(
        "tenstorrent.commands.update._stdin_is_interactive", lambda: True
    )
    result = runner.invoke(app, ["update", "--quiet"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "will flash and reset" in result.output
    assert "running AI model workloads" in result.output
    assert not fake_uv.exists()
    assert not fake_installer.exists()


@pytest.mark.fakes_only
def test_update_second_run_is_up_to_date(runner, fake_uv, fake_installer):
    assert runner.invoke(app, ["update"]).exit_code == 0
    first_uv_calls = len(fake_uv.read_text().splitlines())
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0
    payload = _json(result)
    assert set(payload["result"]["up_to_date"]) == {"tt-smi", "tt-flash", "tt-model"}
    assert payload["result"]["updated"] == []
    assert len(fake_uv.read_text().splitlines()) == first_uv_calls  # no re-pin calls
    assert payload["result"]["installer_ran"] is True  # system converge is idempotent


@pytest.mark.fakes_only
def test_update_respects_overrides_as_external(runner, fake_uv, fake_installer, tmp_path):
    override = tmp_path / "my-smi"
    override.write_text("")
    assert (
        runner.invoke(
            app, ["config", "set", "tools.override.tt-smi", str(override)]
        ).exit_code
        == 0
    )
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0
    payload = _json(result)
    assert payload["result"]["external"] == ["tt-smi"]
    uv_calls = [json.loads(line) for line in fake_uv.read_text().splitlines()]
    assert not any("tt-smi" in " ".join(c) for c in uv_calls)


@pytest.mark.fakes_only
def test_update_pins_installer_python(runner, fake_uv, fake_installer):
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    installer_argv = fake_installer.read_text().strip()
    # install.sh only honors --python-version together with --use-uv (uv provisions
    # the interpreter); passing one without the other is silently dropped upstream.
    assert "--python-version=3.12" in installer_argv
    assert "--use-uv" in installer_argv


@pytest.mark.fakes_only
def test_update_runs_installer_outside_user_cwd(
    runner, fake_uv, fake_installer, isolated_dirs
):
    """The real install.sh leaks a wget-log into its CWD — keep that under TT_DATA_DIR."""
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    cwd_log = fake_installer.with_suffix(fake_installer.suffix + ".cwd")
    installer_cwd = Path(cwd_log.read_text().strip()).resolve()
    assert installer_cwd != Path.cwd().resolve()
    assert installer_cwd == (isolated_dirs / "data" / "installer-work").resolve()


def test_update_dry_run_reports_installer_python(runner):
    result = runner.invoke(app, ["update", "--dry-run", "--json"])
    assert result.exit_code == 0
    plan = _json(result)
    assert plan["installer_python"] == "3.12"
    system = next(i for i in plan["items"] if i["name"] == "system-stack")
    assert "python 3.12" in system["target"]


@pytest.mark.fakes_only
def test_update_installer_failure_is_tool_failed(runner, fake_uv, fake_installer, monkeypatch):
    monkeypatch.setenv("FAKE_INSTALLER_FAIL", "1")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == ExitCode.TOOL_FAILED


@pytest.mark.fakes_only
def test_update_force_passes_update_firmware_force(runner, fake_uv, fake_installer):
    result = runner.invoke(app, ["update", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    installer_argv = fake_installer.read_text().strip()
    assert "--update-firmware=force" in installer_argv
    assert "--versions=release" in installer_argv  # still converges on the golden stack


@pytest.mark.fakes_only
def test_update_without_force_uses_upgrade_only_firmware_mode(
    runner, fake_uv, fake_installer
):
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    installer_argv = fake_installer.read_text()
    assert "--update-firmware=on" in installer_argv
    assert "--update-firmware=force" not in installer_argv


def test_update_dry_run_shows_version_and_force(runner):
    result = runner.invoke(app, ["update", "3.1.0", "--dry-run", "--json"])
    assert result.exit_code == 0
    plan = _json(result)
    # a specific installer version implies force (user knows what they're doing)
    assert plan["installer_version"] == "3.1.0"
    assert plan["force"] is True
    system = next(i for i in plan["items"] if i["name"] == "system-stack")
    assert "install.sh 3.1.0" in system["target"]
    assert "force" in system["action"]
    # --python-version postdates some installer tags, so a user-requested release
    # runs without the pin rather than risking an unrecognized-option exit.
    assert plan["installer_python"] is None
    assert "python" not in system["target"]


@pytest.mark.fakes_only
def test_update_specific_version_fetches_unpinned_and_forces(
    runner, fake_uv, tmp_path, monkeypatch
):
    # A user-requested (non-golden) install.sh is fetched from the templated URL,
    # unpinned, and run with --update-firmware=force implied.
    log = tmp_path / "installer-argv.log"
    monkeypatch.setenv("FAKE_INSTALLER_LOG", str(log))
    fetched = {}

    def fake_fetch(url):
        fetched["url"] = url
        return (
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$FAKE_INSTALLER_LOG"\n'
            "exit 0\n"
        ).encode()

    monkeypatch.setattr(
        "tenstorrent.tools.installers.ScriptInstaller._fetch_https",
        staticmethod(fake_fetch),
    )
    result = runner.invoke(app, ["update", "v3.1.0", "--yes"])
    assert result.exit_code == 0, result.output
    assert fetched["url"].endswith("/v3.1.0/install.sh")  # leading "v" normalized
    assert "unpinned" in result.output  # warned we can't checksum-verify it
    argv = log.read_text()
    assert "--update-firmware=force" in argv
    assert "--python-version" not in argv  # flag may not exist in an older release


@pytest.mark.fakes_only
def test_update_offline_without_goldens_is_offline_error(runner, fake_uv, monkeypatch):
    # No TT_GOLDEN_PATH override and no cache: the golden pins are unknowable
    # offline, so update refuses up front instead of half-converging.
    monkeypatch.delenv("TT_GOLDEN_PATH")
    result = runner.invoke(app, ["--offline", "update"])
    assert result.exit_code == ExitCode.OFFLINE


@pytest.mark.fakes_only
def test_update_fetches_goldens_when_not_cached(runner, fake_uv, monkeypatch, fakes_dir):
    # Without the override, update fetches golden.json at the pinned tag once,
    # then works from the cache (second call: no fetch). The fixture is the
    # verbatim released asset, so the supplement's recorded sha256 matches it.
    monkeypatch.delenv("TT_GOLDEN_PATH")
    blob = (fakes_dir / "data" / "golden.json").read_bytes()
    fetches = []

    def fake_fetch(url):
        fetches.append(url)
        return blob

    monkeypatch.setattr("tenstorrent.backends.installer.fetch_https", fake_fetch)
    assert runner.invoke(app, ["update", "--dry-run"]).exit_code == 0
    assert len(fetches) == 1 and fetches[0].endswith("/golden.json")
    assert runner.invoke(app, ["update", "--dry-run"]).exit_code == 0
    assert len(fetches) == 1  # cache hit, no re-fetch


@pytest.mark.fakes_only
def test_update_offline_skips_system_stack(runner, fake_uv):
    # offline: uv pins still converge from local caches, but the system stack is
    # skipped — tt-installer needs the network for goldens and distro packages.
    result = runner.invoke(app, ["--offline", "update", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["result"]["installer_ran"] is False
    uv_calls = [json.loads(line) for line in fake_uv.read_text().splitlines()]
    installs = [c for c in uv_calls if c[:2] == ["tool", "install"]]
    assert installs and all("--offline" in c for c in installs)


def test_update_flags_stale_installer_clones(runner, tmp_path, monkeypatch):
    """install.sh used to clone these into ~/.local/lib and never update them
    again; tt does not delete what it did not install, so it says where they are."""
    home = tmp_path / "fake-home"
    (home / ".local" / "lib" / "tt-studio").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0
    output = " ".join(result.output.split())
    assert "~/.local/lib/tt-studio" in output
    assert "rm -rf ~/.local/lib/tt-studio" in output  # the cleanup command


def test_update_says_nothing_when_there_are_no_stale_clones(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0
    assert ".local/lib" not in result.output


def test_update_passes_no_install_flags_for_the_bundled_apps(
    runner, fake_uv, fake_installer
):
    """tt owns tt-inference-server and tt-studio at pinned versions, so install.sh
    must not clone its own unpinned copies (June's call, 2026-09-02)."""
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    argv = fake_installer.read_text()
    assert "--no-install-inference-server" in argv
    assert "--no-install-studio" in argv


@pytest.mark.fakes_only
def test_update_reinstalls_a_tool_whose_files_were_deleted(
    runner, fake_uv, fake_installer, monkeypatch
):
    """installed.toml can outlive the install it records.

    plan() used to trust the state file alone, so a tool whose directory had been
    deleted was reported "up-to-date" and apply() skipped it — leaving `tt update`
    claiming convergence for a tool that is missing. The registry already treats a
    recorded-but-absent path as not installed (_resolve_or_none); plan() must agree.
    """
    assert runner.invoke(app, ["update"]).exit_code == 0

    recorded = ToolState(get_paths()).get("tt-flash")
    assert recorded is not None and recorded.path.exists()
    shutil.rmtree(recorded.path.parent, ignore_errors=True)
    recorded.path.unlink(missing_ok=True)
    assert not recorded.path.exists()

    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0
    payload = _json(result)
    assert "tt-flash" in payload["result"]["updated"]
    assert "tt-flash" not in payload["result"]["up_to_date"]


@pytest.mark.parametrize("version", ["2.2.0", "v1.4.0", "0.9.0"])
def test_update_refuses_installer_releases_below_the_flag_floor(runner, version):
    """tt passes --versions unconditionally, which only exists from install.sh 3.0.0.

    argbash exits non-zero on an unrecognized option, so an older tag would die with
    a raw usage dump partway through a sudo'd run. Fail before fetching anything.
    """
    result = runner.invoke(app, ["update", version])
    assert result.exit_code == ExitCode.USAGE
    assert "--versions" in result.output


def test_update_accepts_a_release_at_the_floor(runner, fake_uv, fake_installer):
    """3.0.0 is old but drivable: --versions, --no-install-{inference-server,studio}
    all exist there, so it must not be caught by the floor check."""
    result = runner.invoke(app, ["update", "3.0.0", "--dry-run"])
    assert result.exit_code == 0


# -- phase structure, timings, and reporting honesty ---------------------------
def test_the_phase_list_is_fixed_at_three():
    """A drifting denominator makes `k/N` worthless, so pin the roadmap."""
    from tenstorrent.commands.update import PHASES

    assert PHASES == ["Checks", "Tools", "System"]


def test_json_carries_timings_for_every_phase(runner, fake_uv, fake_installer):
    from tenstorrent.commands.update import PHASES

    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0, result.output
    timings = _json(result)["timings"]
    assert [p["title"] for p in timings["phases"]] == PHASES
    assert timings["total_seconds"] >= 0
    assert timings["steps"], "no steps were recorded"
    assert all("seconds" in step for step in timings["steps"])


def test_offline_skips_the_system_phase_without_dropping_it(runner, fake_uv):
    """--offline must not change the phase count; it marks one skipped and says why."""
    from tenstorrent.commands.update import PHASES

    result = runner.invoke(app, ["update", "--offline", "--json"])
    phases = {p["title"]: p for p in _json(result)["timings"]["phases"]}
    assert set(phases) == set(PHASES)
    assert phases["System"]["status"] == "skipped"
    assert _json(result)["result"]["installer_ran"] is False


def test_offline_explains_the_skip_in_human_mode(runner, fake_uv):
    result = runner.invoke(app, ["update", "--offline"])
    assert "System skipped" in result.output
    assert "without --offline" in result.output


def test_dry_run_never_enters_the_phase_flow(runner, fake_uv, fake_installer):
    """A dry run does no work, so it reports and returns like a utility flag."""
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Update plan" in result.output
    assert "Phase 1/3" not in result.output


def test_a_failing_tool_is_explained_and_not_silently_dropped(
    runner, fake_uv, fake_installer, monkeypatch
):
    """Regression: the warning sat after a `continue` inside `with ui.step(...)`,
    so the context manager exited and the explanation was never printed — the tool
    showed as failed in the summary with no reason given anywhere."""
    from tenstorrent.tools.registry import ToolRegistry

    real_install = ToolRegistry.install

    def flaky(self, spec, *, offline=False):
        if spec.name == "tt-flash":
            raise TTError("git fetch failed", exit_code=ExitCode.TOOL_FAILED)
        return real_install(self, spec, offline=offline)

    monkeypatch.setattr(ToolRegistry, "install", flaky)
    result = runner.invoke(app, ["update"])
    assert "tt-flash was not updated" in result.output
    assert "git fetch failed" in result.output


def test_a_failing_tool_marks_the_phase_failed(
    runner, fake_uv, fake_installer, monkeypatch
):
    """The command exits non-zero, so an all-green stepper would be dishonest."""
    from tenstorrent.tools.registry import ToolRegistry

    real_install = ToolRegistry.install

    def flaky(self, spec, *, offline=False):
        if spec.name == "tt-flash":
            raise TTError("git fetch failed", exit_code=ExitCode.TOOL_FAILED)
        return real_install(self, spec, offline=offline)

    monkeypatch.setattr(ToolRegistry, "install", flaky)
    result = runner.invoke(app, ["update", "--json"])
    phases = {p["title"]: p for p in _json(result)["timings"]["phases"]}
    assert phases["Tools"]["status"] == "failed"
    # …and the run still converged the system stack.
    assert _json(result)["result"]["installer_ran"] is True


def test_plan_summary_reads_as_a_gist():
    from tenstorrent.commands.update import _plan_summary

    class Item:
        def __init__(self, action):
            self.action = action

    class Plan:
        def __init__(self, actions):
            self.items = [Item(a) for a in actions]

    assert _plan_summary(Plan(["upgrade", "install", "up-to-date"])) == (
        "2 to change, 1 up to date"
    )
    assert _plan_summary(Plan(["up-to-date"])) == "1 up to date"
    assert _plan_summary(Plan([])) == "nothing to do"
