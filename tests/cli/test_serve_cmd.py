# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


SMALL_SUPPORT = (
    Path(__file__).parent.parent / "fakes" / "data" / "model_support_small.json"
)


@pytest.fixture(autouse=True)
def small_spec(monkeypatch):
    """Pin the catalog to the fixture: these tests assert on the exact flags a
    model's support entry produces."""
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))


@pytest.fixture(autouse=True)
def empty_hf_cache(monkeypatch):
    monkeypatch.setattr("tenstorrent.modelhub.catalog.scan_hf_cache", lambda: {})


@pytest.fixture
def docker_present(monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )


@pytest.fixture
def fake_server(inference_bin):
    """Recorded-argv log for the fake run.py. Only wired in fake mode; under
    --hardware the shared inference_bin fixture skips (serving is covered by the
    hardware smoke suite, not by these argv assertions)."""
    return inference_bin


@pytest.mark.fakes_only
def test_serve_streams_run_py(runner, docker_present, fake_server, isolated_dirs):
    # without tt-smi available, auto-detect degrades to a warning and no --device
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    assert "auto-detect skipped" in result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    # --model takes the tt-inference-server model id, not the HF repo; the
    # server workflow must pick a backend mode and containers are ours; the
    # host HF cache is always mounted so pulled weights are reused; and with no
    # --port or SERVICE_PORT, tt serve's own default port is passed explicitly
    assert argv == [
        "--model", "Llama-3.1-8B-Instruct", "--workflow", "server", "--docker-server",
        "--no-auth", "--host-hf-cache", str(isolated_dirs / "hf"),
        "--service-port", "20000",
    ]


@pytest.mark.fakes_only
def test_serve_host_hf_cache_honors_config(runner, docker_present, fake_server):
    assert (
        runner.invoke(
            app, ["config", "set", "paths.hf_model_cache_directory", "/data/hf"]
        ).exit_code
        == 0
    )
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--host-hf-cache") + 1] == "/data/hf"


@pytest.mark.fakes_only
def test_serve_runs_with_repo_root_as_cwd(
    runner, docker_present, fake_server, fakes_dir, tmp_path, monkeypatch
):
    # run.py reads Path("VERSION") etc. relative to CWD; the real repo root is
    # the entry script's directory (the official wrapper script cd's there too).
    cwd_log = tmp_path / "inference-cwd.log"
    monkeypatch.setenv("FAKE_INFERENCE_CWD_LOG", str(cwd_log))
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    recorded = cwd_log.read_text().strip()
    assert recorded == str((fakes_dir / "inference-repo").resolve())


@pytest.mark.fakes_only
def test_serve_autodetects_device_from_tt_smi(
    runner, docker_present, fake_server, fake_bin, monkeypatch
):
    # run.py's own detection is broken on fresh checkouts (v0.18.0 bootstrap
    # ordering bug), so serve derives --device from our tt-smi snapshot.
    monkeypatch.setenv("TT_TOOL_BIN_TT_SMI", str(fake_bin / "tt-smi"))
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "multi")  # 2x P300 QuietBox capture
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    assert "p300x2" in result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--device") + 1] == "p300x2"


@pytest.mark.fakes_only
def test_serve_benchmarks_workflow(runner, docker_present, fake_server):
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--workflow", "benchmarks"])
    assert result.exit_code == 0
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--workflow") + 1] == "benchmarks"
    assert "--docker-server" not in argv  # only the server workflow needs it


@pytest.mark.fakes_only
def test_serve_device_override_passthrough(runner, docker_present, fake_server):
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "p300x2"])
    assert result.exit_code == 0
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--device") + 1] == "p300x2"


@pytest.mark.fakes_only
def test_serve_drives_media_models_too(runner, docker_present, fake_server):
    """run.py builds its --model choices from every MODEL_SPECS entry, and
    run_docker_server has media/forge branches, so a media model is served the
    same way a vLLM one is. whisper-large-v3 is media-only in the bundled spec."""
    result = runner.invoke(app, ["serve", "whisper-large-v3"])
    assert result.exit_code == 0
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--model") + 1] == "whisper-large-v3"


@pytest.mark.fakes_only
def test_serve_warns_when_model_not_cached(runner, docker_present, fake_server):
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert "not in the local model cache" in result.output


@pytest.mark.fakes_only
@pytest.mark.parametrize(
    "model",
    [
        "resnet-50",  # forge CNN: weights ship in the container image
        "whisper-large-v3",  # STT: the container fetches its own weights
        "speecht5_tts",  # TTS: same
    ],
)
def test_serve_does_not_warn_when_the_host_cache_is_not_used(
    runner, docker_present, fake_server, model
):
    """A cache-miss warning has to be actionable. These models never read the
    host HF cache, so `tt model pull` would not change anything."""
    result = runner.invoke(app, ["serve", model])
    assert result.exit_code == 0
    assert "not in the local model cache" not in result.output


def test_serve_without_container_runtime(runner, fake_server, monkeypatch):
    monkeypatch.setattr("tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None)
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "container runtime" in result.output
    assert not fake_server.exists()  # preflight stopped before launching anything


@pytest.mark.fakes_only
def test_serve_tool_failure_surfaces(runner, docker_present, fake_server, monkeypatch):
    monkeypatch.setenv("FAKE_INFERENCE_FAIL", "1")
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == ExitCode.TOOL_FAILED


def test_serve_unknown_model(runner, docker_present):
    result = runner.invoke(app, ["serve", "gpt-17"])
    assert result.exit_code == ExitCode.USAGE
    assert "tt model list" in result.output


def test_serve_not_installed_clone_failure_surfaces(
    runner, docker_present, tmp_path, monkeypatch
):
    # no env override and nothing installed: ensure() clones the pinned ref.
    # Point the supplement at a local nonexistent repo so the failure is hermetic.
    supplement = tmp_path / "supplement.toml"
    supplement.write_text(
        'schema_version = 1\n'
        '[tools.tt-inference-server]\n'
        'kind = "git-venv"\n'
        'golden_version = "v0.0.1"\n'
        f'repo = "file://{tmp_path}/does-not-exist"\n'
        'entry = "run.py"\n'
    )
    monkeypatch.setenv("TT_MANIFEST_PATH", str(supplement))
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "git" in result.output


@pytest.fixture
def fake_model_manager(model_manager_bin):
    """Recorded-argv log for the fake tt-model binary (fake mode)."""
    return model_manager_bin


@pytest.mark.fakes_only
def test_serve_falls_back_to_tt_model_for_a_bundle_id(
    runner, fake_model_manager, isolated_dirs
):
    """A Hub-style id the released spec does not know is served by tt-model, with
    our HF cache root passed through so both paths share one copy of the weights."""
    result = runner.invoke(app, ["serve", "raahemnabeel/qwen3-coder-30b-a3b"])
    assert result.exit_code == 0, result.output
    assert "not in the model catalog" in result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "raahemnabeel/qwen3-coder-30b-a3b"]
    assert record["hf_home"]  # HF_HOME exported for the child


@pytest.mark.fakes_only
def test_serve_prefers_the_inference_server_on_a_catalog_match(
    runner, docker_present, fake_server, fake_model_manager, isolated_dirs
):
    """A catalog name wins even though tt-model is installed: the spec is checked
    first, and only an unknown name falls through."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    assert fake_server.read_text().strip()  # run.py ran
    assert not fake_model_manager.exists()  # tt-model did not


@pytest.mark.fakes_only
def test_serve_offline_makes_tt_model_use_the_local_bundle(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["serve", "ns/bundle", "--offline"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--local-only"]


def test_serve_unknown_name_that_is_not_a_bundle_id_is_usage(runner, isolated_dirs):
    """A catalog typo must keep the catalog's error rather than becoming a Hub 404."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instrukt"])
    assert result.exit_code == ExitCode.USAGE
    assert "Unknown model" in result.output
    assert "tt model list" in result.output


@pytest.mark.fakes_only
def test_serve_tt_model_rejects_inference_server_workflows(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["serve", "ns/bundle", "--workflow", "benchmarks"])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "benchmarks" in result.output


@pytest.mark.fakes_only
def test_serve_tt_model_warns_that_device_is_not_its_flag(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["serve", "ns/bundle", "--device", "p300x2"])
    assert result.exit_code == 0, result.output
    assert "--device is a tt-inference-server option" in result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert "--device" not in record["argv"]


@pytest.mark.fakes_only
def test_serve_passes_unknown_flags_through_to_tt_model(
    runner, fake_model_manager, isolated_dirs
):
    """tt-model's own flags (--port, --follow) and its vLLM passthrough reach it
    verbatim, after the bundle id where it expects them."""
    result = runner.invoke(
        app, ["serve", "ns/bundle", "--", "--port", "8080", "--follow"]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--port", "8080", "--follow"]


@pytest.mark.fakes_only
def test_serve_passthrough_works_without_the_separator(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["serve", "ns/bundle", "--port", "8080"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--port", "8080"]


@pytest.mark.fakes_only
def test_serve_offline_keeps_local_only_before_the_passthrough(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["serve", "ns/bundle", "--offline", "--", "--port", "9"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--local-only", "--port", "9"]


def test_serve_rejects_passthrough_for_a_catalog_model(runner, isolated_dirs):
    """tt-inference-server takes its options from tt serve itself, so extra args
    there are a mistake rather than a passthrough."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--", "--port", "8080"])
    assert result.exit_code == ExitCode.USAGE
    assert "Unrecognized arguments" in result.output


@pytest.mark.fakes_only
def test_serve_port_becomes_service_port_for_a_catalog_model(
    runner, docker_present, fake_server, isolated_dirs
):
    """run.py takes the host port as --service-port (it publishes
    <bind_host>:<service_port>:8000 on the docker path)."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--port", "8080"])
    assert result.exit_code == 0, result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--service-port") + 1] == "8080"


@pytest.mark.fakes_only
def test_serve_port_is_tt_models_own_flag_for_a_bundle(
    runner, fake_model_manager, isolated_dirs
):
    """The same tt flag on the other path: tt-model declares --port itself."""
    result = runner.invoke(app, ["serve", "ns/bundle", "--port", "8080"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--port", "8080"]


@pytest.mark.fakes_only
def test_serve_passthrough_port_wins_over_the_tt_flag(
    runner, fake_model_manager, isolated_dirs
):
    """Ours is emitted first so an explicit passthrough --port still wins
    (tt-model's launch command is last-wins)."""
    result = runner.invoke(
        app, ["serve", "ns/bundle", "--port", "8080", "--", "--port", "9999"]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["serve", "ns/bundle", "--port", "8080", "--port", "9999"]


def test_serve_rejects_an_out_of_range_port(runner, isolated_dirs):
    result = runner.invoke(app, ["serve", "ns/bundle", "--port", "70000"])
    assert result.exit_code == ExitCode.USAGE


# -- launch settings forwarded from the model support list -----------------------------
def _argv(fake_server):
    return json.loads(fake_server.read_text().splitlines()[-1])


@pytest.mark.fakes_only
def test_serve_enables_tool_calling_from_the_spec_parsers(
    runner, docker_present, fake_server
):
    """Nothing in tt-inference-server reads tool_call_parser_name, so a container
    started without these flags answers `tool_choice: "auto"` with nothing."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "n150"])
    assert result.exit_code == 0, result.output
    argv = _argv(fake_server)
    overrides = json.loads(argv[argv.index("--vllm-override-args") + 1])
    assert overrides == {
        "enable-auto-tool-choice": True,
        "tool-call-parser": "llama3_json",
    }


@pytest.mark.fakes_only
def test_serve_adds_the_reasoning_parser_when_the_spec_has_one(
    runner, docker_present, fake_server
):
    result = runner.invoke(app, ["serve", "Qwen3-32B", "--device", "galaxy"])
    assert result.exit_code == 0, result.output
    argv = _argv(fake_server)
    overrides = json.loads(argv[argv.index("--vllm-override-args") + 1])
    assert overrides["reasoning-parser"] == "qwen3"
    assert overrides["tool-call-parser"] == "hermes"


@pytest.mark.fakes_only
def test_serve_sends_no_vllm_overrides_for_a_media_model(
    runner, docker_present, fake_server
):
    """run_docker_server only reads --vllm-override-args on the vLLM branch, and a
    media entry has no parser to send anyway."""
    result = runner.invoke(app, ["serve", "whisper-large-v3", "--device", "n150"])
    assert result.exit_code == 0, result.output
    assert "--vllm-override-args" not in _argv(fake_server)


@pytest.mark.fakes_only
def test_serve_forwards_a_forced_docker_image(runner, docker_present, fake_server):
    """The spec pins an image tt cannot drive on this board; the support list
    records the replacement and serve has to pass it."""
    result = runner.invoke(app, ["serve", "whisper-large-v3", "--device", "n150"])
    assert result.exit_code == 0, result.output
    argv = _argv(fake_server)
    assert argv[argv.index("--override-docker-image") + 1] == "ghcr.io/example/media:0.17.0"


@pytest.mark.fakes_only
def test_serve_never_passes_back_the_specs_own_tt_config(
    runner, docker_present, fake_server
):
    """DeviceModelSpec.__post_init__ already folds override_tt_config into
    vllm_args, so tt passes the flag only where it is correcting one."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "n150"])
    assert result.exit_code == 0, result.output
    assert "--override-tt-config" not in _argv(fake_server)


@pytest.mark.fakes_only
def test_serve_sends_the_device_name_the_spec_actually_has(
    runner, docker_present, fake_server
):
    """Llama has no p150x4 spec in the fixture; it is reached through the p300x2
    one, and asking for p150x4 would 404 on the server's spec lookup."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "p150x4"])
    assert result.exit_code == 0, result.output
    assert "serving it through the p300x2 one" in result.output
    argv = _argv(fake_server)
    assert argv[argv.index("--device") + 1] == "p300x2"


def test_serve_refuses_a_board_the_model_is_known_to_fail_on(
    runner, docker_present, fake_server
):
    result = runner.invoke(app, ["serve", "whisper-large-v3", "--device", "p300x2"])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "does not run on p300x2" in result.output
    assert "healthy state" in result.output  # the recorded reason, not just a refusal
    assert not fake_server.exists()


@pytest.mark.fakes_only
def test_serve_force_overrides_a_known_failure(runner, docker_present, fake_server):
    result = runner.invoke(
        app, ["serve", "whisper-large-v3", "--device", "p300x2", "--force"]
    )
    assert result.exit_code == 0, result.output
    assert _argv(fake_server)[0] == "--model"  # it reached run.py


# -- --dry-run -------------------------------------------------------------------------
def test_serve_dry_run_reports_the_resolved_configuration(runner, fake_server):
    result = runner.invoke(
        app, ["serve", "Llama-3.1-8B-Instruct", "--device", "n150", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["device_sent"] == "n150"
    assert plan["tool_call_parser"] == "llama3_json"
    assert "--vllm-override-args" in plan["argv"]
    assert not fake_server.exists()  # nothing was run


def test_serve_dry_run_needs_no_container_runtime(runner, fake_server, monkeypatch):
    """A dry run describes what would happen; requiring docker to do that would
    make it useless on exactly the machines where you want to check first."""
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--dry-run"])
    assert result.exit_code == 0, result.output


def test_serve_dry_run_shows_the_substituted_device(runner, fake_server):
    result = runner.invoke(
        app, ["serve", "Llama-3.1-8B-Instruct", "--device", "p150x4", "--dry-run", "--json"]
    )
    plan = json.loads(result.stdout)
    assert plan["device_requested"] == "p150x4"
    assert plan["device_sent"] == "p300x2"
    assert plan["served_through"] == "p300x2"


def test_serve_dry_run_distinguishes_a_forced_setting_from_the_specs_own(
    runner, fake_server
):
    """The spec's tt config is applied by the server itself; only a forced one is
    passed as a flag. The dry run has to show which is which or it misleads."""
    forced = json.loads(
        runner.invoke(
            app, ["serve", "whisper-large-v3", "--device", "n150", "--dry-run", "--json"]
        ).stdout
    )
    assert forced["docker_image"] == "ghcr.io/example/media:0.17.0"
    assert forced["spec_docker_image"] == "ghcr.io/example/media:0.10.0"
    assert forced["forced_tt_config"] is None


def test_serve_dry_run_still_refuses_a_known_bad_board(runner, fake_server):
    result = runner.invoke(
        app, ["serve", "whisper-large-v3", "--device", "p300x2", "--dry-run"]
    )
    assert result.exit_code == ExitCode.UNSUPPORTED


@pytest.mark.fakes_only
def test_serve_dry_run_for_a_bundle_does_not_install_tt_model(
    runner, isolated_dirs, monkeypatch
):
    """tt-model is lazily installed; describing a serve must not fetch its git ref."""
    def boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("tt-model was installed for a dry run")

    monkeypatch.setattr("tenstorrent.tools.registry.ToolRegistry.ensure", boom)
    result = runner.invoke(app, ["serve", "acme/some-bundle", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["backend"] == "tt-model"
    assert plan["installed"] is False
    assert plan["argv"][:3] == ["<tt-model>", "serve", "acme/some-bundle"]


def test_serve_dry_run_names_the_image_even_without_an_override(runner, fake_server):
    """"from the model spec" is not an answer to "which image will this run" —
    the spec's own ref is in the support list, so name it."""
    plan = json.loads(
        runner.invoke(
            app,
            ["serve", "Llama-3.1-8B-Instruct", "--device", "n150", "--dry-run", "--json"],
        ).stdout
    )
    assert plan["docker_image"] is None  # nothing forced
    assert plan["spec_docker_image"] == "ghcr.io/example/vllm:0.17.0"


def test_serve_dry_run_reports_the_effective_default_port(runner, fake_server, monkeypatch):
    """tt serve defaults --service-port to SERVICE_PORT or 20000, and it inherits
    our environment — so 20000 is not always what a plain serve would use."""
    monkeypatch.delenv("SERVICE_PORT", raising=False)
    argv = ["serve", "Llama-3.1-8B-Instruct", "--device", "n150", "--dry-run", "--json"]
    assert json.loads(runner.invoke(app, argv).stdout)["default_port"] == "20000"
    monkeypatch.setenv("SERVICE_PORT", "7777")
    plan = json.loads(runner.invoke(app, argv).stdout)
    assert plan["default_port"] == "7777"
    assert plan["default_port_from_env"] is True


@pytest.fixture
def pulled_bundle(monkeypatch, tmp_path):
    """A bundle tt-model has pulled: its index entry plus the on-disk manifest."""
    root = tmp_path / "tt-model"
    (root / "pulled" / "acme__demo").mkdir(parents=True)
    (root / "installed.json").write_text(json.dumps({"acme/demo": {}}))
    (root / "pulled" / "acme__demo" / "tt_kernel_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "5.1",
                "arch": "blackhole",
                "device_count": 4,
                "tt_metal_version": "0.65.2",
                "weights": {"repo_id": "acme/demo-weights"},
                "container": {
                    "kind": "vllm-plugin",
                    "image": {"tag": "tt-model/demo:abc123"},
                    "serve": {
                        "hardware": "p300x2",
                        "port": 8000,
                        "max_model_len": 256000,
                        "capabilities": {"tool_parser": "qwen3_coder"},
                        "additional_config": {"tt": {"trace_region_size": 50331648}},
                    },
                    "serve_profiles": [{"name": "default"}],
                },
            }
        )
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return root


def test_serve_dry_run_reads_a_pulled_bundles_own_manifest(runner, pulled_bundle):
    """tt passes no launch flags on this path, but a pulled bundle records the same
    settings on disk — so the preview can say as much as it does for the other
    backend instead of being empty."""
    result = runner.invoke(app, ["serve", "acme/demo", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.stdout)["bundle"]
    assert bundle["engine"] == "vllm-plugin"
    assert bundle["image"] == "tt-model/demo:abc123"
    assert bundle["tool_call_parser"] == "qwen3_coder"
    assert bundle["tt_config"] == {"trace_region_size": 50331648}
    assert bundle["weights_repo"] == "acme/demo-weights"
    assert bundle["hardware"] == "p300x2"


def test_serve_dry_run_does_not_fetch_a_manifest_for_an_unpulled_bundle(
    runner, isolated_dirs, monkeypatch
):
    """No manifest on disk means no preview — fetching one from the Hub would turn
    a dry run into a download."""
    def boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("the Hub was queried for a dry run")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", boom)
    plan = json.loads(
        runner.invoke(app, ["serve", "acme/never-pulled", "--dry-run", "--json"]).stdout
    )
    assert plan["bundle"] is None


# -- pre-seeded persistent volumes -----------------------------------------------------
@pytest.fixture
def volume_root(tmp_path):
    """A root laid out the way tt-inference-server expects, with a volume for
    Qwen3-32B only. The server appends volume_id_<impl>-<model>-v<version> itself,
    so the directory name carries an impl and a version tt never computes."""
    root = tmp_path / "tt-cache"
    (root / "volume_id_tt_transformers-Qwen3-32B-v0.17.0" / "weights").mkdir(parents=True)
    return root


VOLUME_DIR = "volume_id_tt_transformers-Qwen3-32B-v0.17.0"


def _configure_volume(runner, path):
    assert (
        runner.invoke(
            app, ["config", "set", "paths.preloaded_volume_directory", str(path)]
        ).exit_code
        == 0
    )


def _plan(runner, model, device="galaxy"):
    result = runner.invoke(app, ["serve", model, "--device", device, "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


@pytest.mark.fakes_only
def test_serve_uses_a_preloaded_volume_when_one_is_there(runner, volume_root):
    """Passing the root lets the server reuse both the weights and the far more
    expensive tt_metal_cache instead of rebuilding them in a docker volume."""
    _configure_volume(runner, volume_root)
    plan = _plan(runner, "Qwen3-32B")
    assert plan["host_volume"] == str(volume_root)
    assert plan["argv"][plan["argv"].index("--host-volume") + 1] == str(volume_root)
    # the plan is the dry run's contract: reporting a cache the argv does not
    # pass would tell a --json consumer the wrong thing
    assert plan["host_hf_cache"] is None
    # setup_host.check_setup() returns on host_hf_cache before it ever looks at
    # host_model_volume_root, so passing both leaves the volume unconsulted and
    # re-downloads the weights. They are alternatives, not layers.
    assert "--host-hf-cache" not in plan["argv"]


@pytest.mark.fakes_only
def test_serve_skips_the_volume_when_the_model_has_none_there(runner, volume_root):
    """Preloading is per model. Passing the root for a model with no directory in
    it would redirect that model's storage there as a side effect, which is a
    bigger change than declining to use a cache."""
    _configure_volume(runner, volume_root)
    (volume_root / VOLUME_DIR).rename(
        volume_root / "volume_id_tt_transformers-Something-Else-v0.17.0"
    )
    plan = _plan(runner, "Qwen3-32B")
    assert plan["host_volume"] is None
    assert "--host-volume" not in plan["argv"]
    assert "--host-hf-cache" in plan["argv"]  # back to the normal path


@pytest.mark.fakes_only
def test_serve_ignores_a_volume_belonging_to_another_model(runner, volume_root):
    """The directory is Qwen3-32B's; Llama has none there, so it stays on the
    server's own docker volume."""
    _configure_volume(runner, volume_root)
    plan = _plan(runner, "Llama-3.1-8B-Instruct", device="n150")
    assert plan["host_volume"] is None


@pytest.mark.fakes_only
def test_serve_passes_no_volume_by_default(runner, volume_root):
    """Unconfigured is the default and must be byte-identical to before: the
    server keeps using its own docker volume."""
    plan = _plan(runner, "Qwen3-32B")
    assert plan["host_volume"] is None
    assert "--host-volume" not in plan["argv"]


@pytest.mark.fakes_only
def test_serve_tolerates_a_configured_root_that_does_not_exist(runner, tmp_path):
    _configure_volume(runner, tmp_path / "nope")
    assert _plan(runner, "Qwen3-32B")["host_volume"] is None


@pytest.mark.fakes_only
def test_serve_skips_a_volume_seeded_for_a_different_version(runner, volume_root):
    """The server derives volume_id_<impl>-<model>-v<version> and each board has
    its own impl and version — Qwen3-32B spans four names across its devices. A
    match on the model alone would pass the flag, find nothing under the name the
    server actually wants, and build a second copy inside the user's directory."""
    _configure_volume(runner, volume_root)
    (volume_root / VOLUME_DIR).rename(
        volume_root / "volume_id_tt_transformers-Qwen3-32B-v0.9.0"
    )
    plan = _plan(runner, "Qwen3-32B")
    assert plan["host_volume"] is None
    assert "--host-volume" not in plan["argv"]


# -- models whose container fetches its own weights ------------------------------------
@pytest.mark.fakes_only
@pytest.mark.parametrize(
    "model",
    [
        "whisper-large-v3",  # STT
        "speecht5_tts",  # TTS
        "resnet-50",  # forge: weights ship in the image
    ],
)
def test_serve_does_not_make_the_host_download_unused_weights(
    runner, docker_present, fake_server, model
):
    """setup_host with MODEL_SOURCE=huggingface downloads the whole repo to the
    host and mounts it readonly — 12 GB for distil-large-v3 — and these containers
    then ignore it and fetch their own copy. `noaction` skips the host download;
    HF_TOKEN still reaches the container, since run.py writes it to the env-file
    for any --docker-server run regardless of the model source."""
    result = runner.invoke(app, ["serve", model, "--device", "n150"])
    assert result.exit_code == 0, result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert "--host-hf-cache" not in argv


@pytest.mark.fakes_only
def test_serve_still_shares_the_host_cache_for_models_that_read_it(
    runner, docker_present, fake_server
):
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "n150"])
    assert result.exit_code == 0, result.output
    assert "--host-hf-cache" in json.loads(fake_server.read_text().splitlines()[-1])


@pytest.mark.fakes_only
@pytest.mark.parametrize(
    ("model", "device"),
    [("Qwen3-32B", "galaxy"), ("Llama-3.1-8B-Instruct", "n150"), ("resnet-50", "n150")],
)
def test_serve_dry_run_reports_the_weights_source_it_actually_passes(
    runner, volume_root, model, device
):
    """host_hf_cache and host_volume are what --json consumers read; either one
    set without the matching flag in argv is a lie about what would run."""
    _configure_volume(runner, volume_root)
    plan = _plan(runner, model, device=device)
    assert (plan["host_hf_cache"] is not None) == ("--host-hf-cache" in plan["argv"])
    assert (plan["host_volume"] is not None) == ("--host-volume" in plan["argv"])
    assert not (plan["host_hf_cache"] and plan["host_volume"])


@pytest.mark.fakes_only
def test_serve_runs_the_server_unauthenticated(runner, docker_present, fake_server):
    """One flag covers both schemes: it drops the JWT_SECRET requirement on the
    vLLM path — and with it the getpass prompt and the secret run.py persists in
    the checkout — and sets NO_AUTH for media/forge, which would otherwise fall
    back to a bearer token baked into the image."""
    for model, device in (("Llama-3.1-8B-Instruct", "n150"), ("whisper-large-v3", "n150")):
        assert runner.invoke(app, ["serve", model, "--device", device]).exit_code == 0
        assert "--no-auth" in json.loads(fake_server.read_text().splitlines()[-1])


@pytest.mark.fakes_only
def test_serve_does_not_disable_auth_for_client_side_workflows(
    runner, docker_present, fake_server
):
    """--no-auth is about the server tt starts; benchmarks and evals run against
    one and never take it."""
    result = runner.invoke(
        app, ["serve", "Llama-3.1-8B-Instruct", "--device", "n150", "--workflow", "benchmarks"]
    )
    assert result.exit_code == 0
    assert "--no-auth" not in json.loads(fake_server.read_text().splitlines()[-1])


@pytest.mark.fakes_only
def test_serve_accepts_a_device_in_any_casing(runner, docker_present, fake_server):
    """Spec device keys are lowercase and every per-device lookup uses them, so an
    uppercase --device used to resolve to no support entry: the parsers, the image
    override and a pre-seeded volume were all silently skipped."""
    result = runner.invoke(app, ["serve", "Llama-3.1-8B-Instruct", "--device", "N150"])
    assert result.exit_code == 0, result.output
    argv = json.loads(fake_server.read_text().splitlines()[-1])
    assert argv[argv.index("--device") + 1] == "n150"
    assert "--vllm-override-args" in argv


def test_serve_rejects_a_device_the_model_has_no_entry_for(runner, docker_present, fake_server):
    """run.py would reject it as an unknown choice anyway, but only after tt had
    quietly dropped everything keyed on the device."""
    result = runner.invoke(app, ["serve", "Qwen3-32B", "--device", "n300"])
    assert result.exit_code == ExitCode.USAGE
    assert "no support entry for n300" in result.output
    assert "galaxy" in result.output  # it lists what the model does have
    assert not fake_server.exists()
