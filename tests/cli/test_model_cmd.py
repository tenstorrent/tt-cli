# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode



@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich truncates cells to the terminal width; pin it so table assertions test
    content rather than the width of whoever runs the suite."""
    monkeypatch.setenv("COLUMNS", "160")


@pytest.fixture(autouse=True)
def empty_hf_cache(monkeypatch):
    """Default: nothing cached. Tests override the scan to simulate cached models."""
    monkeypatch.setattr("tenstorrent.modelhub.catalog.scan_hf_cache", lambda: {})


SMALL_SUPPORT = (
    Path(__file__).parent.parent / "fakes" / "data" / "model_support_small.json"
)


@pytest.fixture(autouse=True)
def small_spec(monkeypatch):
    """Pin the catalog to the 4-model fixture so exact assertions stay stable when
    the shipped model_support.json is regenerated."""
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))


def _set_cache(monkeypatch, sizes):
    monkeypatch.setattr("tenstorrent.modelhub.catalog.scan_hf_cache", lambda: sizes)


def _names(result) -> list[str]:
    return [m["name"] for m in json.loads(result.output)["models"]]


def test_model_list_shows_catalog(runner):
    # no tt-smi wired: detection degrades to a warning and the full list.
    # Short names only — long ones truncate at the test terminal's 80 columns.
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "Qwen3-32B" in result.output
    assert "resnet-50" in result.output


def test_model_list_detection_failure_warns_and_shows_all(runner):
    result = runner.invoke(app, ["model", "list", "--json"])
    assert result.exit_code == 0
    assert "device detection skipped" in result.output
    payload_start = result.output.index("{")
    payload = json.loads(result.output[payload_start:])
    assert payload["device"] is None
    assert len(payload["models"]) == 5


@pytest.mark.fakes_only
@pytest.mark.parametrize(
    ("scenario", "device", "expected"),
    [
        ("normal", "p300", ["Llama-3.1-8B-Instruct"]),
        # whisper is marked broken on p300x2 in the fixture, so it is hidden
        # there while staying listed on n150 — see the test below.
        ("multi", "p300x2", ["Llama-3.1-8B-Instruct"]),
    ],
)
def test_model_list_filters_to_detected_device(
    runner, smi_bin, monkeypatch, scenario, device, expected
):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", scenario)
    result = runner.invoke(app, ["model", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["device"] == device
    assert [m["name"] for m in payload["models"]] == expected


@pytest.mark.fakes_only
def test_model_list_all_skips_detection(runner, smi_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_SCENARIO", "normal")
    result = runner.invoke(app, ["model", "list", "--all", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["device"] is None
    assert "resnet-50" in [m["name"] for m in payload["models"]]
    assert smi_bin.exists() is False  # --all never shells tt-smi


def test_model_list_shows_every_engine_without_a_serve_column(runner):
    """n150 in the fixture has Llama (vLLM), whisper (media) and resnet (forge).
    All three are servable, so the old serve column would be a constant ✓."""
    result = runner.invoke(app, ["model", "list", "--hw", "n150"])
    assert result.exit_code == 0
    assert "serve" not in result.output
    for name in ("Llama-3.1-8B-Instruct", "whisper-large-v3", "resnet-50"):
        assert name in result.output


def test_model_list_hides_a_model_marked_broken_on_that_device(runner):
    """The support list exists so `tt model list` never offers something that
    will not start. --all still shows it, since a mark is per device."""
    on_board = runner.invoke(app, ["model", "list", "--hw", "p300x2", "--json"])
    assert "whisper-large-v3" not in on_board.output
    assert "whisper-large-v3" in runner.invoke(app, ["model", "list", "--hw", "n150"]).output
    assert "whisper-large-v3" in runner.invoke(app, ["model", "list", "--all"]).output


def test_model_list_hw_filter_skips_detection(runner):
    # no tt-smi wired, yet no warning: --hw bypasses detection entirely
    result = runner.invoke(app, ["model", "list", "--hw", "galaxy", "--json"])
    assert result.exit_code == 0
    assert "detection" not in result.output
    payload = json.loads(result.output)
    assert payload["device"] == "galaxy"
    assert [m["name"] for m in payload["models"]] == ["Qwen3-32B"]


def test_model_list_json_and_cache_merge(runner, monkeypatch):
    _set_cache(monkeypatch, {"openai/whisper-large-v3": 2_200_000_000})
    result = runner.invoke(app, ["model", "list", "--all", "--json"])
    assert result.exit_code == 0
    models = {m["name"]: m for m in json.loads(result.output)["models"]}
    assert models["whisper-large-v3"]["cached"] is True
    assert models["whisper-large-v3"]["cache_size_bytes"] == 2_200_000_000
    assert models["Llama-3.1-8B-Instruct"]["cached"] is False


def test_model_list_cached_filter(runner, monkeypatch):
    _set_cache(monkeypatch, {"openai/whisper-large-v3": 1})
    result = runner.invoke(app, ["model", "list", "--all", "--cached", "--json"])
    assert _names(result) == ["whisper-large-v3"]


def test_model_list_type_and_hw_filters(runner):
    result = runner.invoke(app, ["model", "list", "--all", "--type", "audio", "--json"])
    assert _names(result) == ["whisper-large-v3"]
    result = runner.invoke(app, ["model", "list", "--hw", "p300", "--json"])
    assert _names(result) == ["Llama-3.1-8B-Instruct"]


def test_model_info(runner):
    result = runner.invoke(app, ["model", "info", "Llama-3.1-8B-Instruct", "--json"])
    assert result.exit_code == 0
    info = json.loads(result.output)
    assert info["hf_repo"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert info["tt_model_id"] == "Llama-3.1-8B-Instruct"
    assert info["engines"] == ["vLLM"]
    assert info["devices"]["p300"]["status"] == "FUNCTIONAL"
    assert info["devices"]["p300"]["max_context"] == 131072
    assert "benchmarks" not in info


def test_model_info_accepts_hf_repo_alias(runner):
    result = runner.invoke(app, ["model", "info", "meta-llama/llama-3.1-8b-instruct", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["name"] == "Llama-3.1-8B-Instruct"


def test_model_info_unknown_is_usage_error(runner):
    result = runner.invoke(app, ["model", "info", "gpt-17"])
    assert result.exit_code == ExitCode.USAGE
    assert "tt model list" in result.output


def test_model_pull_downloads_via_hf(runner, monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(repo_id, **kwargs):
        calls.update(repo_id=repo_id, **kwargs)
        return str(tmp_path / "snap")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = runner.invoke(app, ["model", "pull", "whisper-large-v3", "--json"])
    assert result.exit_code == 0, result.output
    assert calls["repo_id"] == "openai/whisper-large-v3"
    assert calls["local_files_only"] is False
    # torch-format duplicates are skipped, matching run.py's own exclude
    assert calls["ignore_patterns"] == ["original/*"]
    # stdout, not output: the cache warning for STT models goes to stderr, and the
    # data channel has to stay parseable on its own (`tt --json | jq`).
    assert json.loads(result.stdout)["path"] == str(tmp_path / "snap")


def test_model_pull_of_an_image_shipped_model_is_unsupported(
    runner, monkeypatch
):
    """Forge CNNs name a model label, not a Hub repo. Downloading is impossible,
    so say so instead of making two doomed requests and reporting a 404 with
    "check connectivity" — advice that can never work."""

    def boom(repo_id, **kwargs):  # pragma: no cover - must not run
        raise AssertionError(f"no download should be attempted for {repo_id}")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    result = runner.invoke(app, ["model", "pull", "resnet-50"])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "ship inside the container image" in result.output.replace("\n", " ")


def test_model_pull_warns_when_the_host_cache_will_not_be_used(
    runner, monkeypatch, tmp_path
):
    """STT/TTS weights are real and cacheable, so the pull proceeds — but the
    serving container fetches its own copy, so it buys nothing for `tt serve`."""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(tmp_path / "snap")
    )
    result = runner.invoke(app, ["model", "pull", "whisper-large-v3"])
    assert result.exit_code == 0, result.output
    assert "ignores the host cache" in result.output.replace("\n", " ")


def test_model_pull_retries_without_xet_backend(runner, monkeypatch, tmp_path):
    """hf-xet chunk failures (seen on hardware) degrade to a plain-HTTP retry."""
    import huggingface_hub

    attempts = []

    def fake_snapshot_download(repo_id, **kwargs):
        attempts.append(huggingface_hub.constants.HF_HUB_DISABLE_XET)
        if len(attempts) == 1:
            raise RuntimeError("Task error: Unable to parse string as hex hash value")
        return str(tmp_path / "snap")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    result = runner.invoke(app, ["model", "pull", "whisper-large-v3"])
    assert result.exit_code == 0, result.output
    assert "Xet backend failed" in result.output
    assert attempts == [False, True]  # second attempt ran with Xet disabled


def test_model_pull_respects_configured_cache_dir(runner, monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(repo_id, **kwargs):
        calls["cache_dir"] = kwargs.get("cache_dir")
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert (
        runner.invoke(
            app, ["config", "set", "paths.hf_model_cache_directory", "/data/hf"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["model", "pull", "whisper-large-v3"]).exit_code == 0
    # the config key has HF_HOME semantics; the hub cache lives under <root>/hub
    # (matching what run.py --host-hf-cache expects when tt serve mounts it)
    assert calls["cache_dir"] == "/data/hf/hub"


def test_model_pull_offline_miss_is_offline_error(runner, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fake_snapshot_download(repo_id, **kwargs):
        assert kwargs.get("local_files_only") is True
        raise LocalEntryNotFoundError("not cached")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = runner.invoke(app, ["model", "pull", "whisper-large-v3", "--offline"])
    assert result.exit_code == ExitCode.OFFLINE
    assert "tt model pull" in result.output


def test_model_compile_is_documented_stub(runner):
    result = runner.invoke(app, ["model", "compile", "whisper-large-v3"])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "not available yet" in result.output


# -- stop / rm ---------------------------------------------------------------------
@pytest.fixture
def fake_model_manager(model_manager_bin):
    return model_manager_bin


@pytest.fixture
def always_tty(monkeypatch):
    """CliRunner replaces sys.stdin, so the confirmation seam is patched instead."""
    monkeypatch.setattr("tenstorrent.commands.model._stdin_isatty", lambda: True)


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    """A stand-in tt-inference-server checkout with per-model leftovers, wired
    through the TT_TOOL_BIN_* seam so nothing writes into the real one."""
    root = tmp_path / "inference-repo"
    logs = root / "workflow_logs" / "docker_server"
    logs.mkdir(parents=True)
    (logs / "vllm_2026-01-01_00-00-00_Llama-3.1-8B-Instruct_p300x2_server.log").write_text(
        "x" * 100
    )
    (logs / "vllm_2026-01-01_00-00-00_Qwen3-32B_p300x2_server.log").write_text("y" * 50)
    volume = root / "persistent_volume" / "volume_id_tt-transformers-Llama-3.1-8B-Instruct-v0.0.1"
    volume.mkdir(parents=True)
    (volume / "blob").write_text("z" * 200)
    entry = root / "run.py"
    entry.write_text("#!/usr/bin/env python3\n")
    monkeypatch.setenv("TT_TOOL_BIN_TT_INFERENCE_SERVER", str(entry))
    return root


@pytest.mark.fakes_only
def test_model_rm_bundle_delegates_to_tt_model(
    runner, fake_model_manager, always_tty, isolated_dirs
):
    result = runner.invoke(
        app, ["model", "rm", "ns/bundle", "--keep-cache", "--include-weights", "--yes"]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["rm", "ns/bundle", "--keep-cache", "--include-weights"]


@pytest.mark.fakes_only
def test_model_rm_bundle_dry_run_runs_nothing(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["model", "rm", "ns/bundle", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would run" in result.output
    assert not fake_model_manager.exists()  # tt-model never invoked


@pytest.mark.fakes_only
def test_model_stop_bundle_delegates_to_tt_model(
    runner, fake_model_manager, isolated_dirs
):
    result = runner.invoke(app, ["model", "stop", "ns/bundle"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["stop", "ns/bundle"]


def test_model_rm_removes_only_the_named_models_artifacts(
    runner, fake_checkout, always_tty, isolated_dirs
):
    """Underscore-delimited matching keeps one model's logs out of another's rm."""
    result = runner.invoke(
        app, ["model", "rm", "Llama-3.1-8B-Instruct", "--yes", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    kinds = sorted(r["kind"] for r in payload["artifacts"])
    assert kinds == ["logs", "volume"]
    assert payload["images_removed"] == []  # images are shared, never removed
    logs = fake_checkout / "workflow_logs" / "docker_server"
    assert not (logs / "vllm_2026-01-01_00-00-00_Llama-3.1-8B-Instruct_p300x2_server.log").exists()
    assert (logs / "vllm_2026-01-01_00-00-00_Qwen3-32B_p300x2_server.log").exists()
    assert list((fake_checkout / "persistent_volume").glob("volume_id_*")) == []


def test_model_rm_dry_run_deletes_nothing(
    runner, fake_checkout, isolated_dirs
):
    result = runner.invoke(
        app, ["model", "rm", "Llama-3.1-8B-Instruct", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True
    logs = fake_checkout / "workflow_logs" / "docker_server"
    assert (logs / "vllm_2026-01-01_00-00-00_Llama-3.1-8B-Instruct_p300x2_server.log").exists()


def test_model_rm_keeps_weights_by_default(
    runner, fake_checkout, always_tty, isolated_dirs, monkeypatch
):
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.cached_weights", lambda repo, config: (["abc"], 16_000_000_000)
    )
    deleted = []
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.delete_cached_weights",
        lambda repo, config: deleted.append(repo) or 0,
    )
    result = runner.invoke(
        app, ["model", "rm", "Llama-3.1-8B-Instruct", "--yes", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["weights_kept_bytes"] == 16_000_000_000
    assert deleted == []  # nothing touched the HF cache
    assert all(r["kind"] != "weights" for r in payload["artifacts"])


def test_model_rm_include_weights_deletes_them(
    runner, fake_checkout, always_tty, isolated_dirs, monkeypatch
):
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.cached_weights", lambda repo, config: (["abc"], 16_000_000_000)
    )
    deleted = []
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.delete_cached_weights",
        lambda repo, config: (deleted.append(repo), 16_000_000_000)[1],
    )
    result = runner.invoke(
        app,
        ["model", "rm", "Llama-3.1-8B-Instruct", "--include-weights", "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["weights_kept_bytes"] == 0
    assert deleted == ["meta-llama/Llama-3.1-8B-Instruct"]


def test_model_rm_without_confirmation_is_usage(runner, fake_checkout, isolated_dirs, monkeypatch):
    """Non-interactive without --yes must not be a silent yes."""
    monkeypatch.setattr("tenstorrent.commands.model._stdin_isatty", lambda: False)
    result = runner.invoke(app, ["model", "rm", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == ExitCode.USAGE
    assert "--dry-run" in result.output


def test_model_rm_unknown_name_is_usage(runner, isolated_dirs):
    result = runner.invoke(app, ["model", "rm", "Llama-3.1-8B-Instrukt", "--yes"])
    assert result.exit_code == ExitCode.USAGE
    assert "Unknown model" in result.output


@pytest.mark.fakes_only
def test_model_rm_bundle_does_not_install_tt_model(runner, uv_bin, isolated_dirs):
    """Teardown must not install the tool: a working fake uv is wired here, so an
    install *would* succeed — the point is that it is never attempted."""
    result = runner.invoke(app, ["model", "rm", "ns/bundle", "--yes"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "not installed" in result.output
    assert not uv_bin.exists()  # uv never ran


@pytest.mark.fakes_only
def test_model_stop_bundle_does_not_install_tt_model(runner, uv_bin, isolated_dirs):
    result = runner.invoke(app, ["model", "stop", "ns/bundle"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert not uv_bin.exists()


@pytest.mark.fakes_only
def test_serve_still_installs_tt_model_on_demand(runner, uv_bin, isolated_dirs):
    """The contrast: serving a bundle *should* install the tool lazily."""
    result = runner.invoke(app, ["serve", "ns/bundle"])
    assert uv_bin.exists(), result.output  # uv was invoked to install tt-model
    argv = json.loads(uv_bin.read_text().splitlines()[0])
    assert argv[:2] == ["tool", "install"]
    assert argv[2].startswith("tt-model @ git+")


# -- stopping catalog-model containers ---------------------------------------------
FAKE_BIN_DIR = Path(__file__).parent.parent / "fakes" / "bin"


def _container(cid, *, snapshot=None, volume=None, name=None, image="img:1"):
    mounts = []
    if snapshot:
        mounts.append({"Type": "bind", "Source": snapshot})
    if volume:
        mounts.append({"Type": "volume", "Name": volume})
    return {
        "Id": cid,
        "Name": f"/{name or 'tt-inference-server-' + cid[:4]}",
        "Config": {"Image": image},
        "Mounts": mounts,
    }


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    """Point the backend's runtime lookup at the fake docker and collect its stops."""
    stop_log = tmp_path / "docker-stop.log"
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which",
        lambda name: str(FAKE_BIN_DIR / "docker") if name == "docker" else None,
    )
    monkeypatch.setenv("FAKE_DOCKER_STOP_LOG", str(stop_log))

    def set_containers(entries):
        monkeypatch.setenv("FAKE_DOCKER_CONTAINERS", json.dumps(entries))

    return set_containers, stop_log


@pytest.mark.fakes_only
def test_model_stop_matches_a_container_by_its_weights_mount(
    runner, fake_docker, isolated_dirs
):
    """`tt serve` always passes --host-hf-cache, so the HF snapshot bind mount is
    what identifies the model — the container name is a random uuid."""
    set_containers, stop_log = fake_docker
    set_containers([
        _container(
            "aaaaaaaaaaaa11",
            snapshot="/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/rev",
        ),
        _container("bbbbbbbbbbbb22", snapshot="/hf/hub/models--Qwen--Qwen3-32B/snapshots/rev"),
    ])
    result = runner.invoke(app, ["model", "stop", "Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    assert stop_log.read_text().split() == ["aaaaaaaaaaaa"]  # the Qwen one untouched


@pytest.mark.fakes_only
def test_model_stop_matches_a_container_by_its_named_volume(
    runner, fake_docker, isolated_dirs
):
    """Without a host HF cache the weights live in a volume named
    volume_id_<impl>-<model> — generate_docker_volume_name drops the version so an
    image upgrade reuses the volume. impl_id contains hyphens, so the match is on
    the model-name suffix."""
    set_containers, stop_log = fake_docker
    set_containers([_container("cccccccccccc33", volume="volume_id_tt-transformers-Qwen3-32B")])
    result = runner.invoke(app, ["model", "stop", "Qwen3-32B"])
    assert result.exit_code == 0, result.output
    assert stop_log.read_text().split() == ["cccccccccccc"]


@pytest.mark.fakes_only
def test_model_stop_accepts_the_hf_repo_as_the_name(
    runner, fake_docker, isolated_dirs
):
    set_containers, stop_log = fake_docker
    set_containers([
        _container(
            "dddddddddddd44",
            snapshot="/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/rev",
        )
    ])
    result = runner.invoke(app, ["model", "stop", "meta-llama/Llama-3.1-8B-Instruct"])
    assert result.exit_code == 0, result.output
    assert stop_log.read_text().strip() == "dddddddddddd"


@pytest.mark.fakes_only
def test_model_stop_stops_every_container_for_the_model(
    runner, fake_docker, isolated_dirs
):
    """The same model served twice (different ports) means two containers."""
    set_containers, stop_log = fake_docker
    snapshot = "/hf/hub/models--Qwen--Qwen3-32B/snapshots/rev"
    set_containers([
        _container("eeeeeeeeeeee55", snapshot=snapshot),
        _container("ffffffffffff66", snapshot=snapshot),
    ])
    result = runner.invoke(app, ["model", "stop", "Qwen3-32B", "--json"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["stopped"]) == 2
    assert sorted(stop_log.read_text().split()) == ["eeeeeeeeeeee", "ffffffffffff"]


@pytest.mark.fakes_only
def test_model_stop_when_nothing_is_running_is_ok(
    runner, fake_docker, isolated_dirs
):
    set_containers, stop_log = fake_docker
    set_containers([])
    result = runner.invoke(app, ["model", "stop", "Qwen3-32B"])
    assert result.exit_code == ExitCode.OK
    assert not stop_log.exists()


@pytest.mark.fakes_only
def test_model_stop_reports_unidentifiable_containers_instead_of_guessing(
    runner, fake_docker, isolated_dirs
):
    """A --host-weights-dir container may encode nothing about its model. Never
    stop a server on a guess."""
    set_containers, stop_log = fake_docker
    set_containers([_container("999999999999aa", snapshot="/opt/my-weights")])
    result = runner.invoke(app, ["model", "stop", "Qwen3-32B"])
    assert result.exit_code == ExitCode.ERROR
    assert "could not be identified" in result.output
    assert not stop_log.exists()


def test_model_stop_without_a_container_runtime_is_tool_missing(
    runner, isolated_dirs, monkeypatch
):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )
    result = runner.invoke(app, ["model", "stop", "Qwen3-32B"])
    assert result.exit_code == ExitCode.TOOL_MISSING


# -- community bundle listing ------------------------------------------------------
def _stub_bundles(monkeypatch, entries):
    """Replace the Hub query; the suite must stay network-free."""
    from tenstorrent.modelhub.bundles import BundleInfo

    made = [BundleInfo(**e) for e in entries]
    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles.search_community", lambda **kw: made
    )
    return made


def _table_header(output: str) -> str:
    """The rendered header row. Matched on the box-drawing column separator, not
    on a column name: the caption mentions the dropped columns, so searching for
    one of those names finds a wrapped caption line and asserts nothing."""
    return next(line for line in output.splitlines() if "┃" in line)


def test_model_list_community_shows_bundles(runner, monkeypatch, isolated_dirs):
    _stub_bundles(monkeypatch, [
        {"name": "ns/alpha", "kind": "container", "engine": "vLLM",
         "arch": ["blackhole"], "downloads": 3, "installed": True},
        {"name": "ns/beta", "kind": "thin", "engine": "vLLM",
         "arch": ["wormhole_b0", "1x4"], "installed": False},
    ])
    result = runner.invoke(app, ["model", "list", "--community"])
    assert result.exit_code == 0, result.output
    assert "ns/alpha" in result.output and "ns/beta" in result.output
    assert "wormhole_b0" in result.output
    # The table answers "can I run this, and is it here already". kind and engine
    # are how a bundle is built, not something you pick one on; both stay in --json.
    header = _table_header(result.output)
    assert [c.strip() for c in header.strip("┃").split("┃")] == [
        "name",
        "source",
        "arch",
        "weights",
    ]


def test_model_list_community_writes_the_completion_cache(
    runner, monkeypatch, isolated_dirs
):
    # Tab completion must never query the Hub, so the listing is what teaches
    # `tt serve <TAB>` the community bundle ids.
    from tenstorrent.modelhub import bundles, completions

    _stub_bundles(monkeypatch, [
        {"name": "ns/alpha", "kind": "container", "engine": "vLLM",
         "arch": ["blackhole"], "installed": False},
    ])
    result = runner.invoke(app, ["model", "list", "--community"])
    assert result.exit_code == 0, result.output
    assert bundles.cached_community_names() == ["ns/alpha"]
    assert completions.complete_model("ns/al") == ["ns/alpha"]


def test_model_list_community_json_contract(runner, monkeypatch, isolated_dirs):
    _stub_bundles(monkeypatch, [
        {"name": "ns/alpha", "kind": "container", "engine": "vLLM",
         "arch": ["blackhole"], "downloads": 3, "installed": True},
    ])
    result = runner.invoke(app, ["model", "list", "--community", "--json"])
    payload = json.loads(result.output)
    assert payload["source"] == "tt-model-catalog"
    assert payload["bundles"][0]["name"] == "ns/alpha"
    assert payload["bundles"][0]["installed"] is True


def test_model_list_community_cached_filters_to_installed(
    runner, monkeypatch, isolated_dirs
):
    """--cached means "what is on this machine", so it keeps the local rows — one
    per installed bundle — and drops the Hub listing entirely."""
    _stub_bundles(monkeypatch, [
        {"name": "ns/alpha", "installed": True},
        {"name": "ns/beta", "installed": False},
    ])
    _stub_local(monkeypatch, [{"name": "ns/alpha"}])
    result = runner.invoke(app, ["model", "list", "--community", "--cached", "--json"])
    rows = json.loads(result.output)["bundles"]
    assert [(b["name"], b["source"]) for b in rows] == [("ns/alpha", "local")]


def test_model_list_community_does_not_read_the_support_list(
    runner, monkeypatch, isolated_dirs
):
    """The two listings are independent: no spec parse, no tt-smi detection."""
    _stub_bundles(monkeypatch, [{"name": "ns/alpha"}])

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("the support list was read for --community")

    monkeypatch.setattr("tenstorrent.modelhub.catalog.ModelSupportSource", boom)
    result = runner.invoke(app, ["model", "list", "--community"])
    assert result.exit_code == 0, result.output


def test_model_list_community_rejects_device_filters(runner, isolated_dirs):
    for argv in (["--hw", "p300x2"], ["--all"]):
        result = runner.invoke(app, ["model", "list", "--community", *argv])
        assert result.exit_code == ExitCode.USAGE, argv
        assert "does not apply" in result.output


def test_model_list_community_rejects_type_filter(runner, isolated_dirs):
    result = runner.invoke(app, ["model", "list", "--community", "--type", "llm"])
    assert result.exit_code == ExitCode.USAGE
    assert "not published as a repo tag" in result.output


def test_model_list_community_offline_shows_local_bundles_only(
    runner, monkeypatch, isolated_dirs
):
    """Local installs are entirely on disk, so --offline degrades to them instead of
    refusing the command."""
    from tenstorrent.modelhub.bundles import BundleInfo

    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles.local_bundles",
        lambda **kw: [BundleInfo(name="ns/local", source="local", installed=True)],
    )

    def boom(**kw):  # pragma: no cover - the Hub must not be reached
        raise AssertionError("the Hub was queried under --offline")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.search_community", boom)
    result = runner.invoke(app, ["--offline", "model", "list", "--community"])
    assert result.exit_code == 0, result.output
    assert "ns/local" in result.output
    assert "only bundles installed on this machine" in result.output


def test_model_list_community_weights_cell_states(runner, monkeypatch, isolated_dirs):
    """Three distinct states: cached with a size, referenced but absent, unknown."""
    _stub_bundles(monkeypatch, [
        {"name": "ns/cached", "installed": True,
         "weights_repo": "org/w", "weights_bytes": 2_000_000_000},
        {"name": "ns/nocache", "installed": True, "weights_repo": "org/w2"},
        {"name": "ns/unpulled", "installed": False},
    ])
    result = runner.invoke(app, ["model", "list", "--community"])
    assert result.exit_code == 0, result.output
    # last column is `weights`; compare that cell alone, not the whole row
    weights = {}
    for line in result.output.splitlines():
        if line.startswith("│") and "ns/" in line:
            cells = [c.strip() for c in line.strip("│").split("│")]
            weights[cells[0]] = cells[-1]
    assert weights["ns/cached"] == "✓ 1.9 GB"
    assert weights["ns/nocache"] == "—"  # referenced, but not in the cache
    assert weights["ns/unpulled"] == "?"  # not pulled, so the reference is unknown


def _stub_local(monkeypatch, entries):
    from tenstorrent.modelhub.bundles import BundleInfo

    made = [BundleInfo(source="local", installed=True, **e) for e in entries]
    monkeypatch.setattr("tenstorrent.modelhub.bundles.local_bundles", lambda **kw: made)
    return made


def test_model_list_community_includes_unpublished_local_bundles(
    runner, monkeypatch, isolated_dirs
):
    """A bundle someone shared privately is installed here but absent from the
    catalog — it must still be listed, marked as local."""
    _stub_bundles(monkeypatch, [{"name": "ns/published"}])
    _stub_local(monkeypatch, [{"name": "someone/private", "kind": "container"}])
    result = runner.invoke(app, ["model", "list", "--community", "--json"])
    assert result.exit_code == 0, result.output
    rows = {b["name"]: b["source"] for b in json.loads(result.output)["bundles"]}
    assert rows == {"ns/published": "HF", "someone/private": "local"}


def test_model_list_community_lists_a_bundle_once_per_source(
    runner, monkeypatch, isolated_dirs
):
    """Published and installed are two different facts about a bundle. Collapsing
    them to one row loses whichever one the merge did not pick."""
    _stub_bundles(monkeypatch, [{"name": "ns/both", "downloads": 7}])
    _stub_local(monkeypatch, [{"name": "ns/both"}])
    result = runner.invoke(app, ["model", "list", "--community", "--json"])
    payload = json.loads(result.output)["bundles"]
    assert [b["source"] for b in payload] == ["HF", "local"]
    assert [b["name"] for b in payload] == ["ns/both", "ns/both"]
    assert payload[0]["downloads"] == 7  # only the Hub publishes this


def test_model_list_community_marks_local_rows_in_the_table(
    runner, monkeypatch, isolated_dirs
):
    _stub_bundles(monkeypatch, [])
    _stub_local(monkeypatch, [{"name": "someone/private"}])
    result = runner.invoke(app, ["model", "list", "--community"])
    assert "someone/private" in result.output
    assert "local" in result.output
    # the installed column is gone — source carries it now
    assert "installed" not in _table_header(result.output)


# -- pulling things the released spec does not list --------------------------------
@pytest.fixture
def no_hub_probe(monkeypatch):
    """Default: the bundle probe says "not a bundle" without touching the network."""
    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", lambda name: False)


@pytest.mark.fakes_only
def test_model_pull_bundle_delegates_to_tt_model(
    runner, fake_model_manager, monkeypatch, isolated_dirs
):
    """A bundle id installs the bundle, with weights, into the shared HF cache."""
    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", lambda name: True)
    result = runner.invoke(app, ["model", "pull", "ns/bundle"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["pull", "ns/bundle", "--with-weights"]
    assert record["hf_home"]  # same cache everything else uses


def test_model_pull_plain_hf_repo_fetches_weights_with_a_warning(
    runner, no_hub_probe, monkeypatch, isolated_dirs
):
    """An ordinary HF repo is pullable, but must not imply it is servable."""
    seen = {}

    def fake_download(model, config, *, offline, output):
        seen["repo"] = model.hf_repo
        return Path("/tmp/weights")

    monkeypatch.setattr("tenstorrent.modelhub.hub.download_weights", fake_download)
    result = runner.invoke(app, ["model", "pull", "microsoft/phi-4", "--json"])
    assert result.exit_code == 0, result.output
    assert seen["repo"] == "microsoft/phi-4"
    # Rich wraps warnings at the terminal width, so compare on normalized whitespace
    assert "cannot serve it" in " ".join(result.output.split())  # the warning, on stderr
    # CliRunner merges stderr into output, so parse the JSON document itself
    assert json.loads(result.output[result.output.index("{"):])["kind"] == "weights"


@pytest.mark.fakes_only
def test_model_pull_weights_only_skips_the_bundle_install(
    runner, fake_model_manager, monkeypatch, isolated_dirs
):
    """--weights-only never invokes tt-model, even for a real bundle id."""
    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", lambda name: True)
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.download_weights",
        lambda model, config, **kw: Path("/tmp/w"),
    )
    result = runner.invoke(app, ["model", "pull", "ns/bundle", "--weights-only"])
    assert result.exit_code == 0, result.output
    assert not fake_model_manager.exists()


def test_model_pull_offline_does_not_probe_the_hub(runner, monkeypatch, isolated_dirs):
    def boom(name):  # pragma: no cover - must not run
        raise AssertionError("the Hub was probed under --offline")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", boom)
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.download_weights",
        lambda model, config, **kw: Path("/tmp/w"),
    )
    result = runner.invoke(app, ["--offline", "model", "pull", "ns/whatever"])
    assert result.exit_code == 0, result.output


def test_model_pull_unknown_non_repo_name_is_usage(runner, isolated_dirs):
    result = runner.invoke(app, ["model", "pull", "Llama-3.1-8B-Instrukt"])
    assert result.exit_code == ExitCode.USAGE
    assert "Unknown model" in result.output


def test_model_pull_catalog_model_is_unchanged(
    runner, monkeypatch, isolated_dirs
):
    """The spec path must not be disturbed by the new routing."""
    seen = {}
    monkeypatch.setattr(
        "tenstorrent.modelhub.hub.download_weights",
        lambda model, config, **kw: seen.setdefault("repo", model.hf_repo) or Path("/tmp/w"),
    )
    result = runner.invoke(app, ["model", "pull", "Llama-3.1-8B-Instruct", "--json"])
    assert result.exit_code == 0, result.output
    assert seen["repo"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert json.loads(result.output)["hf_repo"] == "meta-llama/Llama-3.1-8B-Instruct"


@pytest.mark.fakes_only
def test_model_pull_bundle_flag_skips_detection(
    runner, fake_model_manager, monkeypatch, isolated_dirs
):
    """--bundle is a claim about the name, so no Hub probe happens — the point is a
    private or unreachable repo that detection cannot answer for."""
    def boom(name):  # pragma: no cover - must not run
        raise AssertionError("the Hub was probed despite --bundle")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", boom)
    result = runner.invoke(app, ["model", "pull", "ns/private", "--bundle"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["pull", "ns/private", "--with-weights"]


@pytest.mark.fakes_only
def test_model_pull_bundle_flag_overrides_a_catalog_match(
    runner, fake_model_manager, isolated_dirs
):
    """A spec entry's hf_repo is also repo-shaped; --bundle says which one is meant."""
    result = runner.invoke(
        app, ["model", "pull", "meta-llama/Llama-3.1-8B-Instruct", "--bundle"]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"][:2] == ["pull", "meta-llama/Llama-3.1-8B-Instruct"]


def test_model_pull_bundle_flag_rejects_a_non_repo_name(runner, isolated_dirs):
    result = runner.invoke(app, ["model", "pull", "Llama-3.1-8B-Instruct", "--bundle"])
    assert result.exit_code == ExitCode.USAGE
    assert "not a bundle id" in result.output


def test_model_pull_rejects_both_direction_flags(runner, isolated_dirs):
    result = runner.invoke(app, ["model", "pull", "ns/x", "--bundle", "--weights-only"])
    assert result.exit_code == ExitCode.USAGE
    assert "opposites" in result.output


@pytest.mark.fakes_only
@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "ns/bundle"],
        ["model", "pull", "ns/bundle", "--bundle"],
        ["model", "stop", "ns/bundle"],
        ["model", "rm", "ns/bundle", "--yes"],
    ],
    ids=["serve", "pull", "stop", "rm"],
)
def test_every_tt_model_call_gets_the_configured_hf_cache(
    runner, fake_model_manager, always_tty, isolated_dirs, tmp_path, argv
):
    """A configured cache must reach teardown too: `tt-model rm --include-weights`
    deletes from whatever cache its own process resolves, so a stop/rm that skipped
    HF_HOME would act on the default cache and orphan the real weights."""
    configured = tmp_path / "elsewhere"
    assert runner.invoke(
        app, ["config", "set", "paths.hf_model_cache_directory", str(configured)]
    ).exit_code == 0
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["hf_home"] == str(configured)


@pytest.mark.fakes_only
def test_model_pull_bundle_flag_honours_offline(
    runner, fake_model_manager, isolated_dirs
):
    """--offline means never download; installing a bundle inherently downloads, and
    tt-model's pull has no local-only mode. Refuse rather than reaching the network."""
    result = runner.invoke(app, ["--offline", "model", "pull", "ns/x", "--bundle"])
    assert result.exit_code == ExitCode.OFFLINE
    assert not fake_model_manager.exists()  # tt-model never invoked


# -- info on a tt-model bundle id --------------------------------------------------
# `tt model info` was the one model verb that rejected an id the community listing,
# `tt model pull` and `tt serve` all accept ("Unknown model … run tt model list").
def _pull_bundle_to_disk(tmp_path, repo_id, manifest):
    """What tt-model leaves behind after `pull`: its index entry and the manifest."""
    root = Path(tmp_path) / "xdg-cache" / "tt-model"  # isolated_dirs' XDG_CACHE_HOME
    pulled = root / "pulled" / repo_id.replace("/", "__")
    pulled.mkdir(parents=True, exist_ok=True)
    (pulled / "tt_kernel_manifest.json").write_text(json.dumps(manifest))
    (root / "installed.json").write_text(
        json.dumps({repo_id: {"repo_id": repo_id, "container": True, "arch": "blackhole"}})
    )


@pytest.mark.fakes_only
def test_model_info_bundle_delegates_to_tt_model_when_installed(
    runner, fake_model_manager, isolated_dirs
):
    """tt-model prints the manifest and the compatibility verdict; tt does not
    reimplement either."""
    result = runner.invoke(app, ["model", "info", "ns/bundle"])
    assert result.exit_code == 0, result.output
    record = json.loads(fake_model_manager.read_text().splitlines()[-1])
    assert record["argv"] == ["info", "ns/bundle"]
    assert record["hf_home"]  # same cache everything else uses


@pytest.mark.fakes_only
def test_model_info_bundle_does_not_install_tt_model(
    runner, uv_bin, monkeypatch, isolated_dirs
):
    """Inspection must not clone-and-build a tool: with tt-model absent the catalog
    row is rendered instead, and uv (which *would* succeed here) never runs."""
    _stub_bundles(monkeypatch, [
        {"name": "ns/bundle", "kind": "container", "engine": "vllm-plugin",
         "arch": ["blackhole"], "downloads": 7, "installed": False},
    ])
    result = runner.invoke(app, ["model", "info", "ns/bundle"])
    assert result.exit_code == 0, result.output
    assert not uv_bin.exists()
    assert "tt-model bundle (container)" in result.output
    assert "blackhole" in result.output and "vllm-plugin" in result.output
    assert "tt model pull ns/bundle" in result.output  # not installed → how to get it
    assert "tt serve ns/bundle --dry-run" in result.output


@pytest.mark.fakes_only
def test_model_info_bundle_json_is_the_catalog_row_even_with_tt_model_installed(
    runner, fake_model_manager, monkeypatch, isolated_dirs
):
    """tt-model's info output is a manifest followed by prose, not one JSON document,
    so --json always carries tt's own contract."""
    _stub_bundles(monkeypatch, [
        {"name": "ns/bundle", "kind": "container", "engine": "vllm-plugin",
         "arch": ["blackhole"], "downloads": 7, "installed": False},
    ])
    result = runner.invoke(app, ["model", "info", "ns/bundle", "--json"])
    assert result.exit_code == 0, result.output
    assert not fake_model_manager.exists()  # tt-model never invoked
    payload = json.loads(result.output)
    assert payload["source"] == "tt-model-catalog"
    assert payload["bundle"]["name"] == "ns/bundle"
    assert payload["bundle"]["engine"] == "vllm-plugin"
    assert payload["in_catalog"] is True
    assert payload["serve"] is None  # not pulled: no manifest on disk
    assert payload["tt_model_installed"] is True


def test_model_info_bundle_matches_the_id_case_insensitively(
    runner, monkeypatch, isolated_dirs
):
    _stub_bundles(monkeypatch, [{"name": "NS/Bundle", "arch": ["blackhole"]}])
    result = runner.invoke(app, ["model", "info", "ns/bundle", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["bundle"]["name"] == "NS/Bundle"


def test_model_info_pulled_bundle_shows_its_launch_settings(
    runner, monkeypatch, tmp_path, isolated_dirs
):
    """Once pulled, the manifest on disk says what `tt serve` will run — the same
    facts `tt serve --dry-run` reports — so info shows them without a Hub fetch."""
    _pull_bundle_to_disk(tmp_path, "ns/dit", {
        "arch": "blackhole", "device_count": 1,
        "tt_metal_version": "0.65.2",
        "container": {
            "kind": "tt-dit-server",
            "image": {"repository": "tt-model/dit", "tag": "tt-model/dit:abc"},
            "serve": {"port": 8000},
            "serve_profiles": [{"name": "p150"}, {"name": "p300"}],
        },
        "weights": {"repo_id": "org/w"},
    })
    _stub_bundles(monkeypatch, [])  # unpublished: the local index is the only source
    result = runner.invoke(app, ["model", "info", "ns/dit"])
    assert result.exit_code == 0, result.output
    assert "installed here" in result.output
    assert "tt-model/dit:abc" in result.output
    assert "tt-dit-server" in result.output
    assert "p150, p300" in result.output
    assert "1 chip" in result.output and "1 chips" not in result.output
    assert "org/w" in result.output


def test_model_info_bundle_offline_reads_only_the_local_index(
    runner, monkeypatch, tmp_path, isolated_dirs
):
    _pull_bundle_to_disk(tmp_path, "ns/dit", {"container": {"kind": "tt-dit-server"}})

    def boom(**kw):  # pragma: no cover - must never run
        raise AssertionError("the Hub was queried under --offline")

    monkeypatch.setattr("tenstorrent.modelhub.bundles.search_community", boom)
    result = runner.invoke(app, ["--offline", "model", "info", "ns/dit", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["bundle"]["source"] == "local"
    assert payload["in_catalog"] is None  # not asked, so neither yes nor no
    assert payload["offline"] is True


def test_model_info_bundle_offline_and_not_installed_is_an_offline_error(
    runner, monkeypatch, isolated_dirs
):
    monkeypatch.setattr(
        "tenstorrent.modelhub.bundles.search_community",
        lambda **kw: pytest.fail("the Hub was queried under --offline"),
    )
    result = runner.invoke(app, ["--offline", "model", "info", "ns/absent"])
    assert result.exit_code == ExitCode.OFFLINE
    assert "tt model pull ns/absent" in result.output


def test_model_info_unlisted_published_bundle_warns_and_shows_the_id(
    runner, monkeypatch, isolated_dirs
):
    """Pushed to the Hub but never `tt-model publish`ed: still a bundle, still
    servable, so info says so rather than calling it unknown."""
    _stub_bundles(monkeypatch, [])
    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", lambda name: True)
    result = runner.invoke(app, ["model", "info", "someone/private", "--json"])
    assert result.exit_code == 0, result.output
    assert "not in the community catalog" in result.output  # the warning
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["bundle"]["name"] == "someone/private"
    assert payload["in_catalog"] is False


def test_model_info_plain_hf_repo_is_not_a_bundle(runner, no_hub_probe, monkeypatch, isolated_dirs):
    _stub_bundles(monkeypatch, [])
    result = runner.invoke(app, ["model", "info", "org/plain-weights"])
    assert result.exit_code == ExitCode.USAGE
    assert "not a tt-model bundle" in result.output
    assert "--weights-only" in result.output


def test_model_info_unreachable_hub_is_not_a_usage_error(runner, monkeypatch, isolated_dirs):
    _stub_bundles(monkeypatch, [])
    monkeypatch.setattr("tenstorrent.modelhub.bundles.is_bundle_repo", lambda name: None)
    result = runner.invoke(app, ["model", "info", "org/maybe"])
    assert result.exit_code == ExitCode.ERROR
    assert "Could not check" in result.output


def test_model_info_catalog_typo_is_still_unknown_model(runner, isolated_dirs):
    """Only a Hub-shaped name may fall through to the bundle path; a spec typo
    keeps the catalog error it always had."""
    result = runner.invoke(app, ["model", "info", "Llama-3.1-8B-Instrukt"])
    assert result.exit_code == ExitCode.USAGE
    assert "Unknown model" in result.output and "tt model list" in result.output


def test_model_info_spec_hf_repo_alias_still_wins_over_the_bundle_path(
    runner, fake_model_manager, isolated_dirs
):
    """A spec entry's hf_repo is also namespace/name; the spec wins, as for serve."""
    result = runner.invoke(app, ["model", "info", "meta-llama/Llama-3.1-8B-Instruct", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "Llama-3.1-8B-Instruct"
    assert not fake_model_manager.exists()
