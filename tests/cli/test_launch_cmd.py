# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode, TTError


LAUNCH_SUPPORT = (
    Path(__file__).parent.parent / "fakes" / "data" / "model_support_launch.json"
)
FAKE_BIN_DIR = Path(__file__).parent.parent / "fakes" / "bin"


@pytest.fixture
def fake_docker_containers(monkeypatch):
    """Point both serving backends' docker lookup at the fake docker binary, for
    the no-port/--url discovery path that reads ports off running containers."""
    for module in ("inference_server", "model_manager"):
        monkeypatch.setattr(
            f"tenstorrent.backends.serving.{module}.shutil.which",
            lambda name, _bin=str(FAKE_BIN_DIR / "docker"): _bin if name == "docker" else None,
        )

    def set_containers(entries):
        monkeypatch.setenv("FAKE_DOCKER_CONTAINERS", json.dumps(entries))

    set_containers([])
    return set_containers


@pytest.fixture(autouse=True)
def launch_support(monkeypatch):
    """Pin the catalog to the launch fixture: these tests assert on exact verdicts."""
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(LAUNCH_SUPPORT))


@pytest.fixture(autouse=True)
def approve_prompts(monkeypatch):
    """Answer yes at the confirmation. Tests for the prompt itself patch these back."""
    monkeypatch.setattr("tenstorrent.commands.launch._stdin_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: True)


@pytest.fixture(autouse=True)
def container_side_effects(monkeypatch):
    """Ports are free and containers answer at once, so no test waits on a poll."""
    monkeypatch.setattr("tenstorrent.launchers.container.port_is_free", lambda port: True)
    monkeypatch.setattr(
        "tenstorrent.launchers.container.wait_until_ready", lambda url, timeout_s: True
    )


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    """An installed `opencode`, reached through the TT_TOOL_BIN_* seam."""
    exe = tmp_path / "opencode"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_OPENCODE", str(exe))
    return exe


@pytest.fixture(autouse=True)
def execed(monkeypatch):
    """Capture the exec hand-off instead of replacing the test process.

    Autouse deliberately: a test that reaches hand-off without this would exec the
    fake client over the pytest process, ending the run with no failure reported.
    """
    calls: list[list[str]] = []

    def fake_execvpe(file, argv, env):
        calls.append(list(argv))
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    return calls


def opencode_config() -> Path:
    import os

    return Path(os.environ["XDG_CONFIG_HOME"]) / "opencode" / "opencode.json"


@pytest.mark.fakes_only
def test_configures_and_hands_over(runner, served, fake_client, execed):
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "opencode", "--url", base])
    assert result.exit_code == 0, result.output
    # The served id is what the API expects (an HF repo here), not the short name.
    assert execed == [[str(fake_client), "--model", "tenstorrent/Qwen/Qwen3-32B"]]
    doc = json.loads(opencode_config().read_text())
    provider = doc["provider"]["tenstorrent"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == base
    assert provider["models"] == {
        "Qwen/Qwen3-32B": {
            "name": "Qwen/Qwen3-32B",
            "limit": {"context": 131072, "output": 32768},
        }
    }


@pytest.mark.fakes_only
def test_dry_run_changes_nothing_and_needs_no_client(runner, served, execed):
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "opencode", "--url", base, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not opencode_config().exists()
    assert execed == []
    assert "@ai-sdk/openai-compatible" in result.output


@pytest.mark.fakes_only
def test_existing_config_is_merged_not_replaced(runner, served, fake_client, execed):
    config = opencode_config()
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "theme": "tokyonight",
                "provider": {
                    "tenstorrent": {"models": {"Old-Model": {"name": "Old-Model"}}},
                    "other": {"name": "Someone else"},
                },
            }
        )
    )
    result = runner.invoke(
        app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")]
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(config.read_text())
    assert doc["theme"] == "tokyonight"  # unrelated settings survive
    assert doc["provider"]["other"] == {"name": "Someone else"}
    assert set(doc["provider"]["tenstorrent"]["models"]) == {
        "Old-Model",
        "Qwen/Qwen3-32B",
    }


@pytest.mark.fakes_only
def test_refuses_a_model_without_tool_calling(runner, served, fake_client, execed):
    base = served("mistralai/Mistral-7B-Instruct-v0.3")
    result = runner.invoke(app, ["launch", "opencode", "--url", base])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "cannot do tool calling" in result.output
    # The refusal names a model that can, rather than only saying no.
    assert "Qwen3-32B" in result.output
    assert not opencode_config().exists()
    assert execed == []


@pytest.mark.fakes_only
def test_force_configures_despite_no_tool_calling(runner, served, fake_client, execed):
    base = served("mistralai/Mistral-7B-Instruct-v0.3")
    result = runner.invoke(app, ["launch", "opencode", "--url", base, "--force"])
    assert result.exit_code == 0, result.output
    assert opencode_config().exists()
    assert execed


@pytest.mark.fakes_only
def test_refuses_a_model_that_is_not_a_language_model(runner, served, fake_client):
    base = served("openai/whisper-large-v3")
    result = runner.invoke(app, ["launch", "opencode", "--url", base])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "not a language model" in result.output


@pytest.mark.fakes_only
def test_unknown_served_id_warns_but_proceeds(runner, served, fake_client, execed):
    """A tt-model bundle is not in the catalog, so capability cannot be checked."""
    base = served("acme/some-bundle")
    result = runner.invoke(app, ["launch", "opencode", "--url", base])
    assert result.exit_code == 0, result.output
    assert "not in the model catalog" in result.output
    assert execed


@pytest.mark.fakes_only
def test_selects_among_several_served_models(runner, served, fake_client, execed):
    base = served("openai/whisper-large-v3", "Qwen/Qwen3-32B")
    result = runner.invoke(
        app, ["launch", "opencode", "--url", base, "--model", "Qwen3-32B"]
    )
    assert result.exit_code == 0, result.output
    assert execed == [[str(fake_client), "--model", "tenstorrent/Qwen/Qwen3-32B"]]


@pytest.mark.fakes_only
def test_model_that_is_not_served_is_a_usage_error(runner, served, fake_client):
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(
        app, ["launch", "opencode", "--url", base, "--model", "Llama-3.1-8B-Instruct"]
    )
    assert result.exit_code == ExitCode.USAGE
    assert "is not being served" in result.output


@pytest.mark.fakes_only
def test_missing_client_points_at_its_own_installer(runner, served, monkeypatch):
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name, **_: None)
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: None)
    result = runner.invoke(app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "opencode.ai" in result.output
    assert "tt update" not in result.output


@pytest.mark.fakes_only
def test_no_server_says_how_to_start_one(runner, fake_client):
    # Port 1 is privileged and never listening; discovery must fail cleanly.
    result = runner.invoke(app, ["launch", "opencode", "--port", "1"])
    assert result.exit_code == ExitCode.ERROR
    assert "tt serve" in result.output


@pytest.mark.fakes_only
def test_launch_finds_a_model_via_its_inference_server_container_port(
    runner, served, fake_client, execed, fake_docker_containers
):
    """No --port/--url: the port comes off the running container's own published
    port, exactly like `tt model stop` finds a container to stop."""
    base = served("Qwen/Qwen3-32B")
    port = base.rsplit(":", 1)[1].split("/")[0]
    fake_docker_containers([
        {
            "Id": "aaaaaaaaaaaa",
            "Name": "/tt-inference-server-aaaa",
            "Config": {"Image": "img:1"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": port}]}},
        }
    ])
    result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0, result.output
    doc = json.loads(opencode_config().read_text())
    assert doc["provider"]["tenstorrent"]["options"]["baseURL"] == base


@pytest.mark.fakes_only
def test_launch_finds_a_model_via_its_tt_model_container_port(
    runner, served, fake_client, execed, fake_docker_containers
):
    """Same, for a tt-model bundle: its container carries the org.tenstorrent.tt-model
    label instead, and publishes host:container on the same port number."""
    base = served("Qwen/Qwen3-32B")
    port = base.rsplit(":", 1)[1].split("/")[0]
    fake_docker_containers([
        {
            "Id": "bbbbbbbbbbbb",
            "Name": "/tt-model-demo-default",
            "Labels": {"org.tenstorrent.tt-model": "demo"},
            "Ports": f"0.0.0.0:{port}->{port}/tcp",
        }
    ])
    result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0, result.output
    doc = json.loads(opencode_config().read_text())
    assert doc["provider"]["tenstorrent"]["options"]["baseURL"] == base


@pytest.mark.fakes_only
def test_launch_selects_a_model_from_a_container_other_than_the_first_found(
    runner, served, fake_client, execed, fake_docker_containers
):
    """Two containers running at once: --model must be able to find the one
    that isn't whichever port discovery happens to reach first."""
    first = served("mistralai/Mistral-7B-Instruct-v0.3")
    second = served("Qwen/Qwen3-32B")
    first_port = first.rsplit(":", 1)[1].split("/")[0]
    second_port = second.rsplit(":", 1)[1].split("/")[0]
    fake_docker_containers([
        {
            "Id": "aaaaaaaaaaaa",
            "Name": "/tt-inference-server-aaaa",
            "Config": {"Image": "img:1"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": first_port}]}},
        },
        {
            "Id": "cccccccccccc",
            "Name": "/tt-inference-server-cccc",
            "Config": {"Image": "img:1"},
            "Mounts": [],
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": second_port}]}},
        },
    ])
    result = runner.invoke(app, ["launch", "opencode", "--model", "Qwen/Qwen3-32B"])
    assert result.exit_code == 0, result.output
    doc = json.loads(opencode_config().read_text())
    assert doc["provider"]["tenstorrent"]["options"]["baseURL"] == second


@pytest.mark.fakes_only
def test_launch_falls_back_to_the_default_port_when_docker_finds_nothing(
    runner, fake_client, fake_docker_containers, monkeypatch
):
    """No matching container (or no docker at all) must not crash discovery — it
    falls back to tt's plain default port. `discover` is faked rather than relying
    on nothing actually listening on 20000, which the host running this suite
    cannot guarantee (e.g. a real `tt serve` left running on its own default)."""
    attempted = []

    def fake_discover(base_url):
        attempted.append(base_url)
        raise TTError("no server", exit_code=ExitCode.ERROR)

    monkeypatch.setattr("tenstorrent.commands.launch.discover", fake_discover)
    result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == ExitCode.ERROR
    assert attempted == ["http://127.0.0.1:20000/v1"]


@pytest.mark.fakes_only
def test_launch_falls_back_when_docker_itself_is_not_installed(
    runner, fake_client, monkeypatch
):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )
    monkeypatch.setattr(
        "tenstorrent.backends.serving.model_manager.shutil.which", lambda name: None
    )
    attempted = []

    def fake_discover(base_url):
        attempted.append(base_url)
        raise TTError("no server", exit_code=ExitCode.ERROR)

    monkeypatch.setattr("tenstorrent.commands.launch.discover", fake_discover)
    result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == ExitCode.ERROR
    assert attempted == ["http://127.0.0.1:20000/v1"]


@pytest.mark.fakes_only
def test_unparseable_client_config_is_never_overwritten(runner, served, fake_client):
    config = opencode_config()
    config.parent.mkdir(parents=True)
    config.write_text("{ not json ")
    result = runner.invoke(app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == ExitCode.CONFIG
    assert config.read_text() == "{ not json "


@pytest.mark.fakes_only
def test_a_json_config_that_is_not_an_object_is_a_clean_error(runner, served, fake_client):
    """Valid JSON, wrong shape: must be a CONFIG error, not an AttributeError crash."""
    config = opencode_config()
    config.parent.mkdir(parents=True)
    config.write_text("[]")
    result = runner.invoke(app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == ExitCode.CONFIG
    assert "not a JSON object" in result.output
    assert config.read_text() == "[]"


@pytest.mark.fakes_only
def test_url_and_port_together_is_a_usage_error(runner):
    result = runner.invoke(
        app, ["launch", "opencode", "--url", "http://x/v1", "--port", "8000"]
    )
    assert result.exit_code == ExitCode.USAGE


@pytest.mark.fakes_only
def test_unknown_client_is_a_usage_error(runner):
    """An unknown client is now an unknown subcommand, so click rejects it."""
    result = runner.invoke(app, ["launch", "notacoder"])
    assert result.exit_code == ExitCode.USAGE


@pytest.mark.fakes_only
def test_unknown_client_for_a_subcommand_names_the_supported_ones(runner):
    result = runner.invoke(app, ["launch", "stop", "notacoder"])
    assert result.exit_code == ExitCode.USAGE
    assert "opencode" in result.output


@pytest.mark.fakes_only
def test_json_mode_emits_the_payload_and_does_not_exec(
    runner, served, fake_client, execed
):
    base = served("Qwen/Qwen3-32B")
    # --json cannot ask, so a config edit needs --yes; without it, see
    # test_json_mode_without_yes_refuses_to_edit_the_config.
    result = runner.invoke(app, ["launch", "opencode", "--url", base, "--json", "--yes"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["model"] == "Qwen/Qwen3-32B"
    assert payload["catalog_name"] == "Qwen3-32B"
    assert payload["tool_call_parser"] == "hermes"
    assert payload["base_url"] == base
    assert payload["applied"] is True
    assert execed == []


@pytest.mark.fakes_only
def test_no_exec_writes_config_only(runner, served, fake_client, execed):
    result = runner.invoke(
        app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B"), "--no-exec"]
    )
    assert result.exit_code == 0, result.output
    assert opencode_config().exists()
    assert execed == []


# -- openwebui: a service, pulled and run only with consent ----------------------
@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    """A `docker` whose argv is recorded. `inspect` fails, i.e. no container yet."""
    log = tmp_path / "docker-argv.jsonl"
    exe = tmp_path / "docker"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_DOCKER_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if sys.argv[1:2] == ['inspect'] else 0)\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_OPENWEBUI", str(exe))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    return log




@pytest.mark.fakes_only
def test_openwebui_pulls_and_runs_pointed_at_the_host(
    runner, served, fake_docker
):
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(
        app, ["launch", "openwebui", "--url", base, "--web-port", "3080"]
    )
    assert result.exit_code == 0, result.output
    calls = [json.loads(line) for line in fake_docker.read_text().splitlines()]
    assert calls[0][:1] == ["inspect"]
    assert calls[1] == ["pull", "ghcr.io/open-webui/open-webui:main"]
    run = calls[2]
    assert run[:2] == ["run", "-d"]
    assert "3080:8080" in run
    # 127.0.0.1 would be Open WebUI itself inside the container.
    assert "OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1" not in run
    port = base.rsplit(":", 1)[1].split("/")[0]
    assert f"OPENAI_API_BASE_URL=http://host.docker.internal:{port}/v1" in run
    assert "http://localhost:3080" in result.output


@pytest.mark.fakes_only
def test_openwebui_does_not_need_tool_calling(
    runner, served, fake_docker
):
    """A chat-only client works with a model opencode would be refused for."""
    base = served("mistralai/Mistral-7B-Instruct-v0.3")
    result = runner.invoke(app, ["launch", "openwebui", "--url", base])
    assert result.exit_code == 0, result.output


@pytest.mark.fakes_only
def test_openwebui_declining_the_prompt_starts_nothing(
    runner, served, fake_docker, monkeypatch
):
    monkeypatch.setattr("tenstorrent.commands.launch._stdin_isatty", lambda: True)
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: False)
    result = runner.invoke(app, ["launch", "openwebui", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == ExitCode.OK
    calls = [json.loads(line) for line in fake_docker.read_text().splitlines()]
    assert [c for c in calls if c[:1] in (["pull"], ["run"])] == []


@pytest.mark.fakes_only
def test_openwebui_non_interactive_needs_yes(runner, served, fake_docker, monkeypatch):
    monkeypatch.setattr("tenstorrent.commands.launch._stdin_isatty", lambda: False)
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "openwebui", "--url", base])
    assert result.exit_code == ExitCode.USAGE
    assert "--yes" in result.output
    assert runner.invoke(app, ["launch", "openwebui", "--url", base, "--yes"]).exit_code == 0


@pytest.mark.fakes_only
def test_openwebui_dry_run_needs_no_container_runtime(runner, served, monkeypatch):
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name, **_: None)
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: None)
    result = runner.invoke(
        app, ["launch", "openwebui", "--url", served("Qwen/Qwen3-32B"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "docker pull" in result.output.replace("\n", " ")


@pytest.mark.fakes_only
def test_anythingllm_runs_its_own_container(runner, served, monkeypatch, tmp_path):
    log = tmp_path / "docker-argv.jsonl"
    exe = tmp_path / "docker"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_DOCKER_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if sys.argv[1:2] == ['inspect'] else 0)\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_ANYTHINGLLM", str(exe))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    result = runner.invoke(app, ["launch", "anythingllm", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == 0, result.output
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[1] == ["pull", "mintplexlabs/anythingllm:latest"]
    run = calls[2]
    assert run[run.index("--name") + 1] == "tt-anythingllm"
    assert "GENERIC_OPEN_AI_MODEL_PREF=Qwen/Qwen3-32B" in run


# -- pi and aider: installed clients, like opencode -------------------------------
@pytest.fixture
def fake_client_named(monkeypatch, tmp_path):
    def make(tool: str):
        exe = tmp_path / tool
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        monkeypatch.setenv(f"TT_TOOL_BIN_{tool.upper()}", str(exe))
        return exe

    return make


@pytest.mark.fakes_only
def test_pi_writes_its_models_json_and_hands_over(
    runner, served, fake_client_named, execed, isolated_dirs
):
    exe = fake_client_named("pi")
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "pi", "--url", base])
    assert result.exit_code == 0, result.output
    assert execed == [[str(exe), "--provider", "tenstorrent", "--model", "Qwen/Qwen3-32B"]]
    doc = json.loads((isolated_dirs / "home" / ".pi" / "agent" / "models.json").read_text())
    provider = doc["providers"]["tenstorrent"]
    assert provider["baseUrl"] == base
    assert provider["api"] == "openai-completions"
    # Qwen3-32B publishes a reasoning parser, so pi is told the model thinks.
    assert provider["models"][0]["reasoning"] is True


@pytest.mark.fakes_only
def test_aider_passes_the_endpoint_in_the_environment(
    runner, served, fake_client_named, monkeypatch
):
    exe = fake_client_named("aider")
    passed: dict = {}

    def fake_execvpe(file, argv, env):
        passed["argv"] = list(argv)
        passed["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "aider", "--url", base])
    assert result.exit_code == 0, result.output
    assert passed["argv"] == [str(exe), "--model", "openai/Qwen/Qwen3-32B"]
    assert passed["env"]["OPENAI_API_BASE"] == base
    # exec_tty merges rather than replaces, so the child keeps the real environment.
    assert "PATH" in passed["env"]


@pytest.mark.fakes_only
def test_qwencode_passes_the_endpoint_as_cli_arguments(
    runner, served, fake_client_named, execed
):
    exe = fake_client_named("qwencode")
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "qwencode", "--url", base])
    assert result.exit_code == 0, result.output
    assert execed == [
        [
            str(exe),
            "--auth-type",
            "openai",
            "--openai-api-key",
            "tt-local",
            "--openai-base-url",
            base,
            "--model",
            "Qwen/Qwen3-32B",
        ]
    ]


@pytest.mark.fakes_only
def test_hermes_passes_the_endpoint_in_the_environment(
    runner, served, fake_client_named, monkeypatch
):
    exe = fake_client_named("hermes")
    passed: dict = {}

    def fake_execvpe(file, argv, env):
        passed["argv"] = list(argv)
        passed["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "hermes", "--url", base])
    assert result.exit_code == 0, result.output
    assert passed["argv"] == [
        str(exe), "chat", "--provider", "openai", "--model", "Qwen/Qwen3-32B"
    ]
    assert passed["env"]["OPENAI_BASE_URL"] == base
    assert passed["env"]["OPENAI_API_KEY"] == "tt-local"


@pytest.mark.fakes_only
def test_hermes_web_starts_the_dashboard_on_the_same_endpoint(
    runner, served, fake_client_named, monkeypatch
):
    exe = fake_client_named("hermes")
    passed: dict = {}

    def fake_execvpe(file, argv, env):
        passed["argv"] = list(argv)
        passed["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("tenstorrent.tools.runner.os.execvpe", fake_execvpe)
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(
        app, ["launch", "hermes", "--url", base, "--web", "--web-port", "9200"]
    )
    assert result.exit_code == 0, result.output
    assert passed["argv"] == [str(exe), "dashboard", "--port", "9200"]
    assert passed["env"]["OPENAI_BASE_URL"] == base
    assert passed["env"]["HERMES_TUI_PROVIDER"] == "openai"
    assert passed["env"]["HERMES_MODEL"] == "Qwen/Qwen3-32B"


@pytest.mark.fakes_only
def test_web_is_refused_for_a_client_without_a_web_ui(
    runner, served, fake_client_named
):
    fake_client_named("aider")
    result = runner.invoke(app, ["launch", "aider", "--url", served("Qwen/Qwen3-32B"), "--web"])
    assert result.exit_code == ExitCode.USAGE
    assert "no web UI" in result.output


@pytest.mark.fakes_only
def test_pi_refuses_a_model_without_tool_calling(runner, served, fake_client_named):
    fake_client_named("pi")
    base = served("mistralai/Mistral-7B-Instruct-v0.3")
    result = runner.invoke(app, ["launch", "pi", "--url", base])
    assert result.exit_code == ExitCode.UNSUPPORTED
    assert "cannot do tool calling" in result.output


@pytest.mark.fakes_only
def test_group_help_lists_every_client(runner):
    result = runner.invoke(app, ["launch", "--help"])
    assert result.exit_code == 0
    for tool in (
        "opencode", "pi", "aider", "qwencode", "hermes", "openwebui", "anythingllm",
        "deep-research",
    ):
        assert tool in result.output
    for subcommand in ("list", "stop", "disconnect"):
        assert subcommand in result.output


# -- listing and teardown ---------------------------------------------------------
@pytest.mark.fakes_only
def test_list_shows_every_client_without_a_server(runner, monkeypatch):
    """--list must work with nothing serving and nothing installed."""
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name, **_: None)
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: None)
    result = runner.invoke(app, ["launch", "list"])
    assert result.exit_code == 0, result.output
    for tool in (
        "opencode", "pi", "aider", "qwencode", "hermes", "openwebui", "anythingllm",
        "deep-research",
    ):
        assert tool in result.output
    # Rich wraps the column, so match on a fragment rather than the whole phrase.
    assert "needs docker" in result.output.replace("\n", " ")


@pytest.mark.fakes_only
def test_list_json_reports_availability(runner, fake_client_named, monkeypatch):
    fake_client_named("opencode")
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name, **_: None)
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: None)
    result = runner.invoke(app, ["launch", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["tool"]: row for row in json.loads(result.stdout)}
    assert rows["opencode"]["available"] is True
    assert rows["aider"]["available"] is False
    assert rows["openwebui"]["kind"] == "web service"
    assert rows["aider"]["requires_tool_calling"] is True
    assert rows["anythingllm"]["requires_tool_calling"] is False




@pytest.mark.fakes_only
def test_stop_stops_a_running_container(runner, monkeypatch, tmp_path):
    log = tmp_path / "docker.jsonl"
    exe = tmp_path / "docker"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_DOCKER_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:2] == ['inspect']:\n"
        "    print(json.dumps({'State': {'Running': True}, 'Config': {'Env': []}}))\n"
        "sys.exit(0)\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_OPENWEBUI", str(exe))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    result = runner.invoke(app, ["launch", "stop", "openwebui"])
    assert result.exit_code == 0, result.output
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    # stop, not rm: the volume holds the user's chats.
    assert ["stop", "tt-open-webui"] in calls
    assert not [c for c in calls if c[:1] == ["rm"]]


@pytest.mark.fakes_only
def test_stop_is_a_usage_error_for_a_terminal_client(runner, fake_client_named):
    fake_client_named("opencode")
    result = runner.invoke(app, ["launch", "stop", "opencode"])
    assert result.exit_code == ExitCode.USAGE
    assert "not something tt runs" in result.output


def _fake_hermes(monkeypatch, tmp_path, status: str):
    log = tmp_path / "hermes.jsonl"
    exe = tmp_path / "hermes"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_HERMES_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if '--status' in sys.argv:\n"
        f"    print({status!r})\n"
        "sys.exit(0)\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_HERMES", str(exe))
    monkeypatch.setenv("FAKE_HERMES_LOG", str(log))
    return log


@pytest.mark.fakes_only
def test_stop_stops_a_running_hermes_dashboard(runner, monkeypatch, tmp_path):
    log = _fake_hermes(monkeypatch, tmp_path, "dashboard pid 4242 on port 9119")
    result = runner.invoke(app, ["launch", "stop", "hermes"])
    assert result.exit_code == 0, result.output
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ["dashboard", "--stop"] in calls


@pytest.mark.fakes_only
def test_stop_leaves_hermes_alone_when_no_dashboard_runs(runner, monkeypatch, tmp_path):
    log = _fake_hermes(monkeypatch, tmp_path, "No hermes dashboard or serve processes running.")
    result = runner.invoke(app, ["launch", "stop", "hermes"])
    assert result.exit_code == 0, result.output
    assert "not running" in result.output
    assert ["dashboard", "--stop"] not in [json.loads(l) for l in log.read_text().splitlines()]


@pytest.mark.fakes_only
def test_stop_does_not_need_a_model_server(runner, monkeypatch, tmp_path):
    """Teardown must work after the served model is long gone."""
    exe = tmp_path / "docker"
    exe.write_text("#!/bin/sh\nexit 1\n")  # inspect fails: no container
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_ANYTHINGLLM", str(exe))
    result = runner.invoke(app, ["launch", "stop", "anythingllm"])
    assert result.exit_code == 0, result.output
    assert "not running" in result.output


# -- asking before editing a config tt does not own -------------------------------
@pytest.mark.fakes_only
def test_the_prompt_names_the_file_it_would_edit(runner, served, fake_client, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(
        "tenstorrent.commands.launch.confirm", lambda q: asked.append(q) or True
    )
    result = runner.invoke(app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert str(opencode_config()) in asked[0]
    assert "provider.tenstorrent" in asked[0]


@pytest.mark.fakes_only
def test_declining_leaves_the_config_untouched(runner, served, fake_client, monkeypatch, execed):
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: False)
    result = runner.invoke(app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B")])
    assert result.exit_code == ExitCode.OK  # declining is not a failure
    assert not opencode_config().exists()
    assert execed == []


@pytest.mark.fakes_only
def test_no_prompt_when_the_entry_is_already_what_we_would_write(
    runner, served, fake_client, monkeypatch, execed
):
    """Re-launching an already-configured client must not nag."""
    base = served("Qwen/Qwen3-32B")
    assert runner.invoke(app, ["launch", "opencode", "--url", base]).exit_code == 0
    asked: list[str] = []
    monkeypatch.setattr(
        "tenstorrent.commands.launch.confirm", lambda q: asked.append(q) or True
    )
    result = runner.invoke(app, ["launch", "opencode", "--url", base])
    assert result.exit_code == 0, result.output
    assert asked == []


@pytest.mark.fakes_only
def test_json_mode_without_yes_refuses_to_edit_the_config(runner, served, fake_client):
    base = served("Qwen/Qwen3-32B")
    result = runner.invoke(app, ["launch", "opencode", "--url", base, "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert "--yes" in result.output
    assert not opencode_config().exists()


@pytest.mark.fakes_only
def test_dry_run_never_asks(runner, served, monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.commands.launch.confirm",
        lambda _: pytest.fail("a dry run must not ask"),
    )
    result = runner.invoke(
        app, ["launch", "opencode", "--url", served("Qwen/Qwen3-32B"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output


# -- disconnect -------------------------------------------------------------------
@pytest.mark.fakes_only
def test_disconnect_removes_only_our_block(runner, served, fake_client):
    base = served("Qwen/Qwen3-32B")
    assert runner.invoke(app, ["launch", "opencode", "--url", base]).exit_code == 0
    config = opencode_config()
    doc = json.loads(config.read_text())
    doc["theme"] = "tokyonight"
    doc["provider"]["other"] = {"name": "Someone else"}
    config.write_text(json.dumps(doc))

    result = runner.invoke(app, ["launch", "disconnect", "opencode"])
    assert result.exit_code == 0, result.output
    doc = json.loads(config.read_text())
    assert "tenstorrent" not in doc["provider"]
    assert doc["provider"]["other"] == {"name": "Someone else"}
    assert doc["theme"] == "tokyonight"


@pytest.mark.fakes_only
def test_disconnect_with_nothing_configured_is_not_an_error(runner, fake_client):
    result = runner.invoke(app, ["launch", "disconnect", "opencode"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output


@pytest.mark.fakes_only
def test_disconnect_dry_run_changes_nothing(runner, served, fake_client):
    base = served("Qwen/Qwen3-32B")
    assert runner.invoke(app, ["launch", "opencode", "--url", base]).exit_code == 0
    result = runner.invoke(app, ["launch", "disconnect", "opencode", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would undo" in result.output
    assert "tenstorrent" in json.loads(opencode_config().read_text())["provider"]


@pytest.mark.fakes_only
def test_disconnect_asks_first(runner, served, fake_client, monkeypatch):
    base = served("Qwen/Qwen3-32B")
    assert runner.invoke(app, ["launch", "opencode", "--url", base]).exit_code == 0
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: False)
    result = runner.invoke(app, ["launch", "disconnect", "opencode"])
    assert result.exit_code == ExitCode.OK
    assert "tenstorrent" in json.loads(opencode_config().read_text())["provider"]


@pytest.mark.fakes_only
def test_disconnect_aider_has_nothing_to_undo(runner, fake_client_named):
    fake_client_named("aider")
    result = runner.invoke(app, ["launch", "disconnect", "aider"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output


@pytest.mark.fakes_only
def test_disconnect_removes_a_container_but_not_its_volume(runner, monkeypatch, tmp_path):
    log = tmp_path / "docker.jsonl"
    exe = tmp_path / "docker"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_DOCKER_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:2] == ['inspect']:\n"
        "    print(json.dumps({'State': {'Running': True}, 'Config': {'Env': []}}))\n"
        "sys.exit(0)\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_OPENWEBUI", str(exe))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    result = runner.invoke(app, ["launch", "disconnect", "openwebui"])
    assert result.exit_code == 0, result.output
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ["stop", "tt-open-webui"] in calls  # stopped before removal
    assert ["rm", "tt-open-webui"] in calls
    # The volume is the user's data and is never touched.
    assert not [c for c in calls if c[:1] == ["volume"]]
    assert not [c for c in calls if "-f" in c]


@pytest.mark.fakes_only
def test_bare_launch_shows_help(runner):
    result = runner.invoke(app, ["launch"])
    assert result.exit_code == 0
    assert "opencode" in result.output


# -- a server that is not up yet --------------------------------------------------
@pytest.mark.fakes_only
def test_no_server_blames_loading_first_because_serve_returns_early(runner, fake_client):
    """`tt serve` returns once the container is listed, minutes before a large model
    answers, so a still-loading server is the likelier cause than a missing one."""
    result = runner.invoke(app, ["launch", "opencode", "--port", "1"])
    assert result.exit_code == ExitCode.ERROR
    flat = result.output.replace("\n", " ")
    assert "still loading" in flat
    assert "docker logs" in flat


# -- deep-research: started from a local checkout ---------------------------------
COHOST = "raahemnabeel/cohost-ornith-35b-qwen3vl"


@pytest.fixture
def deep_research(monkeypatch, tmp_path, fake_docker_containers, model_manager_bin):
    """A checkout, a fake `uv` that records what it ran and where, the three
    endpoints answering (`serve.answering`; None: only once tt-model has served),
    and `serve(*bundles, app_running=False)` to set what docker reports."""
    from tenstorrent.launchers.base import RunningModel

    checkout = tmp_path / "tt-deep-research"
    (checkout / "recipes").mkdir(parents=True)
    (checkout / "recipe.toml").write_text("")
    for recipe in ("ornith-35b-cohost", "rerank"):
        (checkout / "recipes" / f"{recipe}.toml").write_text("")
    log = tmp_path / "uv.jsonl"
    exe = tmp_path / "uv"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['FAKE_UV_LOG'], 'a').write(\n"
        "    json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("TT_TOOL_BIN_DEEP_RESEARCH", str(exe))
    monkeypatch.setenv("FAKE_UV_LOG", str(log))
    llm = RunningModel(
        served_id="ornith-ai/Ornith-1.5-35B-A3B", base_url="http://127.0.0.1:8000/v1"
    )
    monkeypatch.setattr("tenstorrent.launchers.apps.deep_research.probe", lambda url, **_: (
        [llm] if serve.answering or (serve.answering is None and tt_model_calls()) else None
    ))
    monkeypatch.setattr("tenstorrent.launchers.apps.deep_research.port_is_free", lambda p: True)
    # Nothing answers on the cohost's status port: readiness comes from the endpoints.
    monkeypatch.setattr(
        "tenstorrent.launchers.apps.deep_research.STATUS_URL", "http://127.0.0.1:1"
    )
    monkeypatch.setattr("tenstorrent.launchers.apps.deep_research._POLL_S", 0)

    def tt_model_calls() -> list[list[str]]:
        if not model_manager_bin.exists():
            return []
        return [json.loads(line)["argv"] for line in model_manager_bin.read_text().splitlines()]

    def serve(*bundles: str, app_running: bool = False):
        containers = [
            {"Id": f"{i:012d}", "Labels": {
                "org.tenstorrent.tt-model": bundle.split("/")[1],
                "org.tenstorrent.tt-model.repo": bundle,
            }}
            for i, bundle in enumerate(bundles)
        ]
        if app_running:
            containers.append({"Id": "c0ffee000000", "Labels": {
                "com.docker.compose.project": "ttdr",
                "com.docker.compose.service": "ui-deepagents",
                "com.docker.compose.project.working_dir": str(checkout),
            }})
        fake_docker_containers(containers)

    def calls() -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    serve.checkout, serve.calls, serve.tt_model_calls = checkout, calls, tt_model_calls
    serve.answering = True
    return serve


@pytest.mark.fakes_only
def test_deep_research_runs_ttdr_up_with_the_recipe_for_the_served_bundle(runner, deep_research):
    deep_research(COHOST)
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert deep_research.calls() == [{
        "argv": ["run", "ttdr", "up", "--recipe", "recipes/ornith-35b-cohost.toml"],
        "cwd": str(deep_research.checkout),
    }]
    assert "http://localhost:3000" in result.output


@pytest.mark.fakes_only
def test_deep_research_reads_its_checkout_from_config(runner, deep_research):
    deep_research(COHOST)
    set_result = runner.invoke(
        app, ["config", "set", "launch.deep_research_path", str(deep_research.checkout)]
    )
    assert set_result.exit_code == 0, set_result.output
    result = runner.invoke(app, ["launch", "deep-research", "--yes"])
    assert result.exit_code == 0, result.output
    assert deep_research.calls()[0]["cwd"] == str(deep_research.checkout)


@pytest.mark.fakes_only
def test_deep_research_serves_its_default_bundle_when_none_is_running(runner, deep_research):
    deep_research()
    deep_research.answering = None
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert deep_research.tt_model_calls() == [
        ["serve", COHOST, "--local-only", "--port", "8000", "--detach"]
    ]
    assert deep_research.calls() == [{
        "argv": ["run", "ttdr", "up", "--recipe", "recipes/ornith-35b-cohost.toml"],
        "cwd": str(deep_research.checkout),
    }]
    assert "http://localhost:3000" in result.output


@pytest.mark.fakes_only
def test_deep_research_asks_once_and_warns_a_cold_start_is_slow(
    runner, deep_research, monkeypatch
):
    deep_research()
    deep_research.answering = None
    asked: list[str] = []
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda q: asked.append(q) or True)
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert COHOST in asked[0] and "10-15 minutes" in asked[0]


@pytest.mark.fakes_only
def test_deep_research_declining_serves_nothing(runner, deep_research, monkeypatch):
    deep_research()
    deep_research.answering = None
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: False)
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert deep_research.tt_model_calls() == [] and deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_no_serve_says_how_to_serve_it(runner, deep_research):
    deep_research()
    deep_research.answering = None
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout), "--no-serve"]
    )
    assert result.exit_code == ExitCode.ERROR
    assert "cohost-ornith-35b-qwen3vl" in result.output
    assert deep_research.tt_model_calls() == [] and deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_does_not_serve_over_another_model(runner, deep_research):
    deep_research()  # nothing of its own running, but port 8000 answers
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == ExitCode.ERROR
    assert "will not serve over it" in result.output.replace("\n", " ")
    assert deep_research.tt_model_calls() == [] and deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_reports_a_bundle_that_exits_while_booting(runner, deep_research):
    deep_research()
    deep_research.answering = False  # never ready, and docker never lists it
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout), "--yes"]
    )
    assert result.exit_code == ExitCode.TOOL_FAILED
    assert "tt model logs" in result.output
    assert deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_dry_run_shows_the_serve_and_runs_nothing(runner, deep_research):
    deep_research()
    deep_research.answering = None
    result = runner.invoke(app, [
        "launch", "deep-research", "--path", str(deep_research.checkout), "--dry-run", "--json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["serve"] == f"tt-model serve {COHOST} --local-only --port 8000 --detach"
    assert payload["recipe"] == "recipes/ornith-35b-cohost.toml"
    assert deep_research.tt_model_calls() == [] and deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_without_a_checkout_says_how_to_clone_it(runner, deep_research, tmp_path):
    deep_research(COHOST)
    result = runner.invoke(app, ["launch", "deep-research", "--path", str(tmp_path / "none")])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "git clone" in result.output
    assert deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_recipe_override_skips_matching(runner, deep_research):
    deep_research()  # nothing matched by bundle, but the endpoints answer
    result = runner.invoke(app, [
        "launch", "deep-research", "--path", str(deep_research.checkout),
        "--recipe", "recipes/rerank.toml",
    ])
    assert result.exit_code == 0, result.output
    assert deep_research.calls()[0]["argv"][-1] == "recipes/rerank.toml"


@pytest.mark.fakes_only
def test_deep_research_dry_run_starts_nothing(runner, deep_research, monkeypatch):
    deep_research(COHOST)
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: pytest.fail("asked"))
    result = runner.invoke(app, [
        "launch", "deep-research", "--path", str(deep_research.checkout), "--dry-run", "--json",
    ])
    assert result.exit_code == 0, result.output
    assert deep_research.calls() == []
    payload = json.loads(result.stdout)
    assert payload["steps"][0][1:] == [
        "run", "ttdr", "up", "--recipe", "recipes/ornith-35b-cohost.toml"
    ]
    assert payload["applied"] is False
    assert payload["url"] == "http://localhost:3000"


@pytest.mark.fakes_only
def test_deep_research_declining_starts_nothing(runner, deep_research, monkeypatch):
    deep_research(COHOST)
    monkeypatch.setattr("tenstorrent.commands.launch.confirm", lambda _: False)
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert deep_research.calls() == []


@pytest.mark.fakes_only
def test_deep_research_already_running_is_not_restarted(runner, deep_research):
    deep_research(COHOST, app_running=True)
    (deep_research.checkout / ".env").write_text("TTDR_RECIPE=recipes/ornith-35b-cohost.toml\n")
    result = runner.invoke(
        app, ["launch", "deep-research", "--path", str(deep_research.checkout)]
    )
    assert result.exit_code == 0, result.output
    assert deep_research.calls() == []
    assert "http://localhost:3000" in result.output


@pytest.mark.fakes_only
def test_stop_deep_research_runs_ttdr_down_where_it_was_started(runner, deep_research):
    deep_research(COHOST, app_running=True)
    result = runner.invoke(app, ["launch", "stop", "deep-research"])
    assert result.exit_code == 0, result.output
    assert deep_research.calls() == [
        {"argv": ["run", "ttdr", "down"], "cwd": str(deep_research.checkout)}
    ]


@pytest.mark.fakes_only
def test_stop_deep_research_models_stops_its_bundle_too(runner, deep_research):
    deep_research(COHOST, "someone/else", app_running=True)
    result = runner.invoke(app, ["launch", "stop", "deep-research", "--models", "--json"])
    assert result.exit_code == 0, result.output
    assert deep_research.calls()[0]["argv"] == ["run", "ttdr", "down"]
    assert deep_research.tt_model_calls() == [["stop", COHOST]]
    assert json.loads(result.stdout)["models_stopped"] == [COHOST]


@pytest.mark.fakes_only
def test_stop_models_is_a_usage_error_for_other_clients(runner, fake_client_named):
    fake_client_named("hermes")
    result = runner.invoke(app, ["launch", "stop", "hermes", "--models"])
    assert result.exit_code == ExitCode.USAGE


@pytest.mark.fakes_only
def test_stop_deep_research_when_it_is_not_running(runner, deep_research):
    deep_research(COHOST)
    result = runner.invoke(app, ["launch", "stop", "deep-research"])
    assert result.exit_code == 0, result.output
    assert "not running" in result.output
    assert deep_research.calls() == []


@pytest.mark.fakes_only
def test_disconnect_deep_research_has_nothing_to_undo(runner, deep_research):
    result = runner.invoke(app, ["launch", "disconnect", "deep-research"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output


@pytest.mark.fakes_only
def test_list_reports_a_missing_deep_research_checkout(runner, deep_research):
    deep_research()

    def row() -> dict:
        listed = json.loads(runner.invoke(app, ["launch", "list", "--json"]).stdout)
        return next(r for r in listed if r["tool"] == "deep-research")

    assert row()["available"] is False
    assert "checkout" in row()["needs"]
    runner.invoke(app, ["config", "set", "launch.deep_research_path", str(deep_research.checkout)])
    assert row()["available"] is True
    assert row()["container_state"] == "absent"
