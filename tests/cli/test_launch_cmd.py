# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode


LAUNCH_SUPPORT = (
    Path(__file__).parent.parent / "fakes" / "data" / "model_support_launch.json"
)


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
def served():
    """A real loopback server answering GET /v1/models, like a served model does.

    Returns a callable: served(*ids) -> base_url. Real HTTP rather than a patched
    urlopen, so the discovery path that runs on hardware is the one under test.
    """
    servers: list[ThreadingHTTPServer] = []

    def start(*ids: str) -> str:
        payload = json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": i, "object": "model", "max_model_len": 131072} for i in ids
                ],
            }
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
                body = payload if self.path.endswith("/models") else b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/v1"

    yield start
    for server in servers:
        server.shutdown()


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
    assert provider["models"] == {"Qwen/Qwen3-32B": {"name": "Qwen/Qwen3-32B"}}


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
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name: None)
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
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name: None)
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
    for tool in ("opencode", "pi", "aider", "openwebui", "anythingllm"):
        assert tool in result.output
    for subcommand in ("list", "stop", "disconnect"):
        assert subcommand in result.output


# -- listing and teardown ---------------------------------------------------------
@pytest.mark.fakes_only
def test_list_shows_every_client_without_a_server(runner, monkeypatch):
    """--list must work with nothing serving and nothing installed."""
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name: None)
    result = runner.invoke(app, ["launch", "list"])
    assert result.exit_code == 0, result.output
    for tool in ("opencode", "pi", "aider", "openwebui", "anythingllm"):
        assert tool in result.output
    # Rich wraps the column, so match on a fragment rather than the whole phrase.
    assert "needs docker" in result.output.replace("\n", " ")


@pytest.mark.fakes_only
def test_list_json_reports_availability(runner, fake_client_named, monkeypatch):
    fake_client_named("opencode")
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name: None)
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
