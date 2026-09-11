# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

import json
import os
import socket

import pytest

from tenstorrent.errors import ExitCode, TTError
from tenstorrent.launchers import LAUNCHERS
from tenstorrent.launchers.base import (
    VLLM_ENGINE,
    LaunchEnv,
    LaunchOptions,
    RunningModel,
    read_json_config,
    resolve_executable,
    tool_call_parser,
    write_json_config,
)
from tenstorrent.launchers.container import container_base_url, port_is_free
from tenstorrent.launchers.discovery import DEFAULT_BASE_URL, base_url_for, probe
from tenstorrent.launchers.apps.openwebui import IMAGE
from tenstorrent.modelhub.catalog import ModelCatalog
from tenstorrent.output import OutputManager
from tenstorrent.tools.runner import CaptureResult


@pytest.fixture(autouse=True)
def container_side_effects(monkeypatch):
    """Assume the host port is free and the service answers at once. The tests for
    those two behaviours patch them back."""
    monkeypatch.setattr("tenstorrent.launchers.container.port_is_free", lambda port: True)
    monkeypatch.setattr(
        "tenstorrent.launchers.container.wait_until_ready", lambda url, timeout_s: True
    )


def test_port_is_free_sees_a_bound_port(monkeypatch):
    monkeypatch.undo()  # the real implementation, not the autouse stand-in
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        assert port_is_free(taken.getsockname()[1]) is False
    # A port nothing listens on is bindable again.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert port_is_free(free) is True


def test_container_refuses_a_taken_port_before_pulling(monkeypatch):
    monkeypatch.setattr("tenstorrent.launchers.container.port_is_free", lambda port: False)

    class Absent:
        def capture(self, argv, **kwargs):
            return CaptureResult(returncode=1, stdout="", stderr="")

    with pytest.raises(TTError) as err:
        LAUNCHERS["openwebui"].plan(
            RunningModel(served_id="m", base_url="http://127.0.0.1:8000/v1"),
            LaunchOptions(web_port=3000),
            executable="/usr/bin/docker",
            runner=Absent(),
        )
    assert err.value.exit_code == ExitCode.CONFIG
    # The next step has to offer a way out, not just the complaint.
    assert "--web-port" in (err.value.next_step or "")


def test_an_already_running_container_does_not_check_the_port(monkeypatch):
    """Adopting our own container must not fail on the port it is itself using."""
    monkeypatch.setattr("tenstorrent.launchers.container.port_is_free", lambda port: False)

    class Running:
        def capture(self, argv, **kwargs):
            doc = {"State": {"Running": True}, "Config": {"Env": []}}
            return CaptureResult(returncode=0, stdout=json.dumps(doc), stderr="")

    prep = LAUNCHERS["openwebui"].plan(
        RunningModel(served_id="m", base_url="http://127.0.0.1:8000/v1"),
        LaunchOptions(web_port=3000),
        executable="/usr/bin/docker",
        runner=Running(),
    )
    assert prep.steps == [] and prep.consent is None


class _Existing:
    """An inspect record for a container created with -p 3080:<port>."""

    def __init__(self, launcher, *, running: bool):
        self.launcher, self.running = launcher, running

    def capture(self, argv, **kwargs):
        doc = {
            "State": {"Running": self.running},
            "Config": {"Env": []},
            "HostConfig": {
                "PortBindings": {
                    f"{self.launcher.container_port}/tcp": [{"HostPort": "3080"}]
                }
            },
        }
        return CaptureResult(returncode=0, stdout=json.dumps(doc), stderr="")


@pytest.mark.parametrize("running", [True, False])
def test_an_existing_container_reports_the_port_it_actually_publishes(running):
    """--web-port cannot move a container's published port: it was fixed at create
    time. Reporting the requested one would print a URL that answers nothing."""
    launcher = LAUNCHERS["openwebui"]
    prep = launcher.plan(
        RunningModel(served_id="m", base_url="http://127.0.0.1:8000/v1"),
        LaunchOptions(web_port=9999),
        executable="/usr/bin/docker",
        runner=_Existing(launcher, running=running),
    )
    assert prep.url == "http://localhost:3080"


def test_starting_a_stopped_container_checks_its_own_port(monkeypatch):
    """Something else may have taken the port while it was down, and `docker start`
    would fail — so the guard covers the start path too, on the container's port."""
    checked: list[int] = []
    monkeypatch.setattr(
        "tenstorrent.launchers.container.port_is_free",
        lambda port: checked.append(port) or False,
    )
    launcher = LAUNCHERS["openwebui"]
    with pytest.raises(TTError) as err:
        launcher.plan(
            RunningModel(served_id="m", base_url="http://127.0.0.1:8000/v1"),
            LaunchOptions(web_port=9999),
            executable="/usr/bin/docker",
            runner=_Existing(launcher, running=False),
        )
    assert checked == [3080]  # its own port, not the requested one
    assert err.value.exit_code == ExitCode.CONFIG
    # --web-port cannot move an existing binding, so the advice must not offer it
    # as the whole fix.
    assert "docker rm" in (err.value.next_step or "")


def test_no_client_is_named_after_a_subcommand():
    """Clients are registered as subcommands of `tt launch`, so a client called
    "list" or "stop" would shadow the group's own command."""
    from tenstorrent.commands.launch import SUBCOMMANDS

    assert not set(LAUNCHERS) & set(SUBCOMMANDS)


def test_every_client_can_be_asked_what_disconnect_would_undo(tmp_path, monkeypatch):
    # Point every config location at an empty dir: under --hardware the real HOME
    # is not redirected, and a client the developer configured would fail this.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    for launcher in LAUNCHERS.values():
        # Nothing configured and no runtime resolved: never an error, just None.
        assert launcher.disconnect_plan(None, None) is None


def test_every_client_says_what_it_configures():
    for launcher in LAUNCHERS.values():
        assert launcher.target()


def test_tool_call_parser_does_not_vary_by_device():
    """`tt launch` reads tool calling as a per-model property, with no board
    detection. That is only sound while the support list agrees, so pin it: if a
    regenerated list ever publishes different parsers per device, this fails and
    the capability check has to take a device."""
    for model in ModelCatalog().list(cached_sizes={}):
        parsers = {
            support.tool_call_parser
            for support in model.devices.values()
            if VLLM_ENGINE in support.engines
        }
        assert len(parsers) <= 1, f"{model.name} publishes {parsers}"


def test_every_launcher_has_the_fields_the_command_reads():
    for name, launcher in LAUNCHERS.items():
        assert launcher.id == name
        assert launcher.binaries and launcher.install_hint
        assert isinstance(launcher.requires_tool_calling, bool)
        assert isinstance(launcher.hands_over_terminal, bool)
        # A miss must send the user to the tool's own installer, never to tt update.
        assert "tt update" not in launcher.install_hint


def test_base_url_defaults_and_port_shorthand():
    assert base_url_for(None, None) == DEFAULT_BASE_URL
    assert base_url_for(None, 8123) == "http://127.0.0.1:8123/v1"
    assert base_url_for("http://box:9000/v1/", None) == "http://box:9000/v1"
    with pytest.raises(TTError) as err:
        base_url_for("http://box/v1", 8000)
    assert err.value.exit_code == ExitCode.USAGE


def test_resolve_executable_prefers_env_then_override(tmp_path, monkeypatch):
    class Config:
        def __init__(self, value=None):
            self.value = value

        def get(self, key):
            assert key == "tools.override.opencode"
            return self.value

    output = OutputManager()
    launcher = LAUNCHERS["opencode"]
    monkeypatch.setenv("TT_TOOL_BIN_OPENCODE", "/from/env")
    assert resolve_executable(launcher, Config("/from/config"), output) == "/from/env"
    monkeypatch.delenv("TT_TOOL_BIN_OPENCODE")
    assert resolve_executable(launcher, Config("/from/config"), output) == "/from/config"
    monkeypatch.setattr("tenstorrent.launchers.base.shutil.which", lambda name, **_: None)
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: None)
    with pytest.raises(TTError) as err:
        resolve_executable(launcher, Config(), output)
    assert err.value.exit_code == ExitCode.TOOL_MISSING


def test_resolve_executable_retries_with_a_freshly_sourced_path(tmp_path, monkeypatch):
    """A client just installed by a script often isn't on this process's PATH yet
    — only a shell that re-reads its rc files sees it. One retry with a fresh
    shell's PATH must find it without the user opening a new terminal."""
    exe = tmp_path / "opencode"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)

    class Config:
        def get(self, key):
            return None

    launcher = LAUNCHERS["opencode"]
    monkeypatch.delenv("TT_TOOL_BIN_OPENCODE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path.parent))  # opencode is not here yet
    monkeypatch.setattr("tenstorrent.launchers.base._shell_path", lambda: str(tmp_path))
    assert resolve_executable(launcher, Config(), OutputManager()) == str(exe)
    # A tool found this way may itself need PATH at run time (a `#!/usr/bin/env
    # node` shebang, or a subprocess it shells out to) — not just this lookup.
    assert os.environ["PATH"] == str(tmp_path)


def test_write_json_config_creates_parents_and_leaves_no_temp(tmp_path):
    path = tmp_path / "nested" / "client.json"
    write_json_config(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in path.parent.iterdir()] == ["client.json"]


def test_read_json_config_reports_a_new_file_and_an_empty_one(tmp_path):
    assert read_json_config(tmp_path / "missing.json") == ({}, True)
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert read_json_config(empty) == ({}, False)


@pytest.mark.parametrize("content", ["[]", '"a string"', "42", "null"])
def test_read_json_config_refuses_valid_json_that_is_not_an_object(tmp_path, content):
    """Callers merge with .get()/.setdefault(), so a non-object top level has to be a
    controlled CONFIG error rather than an AttributeError."""
    path = tmp_path / "client.json"
    path.write_text(content)
    with pytest.raises(TTError) as err:
        read_json_config(path)
    assert err.value.exit_code == ExitCode.CONFIG
    assert path.read_text() == content  # never rewritten


def test_opencode_plan_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    launcher = LAUNCHERS["opencode"]
    model = RunningModel(served_id="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1")
    prep = launcher.plan(model, LaunchOptions(), executable=None, runner=None)
    assert prep.config is not None
    assert prep.config.created is True
    assert not prep.config.path.exists()  # planning writes nothing
    assert prep.config.block["options"]["baseURL"] == model.base_url
    # Editing a file tt does not own is asked about, and the question names it.
    assert prep.consent and str(prep.config.path) in prep.consent
    assert prep.steps == [["opencode", "--model", "tenstorrent/Qwen/Qwen3-32B"]]

    launcher.apply(model, prep, LaunchEnv("/usr/bin/opencode", None, None))
    doc = json.loads(prep.config.path.read_text())
    assert doc["$schema"]  # only stamped on a file tt created
    assert doc["provider"]["tenstorrent"]["models"] == {
        "Qwen/Qwen3-32B": {"name": "Qwen/Qwen3-32B"}
    }


def test_openwebui_rewrites_a_loopback_endpoint_for_the_container():
    # localhost inside the container is Open WebUI itself, not the model server.
    assert (
        container_base_url("http://127.0.0.1:8000/v1")
        == "http://host.docker.internal:8000/v1"
    )
    assert (
        container_base_url("http://localhost:8123/v1")
        == "http://host.docker.internal:8123/v1"
    )
    # A real host is already reachable from inside; leave it alone.
    assert container_base_url("http://quietbox:8000/v1") == "http://quietbox:8000/v1"


def test_openwebui_plans_a_pull_and_a_run_and_asks_first():
    launcher = LAUNCHERS["openwebui"]
    model = RunningModel(served_id="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1")

    class Absent:
        def capture(self, argv, **kwargs):
            return CaptureResult(returncode=1, stdout="", stderr="no such container")

    prep = launcher.plan(
        model, LaunchOptions(web_port=3080), executable="/usr/bin/docker", runner=Absent()
    )
    assert prep.consent and "Pull" in prep.consent
    assert prep.url == "http://localhost:3080"
    pull, run = prep.steps
    assert pull == ["/usr/bin/docker", "pull", IMAGE]
    assert run[:3] == ["/usr/bin/docker", "run", "-d"]
    assert "3080:8080" in run
    assert f"OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1" in run
    # Otherwise the first run's settings are baked into its database.
    assert "ENABLE_PERSISTENT_CONFIG=false" in run


def test_openwebui_refuses_a_container_built_for_another_server():
    launcher = LAUNCHERS["openwebui"]
    model = RunningModel(served_id="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1")

    class Stale:
        def capture(self, argv, **kwargs):
            doc = {
                "State": {"Running": True},
                "Config": {"Env": ["OPENAI_API_BASE_URL=http://host.docker.internal:9999/v1"]},
            }
            return CaptureResult(returncode=0, stdout=json.dumps(doc), stderr="")

    with pytest.raises(TTError) as err:
        launcher.plan(
            model, LaunchOptions(), executable="/usr/bin/docker", runner=Stale()
        )
    assert err.value.exit_code == ExitCode.CONFIG
    assert "docker rm -f" in (err.value.next_step or "")


def test_pi_provider_block_matches_its_documented_shape(tmp_path, monkeypatch):
    """Shape from pi's own docs/models.json reference (docs/models.md)."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    launcher = LAUNCHERS["pi"]
    model = RunningModel(
        served_id="Qwen/Qwen3-32B",
        base_url="http://127.0.0.1:8000/v1",
        max_context=131072,
    )
    prep = launcher.plan(model, LaunchOptions(), executable=None, runner=None)
    block = prep.config.block
    assert block["api"] == "openai-completions"
    assert block["baseUrl"] == model.base_url
    assert block["apiKey"]  # keyless servers still need a placeholder
    # pi's docs name vLLM among the servers that reject both of these.
    assert block["compat"] == {
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
    }
    (entry,) = block["models"]
    assert entry["id"] == "Qwen/Qwen3-32B"
    assert entry["contextWindow"] == 131072
    assert entry["input"] == ["text"]  # image input only for a vlm entry
    # --provider/--model rather than "provider/id", which a repo-shaped id breaks.
    assert prep.steps == [["pi", "--provider", "tenstorrent", "--model", "Qwen/Qwen3-32B"]]


def test_pi_merges_models_instead_of_replacing_the_provider_list(tmp_path, monkeypatch):
    """pi's `models` key replaces the list, so the adapter has to merge by id."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "tenstorrent": {"models": [{"id": "Old-Model", "name": "Old"}]},
                    "ollama": {"baseUrl": "http://localhost:11434/v1"},
                }
            }
        )
    )
    launcher = LAUNCHERS["pi"]
    model = RunningModel(served_id="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1")
    prep = launcher.plan(model, LaunchOptions(), executable=None, runner=None)
    assert [m["id"] for m in prep.config.block["models"]] == ["Old-Model", "Qwen/Qwen3-32B"]
    assert prep.config.document["providers"]["ollama"]["baseUrl"]  # untouched


def test_pi_declares_image_input_for_a_vision_model(tmp_path, monkeypatch):
    from tenstorrent.models.model import DeviceSupport, ModelInfo

    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    vlm = ModelInfo(
        name="Qwen3-VL-32B-Instruct",
        hf_repo="Qwen/Qwen3-VL-32B-Instruct",
        model_type="vlm",
        engines=["vLLM"],
        devices={"p300x2": DeviceSupport(engines=["vLLM"], status="FUNCTIONAL")},
    )
    model = RunningModel(
        served_id="Qwen/Qwen3-VL-32B-Instruct",
        base_url="http://127.0.0.1:8000/v1",
        entry=vlm,
    )
    prep = LAUNCHERS["pi"].plan(model, LaunchOptions(), executable=None, runner=None)
    assert prep.config.block["models"][0]["input"] == ["text", "image"]


def test_aider_configures_by_environment_and_writes_nothing(tmp_path, monkeypatch):
    home = tmp_path / "aider-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    launcher = LAUNCHERS["aider"]
    model = RunningModel(served_id="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1")
    prep = launcher.plan(model, LaunchOptions(), executable=None, runner=None)
    assert prep.config is None
    assert prep.env == {
        "OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
        "OPENAI_API_KEY": "tt-local",
    }
    # litellm reaches an OpenAI-compatible endpoint through the openai/ prefix.
    assert prep.steps == [["aider", "--model", "openai/Qwen/Qwen3-32B"]]
    assert list(home.iterdir()) == []


def test_anythingllm_pins_the_model_because_it_has_no_discovery():
    launcher = LAUNCHERS["anythingllm"]
    model = RunningModel(
        served_id="Qwen/Qwen3-32B",
        base_url="http://127.0.0.1:8000/v1",
        max_context=131072,
    )

    class Absent:
        def capture(self, argv, **kwargs):
            return CaptureResult(returncode=1, stdout="", stderr="")

    prep = launcher.plan(
        model, LaunchOptions(web_port=3081), executable="/usr/bin/docker", runner=Absent()
    )
    _, run = prep.steps
    assert "LLM_PROVIDER=generic-openai" in run
    assert "GENERIC_OPEN_AI_BASE_PATH=http://host.docker.internal:8000/v1" in run
    assert "GENERIC_OPEN_AI_MODEL_PREF=Qwen/Qwen3-32B" in run
    assert "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=131072" in run
    assert "3081:3001" in run
    # Required for the bundled LanceDB and native embedder.
    assert run[run.index("--cap-add") + 1] == "SYS_ADMIN"


def test_anythingllm_falls_back_when_the_server_publishes_no_context():
    launcher = LAUNCHERS["anythingllm"]
    model = RunningModel(served_id="m", base_url="http://127.0.0.1:8000/v1")
    env = launcher.container_env(model, "http://host.docker.internal:8000/v1")
    assert env["GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT"] == "32768"


def test_container_clients_do_not_share_a_container_name():
    containers = [
        launcher.container for launcher in LAUNCHERS.values() if hasattr(launcher, "container")
    ]
    assert len(containers) == len(set(containers))


def test_tool_call_parser_reads_only_vllm_devices():
    from tenstorrent.models.model import DeviceSupport, ModelInfo

    media_only = ModelInfo(
        name="m",
        hf_repo="o/m",
        model_type="audio",
        engines=["media"],
        devices={"p300x2": DeviceSupport(engines=["media"], status="FUNCTIONAL",
                                         tool_call_parser="hermes")},
    )
    assert tool_call_parser(media_only) is None


# -- discovery.probe: the non-raising view `tt model ps` uses -----------------------------
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_probe_returns_none_for_a_dead_port():
    assert probe(f"http://127.0.0.1:{_free_port()}/v1", timeout_s=0.5) is None


def test_probe_returns_models_from_a_live_server(served):
    got = probe(served("Qwen/Qwen3.5-9B"))
    assert [m.served_id for m in got] == ["Qwen/Qwen3.5-9B"]
    assert got[0].max_context == 131072


def test_probe_returns_none_for_a_404_html_server(served):
    assert probe(served.reject()) is None


def test_probe_returns_none_for_foreign_json(loopback):
    # a JSON array where an object is expected used to raise AttributeError
    assert probe(loopback(b"[]"), timeout_s=1) is None
    assert probe(loopback(b'{"data": "nope"}'), timeout_s=1) is None
    assert probe(loopback(b'{"data": [1, {"name": "no id"}]}'), timeout_s=1) is None


def test_probe_honours_its_timeout(loopback):
    import time

    t0 = time.monotonic()
    assert probe(loopback(b"{}", delay_s=3), timeout_s=0.5) is None
    assert time.monotonic() - t0 < 2


@pytest.fixture
def loopback():
    """A loopback server answering every GET with a fixed body, optionally slowly."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    servers = []

    def start(body: bytes, *, delay_s: float = 0) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                import time

                time.sleep(delay_s)
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
