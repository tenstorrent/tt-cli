# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt model ps`: classification of the three backends' containers, the health
probe, and the --json contract TT-Studio and `tt report issue` read."""

from __future__ import annotations

import json
import re
import socket
import time
from pathlib import Path

import pytest

from tenstorrent.cli import app
from tenstorrent.errors import ExitCode

SMALL_SUPPORT = Path(__file__).parent.parent / "fakes" / "data" / "model_support_small.json"
STUDIO_IMAGE = "ghcr.io/tenstorrent/tt-studio/studio_images:qwen35-9b-blackhole-20260810"
TT_MODEL_LABELS = {
    "org.tenstorrent.tt-model": "qwen3-coder-30b-a3b",
    "org.tenstorrent.tt-model.repo": "raahemnabeel/qwen3-coder-30b-a3b",
    "org.tenstorrent.tt-model.profile": "default",
    "org.tenstorrent.tt-model.kind": "vllm-plugin",
    "org.tenstorrent.tt-model.arch": "blackhole",
    "org.tenstorrent.tt-model.weights": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}
CONTRACT_KEYS = {
    "name", "backend", "container", "container_id", "image", "port", "base_url",
    "status", "health", "started_at", "uptime_s", "profile", "served_id",
}


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture(autouse=True)
def small_spec(monkeypatch):
    monkeypatch.setenv("TT_MODEL_SUPPORT_PATH", str(SMALL_SUPPORT))


@pytest.fixture(autouse=True)
def no_default_endpoint(monkeypatch):
    """Point the default probe at a port nothing listens on, so a model server
    that happens to run on 8000 on the developer's box cannot leak into a test.
    Tests about the default endpoint patch it again themselves."""
    monkeypatch.setattr(
        "tenstorrent.launchers.discovery.DEFAULT_BASE_URL",
        f"http://127.0.0.1:{free_port()}/v1",
    )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _record(
    cid: str,
    name: str,
    *,
    image: str = "img:1",
    running: bool = True,
    started: str | None = "2026-09-08T10:00:00.123456789Z",
    port: int | None = None,
    labels: dict | None = None,
    mounts: list | None = None,
) -> dict:
    """A `docker inspect` record in the shape tt reads."""
    bindings = (
        {f"{port}/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(port)}]} if port else {}
    )
    return {
        "Id": cid,
        "Name": f"/{name}",
        "Config": {"Image": image, "Labels": labels or {}},
        "Mounts": mounts or [],
        "State": {
            "Status": "running" if running else "exited",
            "Running": running,
            "StartedAt": started or "0001-01-01T00:00:00Z",
        },
        "HostConfig": {"PortBindings": bindings},
    }


def _studio(port=7000, **kw):
    return _record("2a324d170bec" + "0" * 52, "Qwen3.5-9B", image=STUDIO_IMAGE, port=port, **kw)


def _tt_model(port=20000, **kw):
    return _record(
        "68738b652848" + "0" * 52,
        "tt-model-qwen3-coder-30b-a3b-default",
        image="tt-model/qwen3-coder-30b-a3b:7c2773460298",
        port=port,
        labels=TT_MODEL_LABELS,
        **kw,
    )


def _inference_server(port=8000, **kw):
    return _record(
        "abcdef123456" + "0" * 52,
        "tt-inference-server-abcd",
        image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release:0.0.5",
        port=port,
        mounts=[
            {
                "Type": "bind",
                "Source": "/home/someone/.cache/huggingface/hub/"
                "models--Qwen--Qwen3-32B/snapshots/abc",
            }
        ],
        **kw,
    )


def _served(result) -> dict:
    return json.loads(result.output[result.output.index("{"):])


@pytest.mark.fakes_only
def test_ps_empty_is_exit_zero(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([])
    result = runner.invoke(app, ["model", "ps", "--json"])
    assert result.exit_code == 0
    assert _served(result) == {"served": [], "probed": True}

    result = runner.invoke(app, ["model", "ps"])
    assert result.exit_code == 0
    assert "No model servers running" in result.output


@pytest.mark.fakes_only
def test_ps_json_contract(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio(), _tt_model(), _inference_server()])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert result.exit_code == 0
    data = _served(result)
    assert set(data) == {"served", "probed"}
    assert data["probed"] is False
    assert len(data["served"]) == 3
    for row in data["served"]:
        # this key set is the public --json contract; TT-Studio reads it
        assert set(row) == CONTRACT_KEYS
        assert row["started_at"] == "2026-09-08T10:00:00Z"  # fraction stripped, UTC
        assert isinstance(row["uptime_s"], int) and row["uptime_s"] >= 0
        assert row["base_url"] == f"http://127.0.0.1:{row['port']}/v1"
        assert row["health"] == "unknown"  # not probed


def test_ps_without_container_runtime_is_tool_missing(runner, monkeypatch):
    monkeypatch.setattr(
        "tenstorrent.backends.serving.inference_server.shutil.which", lambda name: None
    )
    result = runner.invoke(app, ["model", "ps", "--json"])
    assert result.exit_code == ExitCode.TOOL_MISSING
    assert json.loads(result.output)["error"]["code"] == "TOOL_MISSING"


@pytest.mark.fakes_only
def test_ps_classifies_each_backend(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_inference_server(), _tt_model(), _studio()])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert result.exit_code == 0
    rows = {row["backend"]: row for row in _served(result)["served"]}
    assert set(rows) == {"inference-server", "model-manager", "studio"}

    server = rows["inference-server"]
    assert server["name"] == "Qwen3-32B"  # catalog name, via the weights mount
    assert server["container"] == "tt-inference-server-abcd"
    assert server["port"] == 8000
    assert server["profile"] is None

    bundle = rows["model-manager"]
    assert bundle["name"] == "raahemnabeel/qwen3-coder-30b-a3b"  # the .repo label
    assert bundle["profile"] == "default"
    assert bundle["port"] == 20000  # what it published, not a profile default
    assert bundle["container_id"] == "68738b652848"

    studio = rows["studio"]
    assert studio["name"] == studio["container"] == "Qwen3.5-9B"
    assert studio["port"] == 7000
    assert studio["image"] == STUDIO_IMAGE


@pytest.mark.fakes_only
def test_ps_names_an_unmatched_inference_server_by_its_weights(runner, fake_docker):
    set_containers, _ = fake_docker
    record = _inference_server()
    record["Mounts"][0]["Source"] = "/x/hub/models--acme--Not-In-Catalog/snapshots/abc"
    set_containers([record])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert _served(result)["served"][0]["name"] == "acme/Not-In-Catalog"


@pytest.mark.fakes_only
def test_ps_excludes_studio_infra_and_unrelated_containers(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([
        _record("1" * 64, "tt_studio_backend_api_prod",
                image="ghcr.io/tenstorrent/tt-studio/backend:sha-df611abf13af", port=8000),
        _record("2" * 64, "tt_studio_frontend_prod",
                image="ghcr.io/tenstorrent/tt-studio/frontend:sha-df611abf13af", port=3000),
        _record("3" * 64, "tt_studio_chroma_prod", image="chromadb/chroma:0.5.3", port=8111),
        _record("4" * 64, "tt_studio_litellm", image="ghcr.io/berriai/litellm:main-stable"),
        _record("5" * 64, "somebody-elses-postgres", image="postgres:16", port=5432),
        _studio(),
    ])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert result.exit_code == 0
    rows = _served(result)["served"]
    assert [r["container"] for r in rows] == ["Qwen3.5-9B"]


@pytest.mark.fakes_only
def test_ps_hides_exited_by_default_and_shows_them_with_all(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio(), _tt_model(running=False, started="2026-09-03T20:10:53.5Z")])

    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert [r["container"] for r in _served(result)["served"]] == ["Qwen3.5-9B"]

    result = runner.invoke(app, ["model", "ps", "--no-probe", "--all", "--json"])
    rows = _served(result)["served"]
    assert [r["backend"] for r in rows] == ["studio", "model-manager"]  # running first
    exited = rows[1]
    assert exited["status"] == "exited"
    assert exited["health"] == "stopped"
    assert exited["uptime_s"] is None
    assert exited["started_at"] == "2026-09-03T20:10:53Z"

    result = runner.invoke(app, ["model", "ps", "--no-probe", "-a"])
    assert "stopped" in result.output


@pytest.mark.fakes_only
def test_ps_never_started_container_has_no_started_at(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_tt_model(running=False, started=None)])
    result = runner.invoke(app, ["model", "ps", "--all", "--no-probe", "--json"])
    row = _served(result)["served"][0]
    assert row["started_at"] is None
    assert row["uptime_s"] is None


@pytest.mark.fakes_only
def test_ps_probe_marks_a_live_server_healthy(runner, fake_docker, served):
    set_containers, _ = fake_docker
    port = int(served("Qwen/Qwen3.5-9B").rsplit(":", 1)[1].split("/")[0])
    set_containers([_studio(port=port)])
    result = runner.invoke(app, ["model", "ps", "--json"])
    assert result.exit_code == 0
    row = _served(result)["served"][0]
    assert row["health"] == "healthy"
    assert row["served_id"] == "Qwen/Qwen3.5-9B"
    assert _served(result)["probed"] is True


@pytest.mark.fakes_only
def test_ps_probe_marks_a_silent_running_container_starting(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio(port=free_port()), _tt_model(port=free_port())])
    t0 = time.monotonic()
    result = runner.invoke(app, ["model", "ps", "--json"])
    elapsed = time.monotonic() - t0
    assert result.exit_code == 0
    rows = _served(result)["served"]
    assert [r["health"] for r in rows] == ["starting", "starting"]
    assert all(r["served_id"] is None for r in rows)
    # probes run in parallel with a short timeout: two dead ports must not add up
    assert elapsed < 4


@pytest.mark.fakes_only
def test_ps_probe_ignores_a_non_openai_server_on_the_default_port(
    runner, fake_docker, served, monkeypatch
):
    """TT-Studio's own backend listens on 8000 and answers 404 HTML to /v1/models."""
    set_containers, _ = fake_docker
    set_containers([])
    monkeypatch.setattr("tenstorrent.launchers.discovery.DEFAULT_BASE_URL", served.reject())
    result = runner.invoke(app, ["model", "ps", "--json"])
    assert result.exit_code == 0
    assert _served(result) == {"served": [], "probed": True}


@pytest.mark.fakes_only
def test_ps_unattributed_default_endpoint_becomes_an_unknown_row(
    runner, fake_docker, served, monkeypatch
):
    set_containers, _ = fake_docker
    set_containers([])
    base_url = served("meta-llama/Llama-3.1-8B-Instruct")
    monkeypatch.setattr("tenstorrent.launchers.discovery.DEFAULT_BASE_URL", base_url)
    result = runner.invoke(app, ["model", "ps", "--json"])
    assert result.exit_code == 0
    rows = _served(result)["served"]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == CONTRACT_KEYS
    assert row["backend"] == "unknown"
    assert row["container"] is None and row["container_id"] is None
    assert row["name"] == "Llama-3.1-8B-Instruct"  # catalog name for the served id
    assert row["served_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert row["base_url"] == base_url
    assert row["health"] == "healthy"
    assert row["status"] == "running"
    # the table shows it too, with an em dash where the container would be
    result = runner.invoke(app, ["model", "ps"])
    assert "unknown" in result.output and "—" in result.output


@pytest.mark.fakes_only
def test_ps_default_endpoint_is_not_probed_twice_when_a_container_owns_it(
    runner, fake_docker, served, monkeypatch
):
    set_containers, _ = fake_docker
    base_url = served("Qwen/Qwen3-32B")
    port = int(base_url.rsplit(":", 1)[1].split("/")[0])
    monkeypatch.setattr("tenstorrent.launchers.discovery.DEFAULT_BASE_URL", base_url)
    set_containers([_inference_server(port=port)])
    result = runner.invoke(app, ["model", "ps", "--json"])
    rows = _served(result)["served"]
    assert len(rows) == 1  # attributed to the container, no extra unknown row
    assert rows[0]["backend"] == "inference-server"
    assert rows[0]["health"] == "healthy"


@pytest.mark.fakes_only
def test_ps_no_probe_skips_http_and_reports_unknown(runner, fake_docker, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("--no-probe must not touch the network")

    monkeypatch.setattr("tenstorrent.launchers.discovery.probe", boom)
    set_containers, _ = fake_docker
    set_containers([_studio()])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--json"])
    assert result.exit_code == 0
    data = _served(result)
    assert data["probed"] is False
    assert data["served"][0]["health"] == "unknown"
    assert data["served"][0]["served_id"] is None


@pytest.mark.fakes_only
def test_ps_running_container_without_a_port_is_unknown(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio(port=None)])
    result = runner.invoke(app, ["model", "ps", "--json"])
    row = _served(result)["served"][0]
    assert row["port"] is None and row["base_url"] is None
    assert row["health"] == "unknown"


@pytest.mark.fakes_only
def test_ps_table_columns(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio()])
    result = runner.invoke(app, ["model", "ps", "--no-probe"])
    assert result.exit_code == 0
    for header in ("name", "backend", "container", "port", "health", "uptime"):
        assert header in result.output
    assert "Qwen3.5-9B" in result.output
    assert "7000" in result.output
    assert re.search(r"\d+[smhd]\b", result.output), result.output


@pytest.mark.fakes_only
def test_ps_quiet_prints_nothing(runner, fake_docker):
    set_containers, _ = fake_docker
    set_containers([_studio()])
    result = runner.invoke(app, ["model", "ps", "--no-probe", "--quiet"])
    assert result.exit_code == 0
    assert result.output == ""
