# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Hermes Agent adapter: OpenAI-compatible environment.

`hermes chat` has no base-URL flag, and its `~/.hermes/config.yaml` is YAML that
tt does not parse. `--provider openai` reads the endpoint and key from
OPENAI_BASE_URL and OPENAI_API_KEY, so the setting lasts for this run only.

With --web the same endpoint backs `hermes dashboard`, whose in-browser chat
inherits the dashboard's environment. The provider and model are chosen through
the variables `hermes --tui --provider/--model` itself sets, since a provider
saved in config.yaml would otherwise outrank HERMES_INFERENCE_PROVIDER.

Tenstorrent look: the terminal skin (`skins/`) is installed for every launch and the
dashboard theme (`dashboard-themes/`) for --web. Each is made active through
`hermes config set`, never by editing config.yaml, and only while the user is still
on Hermes's shipped default. A skin or theme the user picked is never overridden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel, free_web_port

_PLACEHOLDER_KEY = "tt-local"
_PROVIDER = "openai"
_NAME = "tenstorrent"
_SHIPPED_DEFAULTS = {"", "default"}


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME", "").strip()
    return Path(home).expanduser() if home else Path.home() / ".hermes"


@dataclass(frozen=True)
class _Look:
    """A bundled Hermes appearance: where it is installed and the setting that selects it."""

    package_dir: str  # directory under launchers/apps holding tenstorrent.yaml
    hermes_dir: str  # directory under the Hermes home that Hermes scans
    config_key: str


_SKIN = _Look("hermes_skin", "skins", "display.skin")
_THEME = _Look("hermes_theme", "dashboard-themes", "dashboard.theme")


class Hermes:
    id = "hermes"
    binaries = ("hermes",)
    install_hint = (
        "Install Hermes Agent (curl -fsSL "
        "https://hermes-agent.nousresearch.com/install.sh | bash): "
        "https://hermes-agent.nousresearch.com/docs/, then re-run."
    )
    requires_tool_calling = True
    hands_over_terminal = True
    web_ui = True

    def target(self) -> str:
        return "environment (this run only)"

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        env = {
            "OPENAI_BASE_URL": model.base_url,
            "OPENAI_API_KEY": _PLACEHOLDER_KEY,
        }
        exe = executable or self.binaries[0]
        if not options.web:
            return Preparation(env=env, steps=[self._argv(exe, model)])
        env |= {
            "HERMES_TUI_PROVIDER": _PROVIDER,
            "HERMES_INFERENCE_PROVIDER": _PROVIDER,
            "HERMES_MODEL": model.served_id,
            "HERMES_INFERENCE_MODEL": model.served_id,
        }
        web_port = free_web_port(options)
        return Preparation(
            env=env,
            steps=[self._dashboard_argv(exe, web_port)],
            url=f"http://localhost:{web_port}",
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        """The environment is applied at hand-off; only the Tenstorrent look is written."""
        self._adopt(_SKIN, env)
        if prep.url is not None:
            self._adopt(_THEME, env)

    def _adopt(self, look: _Look, env: LaunchEnv) -> None:
        """Install the look, and make it active unless the user already chose another.
        Cosmetic, so nothing here may block the launch."""
        source = resources.files(__package__).joinpath(look.package_dir, f"{_NAME}.yaml")
        target = _hermes_home() / look.hermes_dir / f"{_NAME}.yaml"
        try:
            content = source.read_bytes()
            if not (target.exists() and target.read_bytes() == content):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        except OSError:
            return
        exe = env.executable
        current = env.runner.capture([exe, "config", "get", look.config_key], check=False, tool=self.id)
        # `config get` exits non-zero when the key is unset.
        value = current.stdout.strip() if current.returncode == 0 else ""
        if value in _SHIPPED_DEFAULTS:
            env.runner.capture(
                [exe, "config", "set", look.config_key, _NAME], check=False, tool=self.id
            )

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        argv = [env.executable, *prep.steps[0][1:]]
        env.runner.exec_tty(argv, env=prep.env)

    def service_state(self, executable: str | None, runner) -> str:
        """running / stopped / unknown, for `tt launch stop`. Hermes keeps one
        machine-level dashboard, so this is about any dashboard, not only tt's."""
        if executable is None:
            return "unknown"
        result = runner.capture([executable, "dashboard", "--status"], check=False, tool=self.id)
        if result.returncode != 0:
            return "unknown"
        return "stopped" if "no hermes" in result.stdout.lower() else "running"

    def stop(self, env: LaunchEnv) -> None:
        """Stops every Hermes web server process, including one started outside tt."""
        env.runner.stream([env.executable, "dashboard", "--stop"], tool=self.id)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Only the skin and theme persist; both are cosmetic and switchable, so there is nothing to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        return [executable, "chat", "--provider", _PROVIDER, "--model", model.served_id]

    def _dashboard_argv(self, executable: str, web_port: int) -> list[str]:
        return [executable, "dashboard", "--port", str(web_port)]
