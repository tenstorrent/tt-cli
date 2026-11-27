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

For --web the Tenstorrent dashboard theme is dropped into Hermes's
`dashboard-themes/` folder and made the active theme, but only while the user is
still on Hermes's shipped default. A theme the user picked is never overridden.
The active theme is set through `hermes config set`, never by editing config.yaml.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel, free_web_port

_PLACEHOLDER_KEY = "tt-local"
_PROVIDER = "openai"
_THEME_FILE = "tenstorrent.yaml"
_THEME_NAME = "tenstorrent"
_THEME_KEY = "dashboard.theme"
_SHIPPED_DEFAULT_THEMES = {"", "default"}


def _theme_dir() -> Path:
    home = os.environ.get("HERMES_HOME", "").strip()
    return (Path(home).expanduser() if home else Path.home() / ".hermes") / "dashboard-themes"


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
        """The environment is applied at hand-off; only the web dashboard gets the theme."""
        if prep.url is None:
            return
        self._install_theme()
        self._activate_theme(env)

    def _install_theme(self) -> None:
        theme = resources.files(__package__).joinpath("hermes_theme", _THEME_FILE).read_bytes()
        target = _theme_dir() / _THEME_FILE
        try:
            if target.exists() and target.read_bytes() == theme:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(theme)
        except OSError:
            pass  # cosmetic: an unwritable Hermes home must not block the launch

    def _activate_theme(self, env: LaunchEnv) -> None:
        """Make the theme active unless the user already chose another one."""
        if not (_theme_dir() / _THEME_FILE).exists():
            return
        exe = env.executable
        current = env.runner.capture([exe, "config", "get", _THEME_KEY], check=False, tool=self.id)
        # `config get` exits non-zero when the key is unset.
        value = current.stdout.strip() if current.returncode == 0 else ""
        if value in _SHIPPED_DEFAULT_THEMES:
            env.runner.capture(
                [exe, "config", "set", _THEME_KEY, _THEME_NAME], check=False, tool=self.id
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
        """Only the theme persists; it is cosmetic and switchable in the dashboard, so there is nothing to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        return [executable, "chat", "--provider", _PROVIDER, "--model", model.served_id]

    def _dashboard_argv(self, executable: str, web_port: int) -> list[str]:
        return [executable, "dashboard", "--port", str(web_port)]
