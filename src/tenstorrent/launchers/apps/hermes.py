# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Hermes Agent adapter: OpenAI-compatible environment, nothing on disk.

`hermes chat` has no base-URL flag, and its `~/.hermes/config.yaml` is YAML that
tt does not parse. `--provider openai` reads the endpoint and key from
OPENAI_BASE_URL and OPENAI_API_KEY, so the setting lasts for this run only.

With --web the same endpoint backs `hermes dashboard`, whose in-browser chat
inherits the dashboard's environment. The provider and model are chosen through
the variables `hermes --tui --provider/--model` itself sets, since a provider
saved in config.yaml would otherwise outrank HERMES_INFERENCE_PROVIDER.
"""

from __future__ import annotations

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel

_PLACEHOLDER_KEY = "tt-local"
_PROVIDER = "openai"


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
        return Preparation(
            env=env,
            steps=[self._dashboard_argv(exe, options.web_port)],
            url=f"http://localhost:{options.web_port}",
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        """Nothing on disk: the environment is applied at hand-off."""

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        argv = [env.executable, *prep.steps[0][1:]]
        env.runner.exec_tty(argv, env=prep.env)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists, so there is never anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        return [executable, "chat", "--provider", _PROVIDER, "--model", model.served_id]

    def _dashboard_argv(self, executable: str, web_port: int) -> list[str]:
        return [executable, "dashboard", "--port", str(web_port)]
