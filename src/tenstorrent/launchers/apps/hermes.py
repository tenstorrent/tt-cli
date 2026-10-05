# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Hermes Agent adapter: OpenAI-compatible environment, nothing on disk.

`hermes chat` has no base-URL flag, and its `~/.hermes/config.yaml` is YAML that
tt does not parse. `--provider openai` reads the endpoint and key from
OPENAI_BASE_URL and OPENAI_API_KEY, so the setting lasts for this run only.
"""

from __future__ import annotations

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel

_PLACEHOLDER_KEY = "tt-local"


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
        return Preparation(
            env={
                "OPENAI_BASE_URL": model.base_url,
                "OPENAI_API_KEY": _PLACEHOLDER_KEY,
            },
            steps=[self._argv(executable or self.binaries[0], model)],
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        """Nothing on disk: the environment is applied at hand-off."""

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(self._argv(env.executable, model), env=prep.env)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists, so there is never anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        return [executable, "chat", "--provider", "openai", "--model", model.served_id]
