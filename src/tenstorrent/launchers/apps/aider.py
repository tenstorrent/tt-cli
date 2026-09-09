# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Aider adapter: OpenAI-compatible environment plus an `openai/` model ref.

Configured by environment rather than by file. Aider's own config is
`~/.aider.conf.yml`, and merging YAML safely needs a parser tt does not depend
on — refusing to touch a file we cannot re-read would be worse than not writing
one. The trade-off is that the setting lasts for this run: a later bare `aider`
is not configured.
"""

from __future__ import annotations

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel

# Aider routes through litellm, where an OpenAI-compatible endpoint is reached
# with the `openai/` prefix and these two variables.
_PREFIX = "openai"
_PLACEHOLDER_KEY = "tt-local"


class Aider:
    id = "aider"
    binaries = ("aider",)
    install_hint = "Install Aider (https://aider.chat/docs/install.html), then re-run."
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
        env = {
            "OPENAI_API_BASE": model.base_url,
            "OPENAI_API_KEY": _PLACEHOLDER_KEY,
        }
        return Preparation(
            env=env,
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
        return [executable, "--model", f"{_PREFIX}/{model.served_id}"]
