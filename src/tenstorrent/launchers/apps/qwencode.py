# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Qwen Code adapter: passed entirely on the command line, nothing on disk.

Unlike opencode/pi (a JSON config file) or aider (environment variables, for
lack of CLI flags), Qwen Code's own CLI exposes --auth-type, --openai-api-key,
--openai-base-url and --model directly, so there is nothing to write and
nothing to undo.
"""

from __future__ import annotations

from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel

_PLACEHOLDER_KEY = "tt-local"


class QwenCode:
    id = "qwencode"
    binaries = ("qwen",)
    install_hint = (
        "Install Qwen Code (npm install -g @qwen-code/qwen-code; needs Node.js "
        "22+): https://github.com/QwenLM/qwen-code, then re-run."
    )
    requires_tool_calling = True
    hands_over_terminal = True

    def target(self) -> str:
        return "command-line arguments (this run only)"

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        return Preparation(steps=[self._argv(executable or self.binaries[0], model)])

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        """Nothing on disk: everything is passed as CLI arguments at hand-off."""

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(self._argv(env.executable, model))

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists, so there is never anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        return [
            executable,
            "--auth-type",
            "openai",
            "--openai-api-key",
            _PLACEHOLDER_KEY,
            "--openai-base-url",
            model.base_url,
            "--model",
            model.served_id,
        ]
