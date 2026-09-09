# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""OpenCode adapter: an openai-compatible provider block named `tenstorrent`."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..base import (
    ConfigChange,
    LaunchEnv,
    LaunchOptions,
    Preparation,
    RunningModel,
    config_consent,
    json_config_drop,
    json_config_has,
    read_json_config,
    user_config_dir,
    write_json_config,
)

PROVIDER_KEY = "tenstorrent"
_SCHEMA = "https://opencode.ai/config.json"
# The provider schema wants a key; models tt serves are unauthenticated and ignore it.
_PLACEHOLDER_KEY = "tt-local"


class OpenCode:
    id = "opencode"
    binaries = ("opencode",)
    install_hint = "Install OpenCode (https://opencode.ai/docs/), then re-run."
    requires_tool_calling = True
    hands_over_terminal = True

    def config_path(self) -> Path:
        return user_config_dir() / "opencode" / "opencode.json"

    def target(self) -> str:
        return str(self.config_path())

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        path = self.config_path()
        doc, created = read_json_config(path)
        # Snapshot before mutating: setdefault hands back the live dict, and the
        # comparison below decides whether the user is asked at all.
        before = deepcopy(doc.get("provider", {}).get(PROVIDER_KEY))
        provider = doc.setdefault("provider", {}).setdefault(PROVIDER_KEY, {})
        provider.update(
            {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Tenstorrent",
                "options": {"baseURL": model.base_url, "apiKey": _PLACEHOLDER_KEY},
            }
        )
        # Merge rather than replace: a model configured by an earlier launch stays
        # selectable in OpenCode's own picker.
        provider.setdefault("models", {})[model.served_id] = {"name": model.served_id}
        if created:
            doc.setdefault("$schema", _SCHEMA)
        return Preparation(
            config=ConfigChange(
                path=path,
                key=f"provider.{PROVIDER_KEY}",
                block=provider,
                document=doc,
                created=created,
            ),
            # Nothing to approve when the entry is already exactly this.
            consent=None
            if before == provider
            else config_consent(path, f"provider.{PROVIDER_KEY}", existing=before),
            steps=[self._argv(executable or self.binaries[0], model)],
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        assert prep.config is not None
        write_json_config(prep.config.path, prep.config.document)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(self._argv(env.executable, model))

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        path = self.config_path()
        if not json_config_has(path, ("provider",), PROVIDER_KEY):
            return None
        return f"remove provider.{PROVIDER_KEY} from {path}"

    def disconnect(self, env: LaunchEnv) -> None:
        json_config_drop(self.config_path(), ("provider",), PROVIDER_KEY)

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        # served_id is whatever the API expects, so a repo-shaped id makes a ref
        # with two slashes (tenstorrent/Qwen/Qwen3-32B). Correctness with the
        # server wins: a shortened key would be sent verbatim and 404.
        return [executable, "--model", f"{PROVIDER_KEY}/{model.served_id}"]
