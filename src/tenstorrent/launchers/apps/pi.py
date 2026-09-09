# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""pi adapter: a custom provider in `~/.pi/agent/models.json`.

pi documents that file as the way to add vLLM and other OpenAI-compatible
servers (docs/models.md), so no extension has to be installed. It is re-read
whenever `/model` is opened, so an edit lands without a restart.
"""

from __future__ import annotations

import os
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
    reasoning_parser,
    write_json_config,
)

PROVIDER_KEY = "tenstorrent"
# pi treats models as needing auth before they appear in /model, so a keyless
# local server keeps a placeholder (its docs say the same).
_PLACEHOLDER_KEY = "tt-local"


class Pi:
    id = "pi"
    binaries = ("pi",)
    install_hint = "Install pi (https://github.com/earendil-works/pi), then re-run."
    requires_tool_calling = True
    hands_over_terminal = True

    def config_path(self) -> Path:
        root = os.environ.get("PI_CODING_AGENT_DIR")
        return (Path(root) if root else Path.home() / ".pi" / "agent") / "models.json"

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
        before = deepcopy(doc.get("providers", {}).get(PROVIDER_KEY))
        provider = doc.setdefault("providers", {}).setdefault(PROVIDER_KEY, {})
        provider.update(
            {
                "baseUrl": model.base_url,
                "api": "openai-completions",
                "apiKey": _PLACEHOLDER_KEY,
                # pi's docs name vLLM among the servers that reject the `developer`
                # role and `reasoning_effort`.
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
            }
        )
        # `models` replaces the provider's list, so merge by id to keep models a
        # previous launch added.
        models = {entry["id"]: entry for entry in provider.get("models") or []}
        models[model.served_id] = self._model_entry(model)
        provider["models"] = [models[key] for key in sorted(models)]
        return Preparation(
            config=ConfigChange(
                path=path,
                key=f"providers.{PROVIDER_KEY}",
                block=provider,
                document=doc,
                created=created,
            ),
            # Nothing to approve when the entry is already exactly this.
            consent=None
            if before == provider
            else config_consent(path, f"providers.{PROVIDER_KEY}", existing=before),
            steps=[self._argv(executable or self.binaries[0], model)],
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        assert prep.config is not None
        write_json_config(prep.config.path, prep.config.document)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(self._argv(env.executable, model))

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        path = self.config_path()
        if not json_config_has(path, ("providers",), PROVIDER_KEY):
            return None
        return f"remove providers.{PROVIDER_KEY} from {path}"

    def disconnect(self, env: LaunchEnv) -> None:
        json_config_drop(self.config_path(), ("providers",), PROVIDER_KEY)

    def _model_entry(self, model: RunningModel) -> dict:
        vision = bool(model.entry and model.entry.model_type == "vlm")
        entry = {
            "id": model.served_id,
            "name": model.served_id,
            "reasoning": bool(model.entry and reasoning_parser(model.entry)),
            "input": ["text", "image"] if vision else ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
        if model.max_context:
            entry["contextWindow"] = model.max_context
        return entry

    def _argv(self, executable: str, model: RunningModel) -> list[str]:
        # --provider and --model separately, rather than pi's "provider/id" form:
        # a served id is often a repo path, which that form cannot express.
        return [executable, "--provider", PROVIDER_KEY, "--model", model.served_id]
