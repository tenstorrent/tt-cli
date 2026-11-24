# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Qwen Code adapter: the endpoint on the command line, the model's limits as a
`modelProviders` entry in `~/.qwen/settings.json`.

Qwen Code's CLI takes --auth-type, --openai-api-key, --openai-base-url and
--model, but has no flag for the context window or the output budget. Those are
read only from a `modelProviders` entry, which `--auth-type openai --model <id>`
selects (docs/configuration/model-providers.md). The entry is written only when
the server publishes a window to size them from.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ...errors import ExitCode, TTError
from ..base import (
    ConfigChange,
    LaunchEnv,
    LaunchOptions,
    Preparation,
    RunningModel,
    config_consent,
    output_budget,
    read_json_config,
    write_json_config,
)

PROVIDER_KEY = "openai"
DESCRIPTION = "added by tt launch"
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

    def config_path(self) -> Path:
        return Path.home() / ".qwen" / "settings.json"

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
        steps = [self._argv(executable or self.binaries[0], model)]
        if not model.max_context:
            return Preparation(steps=steps)
        path = self.config_path()
        doc, created = read_json_config(path)
        entry = self._model_entry(model)
        before = None
        kept = []
        for existing in _entries(doc, path):
            if _same_route(existing, entry):
                before = deepcopy(existing)
            else:
                kept.append(existing)
        doc.setdefault("modelProviders", {})[PROVIDER_KEY] = [*kept, entry]
        key = f"modelProviders.{PROVIDER_KEY} entry {model.served_id}"
        return Preparation(
            config=ConfigChange(
                path=path, key=key, block=entry, document=doc, created=created
            ),
            consent=None if before == entry else config_consent(path, key, existing=before),
            steps=steps,
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        if prep.config is not None:
            write_json_config(prep.config.path, prep.config.document)

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(self._argv(env.executable, model))

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        path = self.config_path()
        try:
            doc, _ = read_json_config(path)
            ours = [e for e in _entries(doc, path) if _is_ours(e)]
        except TTError:
            return None
        if not ours:
            return None
        return f"remove tt's entries from modelProviders.{PROVIDER_KEY} in {path}"

    def disconnect(self, env: LaunchEnv) -> None:
        path = self.config_path()
        doc, _ = read_json_config(path)
        theirs = [e for e in _entries(doc, path) if not _is_ours(e)]
        doc["modelProviders"][PROVIDER_KEY] = theirs
        write_json_config(path, doc)

    def _model_entry(self, model: RunningModel) -> dict:
        return {
            "id": model.served_id,
            "name": model.served_id,
            "description": DESCRIPTION,
            "baseUrl": model.base_url,
            "generationConfig": {
                "contextWindowSize": model.max_context,
                "samplingParams": {"max_tokens": output_budget(model.max_context)},
            },
        }

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


def _entries(doc: dict, path: Path) -> list[dict]:
    """The provider's model list: Qwen Code's current `ModelConfig[]` shape."""
    providers = doc.get("modelProviders") or {}
    entries = providers.get(PROVIDER_KEY)
    if entries is None:
        return []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise TTError(
            f"modelProviders.{PROVIDER_KEY} in {path} is not a list of models.",
            why="Qwen Code's current settings shape is a bare array per provider; "
            "tt will not rewrite a block it does not recognise.",
            next_step="Update the block to the array form (see Qwen Code's "
            "model-providers docs), or re-run with --dry-run and add the entry by hand.",
            exit_code=ExitCode.CONFIG,
            details={"path": str(path)},
        )
    return entries


def _same_route(a: dict, b: dict) -> bool:
    """Qwen Code identifies a model by its id and baseUrl."""
    return a.get("id") == b.get("id") and a.get("baseUrl") == b.get("baseUrl")


def _is_ours(entry: dict) -> bool:
    return entry.get("description") == DESCRIPTION
