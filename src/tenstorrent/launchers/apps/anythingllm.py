# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""AnythingLLM adapter: its container pointed at the served model.

Unlike Open WebUI it does not discover models from the endpoint, so the served
id and the context window are pinned into its environment.
"""

from __future__ import annotations

from ..base import RunningModel
from ..container import ContainerLauncher

IMAGE = "mintplexlabs/anythingllm:latest"
# AnythingLLM's own default, used when the server does not publish max_model_len.
_FALLBACK_CONTEXT = 32768


class AnythingLLM(ContainerLauncher):
    id = "anythingllm"
    image = IMAGE
    container = "tt-anythingllm"
    volume = "tt-anythingllm-storage"
    volume_path = "/app/server/storage"
    container_port = 3001
    base_url_env = "GENERIC_OPEN_AI_BASE_PATH"
    # Its own docs require this for the bundled LanceDB and native embedder.
    extra_run_args = ("--cap-add", "SYS_ADMIN")

    def container_env(self, model: RunningModel, inner_url: str) -> dict:
        return {
            "STORAGE_DIR": self.volume_path,
            # Embeddings and vectors stay local: no second provider to configure.
            "EMBEDDING_ENGINE": "native",
            "VECTOR_DB": "lancedb",
            "LLM_PROVIDER": "generic-openai",
            self.base_url_env: inner_url,
            "GENERIC_OPEN_AI_API_KEY": "tt-local",
            # It has no model discovery, so the choice is made here.
            "GENERIC_OPEN_AI_MODEL_PREF": model.served_id,
            "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT": str(
                model.max_context or _FALLBACK_CONTEXT
            ),
        }
