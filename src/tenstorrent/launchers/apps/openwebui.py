# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Open WebUI adapter: its container pointed at the served model.

Open WebUI builds its own model picker from GET /v1/models, so it needs the
endpoint but not the model name.
"""

from __future__ import annotations

from ..base import RunningModel
from ..container import ContainerLauncher

IMAGE = "ghcr.io/open-webui/open-webui:main"


class OpenWebUI(ContainerLauncher):
    id = "openwebui"
    image = IMAGE
    container = "tt-open-webui"
    volume = "tt-open-webui-data"
    volume_path = "/app/backend/data"
    container_port = 8080
    base_url_env = "OPENAI_API_BASE_URL"

    def container_env(self, model: RunningModel, inner_url: str) -> dict:
        return {
            self.base_url_env: inner_url,
            # Unauthenticated, but the variable must be set for Open WebUI to
            # enable the OpenAI connection at all.
            "OPENAI_API_KEY": "tt-local",
            "ENABLE_OLLAMA_API": "false",
            # Without this, the first run's settings are persisted into its
            # database and later env changes are ignored.
            "ENABLE_PERSISTENT_CONFIG": "false",
        }
