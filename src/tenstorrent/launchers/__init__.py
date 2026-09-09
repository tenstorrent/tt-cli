# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Adapters that point an OpenAI-compatible client at a served model.

Two shapes, both described by launchers/base.py: a client already installed on
the machine, which tt configures and hands the terminal to (opencode, pi,
aider), and one tt runs as a container, which needs consent (openwebui,
anythingllm — see launchers/container.py). One module per client in apps/;
this registry is the only place that has to change to add one.
"""

from __future__ import annotations

from .apps.aider import Aider
from .apps.anythingllm import AnythingLLM
from .apps.opencode import OpenCode
from .apps.openwebui import OpenWebUI
from .apps.pi import Pi
from .base import Launcher

LAUNCHERS: dict[str, Launcher] = {
    launcher.id: launcher
    for launcher in (OpenCode(), Pi(), Aider(), OpenWebUI(), AnythingLLM())
}

__all__ = ["LAUNCHERS", "Launcher"]
