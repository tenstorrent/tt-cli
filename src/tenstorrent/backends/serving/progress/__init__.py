# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Serve progress: log lines in, a live checklist out.

`phases` and `tracker` are pure (text in, events out); `view` is the only part
that touches a terminal. Backends drive them — see
`boot.py` beside this package for tt-inference-server's two-stage boot.
"""

from .phases import HOST_PHASES, MEDIA_PHASES, VLLM_PHASES, Phase, phases_for
from .tracker import Event, PhaseTracker
from .view import Checklist
from .weights import WeightsProgress

__all__ = [
    "Checklist",
    "Event",
    "HOST_PHASES",
    "MEDIA_PHASES",
    "Phase",
    "PhaseTracker",
    "VLLM_PHASES",
    "WeightsProgress",
    "phases_for",
]
