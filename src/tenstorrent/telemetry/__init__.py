# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Anonymous, opt-in usage telemetry via OpenTelemetry (see session.py)."""

from __future__ import annotations

from .session import NULL_SESSION, TelemetrySession

__all__ = ["NULL_SESSION", "TelemetrySession"]
