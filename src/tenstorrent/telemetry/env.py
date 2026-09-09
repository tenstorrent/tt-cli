# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Environment facts telemetry reads. Its own module so attributes.py (what we send)
and session.py (whether/how we send) can share them without an import cycle."""

from __future__ import annotations

import os

# Markers that mean "a machine, not a person, ran this command". CI invocations are kept
# — they are real usage, and `tt.ci` on the span lets them be filtered downstream — but
# they change how delivery has to work; see flush_mode() in session.py.
CI_ENV_VARS = (
    "CI",
    "CONTINUOUS_INTEGRATION",
    "BUILD_NUMBER",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "BUILDKITE",
    "TF_BUILD",
)


def env_flag(name: str) -> bool:
    """Is an opt-out/CI style env var set? Empty, "0" and "false" all mean unset."""
    value = os.environ.get(name)
    return bool(value) and value.strip().lower() not in ("0", "false")


def is_ci() -> bool:
    return any(env_flag(var) for var in CI_ENV_VARS)
