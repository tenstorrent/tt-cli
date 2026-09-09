# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Typed configuration defaults and the commented first-run template.

DEFAULTS is the single source of truth for keys and fallback values; the template
mirrors it with human-facing comments (comments survive edits thanks to tomlkit).
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    "telemetry": {
        # Opt-in: nothing is collected or sent until the user says yes — via the
        # first-run prompt or `tt config set telemetry.enabled true`.
        "enabled": False,
        # OTLP/HTTP traces endpoint (PostHog). Full path — the exporter must not
        # append /v1/traces. EU users swap the host for eu.i.posthog.com.
        "endpoint": "https://us.i.posthog.com/i/v1/traces",
        # PostHog write-only project key. Empty = telemetry stays inert (nothing sent).
        "posthog_project_key": "phc_kyqdAU5XuGgkcFtoLjj78rNXNnwoQ9KWj47eBs6TADRr",
        # "async" spools spans to disk and uploads them from a detached process, so no
        # command ever waits on the network. "sync" exports in-process (slower, but a
        # span shows up in the collector immediately) — for development.
        "flush_mode": "async",
    },
    "paths": {
        # Empty string = defer to the HuggingFace defaults / env vars.
        "hf_model_cache_directory": "",
        # Empty string = <cache_dir>/models (XDG cache).
        "tt_model_cache_directory": "",
        # Empty string = ~/data/tt-cache, the conventional location.
        "preloaded_volume_directory": "",
    },
    "tools": {
        "sudo_command": "sudo",
        "override": {},
    },
    "device": {
        "backend": "smi",
    },
    "update": {
        # Once a day, in a detached background process, look up the newest tt release
        # and mention it on the next interactive run. Off: never look.
        "check": True,
    },
}

TEMPLATE = """\
# Tenstorrent CLI configuration.
# Edit freely - comments are preserved. `tt config list` shows effective values and
# where each came from; `tt config set <dotted.key> <value>` edits a single key.
#
# Only the keys you want to override need to be here: anything missing (including
# settings added by a newer tt) falls back to its built-in default, and `tt config list`
# lists them all. When adding a key by hand, put it inside the right [table] - TOML
# assigns a key appended to the end of the file to the LAST table, not the one you meant.
# `tt config set` gets this right for you, and tt warns about keys it doesn't recognize.

[telemetry]
# Anonymous usage data helps us improve the developer experience. OPT-IN: nothing is
# collected or sent unless you enable it (tt asks once on first interactive use).
# What is collected: command names, exit codes, coarse OS info, and argument values
# only when they match a known list (e.g. catalog model names) - never free-form
# values, paths, or anything identifying.
# Details: https://github.com/tenstorrent/tt-cli/blob/main/TELEMETRY.md
# Opt in / out any time: tt config set telemetry.enabled true|false
enabled = false
# OpenTelemetry OTLP/HTTP traces endpoint (PostHog). Full path, incl. /i/v1/traces.
# Override per-run with TT_TELEMETRY_ENDPOINT; TT_TELEMETRY_DISABLED=1 turns it all off.
endpoint = "https://us.i.posthog.com/i/v1/traces"
# PostHog write-only project key. Empty = nothing is sent. Override with
# TT_TELEMETRY_POSTHOG_KEY.
posthog_project_key = "phc_kyqdAU5XuGgkcFtoLjj78rNXNnwoQ9KWj47eBs6TADRr"
# How spans are delivered. "async" (default) appends each span to a local spool and
# uploads batches from a detached process, so no command waits on the network.
# "sync" exports in-process: ~400ms slower per command, but spans reach the collector
# immediately, which is what you want when developing against scripts/otlp_sink.py.
# Override per-run with TT_TELEMETRY_FLUSH_MODE=sync. Set TT_TELEMETRY_LOG_FILE=<path>
# to also write every span to a file in OTLP/JSON format and see exactly what is sent.
flush_mode = "async"

[paths]
# HuggingFace cache root (HF_HOME semantics; weights land in <dir>/hub).
# Empty = HF_HOME env or ~/.cache/huggingface. `tt serve` mounts this same
# cache into the inference server, so pulled models are never re-downloaded.
hf_model_cache_directory = ""
# Directory for TT-compiled model artifacts. Empty = XDG cache dir.
tt_model_cache_directory = ""
# Root holding pre-seeded tt-inference-server volumes (weights + tt_metal_cache),
# laid out as <root>/volume_id_<impl>-<model>-v<version>. A volume is used only
# when the directory for that exact model, impl and version is present; otherwise
# the server falls back to its own docker volume as usual.
# Empty = ~/data/tt-cache, the conventional location.
preloaded_volume_directory = ""

[tools]
# Command used to elevate privileges (device reset, driver install).
# Set to "" to never use sudo.
sudo_command = "sudo"

# Per-tool binary overrides, e.g.:  tt-smi = "/usr/local/bin/tt-smi"
[tools.override]

[device]
# Device backend: "smi" delegates to tt-smi. A native backend will land later.
backend = "smi"

[update]
# Once a day tt looks up the newest release (a single request to PyPI, in a detached
# background process, carrying nothing but tt's version) and mentions it on your
# next interactive run. `tt self update` applies it where tt owns its environment.
# false = never look. Per-run: TT_NO_UPDATE_CHECK=1; --offline also skips it.
check = true
"""


def flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys. Leaves empty tables out."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def known_keys() -> list[str]:
    """Every documented dotted key, sorted. Excludes the dynamic tools.override.*
    namespace, which has no fixed key set (see is_known_key)."""
    return sorted(flatten(DEFAULTS))


def default_for(dotted_key: str) -> Any:
    """Return the default value for a dotted key. KeyError if unknown or a table."""
    node: Any = DEFAULTS
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    if isinstance(node, dict):
        raise KeyError(dotted_key)
    return node


def is_known_key(dotted_key: str) -> bool:
    """A key is known if it has a static default, or lives under the dynamic
    tools.override table (per-tool binary paths have no fixed key set)."""
    parts = dotted_key.split(".")
    if parts[:2] == ["tools", "override"] and len(parts) == 3 and parts[2]:
        return True
    try:
        default_for(dotted_key)
        return True
    except KeyError:
        return False
