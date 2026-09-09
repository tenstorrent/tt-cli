# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Span-attribute builders — the single chokepoint for what leaves the machine.

Everything a span carries is produced here, so the anonymization policy has exactly
one place to audit: command path, *which* options were set (names only), a small
allowlist of argument values, the exit-code category, and coarse host/version facts.
No file paths, tokens, hostnames, or usernames.

The value allowlist (`_SAFE_VALUES`) works by **validate then record, never record then
sanitize**: an argument is only exported when its value is a member of a closed
vocabulary the CLI already holds in code — catalog model names, catalog model-type and
hardware sets, an Enum's members, the device-config map, the config schema's own keys,
or a strict semver. Anything else is *dropped entirely*, so a typo, a filesystem path
or a pasted token can never leave the box. Values are never truncated and never hashed:
a hash of a path is still a stable per-user identifier, which would destroy the analytic
value while keeping the re-identification risk.

Deliberately absent, and must stay absent:
- `tt config set <value>` — the schema's own keys include telemetry.posthog_project_key
  (a secret) plus path keys and tools.override.* (absolute paths with the username).
  The *key* is recorded; the value never is.
- `tt compile` / `tt train` argv — permissive by design, so arbitrary local filenames.
- `tt report feedback` — specced to carry free-text feedback and optionally an email.
"""

from __future__ import annotations

import functools
import platform
import re
from enum import Enum
from typing import Any, Callable

from .. import __version__
from ..errors import ExitCode
from .env import is_ci


def resource_attributes(instance_id: str) -> dict[str, Any]:
    """Process-wide facts. `instance_id` is a random per-install UUID (a rotation-safe
    pseudonym, not a fingerprint) — no hostname, username, or environment dump."""
    return {
        "service.name": "tt",
        "service.version": __version__,
        "os.type": platform.system(),
        "os.arch": platform.machine(),
        "process.runtime.version": platform.python_version(),
        "tt.instance_id": instance_id,
        # Whether a person or a build ran this. A boolean derived from a fixed list of
        # env var *names* (see env.py) — never the values, which carry repo slugs, branch
        # names, build URLs and job numbers. Recorded rather than used to drop the span:
        # CI usage is real usage, and this lets it be separated out downstream.
        "tt.ci": is_ci(),
    }


def command_attributes(click_ctx: Any) -> dict[str, Any]:
    """Per-command facts: command path, names of options set, allowlisted arg values."""
    attrs: dict[str, Any] = {}
    if click_ctx is None:
        return attrs
    command = getattr(click_ctx, "command_path", None)
    if command:
        attrs["tt.command"] = command
    options = _options_set(click_ctx)
    if options:
        attrs["tt.options_set"] = options
    attrs.update(_safe_values(click_ctx))
    return attrs


# -- the value allowlist -------------------------------------------------------------
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@functools.lru_cache(maxsize=1)
def _catalog_vocabularies() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(names, model types, hardware) from the bundled compat spec.

    Derived from ModelCatalog rather than copied here, so a spec refresh can't
    leave a stale allowlist behind. Loaded lazily and cached: only the
    model-bearing commands ever pay for it. cached_sizes={} skips the HF cache
    scan — cache state is irrelevant to the vocabulary.
    """
    from ..modelhub.catalog import ModelCatalog

    entries = ModelCatalog().list(cached_sizes={})
    names = frozenset(e.name for e in entries)
    model_types = frozenset(e.model_type for e in entries)
    hardware = frozenset(hw for e in entries for hw in e.hardware)
    return names, model_types, hardware


@functools.lru_cache(maxsize=1)
def _device_configs() -> frozenset[str]:
    from ..backends.serving.inference_server import _BOARDS_TO_DEVICE

    return frozenset(_BOARDS_TO_DEVICE.values())


@functools.lru_cache(maxsize=1)
def _config_keys() -> frozenset[str]:
    from ..config import schema

    return frozenset(schema.flatten(schema.DEFAULTS))


def _model_name(value: Any) -> str | None:
    return _member(value, _catalog_vocabularies()[0])


def _model_type(value: Any) -> str | None:
    return _member(value, _catalog_vocabularies()[1])


def _hardware(value: Any) -> str | None:
    return _member(value, _catalog_vocabularies()[2])


def _device_config(value: Any) -> str | None:
    return _member(value, _device_configs())


def _enum_value(value: Any) -> str | None:
    """Enum-typed params are bounded by construction — Typer already rejected anything
    outside the members, so no vocabulary of our own is needed."""
    return str(value.value) if isinstance(value, Enum) else None


def _config_key(value: Any) -> str | None:
    key = str(value)
    if key in _config_keys():
        return key
    # tools.override.<tool> is a dynamic namespace that flatten() drops (empty table).
    # Record the namespace only: the suffix is a tool name, but collapsing avoids
    # depending on the manifest here, and the interesting signal is "an override was set".
    if key.startswith("tools.override."):
        return "tools.override.*"
    return None


def _installer_version(value: Any) -> str | None:
    version = str(value)
    return version if _SEMVER_RE.match(version) else None


def _index_count(value: Any) -> int | None:
    """Device indices are harmless ints, but the count is the useful signal."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return len(value)


def _member(value: Any, vocabulary: frozenset[str]) -> str | None:
    text = str(value)
    return text if text in vocabulary else None


# (command path without the program name) -> {param name: (attribute, validator)}
_SAFE_VALUES: dict[str, dict[str, tuple[str, Callable[[Any], Any]]]] = {
    "model pull": {"name": ("tt.model", _model_name)},
    "model info": {"name": ("tt.model", _model_name)},
    "model compile": {"name": ("tt.model", _model_name)},
    "model list": {
        "model_type": ("tt.model_type", _model_type),
        "hardware": ("tt.hardware", _hardware),
    },
    "serve": {
        "model": ("tt.model", _model_name),
        "workflow": ("tt.workflow", _enum_value),
        "device": ("tt.device_config", _device_config),
    },
    "config get": {"key": ("tt.config_key", _config_key)},
    # NB: no "value" entry, and there must never be one.
    "config set": {"key": ("tt.config_key", _config_key)},
    "update": {"version": ("tt.installer_version", _installer_version)},
    "device status": {"index": ("tt.device_count", _index_count)},
    "device info": {"index": ("tt.device_count", _index_count)},
    "device reset": {"index": ("tt.device_count", _index_count)},
}


def _safe_values(click_ctx: Any) -> dict[str, Any]:
    """Allowlisted argument values for this command, dropping anything unrecognised.

    Records regardless of whether the user supplied the value or it came from a default:
    a bounded default (`--workflow server`) is safe and worth knowing. `tt.options_set`
    already distinguishes supplied from defaulted.
    """
    command = getattr(click_ctx, "command_path", None)
    params = getattr(click_ctx, "params", None)
    if not command or not params:
        return {}
    # Drop the program name so `python -m tenstorrent` and `tt` map the same way.
    allowed = _SAFE_VALUES.get(" ".join(str(command).split()[1:]))
    if not allowed:
        return {}
    attrs: dict[str, Any] = {}
    for name, (attribute, validate) in allowed.items():
        if name not in params or params[name] is None:
            continue
        try:
            safe = validate(params[name])
        except Exception:
            # A broken vocabulary must never break the command, and must never
            # fall through to recording the raw value.
            continue
        if safe is not None:
            attrs[attribute] = safe
    return attrs


def error_attributes(exit_code: ExitCode) -> dict[str, Any]:
    """Only the exit-code *category* — never the error's what/why text, which can
    contain paths or tool argv (see errors.py)."""
    return {
        "tt.exit_code": int(exit_code),
        "tt.exit_code_name": exit_code.name,
    }


def _options_set(click_ctx: Any) -> list[str]:
    """Names (never values) of parameters the user actually supplied.

    Uses click's parameter-source tracking to distinguish user-supplied values from
    defaults, so we record *that* `--json` was used, not what any option was set to.
    Falls back to emitting nothing if the click API isn't available.
    """
    get_source = getattr(click_ctx, "get_parameter_source", None)
    params = getattr(click_ctx, "params", None)
    if get_source is None or not params:
        return []
    names: list[str] = []
    for name in params:
        try:
            source = get_source(name)
        except Exception:
            continue
        # ParameterSource.DEFAULT means the user didn't supply it; anything else
        # (COMMANDLINE / ENVIRONMENT / PROMPT / DEFAULT_MAP) counts as "set".
        if getattr(source, "name", "DEFAULT") != "DEFAULT":
            names.append(name)
    return sorted(names)
