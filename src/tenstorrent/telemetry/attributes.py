# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Event builders — the single chokepoint for what leaves the machine.

Everything a telemetry event carries is produced here, so the anonymization policy has
exactly one place to audit: command path, *which* options were set (names only), a
small allowlist of argument values, the exit-code category, the duration, and coarse
host/version facts. No file paths, tokens, hostnames, or usernames — and no IP: every
event carries `$geoip_disable`, which tells PostHog to neither keep the client address
nor derive a location from it, and the project is configured to discard the address at
ingest as a second layer (see TELEMETRY.md). What no client can prevent is the server
*seeing* the address of the connection; only a relay in front of PostHog would.

The value allowlist (`_SAFE_VALUES`) works by **validate then record, never record then
sanitize**: an argument is only exported when its value is a member of a closed
vocabulary the CLI already holds in code — catalog model names, catalog model-type and
hardware sets, an Enum's members, the device-config map, the config schema's own keys,
or a strict semver. Anything else is *dropped entirely*, so a typo, a filesystem path
or a pasted token can never leave the box. Values are never truncated and never hashed:
a hash of a path is still a stable per-user identifier, which would destroy the analytic
value while keeping the re-identification risk.

The same rule covers errors: the exit-code *category*, an optional `reason` slug a
raise site attaches on purpose (validated against a closed grammar), and for a crash
the exception's *class name*. Never the message, never a stack frame — both routinely
carry paths and tool argv.

Deliberately absent, and must stay absent:
- `tt config set <value>` — the schema's own keys include telemetry.posthog_project_key
  (a secret) plus path keys and tools.override.* (absolute paths with the username).
  The *key* is recorded; the value never is.
- `tt compile` / `tt train` argv — permissive by design, so arbitrary local filenames.
- `tt report feedback` — specced to carry free-text feedback and optionally an email.
- `tt launch <client> --url` — free text; the client is identified by the command path.

`EVENT_PROPERTY_NAMES` is the complete, closed set of property names an event may
carry. A test pins it, so adding a property is a deliberate, reviewed change.
"""

from __future__ import annotations

import functools
import platform
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from .. import __version__
from ..errors import ExitCode, TTError
from .env import is_ci

# The one event tt sends: one per leaf command invocation.
COMMAND_EVENT = "tt_command"
# PostHog's library identification — what its SDK views group on.
LIB_NAME = "tt-cli"


# -- install-level facts --------------------------------------------------------------
def install_properties() -> dict[str, Any]:
    """Process-wide facts stamped on every event. No hostname, username, or
    environment dump."""
    return {
        "tt_version": __version__,
        "os_type": platform.system(),
        "os_arch": platform.machine(),
        "python_version": platform.python_version(),
        # Whether a person or a build ran this. A boolean derived from a fixed list of
        # env var *names* (see env.py) — never the values, which carry repo slugs, branch
        # names, build URLs and job numbers. Recorded rather than used to drop the event:
        # CI usage is real usage, and this lets it be separated out downstream.
        "ci": is_ci(),
    }


def person_properties(*, internal: bool = False) -> dict[str, Any]:
    """Facts kept on the *install's* PostHog person profile (`$set` overwrites on every
    event, `$set_once` sticks from the first one). The profile is keyed by the random
    per-install id, so it describes a machine's tt install, never a person.

    `internal` is the self-declared "this is a Tenstorrent staff install" flag
    (`telemetry.internal` / TT_TELEMETRY_INTERNAL). Kept on the profile as well as on
    each event so a person-property filter in PostHog excludes the install's whole
    history, including events recorded before the flag was set."""
    facts = install_properties()
    return {
        "$set": {
            "tt_version": facts["tt_version"],
            "os_type": facts["os_type"],
            "os_arch": facts["os_arch"],
            "python_version": facts["python_version"],
            "internal": bool(internal),
        },
        "$set_once": {
            "first_seen_version": facts["tt_version"],
            "first_seen_os_type": facts["os_type"],
        },
    }


# -- per-command facts ----------------------------------------------------------------
def command_properties(click_ctx: Any) -> dict[str, Any]:
    """Command path, names of options set, allowlisted arg values."""
    props: dict[str, Any] = {}
    if click_ctx is None:
        return props
    command = getattr(click_ctx, "command_path", None)
    if command:
        props["command"] = command
    options = _options_set(click_ctx)
    if options:
        props["options_set"] = options
    props.update(_safe_values(click_ctx))
    return props


# -- the value allowlist -------------------------------------------------------------
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
# `reason` slugs are dotted snake_case chosen at the raise site. Anything that does not
# match is dropped: a slug is a closed vocabulary by construction, not a message.
_REASON_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
_REASON_MAX_LEN = 64


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


# (command path without the program name) -> {param name: (property, validator)}
_SAFE_VALUES: dict[str, dict[str, tuple[str, Callable[[Any], Any]]]] = {
    "model pull": {"name": ("model", _model_name)},
    "model info": {"name": ("model", _model_name)},
    "model compile": {"name": ("model", _model_name)},
    "model list": {
        "model_type": ("model_type", _model_type),
        "hardware": ("hardware", _hardware),
    },
    "serve": {
        "model": ("model", _model_name),
        "workflow": ("workflow", _enum_value),
        "device": ("device_config", _device_config),
    },
    "config get": {"key": ("config_key", _config_key)},
    # NB: no "value" entry, and there must never be one.
    "config set": {"key": ("config_key", _config_key)},
    "update": {"version": ("installer_version", _installer_version)},
    "device status": {"index": ("device_count", _index_count)},
    "device info": {"index": ("device_count", _index_count)},
    "device reset": {"index": ("device_count", _index_count)},
}


def _safe_values(click_ctx: Any) -> dict[str, Any]:
    """Allowlisted argument values for this command, dropping anything unrecognised.

    Records regardless of whether the user supplied the value or it came from a default:
    a bounded default (`--workflow server`) is safe and worth knowing. `options_set`
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
    props: dict[str, Any] = {}
    for name, (prop, validate) in allowed.items():
        if name not in params or params[name] is None:
            continue
        try:
            safe = validate(params[name])
        except Exception:
            # A broken vocabulary must never break the command, and must never
            # fall through to recording the raw value.
            continue
        if safe is not None:
            props[prop] = safe
    return props


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


# -- outcome --------------------------------------------------------------------------
def error_properties(exit_code: ExitCode, error: Any = None) -> dict[str, Any]:
    """The exit-code *category*, plus one bounded fact about the failure.

    A TTError contributes its `reason` slug when the raise site set one and it passes
    the slug grammar; anything else about it (what/why/next_step) is text that can
    carry paths or tool argv, so it never leaves. Any other exception contributes only
    its class name — `str(exc)`, `exc.args`, and notes are never read.
    """
    props: dict[str, Any] = {
        "exit_code": int(exit_code),
        "exit_code_name": exit_code.name,
    }
    if isinstance(error, TTError):
        reason = _reason(getattr(error, "reason", None))
        if reason is not None:
            props["reason"] = reason
    elif isinstance(error, BaseException):
        props["exception_type"] = type(error).__name__
    return props


def _reason(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > _REASON_MAX_LEN:
        return None
    return value if _REASON_RE.match(value) else None


# -- the event ------------------------------------------------------------------------
def build_event(
    click_ctx: Any,
    *,
    instance_id: str,
    exit_code: ExitCode,
    error: Any = None,
    duration_ms: int = 0,
    internal: bool = False,
) -> dict[str, Any]:
    """One PostHog event for one command. Pure: no I/O, no clock other than `timestamp`.

    - `uuid` is minted here so a batch that is retried after a lost response is
      deduplicated server-side rather than double-counted.
    - `timestamp` is the capture time in UTC (tz-aware ISO 8601). Events sit in the
      spool for minutes to days before upload, and a naive local time would shift them
      by the user's UTC offset.
    - `distinct_id` is the random per-install id from telemetry.toml: identified
      events (a person profile per *install*) are what make retention, lifecycle and
      cohort insights work; the id is generated, not derived from hardware or accounts.
    """
    properties: dict[str, Any] = {}
    properties.update(command_properties(click_ctx))
    properties.update(error_properties(exit_code, error))
    properties["duration_ms"] = max(0, int(duration_ms))
    properties.update(install_properties())
    properties["internal"] = bool(internal)
    properties["$lib"] = LIB_NAME
    properties["$lib_version"] = __version__
    # No location, not even country: PostHog must not enrich from the request address.
    properties["$geoip_disable"] = True
    properties.update(person_properties(internal=internal))
    return {
        "event": COMMAND_EVENT,
        "uuid": str(uuid.uuid4()),
        "distinct_id": instance_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "properties": properties,
    }


# Every property name an event may carry. Closed on purpose: tests assert equality, so
# growing this set is a reviewed change, and nothing (an `$ip`, an SDK-injected default)
# can appear by accident.
EVENT_PROPERTY_NAMES: frozenset[str] = frozenset(
    {
        "command",
        "options_set",
        "exit_code",
        "exit_code_name",
        "reason",
        "exception_type",
        "duration_ms",
        "tt_version",
        "os_type",
        "os_arch",
        "python_version",
        "ci",
        "internal",
        "$lib",
        "$lib_version",
        "$geoip_disable",
        "$set",
        "$set_once",
    }
    | {prop for params in _SAFE_VALUES.values() for prop, _ in params.values()}
)
