# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Shared pieces for the `tt launch` adapters.

An adapter never installs its client. It locates one the user already has, writes
the least config needed to reach a model that is *already* being served, and hands
over the terminal.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..errors import ExitCode, TTError
from ..models.model import ModelInfo

if TYPE_CHECKING:  # pragma: no cover
    from ..output import OutputManager
    from ..tools.runner import Runner

VLLM_ENGINE = "vLLM"  # the only engine that serves /v1/chat/completions
DEFAULT_WEB_PORT = 3000  # host port for a client that runs as a web service


@dataclass(frozen=True)
class RunningModel:
    """One model a live server reports it is serving."""

    served_id: str  # the id the API expects; often an HF repo (Qwen/Qwen3-32B)
    base_url: str  # OpenAI-compatible base, ending in /v1
    max_context: int | None = None
    entry: ModelInfo | None = None  # catalog entry, when tt knows this id


@dataclass(frozen=True)
class ConfigChange:
    """A client's config file as the adapter would leave it."""

    path: Path
    key: str  # dotted location of the block the adapter owns
    block: dict  # just that block, for display
    document: dict  # the whole file, ready to write
    created: bool  # the file did not exist


@dataclass(frozen=True)
class LaunchOptions:
    """Per-run knobs from the command line that an adapter may need."""

    web_port: int = DEFAULT_WEB_PORT


@dataclass(frozen=True)
class Preparation:
    """What an adapter would do, in a shape the command can print or emit."""

    rows: dict = field(default_factory=dict)  # adapter-specific renderer/JSON rows
    config: ConfigChange | None = None
    steps: list = field(default_factory=list)  # commands tt would run, as argv lists
    consent: str | None = None  # the part that needs a yes, if any
    url: str | None = None  # where to reach a client that runs as a service
    # For clients configured by environment rather than by a file. Merged into the
    # child's environment at hand-off, so nothing on disk is touched.
    env: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchEnv:
    """Everything an adapter needs to actually do its work."""

    executable: str
    runner: "Runner"
    output: "OutputManager"


class Launcher(Protocol):
    id: str
    binaries: tuple[str, ...]  # candidate executables, in preference order
    install_hint: str
    # False for clients that only chat (Open WebUI), which any model can back.
    requires_tool_calling: bool
    # True: the client takes over this terminal. False: it runs as a service, so
    # starting it *is* its only action.
    hands_over_terminal: bool

    def target(self) -> str:
        """What this adapter configures, for `tt launch list`."""
        ...

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner: "Runner",
    ) -> Preparation: ...

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None: ...

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None: ...

    def disconnect_plan(self, executable: str | None, runner: "Runner") -> str | None:
        """What `tt launch disconnect` would undo, or None if there is nothing."""
        ...

    def disconnect(self, env: LaunchEnv) -> None:
        """Undo it, leaving the client's own data alone."""
        ...


def resolve_executable(launcher: Launcher, config) -> str:
    """Locate an installed client: TT_TOOL_BIN_<ID> → tools.override.<id> → PATH.

    Deliberately not ToolRegistry: tt neither installs nor pins these tools, so a
    miss has to point at the tool's own installer rather than `tt update`.
    """
    env_key = f"TT_TOOL_BIN_{launcher.id.upper().replace('-', '_')}"
    override = config.get(f"tools.override.{launcher.id}")
    found = os.environ.get(env_key) or (str(override) if override else None)
    for binary in launcher.binaries if not found else ():
        found = shutil.which(binary)
        if found:
            break
    if not found:
        raise TTError(
            f"{' or '.join(launcher.binaries)} is not installed.",
            why=f"`tt launch {launcher.id}` drives software you already have; "
            "it does not install any.",
            next_step=launcher.install_hint,
            exit_code=ExitCode.TOOL_MISSING,
            details={"tool": launcher.id},
        )
    return found


# -- model capability ------------------------------------------------------------
def serves_chat(entry: ModelInfo) -> bool:
    """Whether tt serves this model through vLLM, i.e. /v1/chat/completions."""
    return VLLM_ENGINE in entry.engines


def tool_call_parser(entry: ModelInfo) -> str | None:
    """The vLLM tool-call parser tt would launch this model with, if any.

    Read across devices: no entry in the support list publishes a different parser
    per device (pinned by a test), so tool calling is a property of the model and
    needs no board detection here.
    """
    for support in entry.devices.values():
        if VLLM_ENGINE in support.engines and support.tool_call_parser:
            return support.tool_call_parser
    return None


def reasoning_parser(entry: ModelInfo) -> str | None:
    """The vLLM reasoning parser for this model, if any — clients that model
    thinking as a capability need to know."""
    for support in entry.devices.values():
        if VLLM_ENGINE in support.engines and support.reasoning_parser:
            return support.reasoning_parser
    return None


def tool_calling_models(catalog, limit: int = 3) -> list[str]:
    """A few models that do support tool calling, to name in a refusal."""
    # cached_sizes={} skips the HF cache scan: only names are needed.
    names = [m.name for m in catalog.list(cached_sizes={}) if tool_call_parser(m)]
    return names[:limit]


# -- JSON config files -----------------------------------------------------------
def read_json_config(path: Path) -> tuple[dict, bool]:
    """A client's existing config, plus whether the file has to be created.

    A file that does not parse is never overwritten: tt reports it and leaves the
    merge to the user.
    """
    if not path.exists():
        return {}, True
    try:
        doc = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError as exc:
        raise TTError(
            f"{path} is not valid JSON.",
            why=f"Refusing to overwrite a config tt cannot parse ({exc}).",
            next_step="Fix the file, or re-run with --dry-run and merge the block by hand.",
            exit_code=ExitCode.CONFIG,
            details={"path": str(path)},
        ) from exc
    # Valid JSON that is not an object (a list, a string, a number) would otherwise
    # reach the callers' .get()/.setdefault() and crash with AttributeError.
    if not isinstance(doc, dict):
        raise TTError(
            f"{path} is not a JSON object.",
            why=f"Its top level is {type(doc).__name__}, so tt cannot merge settings "
            "into it.",
            next_step="Fix the file, or re-run with --dry-run and add the block by hand.",
            exit_code=ExitCode.CONFIG,
            details={"path": str(path)},
        )
    return doc, False


def write_json_config(path: Path, doc: dict) -> None:
    """Write through a sibling temp file: a crash cannot truncate the user's config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tt-tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def config_consent(path: Path, key: str, *, existing: dict | None) -> str:
    """What to ask before editing a config file tt does not own. Always names the
    file, so nobody is asked to approve an edit they cannot see the target of."""
    if existing is None:
        return f"{'Add' if path.exists() else 'Create'} {key} in {path}"
    return f"Update {key} in {path}"


def json_config_has(path: Path, parents: tuple[str, ...], key: str) -> bool:
    """Whether a nested key is present, without parsing errors mattering: an
    unreadable file has nothing tt can claim to have put there."""
    try:
        doc, _ = read_json_config(path)
    except TTError:
        return False
    for parent in parents:
        doc = doc.get(parent) or {}
    return key in doc


def json_config_drop(path: Path, parents: tuple[str, ...], key: str) -> None:
    """Delete one key, leaving the rest of the file — including now-empty parents,
    which the client may have had before tt touched anything."""
    doc, _ = read_json_config(path)
    table = doc
    for parent in parents:
        table = table.get(parent) or {}
    table.pop(key, None)
    write_json_config(path, doc)


def user_config_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    return Path(root) if root else Path.home() / ".config"
