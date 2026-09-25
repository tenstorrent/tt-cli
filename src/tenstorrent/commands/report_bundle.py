# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Support-bundle collectors for `tt report bundle`.

Every collector is fenced: a broken stack is exactly when someone needs a bundle,
so a source that fails becomes a note in manifest.json instead of a failed command.
Every text member passes through redact() before it is written. Unlike
`tt report issue`, the bundle keeps hostnames and absolute paths: it is meant to be
handed to Tenstorrent support, not pasted into a public issue.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import platform
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from .. import __version__
from ..backends.device import get_device_backend
from ..backends.serving.inference_server import CONTAINER_PREFIX, InferenceServerBackend
from ..backends.smi import parse_snapshot
from ..context import AppContext
from ..errors import ExitCode, TTError
from ..output import to_jsonable

# Per-file tail cap for tt-cli and workflow logs; a long serve can leave logs of
# hundreds of MB, and the end is where the failure is.
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_WORKFLOW_LOGS = 10
CONTAINER_LOG_TAIL = 5000

# How the three serving backends' containers are recognised (see also
# model_manager._LABEL and the tt-studio image name).
_MODEL_MANAGER_LABEL = "org.tenstorrent.tt-model"
_STUDIO_IMAGE_PREFIX = "ghcr.io/tenstorrent/tt-studio/studio_images"

_ENV_PREFIXES = ("TT_", "HF_")
_ENV_ALWAYS = ("HF_TOKEN", "JWT_SECRET", "SERVICE_PORT")
_ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET")

_REDACTED = "<redacted>"
# Values already rendered as a placeholder (<set>, <redacted>) are left alone.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(JWT_SECRET|HF_TOKEN)\b(\s*[=:]\s*[\"']?)[^\s\"'<]+", re.IGNORECASE),
        rf"\1\2{_REDACTED}",
    ),
    (re.compile(r"(Authorization:\s*Bearer\s+)\S+", re.IGNORECASE), rf"\1{_REDACTED}"),
    (re.compile(r"(\btoken=)[^\s&\"'<]+", re.IGNORECASE), rf"\1{_REDACTED}"),
    # HuggingFace tokens and PostHog project keys (telemetry.posthog_project_key).
    (re.compile(r"\b(?:hf|phc)_[A-Za-z0-9]{20,}\b"), _REDACTED),
)


def redact(text: str) -> str:
    """Blank out known secret shapes, line by line, keeping the line count intact."""
    return "\n".join(_redact_line(line) for line in text.split("\n"))


def _redact_line(line: str) -> str:
    for pattern, replacement in _REDACTIONS:
        line = pattern.sub(replacement, line)
    return line


@dataclasses.dataclass
class BundleEntry:
    name: str  # path inside the archive, e.g. "tt-logs/run.log"
    data: bytes
    notes: list[str] = dataclasses.field(default_factory=list)


Note = Callable[[str], None]


def _text_entry(name: str, text: str, notes: list[str] | None = None) -> BundleEntry:
    return BundleEntry(name, redact(text).encode("utf-8"), list(notes or []))


def _json_entry(name: str, obj, notes: list[str] | None = None) -> BundleEntry:
    return _text_entry(name, json.dumps(to_jsonable(obj), indent=2, sort_keys=True), notes)


def _tail_bytes(path: Path, limit: int) -> tuple[bytes, bool]:
    """The last `limit` bytes of a file and whether anything was cut."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > limit:
            fh.seek(size - limit)
            return fh.read(), True
        return fh.read(), False


def _file_entry(name: str, path: Path, *, limit: int = MAX_LOG_BYTES) -> BundleEntry:
    data, truncated = _tail_bytes(path, limit)
    notes = [f"truncated to the last {limit} bytes"] if truncated else []
    return _text_entry(name, data.decode("utf-8", errors="replace"), notes)


def _why(exc: BaseException) -> str:
    return exc.exit_code.name if isinstance(exc, TTError) else type(exc).__name__


def _inference_backend(appctx: AppContext) -> InferenceServerBackend:
    return InferenceServerBackend(appctx.registry, appctx.runner, appctx.config, appctx.output)


# -- collectors -------------------------------------------------------------------------
def collect_environment(appctx: AppContext, note: Note) -> list[BundleEntry]:
    """environment.json + tt-smi.json: versions, tools, driver and devices.

    One tt-smi run feeds both files; the raw snapshot is kept verbatim because it
    carries fields (firmware, telemetry) that the parsed model deliberately drops."""
    env: dict = {
        "tt_version": __version__,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "tools": None,
        "driver": None,
        "devices": None,
    }
    entries: list[BundleEntry] = []
    try:
        env["tools"] = [dataclasses.asdict(row) for row in appctx.registry.status()]
    except Exception as exc:
        note(f"environment.json: tools unavailable ({_why(exc)})")
    try:
        raw = get_device_backend(appctx).raw_snapshot()
        entries.append(_json_entry("tt-smi.json", raw))
        snap = parse_snapshot(raw)
        env["driver"] = snap.host.get("Driver")
        env["devices"] = snap.devices
    except Exception as exc:
        note(f"tt-smi.json: unavailable ({_why(exc)})")
    entries.insert(0, _json_entry("environment.json", env))
    return entries


def collect_config(appctx: AppContext, note: Note) -> list[BundleEntry]:
    """tt's own state files. golden.json is reduced to its tag: the data is a copy of
    a public manifest. The telemetry spool and id are deliberately not included."""
    paths = appctx.paths
    entries: list[BundleEntry] = []
    for label, path in (
        ("config.toml", paths.config_file),
        ("installed.toml", paths.state_file),
        ("self-update.toml", paths.self_update_file),
    ):
        if path.is_file():
            entries.append(_file_entry(f"config/{label}", path))
        else:
            note(f"config/{label}: not present")
    if paths.golden_file.is_file():
        try:
            tag = json.loads(paths.golden_file.read_text()).get("tag")
        except Exception:
            tag = None
        entries.append(_json_entry("config/golden.json", {"tag": tag}))
    return entries


def collect_tt_logs(appctx: AppContext, note: Note) -> list[BundleEntry]:
    """Whatever tt wrote under its logs directory (per-run tool output, crash logs)."""
    logs_dir = appctx.paths.logs_dir
    if not logs_dir.is_dir():
        note("tt-logs: no logs directory")
        return []
    return [
        _file_entry(f"tt-logs/{path.relative_to(logs_dir).as_posix()}", path)
        for path in sorted(logs_dir.rglob("*"))
        if path.is_file()
    ]


def collect_workflow_logs(appctx: AppContext, note: Note) -> list[BundleEntry]:
    """The newest tt-inference-server workflow logs from the managed checkout."""
    root = _inference_backend(appctx).checkout_root()
    if root is None:
        note("inference-server: not installed")
        return []
    logs_dir = root / "workflow_logs"
    if not logs_dir.is_dir():
        note("inference-server: no workflow_logs directory")
        return []
    files = sorted(
        (p for p in logs_dir.rglob("*.log") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(files) > MAX_WORKFLOW_LOGS:
        note(f"inference-server: {len(files) - MAX_WORKFLOW_LOGS} older logs omitted")
    return [
        _file_entry(f"inference-server/workflow_logs/{p.relative_to(logs_dir).as_posix()}", p)
        for p in files[:MAX_WORKFLOW_LOGS]
    ]


def _is_tt_container(record: dict) -> bool:
    name = str(record.get("Name") or "").lstrip("/")
    config = record.get("Config") or {}
    labels = config.get("Labels") or record.get("Labels") or {}
    image = str(config.get("Image") or "")
    return (
        name.startswith(CONTAINER_PREFIX)
        or _MODEL_MANAGER_LABEL in labels
        or image.startswith(_STUDIO_IMAGE_PREFIX)
    )


def collect_containers(appctx: AppContext, note: Note) -> list[BundleEntry]:
    """Inspect records plus the log tail of every tt container, running or exited.

    Classification happens here rather than in `ps --filter` so one listing covers
    all three backends and podman behaves the same as docker."""
    backend = _inference_backend(appctx)
    try:
        runtime = backend.container_runtime()
    except TTError as err:
        note(f"containers: unavailable ({err.exit_code.name})")
        return []
    listed = appctx.runner.capture([runtime, "ps", "-a", "--format", "{{.ID}}"], tool=runtime)
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    records = [r for r in backend.inspect_containers(ids) if _is_tt_container(r)]
    entries = [_json_entry("containers/ps.json", records)]
    for record in records:
        cid = str(record.get("Id") or "")[:12]
        name = str(record.get("Name") or "").lstrip("/") or cid
        try:
            result = appctx.runner.capture(
                [runtime, "logs", "--tail", str(CONTAINER_LOG_TAIL), cid],
                tool=runtime,
                check=False,
            )
        except Exception as exc:
            note(f"containers/{name}.log: unavailable ({_why(exc)})")
            continue
        entries.append(_text_entry(f"containers/{name}.log", result.stdout + result.stderr))
    return entries


def collect_env_vars(
    appctx: AppContext, note: Note, environ: Mapping[str, str] | None = None
) -> list[BundleEntry]:
    """Which TT_* / HF_* knobs are set. Values are shown only for names that cannot
    hold a secret (paths, flags); anything with KEY/TOKEN/SECRET in it is just <set>."""
    environ = os.environ if environ is None else environ
    names = sorted({n for n in environ if n.startswith(_ENV_PREFIXES)} | set(_ENV_ALWAYS))
    lines = []
    for name in names:
        if name not in environ:
            lines.append(f"{name}=<unset>")
        elif any(marker in name for marker in _ENV_SECRET_MARKERS):
            lines.append(f"{name}=<set>")
        else:
            lines.append(f"{name}={environ[name]}")
    return [_text_entry("env.txt", "\n".join(lines) + "\n")]


COLLECTORS = (
    collect_environment,
    collect_config,
    collect_tt_logs,
    collect_workflow_logs,
    collect_containers,
    collect_env_vars,
)


# -- archive ------------------------------------------------------------------------------
def default_output_path(now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    return Path.cwd() / f"tt-report-{now:%Y%m%dT%H%M%SZ}.tar.gz"


def write_bundle(appctx: AppContext, output: Path) -> dict:
    """Run every collector and write the tar.gz. The only hard failure is not being
    able to write `output`; everything else degrades into manifest notes."""
    now = datetime.now(timezone.utc)
    notes: list[str] = []
    entries: list[BundleEntry] = []
    for collector in COLLECTORS:
        try:
            entries.extend(collector(appctx, notes.append))
        except Exception as exc:  # a collector bug must never cost the user the bundle
            notes.append(f"{collector.__name__}: failed ({type(exc).__name__}: {exc})")
    manifest = {
        "created": now.isoformat(timespec="seconds"),
        "tt_version": __version__,
        "files": [
            {"name": e.name, "size_bytes": len(e.data), "notes": e.notes} for e in entries
        ],
        "notes": notes,
    }
    entries.append(_json_entry("manifest.json", manifest))

    # One top-level directory so extracting never sprays files into the cwd.
    prefix = output.name.removesuffix(".tar.gz").removesuffix(".tgz") or "tt-report"
    try:
        with tarfile.open(output, "w:gz") as tar:
            for entry in entries:
                info = tarfile.TarInfo(f"{prefix}/{entry.name}")
                info.size = len(entry.data)
                info.mtime = int(now.timestamp())
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(entry.data))
    except OSError as exc:
        raise TTError(
            f"Cannot write support bundle to {output}.",
            why=str(exc),
            next_step="Pass --output to a writable location.",
            exit_code=ExitCode.ERROR,
        ) from exc
    return {
        "path": str(output),
        "files": [e.name for e in entries],
        "size_bytes": output.stat().st_size,
        "notes": notes,
    }
