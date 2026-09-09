#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Build modelhub/model_support.json from the pinned release spec + overrides.

    uv run scripts/build_model_support.py            # rebuild from committed inputs
    uv run scripts/build_model_support.py --check    # CI: fail if the output is stale
    uv run scripts/build_model_support.py --sync     # fetch the spec at the pin first

model_support.json is the artifact tt ships and publishes: one flat, per-device
answer to "does this model work on this board", with the launch settings the
server needs alongside it. It is generated, and CI regenerates it, so it must
never be edited by hand — record a fix in model_support_overrides.toml instead.

The output carries no timestamp deliberately: identical inputs must produce an
identical file, or --check turns into noise and every rebuild is a diff.

A maintainer script, not part of the wheel.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys

import tomlkit
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


SPEC_SCHEMA_VERSION = "0.1.0"

# Spec engine keys, in the order a model's engines are reported.
_ENGINE_ORDER = ("vLLM", "media", "forge")
# Spec device_type keys, lowercased; roughly small → large per family. Unknown
# future keys sort after these, alphabetically — never an error.
_DEVICE_ORDER = (
    "n150", "n150x4", "n300", "t3k", "galaxy_t3k", "galaxy", "dual_galaxy",
    "quad_galaxy", "p100", "p150", "p150x4", "p150x8", "p300", "p300x2", "gpu",
)


def _engine_key(engine: str) -> tuple[int, str]:
    order = _ENGINE_ORDER.index(engine) if engine in _ENGINE_ORDER else len(_ENGINE_ORDER)
    return (order, engine)


def _device_key(device: str) -> tuple[int, str]:
    order = _DEVICE_ORDER.index(device) if device in _DEVICE_ORDER else len(_DEVICE_ORDER)
    return (order, device)


def _preferred_impl(impls: dict) -> tuple[str, dict]:
    """Pick one impl for a (device, engine): the default_impl if one is marked,
    else the first by sorted impl id. Returns (impl_id, leaf) — the id is half of
    the volume directory name the server derives, so it cannot be dropped."""
    if not impls:
        return "", {}
    for impl_id in sorted(impls):
        leaf = impls[impl_id]
        if isinstance(leaf, dict) and (leaf.get("device_model_spec") or {}).get("default_impl"):
            return impl_id, leaf
    impl_id = sorted(impls)[0]
    first = impls[impl_id]
    return (impl_id, first) if isinstance(first, dict) else (impl_id, {})


MODELHUB = REPO_ROOT / "src" / "tenstorrent" / "modelhub"
SUPPLEMENT_PATH = REPO_ROOT / "src" / "tenstorrent" / "tools" / "supplement.toml"
SPEC_PATH = MODELHUB / "release_model_spec.json"
OVERRIDES_PATH = MODELHUB / "model_support_overrides.toml"
OUTPUT_PATH = MODELHUB / "model_support.json"

SUPPORT_SCHEMA_VERSION = 1
OVERRIDES_SCHEMA_VERSION = 1
VALID_REASONS = ("broken", "unsupported")


class BuildError(Exception):
    """An input the build refuses to guess its way past."""


@dataclass(frozen=True)
class Overrides:
    """The three kinds of entry in model_support_overrides.toml."""

    marks: list[dict[str, Any]] = dataclasses_field(default_factory=list)
    fallbacks: list[dict[str, Any]] = dataclasses_field(default_factory=list)
    serve: list[dict[str, Any]] = dataclasses_field(default_factory=list)


# -- inputs ---------------------------------------------------------------------------
def load_overrides(path: Path) -> Overrides:
    """Parse and shape-check the overrides file. Cross-checking each entry against
    the spec happens later, in apply_overrides, where the models are known."""
    doc = tomlkit.parse(path.read_text()).unwrap()
    version = doc.get("schema_version")
    if version != OVERRIDES_SCHEMA_VERSION:
        raise BuildError(
            f"{path.name}: schema_version {version!r}, expected {OVERRIDES_SCHEMA_VERSION}."
        )

    entries = doc.get("override") or []
    seen: set[tuple[str, str | None]] = set()
    for entry in entries:
        missing = [f for f in ("model", "reason", "details", "verified_on") if not entry.get(f)]
        if missing:
            raise BuildError(f"{path.name}: entry {entry!r} is missing {', '.join(missing)}.")
        if entry["reason"] not in VALID_REASONS:
            raise BuildError(
                f"{path.name}: {entry['model']} has unknown reason {entry['reason']!r}; "
                f"expected one of {', '.join(VALID_REASONS)}."
            )
        device = entry.get("device")
        if device and device != device.lower():
            raise BuildError(
                f"{path.name}: {entry['model']} device {device!r} must be lowercase "
                f"(spec device_type keys are lowercased) — use {device.lower()!r}."
            )
        key = (entry["model"], device)
        if key in seen:
            raise BuildError(f"{path.name}: duplicate entry for {key[0]} / {key[1] or '*'}.")
        seen.add(key)

    fallbacks = doc.get("device_fallback") or []
    for entry in fallbacks:
        missing = [f for f in ("device", "serve_as", "engines", "details") if not entry.get(f)]
        if missing:
            raise BuildError(
                f"{path.name}: device_fallback {entry!r} is missing {', '.join(missing)}."
            )
        # A fallback is applied by string match and silently fills nothing when it
        # matches nothing, so a casing slip ("P150X4", "vllm") would leave models
        # off a board they run on with no sign anything was wrong.
        for field in ("device", "serve_as"):
            if entry[field] != entry[field].lower():
                raise BuildError(
                    f"{path.name}: device_fallback {field} {entry[field]!r} must be "
                    f"lowercase (spec device_type keys are lowercased) — use "
                    f"{entry[field].lower()!r}."
                )
        unknown = [e for e in entry["engines"] if e not in _ENGINE_ORDER]
        if unknown:
            raise BuildError(
                f"{path.name}: device_fallback for {entry['device']!r} names unknown "
                f"engine(s) {', '.join(repr(e) for e in unknown)}; the spec spells them "
                f"{', '.join(repr(e) for e in _ENGINE_ORDER)}."
            )
        if entry["device"] == entry["serve_as"]:
            raise BuildError(
                f"{path.name}: device_fallback for {entry['device']!r} points at itself."
            )

    serve = doc.get("serve_override") or []
    for entry in serve:
        if not entry.get("model") or not entry.get("details"):
            raise BuildError(f"{path.name}: serve_override {entry!r} needs model and details.")
        if not (entry.get("docker_image") or entry.get("override_tt_config")):
            raise BuildError(
                f"{path.name}: serve_override for {entry['model']} sets nothing; "
                "give it a docker_image or an override_tt_config."
            )
    return Overrides(marks=entries, fallbacks=fallbacks, serve=serve)


def load_spec(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise BuildError(
            f"{path.name}: schema_version {doc.get('schema_version')!r}, "
            f"expected {SPEC_SCHEMA_VERSION}."
        )
    return doc


def fetch_spec(destination: Path) -> str:
    """Download release_model_spec.json at the pinned server version.

    The pin lives in one place (supplement.toml), so --sync cannot fetch a spec
    that disagrees with the server tt actually installs.
    """
    from urllib.request import urlopen

    # Read supplement.toml directly rather than through the manifest layer: the
    # pin is all we need, and the manifest also wants a fetched golden.json cache
    # that a checkout will not have.
    supplement = tomlkit.parse(SUPPLEMENT_PATH.read_text()).unwrap()
    version = supplement["tools"]["tt-inference-server"]["golden_version"]
    # The spec is a tracked file at the tag, not a release asset: the release
    # only publishes <tag>-release_artifacts.zip (per-model workflow logs).
    url = (
        "https://raw.githubusercontent.com/tenstorrent/tt-inference-server/"
        f"{version}/release_model_spec.json"
    )
    print(f"Fetching release_model_spec.json at {version} …")
    with urlopen(url, timeout=60) as response:  # noqa: S310 — fixed https host
        blob = response.read()
    destination.write_bytes(blob)
    print(f"  wrote {destination.relative_to(REPO_ROOT)} ({len(blob) / 1024:.0f} KiB)")
    return version


# -- build ----------------------------------------------------------------------------
def build_models(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the spec's model → device → engine → impl nesting into one record
    per model, with a device map holding everything needed to serve it.

    One leaf is chosen per (device, engine) exactly as the runtime catalog does,
    so the published file and `tt model list` can never disagree about which impl
    a model resolves to.
    """
    models = []
    for repo, device_map in (spec.get("model_specs") or {}).items():
        if not isinstance(device_map, dict):
            continue
        devices: dict[str, dict[str, Any]] = {}
        leaves: list[tuple[str, dict[str, Any]]] = []  # (engine, leaf), for model-level fields
        for device_type in sorted(device_map, key=lambda d: _device_key(d.lower())):
            engine_map = device_map[device_type]
            if not isinstance(engine_map, dict) or not engine_map:
                continue
            engines = sorted(engine_map, key=_engine_key)
            impl_id, leaf = _preferred_impl(engine_map[engines[0]] or {})
            if not leaf.get("model_name"):
                raise BuildError(f"{repo} → {device_type} → {engines[0]}: leaf has no model_name.")
            device_spec = leaf.get("device_model_spec") or {}
            metadata = leaf.get("metadata") or {}
            max_context = device_spec.get("max_context")
            devices[device_type.lower()] = {
                "engines": engines,
                "status": str(leaf.get("status", "")),
                "max_context": int(max_context) if max_context is not None else None,
                "supported": True,
                # What the server would run without an override — recorded so
                # `tt serve --dry-run` can name the image either way.
                "docker_image": leaf.get("docker_image") or None,
                # The server derives its persistent-volume directory as
                # volume_id_<impl_id>-<model_name>-v<version>; both halves are
                # per-device, so a pre-seeded volume can only be matched exactly
                # with them.
                "impl_id": impl_id or None,
                "version": str(leaf.get("version")) if leaf.get("version") else None,
                # Launch settings the server needs but does not apply on its own:
                # tt passes them as --vllm-override-args / --override-tt-config.
                "tool_call_parser": metadata.get("tool_call_parser_name") or None,
                "reasoning_parser": metadata.get("reasoning_parser_name") or None,
                "override_tt_config": device_spec.get("override_tt_config") or None,
            }
            leaves.append((engines[0], leaf))
        if not devices:
            continue

        primary = next((leaf for engine, leaf in leaves if engine == "vLLM"), leaves[0][1])
        engines = sorted({e for d in devices.values() for e in d["engines"]}, key=_engine_key)
        name = str(primary["model_name"])
        models.append(
            {
                "name": name,
                "hf_repo": str(repo),
                "model_type": str(primary.get("model_type", "")).lower(),
                "engines": engines,
                "tt_model_id": name,
                "param_count": _as_int(primary.get("param_count")),
                "min_disk_gb": _as_int(primary.get("min_disk_gb")),
                "min_ram_gb": _as_float(primary.get("min_ram_gb")),
                "devices": devices,
            }
        )
    models.sort(key=lambda m: m["name"].lower())
    return models


def _as_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def apply_overrides(
    models: list[dict[str, Any]], overrides: list[dict[str, Any]]
) -> list[str]:
    """Mark unsupported (model, device) pairs. Returns warnings for dormant entries.

    An override naming a device the model does not claim is an error: that is
    almost always a typo, and a silent no-op leaves a broken model on offer. An
    override naming a model the spec does not have is only a warning — the marks
    outlive spec versions, and one may legitimately be waiting for a pin bump.
    """
    by_name = {model["name"]: model for model in models}
    warnings = []
    for entry in overrides:
        model = by_name.get(entry["model"])
        if model is None:
            warnings.append(
                f"{entry['model']}: not in this spec; the override is dormant "
                f"(drop it, or leave it for a later server pin)."
            )
            continue

        mark = {
            "reason": entry["reason"],
            "details": " ".join(entry["details"].split()),
            "verified_on": entry["verified_on"],
        }
        if entry.get("source"):
            mark["source"] = entry["source"]

        device = entry.get("device")
        if device is None:
            targets = list(model["devices"])
        elif device in model["devices"]:
            targets = [device]
        else:
            near = difflib.get_close_matches(device, model["devices"], n=1)
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise BuildError(
                f"{entry['model']}: device {device!r} is not one of its spec devices "
                f"{sorted(model['devices'])}, so nothing would be marked.{hint}"
            )

        for target in targets:
            model["devices"][target]["supported"] = False
            model["devices"][target]["unsupported"] = mark
    return warnings


def apply_device_fallbacks(
    models: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> list[str]:
    """Offer a model on a board whose spec it lacks but which can still run it.

    Runs after the marks so a board whose own spec is broken can still be rescued
    through its equivalent — which is how Llama-3.3-70B-Instruct reaches p150x4,
    its native spec being stale while the p300x2 one it borrows is maintained.

    The synthesized entry keeps `serve_as` so `tt serve` sends the device name the
    spec actually has; sending the board's own name would 404 on spec lookup.
    """
    notes = []
    for model in models:
        for rule in rules:
            device, source_device = rule["device"], rule["serve_as"]
            source = model["devices"].get(source_device)
            if source is None or not source["supported"]:
                continue
            if not set(rule["engines"]) & set(source["engines"]):
                continue
            existing = model["devices"].get(device)
            if existing is not None and existing["supported"]:
                continue  # its own spec works; never shadow it
            borrowed = {
                **{k: v for k, v in source.items() if k != "unsupported"},
                "serve_as": source_device,
                "support_source": rule.get("support_source", "mesh-equivalent"),
            }
            if existing is not None:
                # Say why, or the file reads as contradicting the mark above.
                borrowed["note"] = (
                    f"its own {device} spec is unusable: "
                    f"{existing['unsupported']['details']}"
                )
            model["devices"][device] = borrowed
            notes.append(f"{model['name']}: {device} filled from {source_device}")
        # A borrowed device is inserted at the end; device order is the display
        # order for `tt model info`, so restore it.
        model["devices"] = {
            name: model["devices"][name]
            for name in sorted(model["devices"], key=_device_key)
        }
    return notes


def apply_serve_overrides(
    models: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> list[str]:
    """Attach flags `tt serve` must pass because the spec's own value is wrong.

    Recorded under `serve_overrides` rather than merged into the spec fields: the
    server already applies its own override_tt_config, so tt passes a flag only
    where it is correcting one, and the file has to show which is which.
    """
    by_name = {model["name"]: model for model in models}
    warnings = []
    for entry in entries:
        model = by_name.get(entry["model"])
        if model is None:
            warnings.append(f"{entry['model']}: serve_override targets a model not in this spec.")
            continue
        device = entry.get("device")
        if device is None:
            targets = list(model["devices"])
        elif device in model["devices"]:
            targets = [device]
        else:
            near = difflib.get_close_matches(device, model["devices"], n=1)
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise BuildError(
                f"{entry['model']}: serve_override device {device!r} is not one of its "
                f"devices {sorted(model['devices'])}.{hint}"
            )
        applied = {
            key: entry[key] for key in ("docker_image", "override_tt_config") if entry.get(key)
        }
        for target in targets:
            model["devices"][target]["serve_overrides"] = applied
    return warnings


def build_document(spec: dict[str, Any], overrides: Overrides) -> tuple[dict, list[str]]:
    models = build_models(spec)
    warnings = apply_overrides(models, overrides.marks)
    apply_device_fallbacks(models, overrides.fallbacks)  # summarized, not warned
    warnings += apply_serve_overrides(models, overrides.serve)
    document = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "release_version": str(spec.get("release_version", "")),
        "models": models,
    }
    return document, warnings


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=1, sort_keys=False) + "\n"


# -- reporting ------------------------------------------------------------------------
def summarize(document: dict[str, Any]) -> str:
    models = document["models"]
    pairs = [(m, d, s) for m in models for d, s in m["devices"].items()]
    unsupported = [p for p in pairs if not p[2]["supported"]]
    parsers = [p for p in pairs if p[2]["tool_call_parser"]]
    tt_configs = [p for p in pairs if p[2]["override_tt_config"]]
    borrowed = [p for p in pairs if p[2].get("serve_as")]
    forced = [p for p in pairs if p[2].get("serve_overrides")]
    return (
        f"tt-inference-server {document['release_version']}: {len(models)} models, "
        f"{len(pairs)} model/device pairs\n"
        f"  {len(unsupported)} marked unsupported\n"
        f"  {len(parsers)} with a tool-call parser\n"
        f"  {len(tt_configs)} with an override_tt_config\n"
        f"  {len(borrowed)} reached through a device fallback\n"
        f"  {len(forced)} with a forced serve-time flag"
    )


def diff_summary(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    """What changed between the committed artifact and the rebuilt one.

    Names the moving parts a reviewer has to judge — models appearing and
    disappearing, and the launch settings that change how a model is served.
    """
    if old is None:
        return "No previous model_support.json; this is the first build."

    lines = []
    if old.get("release_version") != new.get("release_version"):
        lines.append(f"release_version  {old.get('release_version')} → {new['release_version']}")

    old_models = {m["name"]: m for m in old.get("models", [])}
    new_models = {m["name"]: m for m in new["models"]}
    for name in sorted(set(new_models) - set(old_models)):
        lines.append(f"+ model  {name}")
    for name in sorted(set(old_models) - set(new_models)):
        lines.append(f"- model  {name}")

    for name in sorted(set(old_models) & set(new_models)):
        old_devices = old_models[name]["devices"]
        new_devices = new_models[name]["devices"]
        for device in sorted(set(new_devices) - set(old_devices)):
            lines.append(f"+ device {name} / {device}")
        for device in sorted(set(old_devices) - set(new_devices)):
            lines.append(f"- device {name} / {device}")
        for device in sorted(set(old_devices) & set(new_devices)):
            before, after = old_devices[device], new_devices[device]
            for field in (
                "supported", "status", "tool_call_parser", "reasoning_parser", "serve_as",
            ):
                if before.get(field) != after.get(field):
                    lines.append(
                        f"~ {name} / {device}  {field}: "
                        f"{before.get(field)!r} → {after.get(field)!r}"
                    )
            for field in ("override_tt_config", "serve_overrides"):
                if before.get(field) != after.get(field):
                    lines.append(f"~ {name} / {device}  {field} changed")

    return "\n".join(lines) if lines else "No changes."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# -- entry point ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed model_support.json is not what the inputs produce",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="download release_model_spec.json at the pinned server version first",
    )
    args = parser.parse_args(argv)

    try:
        if args.sync:
            if args.check:
                parser.error("--sync writes inputs; it cannot be combined with --check")
            fetch_spec(SPEC_PATH)

        spec = load_spec(SPEC_PATH)
        overrides = load_overrides(OVERRIDES_PATH)
        document, warnings = build_document(spec, overrides)
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    rendered = render(document)
    previous = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else None

    if args.check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print(
                "error: model_support.json is stale — it is generated, not edited.\n"
                "Run `uv run scripts/build_model_support.py` and commit the result.\n",
                file=sys.stderr,
            )
            print(diff_summary(previous, document), file=sys.stderr)
            return 1
        print("model_support.json is up to date.")
        return 0

    print(diff_summary(previous, document))
    OUTPUT_PATH.write_text(rendered)
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)} (sha256 {sha256(OUTPUT_PATH)}…)")
    print(summarize(document))
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
