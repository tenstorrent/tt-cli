#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Build modelhub/community_catalog.json from community_catalog.toml.

    uv run scripts/build_community_catalog.py          # rebuild from the committed input
    uv run scripts/build_community_catalog.py --check  # CI: fail if the output is stale

community_catalog.json is what tt ships and publishes with each release for
downstream tools: the community bundles `tt model list --community` shows, with
kind, engine, arch and hardware as plain fields. It is generated, so it must
never be edited by hand — change community_catalog.toml instead.

Output contract (schema_version 1): adding a field is non-breaking; renaming or
removing one, or changing its meaning, bumps schema_version.

A maintainer script, not part of the wheel.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import ParseError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tenstorrent.modelhub.bundles import _hardware_chips  # noqa: E402

MODELHUB = REPO_ROOT / "src" / "tenstorrent" / "modelhub"
INPUT_PATH = MODELHUB / "community_catalog.toml"
OUTPUT_PATH = MODELHUB / "community_catalog.json"

SCHEMA_VERSION = 1
KINDS = ("container", "self-contained", "thin")
ENGINES = ("vllm-plugin", "vllm-fork", "tt-dit-server")
VALIDATIONS = ("sweep", "separate")


class BuildError(Exception):
    """An input the build refuses to guess its way past."""


def load_input(path: Path) -> dict[str, Any]:
    try:
        return tomlkit.parse(path.read_text()).unwrap()
    except (OSError, ParseError) as exc:
        raise BuildError(f"{path.name}: {exc}") from exc


def _one_of(repo: str, field: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise BuildError(f"{repo}: {field} {value!r} is not one of {', '.join(allowed)}")


def _bundle(raw: dict[str, Any]) -> dict[str, Any]:
    repo = str(raw.get("repo", "")).strip()
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise BuildError(f"{repo!r} is not a namespace/name repo id")

    hardware = sorted({str(tag).lower() for tag in raw.get("hardware") or []})
    if not hardware:
        raise BuildError(f"{repo}: hardware is empty; list the boards it was validated on")
    chips = {tag: _hardware_chips(tag) for tag in hardware}
    unknown = [tag for tag, hw in chips.items() if hw is None]
    if unknown:
        raise BuildError(f"{repo}: unknown hardware {', '.join(unknown)}")
    arches = {arch for arch, _ in chips.values()}
    if len(arches) != 1:
        raise BuildError(f"{repo}: hardware spans architectures ({', '.join(sorted(arches))})")

    kind, engine = raw.get("kind"), raw.get("engine")
    if kind is not None:
        _one_of(repo, "kind", kind, KINDS)
    if engine is not None:
        _one_of(repo, "engine", engine, ENGINES)
    _one_of(repo, "validation", raw.get("validation"), VALIDATIONS)

    validated_on = raw.get("validated_on")
    if validated_on is not None:
        try:
            validated_on = datetime.date.fromisoformat(str(validated_on)).isoformat()
        except ValueError as exc:
            raise BuildError(f"{repo}: validated_on {validated_on!r} is not a YYYY-MM-DD date") from exc

    return {
        "repo": repo,
        "kind": kind,
        "engine": engine,
        "arch": arches.pop(),
        "hardware": hardware,
        "validation": raw["validation"],
        "validated_on": validated_on,
    }


def build_document(doc: dict[str, Any]) -> dict[str, Any]:
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise BuildError(f"schema_version must be {SCHEMA_VERSION}")
    bundles = [_bundle(raw) for raw in doc.get("bundle") or []]
    seen: set[str] = set()
    for bundle in bundles:
        key = bundle["repo"].lower()
        if key in seen:
            raise BuildError(f"{bundle['repo']} is listed more than once")
        seen.add(key)
    bundles.sort(key=lambda b: b["repo"].lower())
    return {"schema_version": SCHEMA_VERSION, "bundles": bundles}


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=1) + "\n"


def diff_summary(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    if old is None:
        return "No previous community_catalog.json; this is the first build."
    before = {b["repo"]: b for b in old.get("bundles", [])}
    after = {b["repo"]: b for b in new["bundles"]}
    lines = [f"+ {repo}" for repo in sorted(set(after) - set(before), key=str.lower)]
    lines += [f"- {repo}" for repo in sorted(set(before) - set(after), key=str.lower)]
    lines += [
        f"~ {repo}"
        for repo in sorted(set(before) & set(after), key=str.lower)
        if before[repo] != after[repo]
    ]
    return "\n".join(lines) if lines else "No changes."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed community_catalog.json is not what the input produces",
    )
    args = parser.parse_args(argv)

    try:
        document = build_document(load_input(INPUT_PATH))
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    rendered = render(document)
    current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else None
    previous = json.loads(current) if current else None

    if args.check:
        if current != rendered:
            print(
                "error: community_catalog.json is stale — it is generated, not edited.\n"
                "Run `uv run scripts/build_community_catalog.py` and commit the result.\n",
                file=sys.stderr,
            )
            print(diff_summary(previous, document), file=sys.stderr)
            return 1
        print("community_catalog.json is up to date.")
        return 0

    print(diff_summary(previous, document))
    OUTPUT_PATH.write_text(rendered)
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(document['bundles'])} bundles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
