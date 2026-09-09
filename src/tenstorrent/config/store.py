# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Comment-preserving TOML config store (tomlkit round-trip).

Reads fall back to schema defaults so the CLI works with no config file at all;
writes create the file from the commented template on first use.

The file is an *override* layer, not a copy of the defaults: a key absent from it
resolves to its schema default, so an old config.toml keeps working when new settings
are added. The hazard that creates is silence, and this module answers it two ways —
`unknown_keys()` finds settings in the file that no longer (or never did) exist, and
`list_entries()` reports where each effective value came from. Both exist because of a
real incident: a hand-added `flush_mode = "sync"` appended to the end of the file became
`device.flush_mode`, since TOML assigns a bare key to the *last* table, and nothing said
a word about it.
"""

from __future__ import annotations

import copy
import difflib
from pathlib import Path
from typing import Any, Callable

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import Comment, Table, Whitespace

from ..errors import ExitCode, TTError
from . import schema
from .paths import Paths

# Where an effective value came from, as reported by list_entries().
SOURCE_DEFAULT = "default"
SOURCE_FILE = "config.toml"
SOURCE_UNRECOGNIZED = "unrecognized"


def coerce_value(raw: str) -> Any:
    """Interpret a CLI-provided string as a TOML scalar: bool, int, float, or string."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def suggest_key(key: str) -> str | None:
    """Best guess at what an unrecognized key was meant to be, or None."""
    known = schema.known_keys()
    # Leaf name first: putting a key under the wrong table is the common mistake, and
    # matching on the last segment is exact where a fuzzy match on the whole dotted
    # path is not ('device.flush_mode' vs 'telemetry.flush_mode' share only a suffix).
    leaf = key.rsplit(".", 1)[-1]
    same_leaf = [candidate for candidate in known if candidate.rsplit(".", 1)[-1] == leaf]
    if len(same_leaf) == 1:
        return same_leaf[0]
    close = difflib.get_close_matches(key, known, n=1, cutoff=0.7)
    return close[0] if close else None


def _unrecognized_in(document: TOMLDocument) -> list[str]:
    raw = document.unwrap() if hasattr(document, "unwrap") else dict(document)
    return sorted(key for key in schema.flatten(raw) if not schema.is_known_key(key))


def _comment_block(table: Any, name: str) -> list[Comment]:
    """The contiguous standalone comments immediately above `name` in `table`.

    tomlkit stores comment lines as their own entries in a table's body rather than as
    trivia on the key below them, so a key's documentation has to be gathered by walking
    backwards from it.
    """
    body = table.value.body
    index = next(
        (i for i, (key, _) in enumerate(body) if key is not None and key.key == name), None
    )
    if index is None:
        return []
    block: list[Comment] = []
    cursor = index - 1
    while cursor >= 0 and isinstance(body[cursor][1], Comment):
        block.insert(0, body[cursor][1])
        cursor -= 1
    return block


def _ends_with_blank_line(table: Any) -> bool:
    body = table.value.body
    return bool(body) and isinstance(body[-1][1], Whitespace)


def _copy_key_with_comments(user_table: Any, template_table: Any, name: str) -> None:
    """Append `name` (and its documentation) from the template into the user's table."""
    if not _ends_with_blank_line(user_table):
        # Keep the new block visually separate from the key above it.
        user_table.add(tomlkit.nl())
    for comment in _comment_block(template_table, name):
        user_table.add(copy.deepcopy(comment))
    user_table[name] = copy.deepcopy(template_table[name])
    # And separate it from whatever table header follows.
    user_table.add(tomlkit.nl())


def _sync_table(user_table: Any, template_table: Any, prefix: str = "") -> list[str]:
    """Recursively add template keys missing from `user_table`. Returns dotted names."""
    added: list[str] = []
    for name in template_table.keys():
        template_child = template_table[name]
        dotted = f"{prefix}{name}"
        if isinstance(template_child, Table):
            if name not in user_table:
                # A whole missing section: copy it wholesale, comments and all. Only its
                # leaf keys count as added settings — a section that holds none (the
                # dynamic [tools.override] namespace) is a header, not a setting, and
                # reporting it would disagree with missing_keys() and so with --dry-run.
                user_table[name] = copy.deepcopy(template_child)
                added.extend(schema.flatten({name: template_child.unwrap()}, prefix))
            else:
                added.extend(_sync_table(user_table[name], template_child, f"{dotted}."))
        elif name not in user_table:
            _copy_key_with_comments(user_table, template_table, name)
            added.append(dotted)
    return added


class ConfigStore:
    def __init__(self, paths: Paths, *, on_warning: Callable[[str], None] | None = None) -> None:
        self.paths = paths
        # Optional sink for "your config file says something we don't understand".
        # Injected rather than imported so ConfigStore stays independent of output
        # plumbing (and so tests constructing a store directly stay silent).
        self._on_warning = on_warning
        self._reported = False

    # -- document handling -------------------------------------------------------
    def _read_document(self) -> TOMLDocument:
        path = self.paths.config_file
        if not path.exists():
            # The template is ours and always valid; nothing to validate against.
            return tomlkit.parse(schema.TEMPLATE)
        try:
            document = tomlkit.parse(path.read_text())
        except TOMLKitError as exc:
            raise TTError(
                f"Config file {path} is not valid TOML.",
                why=str(exc),
                next_step="Fix the file by hand (`tt config` opens it) or delete it to start fresh.",
                exit_code=ExitCode.CONFIG,
            ) from exc
        self._report_unrecognized(document)
        return document

    # -- validation --------------------------------------------------------------
    def unknown_keys(self) -> list[str]:
        """Dotted keys present in the config file that this version doesn't recognize."""
        path = self.paths.config_file
        if not path.exists():
            return []
        try:
            return _unrecognized_in(tomlkit.parse(path.read_text()))
        except TOMLKitError:
            # Unparseable files are reported by _read_document with a proper error.
            return []

    def _report_unrecognized(self, document: TOMLDocument) -> None:
        """Warn once per process about settings we can't act on.

        Every command reads config, so this piggybacks on a parse that was happening
        anyway and fires wherever the mistake matters — not only under `tt config`.
        """
        if self._reported or self._on_warning is None:
            return
        # Set before emitting: a warning handler must never be able to re-enter here.
        self._reported = True
        for key in _unrecognized_in(document):
            message = (
                f"{key} in {self.paths.config_file} is not a setting this version of tt "
                "recognizes, so it has no effect."
            )
            suggestion = suggest_key(key)
            if suggestion:
                # The overwhelmingly likely cause, worth naming outright: TOML puts a
                # bare key appended to the end of a file into the last table, not the
                # one the user had in mind.
                message += f" Did you mean {suggestion}? (`tt config set` puts it in the right table.)"
            self._on_warning(message)

    def _write_document(self, doc: TOMLDocument) -> None:
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.config_file.write_text(tomlkit.dumps(doc))

    def ensure_file(self) -> None:
        """Create the commented template on first run; never overwrite."""
        if not self.paths.config_file.exists():
            self._write_document(tomlkit.parse(schema.TEMPLATE))

    # -- whole-file operations ---------------------------------------------------
    def missing_keys(self) -> list[str]:
        """Documented keys the file doesn't mention. They still resolve to their
        defaults; this is what `sync()` would add."""
        if not self.paths.config_file.exists():
            return schema.known_keys()
        present = set(schema.flatten(self._raw()))
        return [key for key in schema.known_keys() if key not in present]

    def sync(self, *, dry_run: bool = False) -> list[str]:
        """Write documented-but-absent keys into the file, with their comments.

        Purely additive: existing values, ordering, comments and unrecognized keys are
        left exactly as they are. The point is discoverability — a config.toml written by
        an older `tt` is functionally complete (absent keys fall back to defaults), but
        you cannot discover or edit a setting you can't see.
        """
        if not self.paths.config_file.exists():
            if not dry_run:
                self.ensure_file()
            return schema.known_keys()
        added = self.missing_keys()
        if dry_run or not added:
            return added
        document = self._read_document()
        # Report what was actually written rather than trusting the prediction above, and
        # sort it so --dry-run and the real run list the same thing in the same order.
        written = sorted(_sync_table(document, tomlkit.parse(schema.TEMPLATE)))
        self._write_document(document)
        return written

    def reset(self) -> Path | None:
        """Replace the file with a fresh commented template.

        Returns the path of the backup taken first, or None if there was no file. The
        backup is not optional: this discards every value the user set, and a config
        holding a PostHog key or tool overrides is not something to destroy on a
        one-word command.
        """
        backup: Path | None = None
        if self.paths.config_file.exists():
            backup = self.paths.config_file.with_suffix(".toml.bak")
            backup.write_text(self.paths.config_file.read_text())
        self._write_document(tomlkit.parse(schema.TEMPLATE))
        return backup

    def _raw(self) -> dict:
        document = self._read_document()
        return document.unwrap() if hasattr(document, "unwrap") else dict(document)

    # -- typed access -------------------------------------------------------------
    def _unknown_key_error(self, dotted_key: str) -> TTError:
        return TTError(
            f"Unknown config key: {dotted_key!r}.",
            why="Only documented keys (and tools.override.<tool>) are accepted.",
            next_step="Run `tt config list` to see all available keys.",
            exit_code=ExitCode.CONFIG,
        )

    def get(self, dotted_key: str) -> Any:
        if not schema.is_known_key(dotted_key):
            raise self._unknown_key_error(dotted_key)
        node: Any = self._read_document()
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                try:
                    return schema.default_for(dotted_key)
                except KeyError:
                    return None  # known dynamic key (tools.override.*) that is unset
            node = node[part]
        return node.unwrap() if hasattr(node, "unwrap") else node

    def set(self, dotted_key: str, value: Any) -> None:
        if not schema.is_known_key(dotted_key):
            raise self._unknown_key_error(dotted_key)
        self.ensure_file()
        doc = self._read_document()
        *table_parts, leaf = dotted_key.split(".")
        node: Any = doc
        for part in table_parts:
            if part not in node:
                node[part] = tomlkit.table()
            node = node[part]
        node[leaf] = value
        self._write_document(doc)

    def list_entries(self) -> dict[str, tuple[Any, str]]:
        """Effective config as {key: (value, source)}, sorted by key.

        Sources answer "have I actually changed this?", which is the question behind
        "my setting isn't being picked up":

        - SOURCE_DEFAULT — the effective value *is* the schema default. Note this
          includes keys the file restates at their default value: the first-run template
          writes every key out, so "present in the file" would otherwise be true of
          everything and carry no information at all.
        - SOURCE_FILE — the file sets it to something other than the default.
        - SOURCE_UNRECOGNIZED — the file declares a key this version doesn't know, so it
          has no effect. A key added under the wrong table lands here.
        """
        defaults = schema.flatten(schema.DEFAULTS)
        merged: dict[str, tuple[Any, str]] = {
            key: (value, SOURCE_DEFAULT) for key, value in defaults.items()
        }
        if self.paths.config_file.exists():
            on_disk = self._read_document()
            raw = on_disk.unwrap() if hasattr(on_disk, "unwrap") else dict(on_disk)
            for key, value in schema.flatten(raw).items():
                if not schema.is_known_key(key):
                    merged[key] = (value, SOURCE_UNRECOGNIZED)
                elif key in defaults and value == defaults[key]:
                    merged[key] = (value, SOURCE_DEFAULT)
                else:
                    # Differs from the default, or is a dynamic tools.override.* entry
                    # that only ever exists because the user put it there.
                    merged[key] = (value, SOURCE_FILE)
        return dict(sorted(merged.items()))

    def list_flat(self) -> dict[str, Any]:
        """Effective config: defaults overlaid with the file, as sorted dotted keys."""
        return {key: value for key, (value, _) in self.list_entries().items()}

    def source_of(self, dotted_key: str) -> str:
        """Where `get(dotted_key)` takes its value from."""
        entry = self.list_entries().get(dotted_key)
        return entry[1] if entry else SOURCE_DEFAULT
