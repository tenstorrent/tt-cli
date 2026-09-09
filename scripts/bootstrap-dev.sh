#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.
#
# One-command developer bootstrap + runner for a tt-cli checkout.
#
#   scripts/bootstrap-dev.sh                  # set up everything, then show tt help
#   scripts/bootstrap-dev.sh device status    # set up (no-op when already done), then run `tt device status`
#
# For contributors working from a clone. End users should `pip install tenstorrent`
# and call `tt` directly; nothing here is needed for that path.
#
# Installs uv if missing, syncs the project venv, and passes all arguments
# through to `tt`. Idempotent: re-runs skip anything already in place.
# Setup output goes to stderr so `scripts/bootstrap-dev.sh --json ... | jq` stays clean.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"

note() { echo "bootstrap-dev.sh: $*" >&2; }

# --- prerequisites (no sudo here; point at the distro packages instead) -----
missing=()
for cmd in curl git python3; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
# Debian/Ubuntu ship the venv module separately (python3-venv); without it
# `uv sync` fails later with a far less actionable error.
if [ "${#missing[@]}" -eq 0 ] && ! python3 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    missing+=("python3-venv")
fi
if [ "${#missing[@]}" -gt 0 ]; then
    note "missing prerequisites: ${missing[*]}"
    note "install them first, e.g.:"
    note "  Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip git curl ca-certificates"
    note "  Fedora:        sudo dnf install -y python3 python3-pip git curl"
    exit 1
fi

# --- uv (project manager; also a runtime dep, but we need one on PATH) ------
for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    case ":$PATH:" in *":$dir:"*) ;; *) PATH="$dir:$PATH" ;; esac
done
export PATH
if ! command -v uv >/dev/null 2>&1; then
    note "uv not found; installing via https://astral.sh/uv"
    # Download to a file first rather than piping straight into sh, so a
    # truncated or tampered transfer fails to run instead of half-running.
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    sh "$installer" >&2
    command -v uv >/dev/null 2>&1 || { note "uv install failed"; exit 1; }
fi
note "using $(uv --version)"

# --- project environment -----------------------------------------------------
# Plain sync (not --frozen): a fresh clone may not carry uv.lock. Re-runs are
# a fast no-op once .venv exists and matches.
uv sync --project "$repo" 1>&2

# --- put `tt` on PATH so it works without the script ------------------------
# uv sync installs the console script into the project venv; a symlink from
# ~/.local/bin makes plain `tt model ...` work in any shell. Only a missing
# link is created: an existing one is left alone so this never silently
# changes which `tt` a user's shell picks up.
bin_dir="$HOME/.local/bin"
target="$repo/.venv/bin/tt"
link="$bin_dir/tt"
if [ -x "$target" ]; then
    mkdir -p "$bin_dir"
    if [ -L "$link" ]; then
        current="$(readlink -f "$link" || true)"
        if [ "$current" != "$(readlink -f "$target")" ]; then
            note "warning: $link -> $current already exists; not repointing it to this checkout"
        fi
    elif [ -e "$link" ]; then
        note "warning: $link exists and is not a symlink; leaving it alone"
    else
        ln -s "$target" "$link"
        note "linked $link -> $target (open a new shell if 'tt' isn't picked up yet)"
    fi
fi

# Scripts can't answer the one-time telemetry consent prompt; disable it for
# this run only (per-run tier — it never touches the user's config or spool).
export TT_TELEMETRY_DISABLED=1

note "setup OK ($(uv run --project "$repo" tt --version 2>/dev/null | head -n1))"

exec uv run --project "$repo" tt "$@"
