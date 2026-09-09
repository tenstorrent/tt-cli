#!/usr/bin/env bash
# Hardware-free demo of the tt CLI, run entirely against the fake tools in
# tests/fakes/. Safe to run anywhere; writes only to a temp dir.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d /tmp/tt-demo.XXXXXX)"
trap 'rm -rf "$work"' EXIT

export TT_CONFIG_DIR="$work/config"
export TT_DATA_DIR="$work/data"
export TT_CACHE_DIR="$work/cache"
export TT_TOOL_BIN_TT_SMI="$repo/tests/fakes/bin/tt-smi"
export TT_TOOL_BIN_TT_INSTALLER="$repo/tests/fakes/install.sh"
export TT_TOOL_BIN_TT_INFERENCE_SERVER="$repo/tests/fakes/inference-repo/run.py"
export TT_UV_BIN="$repo/tests/fakes/bin/uv"

tt() { uv run --project "$repo" tt "$@"; }

step() { printf '\n\033[1;36m$ tt %s\033[0m\n' "$*"; }

step config set telemetry.enabled false
tt config set telemetry.enabled false

step config list
tt config list

step device status
tt device status

step "device status --json | jq .devices[0].board_type"
tt device status --json | (jq .devices[0].board_type 2>/dev/null || python3 -c "import json,sys; print(json.load(sys.stdin)['devices'][0]['board_type'])")

step device reset 0 --yes
tt config set tools.sudo_command ""   # fake tt-smi needs no privileges
tt device reset 0 --yes

step update --dry-run
tt update --dry-run

step update
tt update

step model list
tt model list

step "serve Llama-3.1-8B-Instruct (fake backend)"
tt serve Llama-3.1-8B-Instruct || true   # warns model uncached, then fake server exits 0

step "compile x (documented stub, exit 7)"
if tt compile x; then
  echo "expected non-zero exit" >&2; exit 1
else
  code=$?
  [ "$code" -eq 7 ] || { echo "expected exit 7, got $code" >&2; exit 1; }
fi

printf '\n\033[1;32mDemo complete.\033[0m\n'
