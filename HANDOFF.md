# Handoff: pinning managed tools (drop this file before merging)

Branch: `raahem/pin-tool-versions`

## Done (commit d8c5d9a)

- `tt self pin <tool> <version>` / `tt self unpin <tool>`; pins are stored in config under `[tools.pin]`.
- Pinnable kinds: `uv-tool` and `git-venv` (tt-inference-server, tt-model, tt-studio, tt-smi,
  tt-flash). tt-installer is excluded; `tt update <version>` covers it.
- The version is checked upstream when pinning and again at install time
  (`remote_version_exists` in `tools/installers.py`: git ls-remote, or PyPI JSON).
  Unknown version → `tool.version_not_found`, exit 2.
- `ToolRegistry.target_spec()` (golden, or the pin) drives `ensure()`, `tt update`'s plan and
  apply step, and the tt-model label in `tt serve`.
- `tt update` asks per diverged pin, default No; without a TTY or under `--json` it keeps the
  pin. It re-plans after an unpin.
- Fake-mode suite green (2034 passed); manually checked against real GitHub/PyPI.

## Open: the model catalog does not follow a pinned tt-inference-server

`tt model list` and `tt serve` read the bundled `modelhub/model_support.json`, which
`scripts/build_model_support.py` builds from the *golden* tag's `release_model_spec.json`.
With tt-inference-server pinned, run.py uses its own spec (docker image etc.), but tt's model
list, device support and serve fixes still come from the golden version. Symptoms:

- Models newer than the pin are listed and accepted, then rejected mid-run by run.py.
- Models that exist only in the older version are missing, and `tt serve` treats them as
  tt-model bundles (`commands/serve.py`, the `entry is None` branch).

Workaround today: `TT_MODEL_SUPPORT_PATH`, pointing at a hand-built file.

### Planned long-term fix

1. Move the spec → support-list parser out of `scripts/build_model_support.py` into the
   package (e.g. `modelhub/`), keeping the script as a thin wrapper so `--check` in CI still
   produces byte-identical output.
2. When tt-inference-server is pinned (`registry.pinned_version(...)`), build the catalog from
   `<pinned checkout>/release_model_spec.json` (every tagged checkout ships it) and cache the
   result per version under the data dir.
3. Apply `model_support_overrides.toml` entries only where they still match models/devices in
   the older spec, since they are tuned to the golden version.
4. If the pinned checkout is not installed yet, fall back to the bundled list with a warning.
5. Interim nicety: warn on `tt self pin tt-inference-server` that the list reflects golden, and
   fail early in `tt serve` when the pinned spec lacks the model.

## Before merging

- Delete this file.
- Decide whether the catalog work above blocks merge or lands as a follow-up.
