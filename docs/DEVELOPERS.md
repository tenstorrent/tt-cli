This page covers advanced usage of tt-cli: alternative installation methods, golden versions and self-updating, the model catalog, environment variables, exit codes, and running the test suite.

# Other ways to install tt-cli

All of these need Python 3.10 or newer.

## Global: Using pipx

```console
pipx install tenstorrent
tt --help
```

We highly recommend placing tt-cli in an isolated venv: `uv tool`, `pipx`, or a regular venv with nothing else installed in it. An isolated venv is what allows tt-cli to self-update; see [Keeping tt up to date](#keeping-tt-up-to-date).

## Local: with venv and pip

Create and/or activate a python venv in your current directory:

```console
python3 -m venv .venv && source .venv/bin/activate
```

Then install and use the package:

```console
pip install tenstorrent
tt --help
```

Note that unless the venv is active, `tt` will not be available on your PATH.

## Directly from main using git

Latest from main may be broken! Only use this if you know what you are doing.

```console
git clone https://github.com/tenstorrent/tt-cli.git && cd tt-cli && python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -e . && tt --help
```

# Self-updating and golden versions

## Golden versions (system software)

`tt update` converges your system onto a *tested set* of versions published by
[tt-sw-manifest](https://github.com/tenstorrent/tt-sw-manifest) as a CI-validated `golden.json`:
one pinned version per component (firmware, kernel driver, tt-smi, tt-flash, ...). `tt update`
fetches that file at the pinned release tag, verifies its checksum, and caches it locally.
The system stack itself is applied via tt-installer, which downloads the matching
distro-specific installer schema (.ttis) for the same release. Note that tt-installer requires sudo.

Tools delegated to by tt-cli (tt-smi, tt-flash, tt-model) are pinned into isolated
per-tool venvs under `~/.local/share/tenstorrent/` so they can never conflict with each
other or with your Python environment. With `--offline`, tool pins converge from local
caches and the system stack is skipped (tt-installer inherently needs the network).

## Keeping tt up to date

Tool pins (tt-model-manager, tt-inference-server, the installer) ship inside the `tenstorrent`
package, so an old `tt` means old pins. Once a day `tt` looks up the newest release in a
detached background process (one request to PyPI carrying nothing but `tt`'s version)
and, on your next interactive run, prints a short notice asking you to update.

`tt self update` updates for you, but only where `tt` provably owns its environment.

| How you installed `tt` | What `tt self update` does |
|---|---|
| `uv tool install tenstorrent` | `uv tool install --force tenstorrent==<new>` into the same tool venv |
| `pipx install tenstorrent` | `pipx upgrade tenstorrent` |
| a venv with nothing else in it | `pip install tenstorrent==<new>` or `uv pip install …`, whichever made the venv |
| a venv shared with other packages | Refuses (exit 7) and prints the exact command, so *you* decide; this can break dependencies for other packages in the venv. |
| `pip install -e .` checkout | Nothing, ever: `git pull` |

`tt update` runs the same check first and, on a terminal, offers to upgrade `tt` before
touching the system stack (the new `tt` then runs the update). Scripts are never
surprised: without a TTY it only prints the notice, `--yes` or not.

Turn the check off with `tt config set update.check false`, per run with
`TT_NO_UPDATE_CHECK=1`; `--offline` and CI skip it as well. `TT_UPDATE_CHECK_URL` points it
at a mirror (a URL or a local file with PyPI's project JSON shape).

# Model catalog

`tt model list` is backed by a generated support list, built from
tt-inference-server's
[release_model_spec.json](https://github.com/tenstorrent/tt-inference-server/blob/main/release_model_spec.json)
at the pinned server version plus tt's own record of what is known not to work
on which board. In the future, the authoritative model catalog will be migrated
to an upstream repository.

By default the list shows only models that run on this machine's detected device
configuration (via tt-smi) — a model marked as failing on that board is hidden,
and `tt model info` says why. `--all` shows every model on every device, and
`--hw <device>` filters to a specific configuration without touching the hardware.
Model names are the spec's short ids (`Llama-3.1-8B-Instruct`); the HuggingFace repo id
works as an alias everywhere a name is accepted. Every engine is servable —
`vLLM`, `media` and `forge`.

## Community bundles

`tt model list --community` lists models packaged with
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager): bundles
published as HuggingFace repos. These are served by engines included in the repos themselves
(`vllm-plugin`, `tt-dit-server`, etc). `tt model pull
<namespace>/<name>` installs one, and `tt serve <namespace>/<name>` serves it,
passing through anything `tt serve` does not recognize (`tt serve repo/model --
--port 8080 --follow`).

`tt model pull` also accepts an ordinary HuggingFace repo id, fetching its weights
into the same cache — useful to pre-warm before serving — with a warning that
weights alone do not make a model servable.

# Environment variables

| Variable | Effect |
|---|---|
| `TT_CONFIG_DIR` / `TT_DATA_DIR` / `TT_CACHE_DIR` | Relocate config / data (tool venvs, state) / cache |
| `TT_TOOL_BIN_<TOOL>` | Use a specific binary for a managed tool (e.g. `TT_TOOL_BIN_TT_SMI=/opt/bin/tt-smi`) |
| `TT_GOLDEN_PATH` | Use a local golden.json instead of the fetched/cached one (air-gapped installs) |
| `TT_MANIFEST_PATH` | Override the bundled CLI-tool supplement manifest |
| `TT_MODEL_SUPPORT_PATH` | Override the bundled generated model support list |
| `TT_UV_BIN` | Use a specific `uv` binary |
| `TT_TOOL_BIN_<CLIENT>` | Also how `tt launch` finds a client (`TT_TOOL_BIN_OPENCODE`, `TT_TOOL_BIN_AIDER`, …), or `tt config set tools.override.<client> <path>` |
| `TT_NO_UPDATE_CHECK` | Skip the daily "newer tt available?" lookup for this run (`tt config set update.check false` to turn it off for good) |
| `TT_UPDATE_CHECK_URL` | Where that lookup reads PyPI-shaped project JSON from (URL or local file; default `https://pypi.org/pypi/tenstorrent/json`) |
| `HF_HOME` | Weights cache root when `paths.hf_model_cache_directory` is unset; exported to tt-model and passed to tt-inference-server so every tool shares one cache |
| `XDG_CACHE_HOME` | Where tt-model keeps its installed-bundle index (`$XDG_CACHE_HOME/tt-model`), which `tt model list --community` reads |
| `VISUAL` / `EDITOR` | Editor opened by bare `tt config` |

Telemetry has its own set of variables (`TT_TELEMETRY_*`, `DO_NOT_TRACK`) — see
[TELEMETRY.md](../TELEMETRY.md).

# Exit codes

Exit codes are a documented contract:

| Code | Name | Meaning |
|---|---|---|
| 0 | OK | Success |
| 1 | ERROR | Generic/unexpected failure |
| 2 | USAGE | Bad arguments |
| 3 | NO_DEVICES | No Tenstorrent devices detected |
| 4 | TOOL_MISSING | A required tool is not installed (run `tt update`) |
| 5 | TOOL_FAILED | A wrapped tool ran and failed |
| 6 | NEEDS_SUDO | Privileged operation; sudo unavailable non-interactively |
| 7 | UNSUPPORTED | Stub / not implemented yet |
| 8 | OFFLINE | Network needed but offline |
| 9 | CONFIG | Invalid configuration |

# Testing the local repo

From a checkout of the repo:

```console
$ uv run pytest                       # default: everything against the fakes, no hardware
$ uv run pytest --hardware            # against the real stack on a TT machine (read-only)
$ uv run pytest --hardware --run-destructive   # also run tests that reset devices
```

By default the suite runs entirely against the stand-in tools in `tests/fakes/`, wired
in through the `TT_TOOL_BIN_*` seams — no hardware, config, or network. `--hardware`
points the tool-driving tests at the real binaries installed on the machine (resolved
via `PATH`, then the tt-installer venv) and runs the real read-only smoke tests in
`tests/hardware/`; tests that assert fake-tool internals are skipped. Add
`--run-destructive` to also test device resets.
