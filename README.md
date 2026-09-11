# tt — the Tenstorrent CLI

One command for everything Tenstorrent.

`tt` is the single entry point to the Tenstorrent software stack. It either implements a
function natively or delegates to an existing tool (tt-smi, tt-installer, etc.) behind a stable interface.

> [!IMPORTANT]
> This is beta software. Expect breaking changes often as we add features. We welcome your feedback to help us improve tt-cli. If you have an issue or a comment, please don't hesitate to let us know through the GitHub [issues page](https://github.com/tenstorrent/tt-cli/issues/new).

## Quick start

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```console
uv tool install tenstorrent
tt --help
```

(`uv tool update-shell` will make sure the `tt` binary is available on your PATH)

If you don't have uv, install it first:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can also install the CLI with any pip-compatible tool, such as pipx. We *highly recommend* placing tt-cli in an isolated venv so it can safely update itself. For other installation methods, see [DEVELOPERS.md](/docs/DEVELOPERS.md#other-ways-to-install-tt-cli).

## Telemetry

Telemetry is **opt-in**: nothing is collected or sent unless you say so. `tt` asks once, on first interactive run, and doesn't bother you about it on later runs.

If you opt in, `tt` records command names, exit codes, coarse OS facts, and argument values only when they match a known list (e.g. catalog model names). We never send free-form arguments, paths, or static IDs. Exactly what is collected, every switch that controls it, and how delivery works: [TELEMETRY.md](/TELEMETRY.md).

If you change your mind about telemetry, use `tt config set telemetry.enabled true` or `false`. If you previously enabled telemetry, setting it to `false` opts you out and deletes anything not yet uploaded.

## Example commands

This is not an exhaustive list. For the full list of commands and options in each group, add `--help`, e.g. `tt model --help`.

| Device info | Functionality |
|---|---|
| `tt device status` | Detected devices: board, temperature, power, clock (`--json` for JSON output) |
| `tt device info [N…]` | Device metadata: PCI IDs, board id, firmware versions |
| `tt device reset [N…]` | Reset devices |
| `tt smi` | Hand this terminal to the interactive tt-smi TUI (`tt device top` also works) |

| Model serving | Functionality |
|---|---|
| `tt model list` | Models that run on this machine's detected hardware and how each is served (`via`: inference-server or studio; `--all` for every device; `--cached`, `--type`, `--hw` filters; `--community` for tt-model-manager bundles) |
| `tt model info NAME` | Model metadata: engines, per-device support/status, requirements, cache state |
| `tt model pull NAME` | Download a catalog model's weights, a tt-model bundle, or any HuggingFace repo's weights (`--bundle` / `--weights-only` override detection; `--offline`) |
| `tt serve [NAME] [-- ARGS…]` | Serve a model via tt-inference-server, TT-Studio, or tt-model-manager for a community bundle id (`--backend auto\|inference-server\|studio\|model-manager` forces a path; with no NAME, pick from what that backend serves) — see [Serving backends](#serving-backends) |
| `tt model stop NAME` | Stop a running model server, through studio for a studio-only model (`--profile` to stop only one profile of a bundle) |

| Other | Functionality |
|---|---|
| `tt update [VERSION]` | Converge system software + tools onto the latest tested "golden" set (`--dry-run` to preview; `--yes` to skip the confirmation; `--force` to allow downgrades to golden; a tt-installer VERSION runs that release instead and implies `--force`) |
| `tt config` | Open the config file in your editor; `list`/`get`/`set`/`path` for scripting, `sync`/`reset` to maintain the file |
| `tt report issue` | Open a prefilled GitHub issue on tt-cli (environment details auto-collected; `--no-browser` to just print the URL) |
| `tt self update` | Upgrade `tt` itself where it owns its environment (`--check` to only look) — see [Keeping tt up to date](/docs/DEVELOPERS.md) |

For a comprehensive view on packaging, publishing and pulling down community models [read more here](/docs/community-models.md)

## Interactive clients with `tt launch`

`tt serve` gives you an OpenAI-compatible endpoint; `tt launch` points a client at it. tt discovers what is running by asking the server itself (`GET /v1/models`) and configures the client's endpoint for you.

```bash
tt launch list                         # what can I connect, and is it usable now?
tt serve Qwen3-32B --port 8000         # in another terminal
tt launch openwebui --web-port 3080    # pull and run its container, after you confirm
tt launch stop openwebui               # stop it, keeping its data
```

## Serving backends

`tt serve NAME` picks the serving path from the model:

- **inference-server** — every model `tt model list` shows with `via inference-server`, driven through tt-inference-server's `run.py`. Preferred whenever it knows the model.
- **studio** — the few models only [TT-Studio](https://github.com/tenstorrent/tt-studio)'s catalog carries (today `Qwen3.5-9B` and `Qwen3.8-27B`). tt clones studio's `dev` branch on first use and runs `run.py run NAME` from it, which brings the stack up, deploys the model and reports the endpoint; `tt model stop NAME` runs its `--stop-model`. A model tt-inference-server serves is never offered through studio, even with `--backend studio`. Single-chip models (a `P150` entry) are listed for the multi-card Blackhole boards too, the way studio runs them — one chip of a P300. Studio allocates chips and ports itself, so `--device` and `--port` are ignored there with a warning.
- **model-manager** — tt-model bundles (`namespace/name`) neither catalog knows.

`tt model list` shows the path in its `via` column; `tt model info` says the same. `--backend` forces a path and refuses one the model does not offer. With no model, `tt serve --backend studio` (or `inference-server`, `model-manager`, `auto`) lists what that path serves on this machine and asks for a number; the picker needs a terminal and is off under `--json`/`--quiet`.

Every path inherits a Hugging Face token: `HF_TOKEN` from the shell if set, else the token `hf auth login` stored (`HF_TOKEN_PATH`, then `<HF_HOME>/token`). `tt serve --dry-run` names the source without printing the token.

## Configuration

Configuration is TOML at `~/.config/tenstorrent/config.toml` (XDG-compliant). `tt config` opens it with
explanatory comments; edits preserve comments. Prefer `tt config set <key> <value>` over hand-editing: it puts the key in the correct
table.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](/CONTRIBUTING.md) for details on:

- Reporting bugs via GitHub Issues
- Submitting pull requests
- Code standards and SPDX header requirements
- Our weekly PR review cadence

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](/LICENSE) file for the overall license, except where specified.

For clarification on how the Apache 2.0 license applies to this project, including hardware and patent considerations, see [LICENSE_understanding.txt](/LICENSE_understanding.txt).
