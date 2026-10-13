# tt — the Tenstorrent CLI

One command for everything Tenstorrent.

`tt` is the single entry point to the Tenstorrent software stack. It either implements a
function natively or delegates to an existing tool (tt-smi, tt-installer, etc.) behind a stable interface.

> [!IMPORTANT]
> This is beta software. Expect breaking changes often as we add features. We welcome your feedback to help us improve tt-cli. If you have an issue or a comment, please don't hesitate to let us know through the GitHub [issues page](https://github.com/tenstorrent/tt-cli/issues/new).

## Prerequisites

To install and run `tt` you need:

- **Linux.** Ubuntu 22.04 or 24.04 LTS is the supported platform. Debian 13 and Fedora 42/43 also work; other distros are untested. macOS and Windows are not supported.
- **Python 3.10 or newer**, with the `venv` module (`python3-venv` on Debian/Ubuntu).
- **An isolated install tool**: [uv](https://docs.astral.sh/uv/) (recommended), pipx, or a dedicated venv. `tt` ships its own copy of `uv` for managing the tools it delegates to, so you do not need `uv` on your PATH beyond installing `tt` itself.
- **git** and **sudo**, for `tt update`: it clones pinned upstream repos and runs tt-installer, which installs the kernel driver, firmware tooling, and HugePages configuration as root.
- **Network access** to PyPI, GitHub, and the Hugging Face Hub on first run.

`tt device …` and `tt serve …` additionally need a Tenstorrent card with the driver and tt-smi installed (`tt update` does this), and serving needs Docker or Podman runnable without sudo plus a Hugging Face token. The full per-command list, including hardware, disk space, and `tt launch` client requirements, is in [docs/prerequisites.md](/docs/prerequisites.md).

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

If you opt in, `tt` records one event per command: the command name, exit code, duration, coarse OS facts, and argument values only when they match a known list (e.g. catalog model names). We never send free-form arguments, paths, error messages, or anything that identifies you or your machine. Exactly what is collected, every switch that controls it, and how delivery works: [TELEMETRY.md](/TELEMETRY.md).

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
| `tt model list` | Models that run on this machine's detected hardware, from two sources: the released catalog (models Tenstorrent ships and tests via tt-inference-server) and community bundles (`--all` for every device; `--cached`, `--type`, `--hw` filters) |
| `tt model list --catalog` / `--community` | Narrow to one source: `--catalog` for the released catalog only; `--community` for bundles anyone has published with tt-model-manager on the Hugging Face Hub, not tested or maintained by Tenstorrent (served with `tt serve <namespace>/<name>`); `--community --cached` for the ones installed here |
| `tt model search [QUERY]` | Search the Hub for published tt-model bundles (`--catalog` for community-catalog listings only; `--arch`, `--limit`) |
| `tt model info NAME` | Model metadata: engines, per-device support/status, requirements, cache state; for a tt-model bundle id, its manifest and compatibility verdict (or catalog row) |
| `tt model pull NAME` | Download a catalog model's weights, a tt-model bundle, or any HuggingFace repo's weights (`--bundle` / `--weights-only` override detection; `--offline`; bundles: `--force`, `--no-weights`) |
| `tt model profiles NAME` | A pulled bundle's serve profiles and its default |
| `tt serve [NAME] [-- ARGS…]` | Serve a model via tt-inference-server, TT-Studio, or tt-model-manager for a community bundle id (`--inference-server`, `--studio` or `--model-manager` forces a path; with no NAME, pick from what that backend serves; bundles: `--profile`, `--detach`, `--print`, `--refresh`, `--no-update-check`, `--no-weights`) — see [Serving backends](#serving-backends) |
| `tt model curl [PROMPT]` | Send a chat completion to the model being served; unknown options go into the request body (`--max-tokens 40`), `--print` shows the curl instead |
| `tt model stop NAME...` | Stop running model servers; when studio deployed one, studio stops the model and then its own containers and services (`--profile` to stop only one profile of a bundle; `tt model stop $(tt model ps --names)` stops everything) |
| `tt model rm NAME` | Remove a model's local artifacts, keeping its weights unless `--include-weights` (`--dry-run`, `--yes`) |
| `tt model login` | Log in to the Hugging Face Hub for gated or private bundles and weights (`--token`) |
| `tt model publish` / `unpublish` | List or delist your pushed bundle in the community catalog; `tt model package` / `package-thin` / `push` forward to tt-model's authoring commands unchanged |
| `tt model ps` | Model servers running on this machine: name, backend, port, health, uptime (`--all` includes stopped containers; `--no-probe` skips the HTTP health check; `--names`/`-n` prints only the names) |
| `tt model logs NAME` | Output of a served model: the newest tt-inference-server log file for a catalog model, or `tt-model logs` for a bundle (`--follow`; `--tail N`; `--since` needs a running container; `--profile` for bundles) |

| Other | Functionality |
|---|---|
| `tt update [VERSION]` | Converge system software + tools onto the latest tested "golden" set (`--dry-run` to preview; `--yes` to skip the confirmation; `--force` to allow downgrades to golden; a tt-installer VERSION runs that release instead and implies `--force`) |
| `tt config` | Open the config file in your editor; `list`/`get`/`set`/`path` for scripting, `sync`/`reset` to maintain the file |
| `tt report issue` | Open a prefilled GitHub issue on tt-cli (environment details auto-collected; `--no-browser` to just print the URL) |
| `tt report bundle` | Collect a redacted support bundle (environment, tt-smi snapshot, config, tt and inference-server logs, container logs) and open a pre-filled email to support@tenstorrent.com with it attached (`--title` for the subject, `--no-open` to only write the files, `--mailto` for webmail, `--output` to choose the path) |
| `tt self update` | Upgrade `tt` itself where it owns its environment (`--check` to only look) — see [Keeping tt up to date](/docs/DEVELOPERS.md) |
| `tt agent [GOAL]` | Set up Claude Code with the Tenstorrent skills for today's task (deploy a model, bring up a model, or develop) and launch it (`--dry-run` to preview; `--no-launch` to only install the plugins) |

For a comprehensive view on packaging, publishing and pulling down community models [read more here](/docs/community-models.md)

## Watching a model come up

A first serve can take ten minutes: the image is pulled, the weights are fetched, the
device is opened, the KV cache is sized and the model is warmed up. `tt serve` shows
those as a live checklist instead of the server's thousands of log lines, and returns
only once the endpoint actually answers — so `tt launch` straight afterwards works.
There is no flag for it; it is what serving looks like.

`tt serve` prints `tt model logs <model> --follow` before it starts, so that can be run in another terminal to watch the raw boot.

`TT_SERVE_READY_TIMEOUT=<seconds>` raises the one-hour bound on that wait. Ctrl-C stops
watching, not the server — the container keeps booting, and `tt model stop <model>`
ends it.

Both backends draw the same checklist. The long half of a serve — engine, device,
weights, KV cache, warmup — is the container's own boot, so tt reads
`docker logs` and classifies it itself rather than trusting either tool to
narrate it; only the preparation differs, and a bundle simply has one extra step.
[docs/serve-progress-contract.md](/docs/serve-progress-contract.md) has the
details, including what tt-model-manager emits.

Serving a catalog model needs a Hugging Face token: tt-inference-server fetches weights
and tokenizers from the Hub and asks for one interactively when `HF_TOKEN` is unset.
Run `hf auth login` once (or export `HF_TOKEN`) and `tt serve` stays non-interactive;
without one it says so up front rather than stopping at a prompt you cannot see.

## Interactive clients with `tt launch`

`tt serve` gives you an OpenAI-compatible endpoint; `tt launch` points a client at it. tt discovers what is running by asking the server itself (`GET /v1/models`) and configures the client's endpoint for you. `tt model ps` lists what is being served and on which port, using the same probe.

```bash
tt launch list                         # what can I connect, and is it usable now?
tt serve Qwen3-32B --port 8000         # in another terminal
tt launch openwebui --web-port 3080    # pull and run its container, after you confirm
tt launch stop openwebui               # stop it, keeping its data
tt launch hermes                       # Hermes Agent's terminal UI on the served model
tt launch hermes --web                 # its web dashboard and chat, on the same model
tt launch stop hermes                  # stop that dashboard (any running Hermes dashboard)
```

## Serving backends

`tt serve NAME` picks the serving path from the model:

- **inference-server** — every model `tt model list` shows with source `tt-inference-server`, driven through tt-inference-server's `run.py`. Preferred whenever it knows the model.
- **studio** — every model in [TT-Studio](https://github.com/tenstorrent/tt-studio)'s catalog: most are tt-inference-server's, which studio deploys from the same images, plus the few only studio carries (today `Qwen3.5-9B` and `Qwen3.8-27B`), for which it is the default. `tt serve NAME --studio` picks it for any of them. tt clones studio's latest tagged release on first use and runs `run.py run NAME` from it, which brings the stack up, deploys the model and reports the endpoint; `tt model stop NAME` runs its `--stop-model` for anything studio deployed, then `--stop` to take studio's containers and services down with it; a deploy that fails is followed by the same `--stop`, so a broken deploy leaves nothing of studio's running. Single-chip models (a `P150` entry) are listed for the multi-card Blackhole boards too, the way studio runs them — one chip of a P300. Studio allocates chips and ports itself, so `--device` and `--port` are ignored there with a warning.
- **model-manager** — tt-model bundles (`namespace/name`) neither catalog knows.

`tt model info` shows the paths a model is served by (`inference-server, studio` for a model both offer). `--inference-server`, `--studio` or `--model-manager` forces a path and refuses one the model does not offer. With no model, `tt serve --studio` (or `--inference-server`, `--model-manager`) lists what that path serves on this machine — for studio, its whole catalog — and asks for a number; the picker needs a terminal and is off under `--json`/`--quiet`.

Every path inherits a Hugging Face token: `HF_TOKEN` from the shell if set, else the token `hf auth login` stored (`HF_TOKEN_PATH`, then `<HF_HOME>/token`). `tt serve --dry-run` names the source without printing the token.

## Coding agents with `tt agent`

`tt agent` gets Claude Code ready for Tenstorrent work. It checks that Claude Code is installed (and offers the documented installer if not), asks what you are looking to do today, installs the matching plugins from the [tenstorrent/skills](https://github.com/tenstorrent/skills) marketplace, and hands the terminal to `claude`.

| Goal | Plugins loaded |
|---|---|
| `deploy` — Deploy a model on your Tenstorrent hardware | `tt-serve-model` |
| `bringup` — Bring up a new model | `tt-model-bringup` and the `tt-autodebug` it requires |
| `develop` — Actively develop | `tt-skills`, `tt-review-skills`, `tt-autodebug` |

```bash
tt agent                       # pick interactively, then launch claude
tt agent deploy --dry-run      # show what would be installed and run
tt agent develop -- --resume   # anything after -- goes to claude
```

While developing plugins, point it at a local checkout: `tt config set agent.marketplace_source ~/src/skills` (or `TT_AGENT_MARKETPLACE=~/src/skills` for one run).

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
