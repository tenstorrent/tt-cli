# Client launchers: connecting apps to your model with `tt launch`

`tt serve` gives you an OpenAI-compatible endpoint. `tt launch <client>` points an app at it — a chat UI in your browser or a coding agent in your terminal — so you can start using the model straight away.

You don't have to tell the app where the model is. tt finds the model servers it started on this machine, asks each one what it serves (`GET /v1/models`), and configures the client for you.

```bash
tt serve Qwen3-32B          # 1. serve a model (returns once it answers)
tt launch list              # 2. see which clients are available here
tt launch openwebui         # 3. connect one
```

There are two kinds of client, and the difference is **who installs it**. tt-deep-research sits in between: tt installs and starts it.

## Web UIs: tt pulls and runs them for you

For these, tt pulls the image and starts the container itself, after asking you once. The only thing you need to install is a container runtime.

| Client | Command | Prerequisites (install first) | Model must support | What tt runs |
|---|---|---|---|---|
| [Open WebUI](https://github.com/open-webui/open-webui) | `tt launch openwebui` | [Docker](https://docs.docker.com/engine/install/) or [Podman](https://podman.io/docs/installation). Nothing else: tt pulls `ghcr.io/open-webui/open-webui:main` | any chat model | container `tt-open-webui` at <http://localhost:3000> (or the next free port); chats are kept in the volume `tt-open-webui-data` |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | `tt launch anythingllm` | [Docker](https://docs.docker.com/engine/install/) or [Podman](https://podman.io/docs/installation). Nothing else: tt pulls `mintplexlabs/anythingllm:latest` | any chat model | container `tt-anythingllm` at <http://localhost:3000> (or the next free port); data is kept in the volume `tt-anythingllm-storage` |

## Deep research: tt installs and starts it

[tt-deep-research](https://github.com/rnabeelTT/tt-deep-research) is a deep-research app (an agent that searches the web and your documents and writes a cited report). `tt launch deep-research` installs it at tt's pinned version on first launch and runs `ttdr up --recipe <recipe>`, after asking you once. The app keeps its files and data in `~/.tt-deep-research` (`$TTDR_HOME`).

| Client | Command | Prerequisites (install first) | Model must support | What tt runs |
|---|---|---|---|---|
| [tt-deep-research](https://github.com/rnabeelTT/tt-deep-research) | `tt launch deep-research` | git; Docker with Compose. Nothing else: tt installs the app | the models of its recipe (`--list-recipes`). With the board free, `ttdr up` serves them itself | its app containers (Compose project `ttdr`) with the UI at <http://localhost:3000> (the next free port if another program holds it); `tt launch stop deep-research` runs `ttdr down`: the app, its data services and its models stop, and its data is kept (`--keep-models` leaves the models serving) |

Its recipes say which models it serves and on which chips: `tt launch deep-research --list-recipes` lists them. tt uses the recipe whose tt-model bundles are running; with nothing running it asks which recipe to start (Enter takes `default`, three bundles on their own chips), or starts the one `--recipe <name>` names (`default` with `--yes` or without a terminal), and `ttdr up` serves its models (tt asks first: a cold start takes 10-15 minutes). It never serves over models it didn't start: if other bundles hold the chips, or another model answers on a cohost recipe's port (`ttdr up` moves a bundle of its own to the next free port), it says what to stop (`tt model stop ...`) and which recipe fits what is running instead. The cohost recipes need the cohost tt-model build (see the tt-deep-research README).

### Hermes on deep-research's tools

The research engine is an MCP server, and the app ships Hermes skills for it, so Hermes can research with the same tools. Once `tt launch deep-research` is up, two commands give them to Hermes (`ttdr up` prints them and the launch command, with the engine's actual port):

```bash
hermes config set mcp_servers.ttdr.url http://localhost:9901/mcp
hermes config set skills.external_dirs '["~/.tt-deep-research/.hermes/skills"]'
tt launch hermes        # picks the recipe's LLM from the models it serves
```

`qb2-deep-research` is the skill for cited research with those tools. Hermes reads it from the app's folder, so it updates with the app.

## Terminal coding agents: you install them, tt configures them

tt never installs these. Install the agent yourself, then `tt launch` writes the smallest config needed to reach your model and hands the terminal over to the agent.

| Client | Command | Prerequisites (install first) | Model must support | What tt writes |
|---|---|---|---|---|
| OpenCode | `tt launch opencode` | OpenCode: [install guide](https://opencode.ai/docs/) | tool calling | a `tenstorrent` provider in `~/.config/opencode/opencode.json` |
| pi | `tt launch pi` | pi: [install guide](https://github.com/earendil-works/pi) | tool calling | a `tenstorrent` provider in `~/.pi/agent/models.json` |
| Aider | `tt launch aider` | Aider: [install guide](https://aider.chat/docs/install.html) | tool calling | nothing on disk (environment variables, this run only) |
| Qwen Code | `tt launch qwencode` | Node.js 22+, then `npm install -g @qwen-code/qwen-code`: [install guide](https://github.com/QwenLM/qwen-code) | tool calling | a model entry under `modelProviders.openai` in `~/.qwen/settings.json`, carrying the served context window and output budget (only when the server publishes its context length); the endpoint goes on the command line |
| Hermes Agent | `tt launch hermes` (terminal UI) or `tt launch hermes --web` (browser dashboard) | Hermes Agent: `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash`, see the [install guide](https://hermes-agent.nousresearch.com/docs/) | tool calling | environment variables (this run only); installs the Tenstorrent skin (and, with `--web`, dashboard theme) and makes it active unless you picked another |

- Coding agents need a model that can do **tool calling**. If tt knows the model you're serving can't, it stops and suggests one that can (`--force` connects anyway). For a model outside tt's catalog it can't check, so it warns and carries on.
- tt shows the change and asks before writing to a client's config file. `tt launch disconnect <client>` removes what tt wrote.

## What about TT Studio?

[TT Studio](https://github.com/tenstorrent/tt-studio) is **not** a `tt launch` client. It's a serving backend with its own web UI: it deploys the model *and* gives you the interface. Start it with:

```bash
tt serve <model> --studio
```

It needs Docker (Podman is not supported). See [Serving backends](../README.md#serving-backends) for how it compares to the other serving paths.

## Commands and options

```bash
tt launch list                       # every client, and whether it's usable now
tt launch <client>                   # connect it to the model you're serving
tt launch <client> --dry-run         # show what would be done, change nothing
tt launch stop openwebui             # stop a web UI (or the Hermes dashboard); its data is kept
tt launch deep-research --list-recipes   # its recipes and what each serves
tt launch deep-research --recipe ornith-35b-cohost   # pick the recipe yourself
tt launch stop deep-research --keep-models  # stop the app, leave its models serving
tt launch disconnect opencode        # undo the config tt wrote for a client
```

| Option | What it does |
|---|---|
| `--model NAME` | Which model to connect to, when more than one is being served |
| `--port N` | Port the model is served on, if tt can't find it |
| `--url URL` | Full OpenAI-compatible base URL ending in `/v1`, for a server on another machine |
| `--web` | Open the client's web UI instead of its terminal UI (Hermes Agent) |
| `--web-port N` | Host port for a web UI (default 3000, or the next free port above it) |
| `--no-exec` | Set the client up but don't start it (not deep-research: there is nothing to set up, so use `--dry-run`) |
| `--yes`, `-y` | Skip the confirmation prompt |
| `--force` | Connect even if the model can't do tool calling |
| `--recipe NAME` | deep-research only: the recipe to start (default: the one whose models are running, else `default`) |
| `--list-recipes` | deep-research only: its recipes and what each serves on which chips; on a terminal, pick one to start |
| `--aiq` | deep-research only: also run AI-Q's own harness beside Deep Agents (`ttdr up --aiq`), its UI on <http://localhost:3001> (or the next free port) |

> [!TIP]
> Web UIs start on port 3000. If it's taken, tt picks the next free port and prints the URL. TT Studio's web UI also wants port 3000 but doesn't move when it's taken, so if you use Studio, start it before any `tt launch` web UI.

For the full system requirements behind each client (how tt finds the binary, disk space, container permissions), see [Prerequisites](prerequisites.md#tt-launch-clients).
