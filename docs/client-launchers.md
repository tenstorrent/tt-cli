# Client launchers: connecting apps to your model with `tt launch`

`tt serve` gives you an OpenAI-compatible endpoint. `tt launch <client>` points an app at it — a chat UI in your browser or a coding agent in your terminal — so you can start using the model straight away.

You don't have to tell the app where the model is. tt finds the model servers it started on this machine, asks each one what it serves (`GET /v1/models`), and configures the client for you.

```bash
tt serve Qwen3-32B          # 1. serve a model (returns once it answers)
tt launch list              # 2. see which clients are available here
tt launch openwebui         # 3. connect one
```

There are two kinds of client, and the difference is **who installs it**. tt-deep-research sits in between: you clone it, tt starts it.

## Web UIs: tt pulls and runs them for you

For these, tt pulls the image and starts the container itself, after asking you once. The only thing you need to install is a container runtime.

| Client | Command | Prerequisites (install first) | Model must support | What tt runs |
|---|---|---|---|---|
| [Open WebUI](https://github.com/open-webui/open-webui) | `tt launch openwebui` | [Docker](https://docs.docker.com/engine/install/) or [Podman](https://podman.io/docs/installation). Nothing else: tt pulls `ghcr.io/open-webui/open-webui:main` | any chat model | container `tt-open-webui` at <http://localhost:3000> (or the next free port); chats are kept in the volume `tt-open-webui-data` |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | `tt launch anythingllm` | [Docker](https://docs.docker.com/engine/install/) or [Podman](https://podman.io/docs/installation). Nothing else: tt pulls `mintplexlabs/anythingllm:latest` | any chat model | container `tt-anythingllm` at <http://localhost:3000> (or the next free port); data is kept in the volume `tt-anythingllm-storage` |

## Deep research: you clone it, tt starts it

[tt-deep-research](https://github.com/rnabeelTT/tt-deep-research) is a deep-research app (an agent that searches the web and your documents and writes a cited report). tt doesn't install it: clone it, and `tt launch deep-research` runs `uv run ttdr up --recipe <recipe>` in that checkout, after asking you once.

| Client | Command | Prerequisites (install first) | Model must support | What tt runs |
|---|---|---|---|---|
| [tt-deep-research](https://github.com/rnabeelTT/tt-deep-research) | `tt launch deep-research` | `git clone https://github.com/rnabeelTT/tt-deep-research ~/tt-deep-research` (or anywhere, with `--path` or `tt config set launch.deep_research_path <dir>`); [uv](https://docs.astral.sh/uv/getting-started/installation/); Docker with Compose | its LLM, embedder and reranker, served by tt: `tt-model serve raahemnabeel/cohost-ornith-35b-qwen3vl --local-only --port 8000 --detach` serves all three | its app containers (Compose project `ttdr`) with the UI at <http://localhost:3000>; `tt launch stop deep-research` runs `ttdr down`, keeping its data and the models |

tt picks the recipe from the bundle it is serving: `cohost-ornith-35b-qwen3vl` → `recipes/ornith-35b-cohost.toml`, `cohost-ornith-122b-qwen3vl` → `recipes/ornith-122b.toml`, `cohost-qwen38-27b-qwen3vl` → `recipes/qwen38-27b.toml`, and the three separate bundles (`ornith-1.5-35b-a3b`, `qwen3-vl-embedding-8b`, `qwen3-vl-reranker-8b`) → `recipes/rerank.toml`. `--recipe` picks one yourself. Either way tt first checks that the LLM (:8000), embedder (:8005) and reranker (:8004) all answer, so `ttdr up` never has to serve a model itself.

## Terminal coding agents: you install them, tt configures them

tt never installs these. Install the agent yourself, then `tt launch` writes the smallest config needed to reach your model and hands the terminal over to the agent.

| Client | Command | Prerequisites (install first) | Model must support | What tt writes |
|---|---|---|---|---|
| OpenCode | `tt launch opencode` | OpenCode: [install guide](https://opencode.ai/docs/) | tool calling | a `tenstorrent` provider in `~/.config/opencode/opencode.json` |
| pi | `tt launch pi` | pi: [install guide](https://github.com/earendil-works/pi) | tool calling | a `tenstorrent` provider in `~/.pi/agent/models.json` |
| Aider | `tt launch aider` | Aider: [install guide](https://aider.chat/docs/install.html) | tool calling | nothing on disk (environment variables, this run only) |
| Qwen Code | `tt launch qwencode` | Node.js 22+, then `npm install -g @qwen-code/qwen-code`: [install guide](https://github.com/QwenLM/qwen-code) | tool calling | nothing on disk (command-line arguments, this run only) |
| Hermes Agent | `tt launch hermes` (terminal UI) or `tt launch hermes --web` (browser dashboard) | Hermes Agent: `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash`, see the [install guide](https://hermes-agent.nousresearch.com/docs/) | tool calling | nothing on disk (environment variables, this run only) |

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
tt launch deep-research --recipe recipes/rerank.toml   # pick the recipe yourself
tt launch disconnect opencode        # undo the config tt wrote for a client
```

| Option | What it does |
|---|---|
| `--model NAME` | Which model to connect to, when more than one is being served |
| `--port N` | Port the model is served on, if tt can't find it |
| `--url URL` | Full OpenAI-compatible base URL ending in `/v1`, for a server on another machine |
| `--web` | Open the client's web UI instead of its terminal UI (Hermes Agent) |
| `--web-port N` | Host port for a web UI (default 3000, or the next free port above it) |
| `--no-exec` | Set the client up but don't start it |
| `--yes`, `-y` | Skip the confirmation prompt |
| `--force` | Connect even if the model can't do tool calling |
| `--path DIR` | deep-research only: its checkout (default `launch.deep_research_path`, else `~/tt-deep-research`) |
| `--recipe FILE` | deep-research only: the recipe to start, relative to the checkout (default: the one matching the served bundle) |

> [!TIP]
> Web UIs start on port 3000. If it's taken, tt picks the next free port and prints the URL. TT Studio's web UI also wants port 3000 but doesn't move when it's taken, so if you use Studio, start it before any `tt launch` web UI.

For the full system requirements behind each client (how tt finds the binary, disk space, container permissions), see [Prerequisites](prerequisites.md#tt-launch-clients).
