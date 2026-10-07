# Prerequisites

Everything `tt` needs from the machine it runs on, and which commands need it. The
short version is in the [README](/README.md#prerequisites); this page is the full
list, with what happens when something is missing. Exit codes referenced below are
the documented contract in [DEVELOPERS.md](DEVELOPERS.md#exit-codes).

`tt` is a thin front end: most of what it does is delegate to tt-smi, tt-installer,
tt-inference-server, tt-model-manager, TT-Studio and the container runtime. The
prerequisites are therefore mostly *theirs*, and `tt update` installs as many of them
as it can. The sections below separate what you need to **install** `tt`, what you
need to **run** each command group, and what the hardware stack itself requires.

## Operating system

`tt` targets Linux. There is no explicit OS check in the CLI, but it relies on POSIX
APIs (`sudo`, `/dev/tenstorrent`, `fcntl` locks, `venv/bin` layouts) and the system
stack it installs is Linux-only.

| OS | Status | Notes |
|---|---|---|
| Ubuntu 22.04 LTS | Supported | Tenstorrent's preferred platform; CI runs the suite here |
| Ubuntu 24.04 LTS | Supported | In CI |
| Debian 13 | Supported | In CI; `curl` is not installed by default |
| Fedora 42 / 43 | Supported | In CI; tt-installer may need a restart after the base packages |
| Ubuntu 20.04 | Not supported | Deprecated by tt-installer |
| Other DEB/RPM distros | Untested | May work; tt-installer falls back to a published apt suite for unknown codenames |
| Arch, NixOS | Not supported | tt-installer does not support them |
| macOS, Windows, WSL | Not supported | No driver, no `/dev/tenstorrent`; nothing hardware-related will work |

Distro-specific logic lives entirely in tt-installer. `tt update` hands it
`--versions=release` and lets it pick the matching installer schema for the running
distro.

## Installing `tt`

| Requirement | Why |
|---|---|
| Python 3.10 or newer | `requires-python = ">=3.10"`; 3.10 through 3.14 are tested |
| `python3-venv` (Debian/Ubuntu) | The `venv` module is a separate package there; without it, isolated installs fail with an unhelpful error |
| `uv`, `pipx`, or a dedicated venv | `tt` must live in an isolated environment to be able to update itself; see [Other ways to install tt-cli](DEVELOPERS.md#other-ways-to-install-tt-cli) |
| Network to PyPI | To fetch the `tenstorrent` package and its dependencies |

The package depends on `typer`, `rich`, `tomlkit`, `platformdirs`, `uv`,
`huggingface_hub`, `httpx` and `packaging`. Two of those matter beyond import time:

- **`uv`** (the PyPI package) ships the `uv` binary that `tt` uses to create the
  per-tool venvs described below. Lookup order is `TT_UV_BIN`, then the bundled wheel
  binary, then `uv` on your PATH. You do not need `uv` installed separately to run
  `tt`; if the bundled binary is somehow missing, `tt` exits 4 and asks you to
  reinstall the package. (`tt self update` on a uv-tool install prefers the `uv` on
  your PATH, since that is the one that made the install.)
- **`huggingface_hub`** provides the Hub client `tt model pull` uses directly, and the
  `hf` / `huggingface-cli` command you log in with.

No compiler, Rust toolchain, Node.js, or other build tooling is needed to install the
CLI itself.

## Running `tt`: system tools by command

A wrapped binary that is missing produces "Cannot run 'X': executable not found" and
exit 4 (`TOOL_MISSING`); one that exists but fails produces exit 5 (`TOOL_FAILED`).

| Tool | Needed by | Notes |
|---|---|---|
| **git** | `tt update` (always: tt-model is installed from a git ref); `tt serve` on first use of the inference-server or studio path; `tt update --include-lazy` | Clones pinned checkouts of tt-inference-server (~180 MB) and TT-Studio (~200 MB) under the data dir |
| **sudo** | `tt device reset`; the system phase of `tt update` | `tt` probes `sudo -n true` and prompts on a terminal. Without a TTY (or under `--json`) it exits 6 (`NEEDS_SUDO`) and prints the command to re-run. Running as root skips sudo. Set `tools.sudo_command` to `""` to disable it |
| **docker or podman** | `tt serve` (inference-server path); `tt model ps` / `stop` / `logs`; `tt launch openwebui` / `anythingllm` | Either runtime works here |
| **docker** only | `tt serve --studio` (docker compose; podman is not supported); `tt serve <namespace>/<name>` (tt-model bundles) | |
| **An editor** | bare `tt config` | `$VISUAL`, then `$EDITOR`, then the first of `nano`, `vim`, `vi` on PATH. None found: exit 9 |
| **A pager** | long output | `TT_PAGER`, `PAGER`, then `less`; optional, falls back to plain printing |
| **A browser / mail client** | `tt report issue`, `tt report bundle` | Best effort: `webbrowser`, `xdg-open` (needs `DISPLAY` or `WAYLAND_DISPLAY`). On a headless box the URL or `.eml` path is printed instead |
| **pipx** / **pip** | `tt self update`, only when `tt` was installed with that tool | |

`curl` is never run by `tt` itself (downloads go through `httpx`); it appears only in
printed hints and in the uv install one-liner.

### Container runtime permissions

`tt` always invokes `docker` / `podman` directly, never through `sudo`. Your user must
therefore be able to talk to the daemon without elevation: either add yourself to the
`docker` group (`sudo usermod -aG docker $USER`, then log out and back in) or use
rootless Docker/Podman. A permission-denied from the socket surfaces as a generic
exit 5. The `tt launch` container clients use `--add-host host.docker.internal:host-gateway`,
which needs Docker 20.10 or newer.

tt-installer installs Docker (or Podman, on request) if neither is present, so on a
supported distro `tt update` is the simplest way to satisfy this.

## Hardware and the system stack

Every `tt device` command delegates to tt-smi, and `tt serve` opens the card through
a container. For those you need:

- **A Tenstorrent card.** Wormhole (n150, n300, T3K / 4×n300) and Blackhole (p100,
  p150, p150x4, p150x8, p300, p300x2) boards are the ones `tt serve` knows how to map
  to a `--device` argument. Grayskull (e150) is still in that mapping, but current
  tt-kmd releases list only Wormhole and Blackhole as supported.
- **The kernel-mode driver (tt-kmd)**, built via DKMS against your running kernel.
  tt-kmd supports Linux 5.4 and newer. `tt` reads the driver version from tt-smi's
  snapshot but never installs or checks it itself.
- **tt-smi** (and tt-flash for firmware). `tt` installs the golden-pinned versions
  into isolated venvs under `~/.local/share/tenstorrent/tools/`, and falls back to
  `~/.tenstorrent-venv/bin/tt-smi` from a prior tt-installer run.
- **Firmware at the golden version**, flashed by tt-flash. `tt update` compares your
  firmware to golden and asks before flashing (`--yes` to skip the prompt; `--force`
  to allow a downgrade).
- **HugePages** configured by tt-installer's system packages.
- **BIOS**: PCIe AER reporting set to "OS First" (the TT-QuietBox defaults to this).

`tt update` provides all of this in one go. It runs three phases:

1. **Checks**: fetches `golden.json` from the pinned tt-sw-manifest release on GitHub
   and verifies its checksum. With `--offline` and no cached copy, exit 8.
2. **Tools**: `uv tool install` of tt-smi, tt-flash and tt-model into per-tool venvs.
   Lazy tools (tt-inference-server, TT-Studio) only with `--include-lazy`.
3. **System**: downloads the sha256-pinned tt-installer `install.sh` and runs it
   non-interactively with `--use-uv --python-version=3.12`. install.sh asks for sudo
   itself and installs the base packages, Tenstorrent apt/dnf repositories, tt-kmd,
   HugePages, firmware, and a container runtime. `--offline` skips this phase.

`tt` passes `--reboot-option=never`, so it never reboots your machine. **After the
first install of the kernel driver or a HugePages change, reboot before using the
card.** `tt device status` exits 3 (`NO_DEVICES`) with "The driver may not be installed,
or no card is seated" until the driver is loaded.

Until `tt update` has run once with network access, golden versions are unknown and
`tt` cannot install any tool: it exits 9 with "Run `tt update` once with network
access." Air-gapped machines can point `TT_GOLDEN_PATH` at a local `golden.json`.

## Network and credentials

| Endpoint | Used by |
|---|---|
| **PyPI** | Installing `tt`; `uv tool install` of tt-smi/tt-flash; the daily "newer tt available?" check (one request, 10 s timeout; off with `tt config set update.check false`, `TT_NO_UPDATE_CHECK=1`, or `--offline`) |
| **GitHub** (`github.com`, release assets) | `golden.json`, `install.sh`, and `git clone` of tt-inference-server, tt-model-manager and TT-Studio |
| **Tenstorrent package repositories** | install.sh adds them for tt-kmd, tenstorrent-tools and firmware |
| **Hugging Face Hub** | `tt model search` / `pull` / `info` (for uninstalled bundles) / `login` / `publish` / `unpublish` / `push`; weights and tokenizers for every serve |
| **ghcr.io / Docker Hub** | Model images pulled by tt-inference-server, tt-model and TT-Studio; `ghcr.io/open-webui/open-webui:main` and `mintplexlabs/anythingllm:latest` for `tt launch` |
| **us.i.posthog.com** | Telemetry, only if you opted in ([TELEMETRY.md](/TELEMETRY.md)) |

The global `--offline` flag makes every command that needs the network exit 8
(`OFFLINE`) instead of trying.

### Hugging Face token

Serving a catalog model through tt-inference-server **requires** a Hugging Face token:
the server fetches weights and tokenizers from the Hub and would otherwise prompt on a
TTY you cannot see. `tt serve` checks up front and exits 9 without one. The token is
resolved from `HF_TOKEN`, then the file at `HF_TOKEN_PATH`, then `<HF_HOME>/token`
(written by `hf auth login`). Run `hf auth login` once, or `export HF_TOKEN=hf_…`.

The studio and tt-model paths pass a token through if one exists but do not require
it. Gated models (most Llama and Qwen repos) also require you to accept the license
on the model's Hub page before `tt model pull` can download it.

## Disk space

Rough sizes from the code and upstream pins; plan for tens to hundreds of gigabytes
if you serve several models.

| Item | Size |
|---|---|
| `tt` and its dependencies | tens of MB |
| tt-smi / tt-flash / tt-model venvs | a few hundred MB total (uv may also download a managed Python 3.10 / 3.12 if none is installed) |
| tt-inference-server checkout | ~180 MB, plus the per-workflow venvs its `run.py` creates |
| TT-Studio checkout | ~200 MB, plus its own `.tt_studio_run_venv` |
| Model container images | several GB each; the release spec shares 41 images across 67 models |
| Model weights | from ~12 GB (Whisper distil-large-v3) to 50 GB and more for a large LLM; cached under `HF_HOME` (default `~/.cache/huggingface`) or `paths.hf_model_cache_directory` |
| `tt launch` web clients | "a few GB" per container image |

A cold first serve also JIT-compiles kernels (around ten minutes); the kernel cache is
bind-mounted to the host so that cost is paid once per model.

## `tt launch` clients

`tt launch` never installs a client; it finds one and points it at the model you are
serving. Lookup order is `TT_TOOL_BIN_<ID>`, then `tools.override.<id>` in the config,
then PATH, then one retry with PATH from a freshly sourced `~/.bashrc` / `~/.zshrc`.
A missing client exits 4 with the install hint below.

Every client needs an OpenAI-compatible server already answering `/v1/models`,
by default at `http://127.0.0.1:20000/v1` (`tt serve` puts one there). The terminal
clients additionally need a model that supports tool calling.

| Client | Needs | Install |
|---|---|---|
| `openwebui` | docker or podman; host port 3000 free | pulled automatically from `ghcr.io/open-webui/open-webui:main` |
| `anythingllm` | docker or podman; host port 3000 free (mapped to the container's 3001); runs with `--cap-add SYS_ADMIN` | pulled automatically from `mintplexlabs/anythingllm:latest` |
| `opencode` | `opencode` on PATH | https://opencode.ai/docs/ |
| `pi` | `pi` on PATH | https://github.com/earendil-works/pi |
| `aider` | `aider` on PATH | https://aider.chat/docs/install.html |
| `qwencode` | `qwen` on PATH; Node.js 22+ | `npm install -g @qwen-code/qwen-code` |
| `hermes` | `hermes` on PATH | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` |

## Interactive prompts and automation

Some commands need a terminal to confirm, and exit 2 (`USAGE`) when run from a script
without `--yes`:

- `tt device reset`
- `tt update`, when a firmware flash or device reset may happen
- `tt self update`
- `tt serve` of a tt-model bundle whose manifest is not verified

The one-time telemetry consent prompt is skipped when there is no TTY. Scripts can also
set `TT_TELEMETRY_DISABLED=1` for the run.

## Checklist for a fresh machine

1. Install a supported Linux distro and seat the card; set PCIe AER to "OS First" in the BIOS.
2. `sudo apt-get install -y python3 python3-venv python3-pip git curl ca-certificates`
   (Ubuntu/Debian) or `sudo dnf install -y python3 python3-pip git curl` (Fedora).
3. Install uv, then `uv tool install tenstorrent` and `uv tool update-shell`.
4. `tt update` (enter your sudo password when install.sh asks), then **reboot** if this
   is the first time the driver was installed.
5. `tt device status` should list your card. If it exits 3, the driver is not loaded.
6. Add yourself to the `docker` group if `tt update` just installed Docker, and log out
   and back in.
7. `hf auth login` (or `export HF_TOKEN=…`) before your first `tt serve`.
