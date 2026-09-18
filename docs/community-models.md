# Running and Packaging Community Models with tt-cli and tt-model-manager

This document explains how `tt` (the Tenstorrent CLI) and [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) work together: how to discover and serve models through one interface, and how to package a model bring-up so anyone else can serve it with the same two commands you use. It covers the full loop — consume, serve, package, publish — and how the pieces connect underneath.

**Who this is for:** anyone with Tenstorrent hardware (including TT-QuietBox 2 and p150 cards) who wants to run modern models like Laguna-2.1, Muse-Glimmer-30B, Ornith-1.0-35B, or Qwen3.8-Flash-Next; anyone who has lost an afternoon in low-level libraries to run a simple model; and anyone who has done a model bring-up and lacked a way to share it with the community.

## How the two tools relate

`tt` is the single entry point to the Tenstorrent software stack. It either implements a function natively or delegates to an existing tool (tt-smi, tt-installer, tt-model-manager, etc.) behind a stable interface. For model work, that means:

- **`tt model` / `tt serve`** are the consumer surface. They discover, pull, and serve both catalog models (via tt-inference-server) and community bundles (via tt-model-manager). If a model exists - official or community - this is how you run it.
- **`tt-model`** (the tt-model-manager CLI) is the author surface. It packages a bring-up into a self-contained bundle on the Hugging Face Hub, and publishes it to the community catalog. tt-cli pins tt-model-manager into an isolated per-tool venv under `~/.local/share/tenstorrent/` and calls it on your behalf when you pull or serve a community bundle.

The result is a symmetric contract: an author runs `tt-model package` and `tt-model push` once; every consumer after that runs `tt model pull` and `tt serve`. Your afternoon of bring-up pain becomes someone else's ninety-second download.

## Prerequisites

You need a machine on a tested software stack. If you haven't set up yet:

```bash
uv tool install tenstorrent   # or pipx / pip in an isolated venv — see README.md
tt update
tt device status
```

`tt update` converges your machine onto a **golden version set** — driver, firmware, and stack versions validated *together* in CI and published as `golden.json` by [tt-sw-manifest](https://github.com/tenstorrent/tt-sw-manifest). This is what removes the "are my driver, firmware, and tt-metal versions actually compatible?" question: it's answered before the release ships. `tt device status` confirms every detected card, driver state, firmware, temperature, and utilization (if you have tt-smi muscle memory, this is that — one front door, nothing discarded). Full installation options are in [DEVELOPERS.md](DEVELOPERS.md); this document assumes you're past that point.

## Running catalog models

To see what your machine can run:

```bash
tt model list
```

The CLI knows your hardware: the list is backed by a generated support list (built from tt-inference-server's `release_model_spec.json` at the pinned server version, plus tt's own record of known-not-working board combinations) and filtered by your detected device configuration. A model known to fail on your board is hidden, and `tt model info` says why. Use `--all` to see every model on every device, `--hw <device>` to filter for a specific configuration without touching the hardware, `--cached` and `--type` to narrow further, and `--community` for tt-model-manager bundles (covered below). Model names are the spec's short ids (`Llama-3.1-8B-Instruct`); a HuggingFace repo id works as an alias anywhere a name is accepted.

Before committing to a large download, inspect the model:

```bash
tt model info Qwen3.6-27B    # architecture, engines, per-device support, requirements, cache state
tt model pull Qwen3.6-27B
tt serve Qwen3.6-27B         # OpenAI-compatible endpoint
```

`tt serve` is the milestone that matters: an inference endpoint on your own silicon, speaking the API your existing tooling already speaks. Point aider at it, point an agent framework at it, point `curl` at it. Anything after `--` is passed through to the underlying server (`tt serve NAME -- --port 8080`), and `tt model stop NAME` shuts it down.

`tt model pull` detects what it's fetching — a catalog model's weights, a tt-model bundle, or an ordinary HuggingFace repo's weights (useful to pre-warm the cache before serving, with a warning that weights alone don't make a model servable). `--bundle` / `--weights-only` override the detection; `--offline` works from local caches. Weights land in a single shared cache: `HF_HOME` (or `paths.hf_model_cache_directory` in tt config) is exported to tt-model and passed to tt-inference-server, so every tool reads and writes the same weights.

The launch lineup is genuinely current: Qwen 3.8 Flash Next, Qwen3-Coder-30B, Muse Glimmer 30B, poolside's Laguna XS 2.1, Ornith 1.0 35B, and NVIDIA Nemotron 3.5 Lightning.

### Connecting a client with `tt launch`

`tt serve` gives you the endpoint; `tt launch` points a client at it. tt discovers what's running by asking the server itself (`GET /v1/models`) and configures the client's endpoint for you:

```bash
tt launch list                         # what can I connect, and is it usable now?
tt serve Qwen3.6-27B --port 8000
tt launch openwebui --web-port 3080    # in another terminal: pull and run its container, after you confirm
tt launch stop openwebui               # stop it, keeping its data
```

## Running community bundles

Model bring-up on new silicon has always had an awkward afterlife. Someone does the hard part — custom kernels, a patched tt-metal tree, the right vLLM incantation — and the work ends up *done* but not *transferable*. It survives as a branch, a HANDOFF.md, a good dev.to post. The next person can read about the journey; they can't skip it.

tt-model-manager changes what a finished bring-up *is*: a self-contained bundle published as a HuggingFace repo, and tt-cli surfaces it in the same commands you already use:

```bash
tt model list --community            # bundles published via tt-model-manager
tt model info you/mymodel            # manifest + compatibility verdict (via tt-model), or the catalog row
tt model pull you/mymodel
tt serve you/mymodel                 # served via tt-model-manager
tt serve you/mymodel --port 8080  # unrecognized args pass through to the bundle's engine
```

What arrives is not a description of the author's machine — it's the engine the author actually built. Depending on the packaging path (see below), the bundle carries the author's built tt-nn wheel with their custom kernels compiled in, the vLLM plugin, and the model code, plus an installer that rebuilds the environment in a fresh per-model venv on your box — or an OCI image with all of that baked in. Bundles are served by engines included in the repos themselves (`vllm-plugin`, `tt-dit-server`, etc.). After the pull, everything lives under one directory; only `pull` touches the network.

**Weights stay a pointer, never embedded.** A bundle's `weights:` field is an HF repo id (optionally pinned to a revision); weights are downloaded into your own HF cache under your own token. This is why a ~2 GB bundle can front a 57 GB model, why weights are shared across bundles and catalog models alike, and why the same bundle works for anyone regardless of which weights revision they're entitled to.

### How compatibility is handled

Compatibility is **checked, not guaranteed**. When you pull or serve a bundle, tt-model checks the two facts that matter: the **architecture** its binaries were built for (a mismatch fails hard, on purpose — there's no forcing a Wormhole build onto Blackhole) and whether your box has **enough chips** for the chosen profile (a warning you can force past). Wheel interpreter/platform tags are checked separately at install time on the self-contained path.

## Packaging your own bring-up

This is the other half of the loop: turning "it works on my box" into a repo id anyone can serve. Authoring happens through the `tt-model` CLI (tt-model-manager) directly.

### Container packages (v5.1)

The container path is the most robust and is fully documented in tt-model-manager's [container_packages.md](https://github.com/tenstorrent/tt-model-manager/blob/main/docs/container_packages.md); this section covers the flow and how it connects to tt-cli.

The entire authoring interface is one YAML file — `tt-model.yaml`, ideally committed next to the model in your tt-metal fork so the serving recipe is reviewed in the same PR as the model code. It declares the weights pointer, the serving stack (`kind`: `vllm-plugin`, `vllm-fork`, or `tt-dit-server`), the target `arch`, an explicit allowlist of exactly the code that ships (`source.code` — under-shipping is an error, never a skip), serve defaults, and optional named serve profiles. You don't have to write it from scratch: the repo ships a Claude Code skill (`/tt-model-yaml models/demos/blackhole/my_model`) that reads a model directory, works out its import closure and serve recipe, and interviews you for what the directory can't tell it. Validation is front-loaded — everything knowable without hardware (arch, mesh vs chip-count cross-check, that every listed path exists) is checked at load time, because the alternative is finding out ten minutes into a build.

The authoring flow is three commands:

```bash
tt-model package --container tt-model.yaml                # build the image, stage the repo dir
tt-model serve build/my-model/tt_kernel_manifest.json      # prove it locally before shipping
tt-model push build/my-model --private                     # publish to the Hub
```

Two properties make the result trustworthy. First, **the image verifies itself at build time**: imports resolve, torch matches tt-metal's pin, and every registered model is actually resolved inside the finished image — on the author's machine, not the consumer's. Second, **provenance is pinned**: your authored YAML may say `ref: main`, but the published manifest pins every floating ref (tt-metal tree, plugin, weights revision) to a commit and records it in a `built:` block. A plugin that moved under a validated model is exactly the failure this guards against.

One image serves *all* of a model's profiles — kernels JIT-compile against whatever mesh is opened at launch, so a device target (`p150x2` vs `p150x4`) and a deployment shape (latency vs capacity) are both just launch arguments, not separate builds.

### Publishing to the community catalog

Catalog listing is a separate, explicit opt-in on top of pushing:

```bash
tt-model push build/my-model --public --publish   # upload + list in one step
tt-model publish   you/my-model                    # list one pushed earlier
tt-model unpublish you/my-model                    # delist (repo untouched)
```

This is not a submission queue. You push to *your* HF account under *your* governance; the catalog is a static index that stores nothing — every entry points back at your repo. Publishing a model for TT hardware needs no one's permission, including Tenstorrent's. Once published, it appears in `tt model list --community` for everyone, and is servable with `tt serve you/my-model`.

### What the consumer sees

On any box with Docker and a card, a container bundle serves with one command (via `tt serve you/my-model`, or `tt-model serve you/my-model` directly). Auto-pull fetches the image and weights, then `serve` watches the boot as a checklist of landmarks parsed from the container log — host ready, image loaded, engine initialised, device opened, weights loaded, KV cache configured, warm-up — never the raw log itself. A boot that fails marks the step it died in and renders a diagnosis (cause, one line of evidence, what to try) instead of dumping the log. A cold first boot JIT-compiles kernels (~10 min); the kernel cache is bind-mounted to the host so that cost is paid once.

Pull and serve can also be split — `tt-model pull` moves bytes only and needs no card, so it can run on a build host. Note the two entry points treat weights differently:

| | image | weights |
| --- | --- | --- |
| `tt-model pull org/name` | yes | **no** (fetched at first load, or add `--with-weights`) |
| `tt-model serve org/name` (nothing installed) | yes | yes |

Around them: `tt-model profiles` lists a bundle's serve profiles, `tt-model serve --profile <name>` picks a non-default one, `tt-model logs -f` follows a boot, `tt-model stop` sends a clean SIGTERM (a SIGKILL would leave the devices needing `tt-smi -r`), and `tt-model rm` removes a pulled package (`--keep-cache` preserves the JIT/weight caches for a fast re-pull).

## When something breaks

It will — this is real hardware on a fast-moving stack, and the tooling is designed for the bad day rather than pretending it away. Errors are written for someone mid-task: what failed, why (when known), and what to do next, with the raw stack trace linked rather than dumped. Exit codes are a documented contract (see [DEVELOPERS.md](DEVELOPERS.md#exit-codes)) — `3 NO_DEVICES`, `4 TOOL_MISSING` (run `tt update`), `5 TOOL_FAILED`, `8 OFFLINE`, and so on — so scripts can have a bad day gracefully too.

Reporting stops being homework:

```bash
tt report issue
```

This prompts for what matters, works out *which repo* the issue belongs to, and opens a prefilled GitHub issue with environment details and repo-specific debug output auto-collected (`--no-browser` to just print the URL). The "please run these six commands and paste the output" round-trip — the one that adds two days to every bug — is gone. `tt report feedback` handles the fuzzier stuff, routed to the product team.

## Scope and limitations

In the spirit of every good bring-up log, the caveats up front rather than discovered later:

- **tt-model-manager is still in beta.** No support guarantees; the bundle format can change. tt-cli itself is also beta software — expect breaking changes as features land.
- Tenstorrent does not maintain community models brought up and posted via Hugging Face; the tooling makes them accessible.
- Compatibility is **checked, not guaranteed**. Architecture mismatch fails hard on purpose; other mismatches warn and can be forced.

What *is* the commitment: open repos, one tool from discovery to serving, and a contribution path where community bring-up work compounds instead of evaporating. A bring-up used to end at "it works on my box." Now it ends at a repo id anyone can serve, accessible directly from tt-cli.

## Telemetry

Telemetry is **opt-in**: nothing is collected or sent unless you say so. If you opt in, `tt` records command names, exit codes, coarse OS facts, and argument values only when they match a known list — never free-form arguments, paths, or secrets. This is the data used to see which commands get used and where installs fail. Change your mind any time:

```bash
tt config set telemetry.enabled false
```

See [TELEMETRY.md](../TELEMETRY.md) for exactly what is collected and every switch that controls it.
