# Serve progress contract

`tt serve` shows one boot checklist for both serving backends, tt-inference-server and tt-model-manager. This document describes how tt builds that checklist and what each backend must report for it.

## Overview

A watched serve has two stages.

| Stage | Read from | Steps |
|---|---|---|
| Preparation | The backend's output | Host checks, weights download, image pull, container start |
| Container boot | The container's log | Engine start, device open, weights load, KV cache, warmup, API server |

Either stage can be long. Preparation takes seconds when everything is cached, but a cold weights download or image pull can take minutes to hours. The boot usually takes minutes, and longer when the container fetches its own weights.

Preparation differs between backends, so each backend reports it through an adapter (`backends/serving/preparation.py`). The boot runs the same stack for both backends (vLLM or the media server on tt-metal), so tt reads it directly with `docker logs --follow` and classifies it with one tracker (`backends/serving/progress/phases.py`). Nothing a backend says about the boot is used, so both paths render the boot from the same code over the same log.

A backend has to report two things:

1. The steps it takes during preparation.
2. The container it started.

## Watch lifecycle

1. **Preparation.** tt runs the backend with its output piped and converts each line into checklist rows.
2. **Handoff.** Once the backend names a container and `docker inspect` finds it, tt closes the preparation rows and follows the container's log. tt waits for `inspect` because tt-inference-server prints the container name before it creates the container.
3. **Boot.** Container log lines drive the boot rows. The template is chosen from the model's engine: `VLLM_PHASES` for vLLM, `MEDIA_PHASES` for media and forge models.
4. **Ready.** The endpoint decides readiness, not the log. A serve is ready when `/v1/models` answers, `/health` returns 200 (or 404), and the container is running and publishes the probed port. The last check stops another server on the same port from passing as this one.
5. **Failure.** The serve fails if the backend exits non-zero, the container's log stream ends before ready, or the log reports a fatal device error. On a fatal device error tt stops the container, which would otherwise keep the device held. The wait is bounded by `TT_SERVE_READY_TIMEOUT` (default 3600 seconds), and on a timeout the container is left running.

Every line from both stages is written to `serve-<model>-<timestamp>.log` in tt's logs directory, and `--verbose` also prints them as they arrive. Ctrl-C stops watching but leaves the container running.

## tt-inference-server

run.py reports in prose, so the adapter (`RunPyPreparation`) matches pinned patterns.

| Signal | Matched line |
|---|---|
| Preparation rows | `HOST_PHASES` patterns |
| Container name | `--name tt-inference-server-…` |
| Container ID | `Created Docker container ID: …` |
| Weights repo | `Downloading model to host HF cache: …` |

The container name comes from the `docker run` command run.py echoes, and the ID is the fallback. If run.py changes these lines, the handoff falls back to "no container to follow": the serve still completes, but tt does not report ready. Re-check these patterns when the tt-inference-server pin changes.

`tt serve` sets two environment variables for run.py, each only if unset.

| Variable | Value | Effect |
|---|---|---|
| `TT_SERVER_BOOT_ATTEMPTS` | `1` | run.py returns once the container starts |
| `PYTHONUNBUFFERED` | `1` | Output reaches tt as it is written |

With more than one boot attempt, run.py polls `/health` itself and tears the container down to retry if the wait runs out. tt already waits on the same endpoint, so a single attempt leaves one process in charge of the boot.

## tt-model-manager

`tt serve <namespace>/<name>` runs:

```
tt-model --verbose serve <repo_id> --detach [options]
```

| Variable | Value | Effect |
|---|---|---|
| `TT_MODEL_PROGRESS` | `ndjson` | Requests progress events |
| `TT_MODEL_NO_PIN` | `1` | Disables tt-model's live view |
| `PYTHONUNBUFFERED` | `1` | Output reaches tt as it is written |

`--detach` stops tt-model from also waiting on the boot, so only one process decides when the serve is ready or has failed. `--verbose` is a global option placed before the subcommand, so it is never forwarded to vLLM.

If the user passes `--detach` or `--print`, tt does not watch the serve. It runs tt-model with the terminal attached and without the variables above.

### Events

tt-model writes one JSON object per line, and tt reads stdout and stderr as one stream. Lines that are not events, unknown `event` values and unknown fields are ignored, so the format can be extended without a coordinated release.

| Event | Fields | Meaning |
|---|---|---|
| `step` | `key`, `state`, `label`, `detail` | A step starts or settles |
| `progress` | `done`, `total`, `unit` | Progress for the active step |
| `container` | `id`, `endpoint` | The container to follow |
| `error` | `cause`, `detail`, `evidence`, `actions` | A diagnosis for a failed boot |

- `step.state` is `start`, `done` or `fail`. A `done` with no prior `start` renders as a step that was already complete.
- `progress.unit` is `bytes` or `count`. The event is ignored when `total` is not positive.
- `container.endpoint` is optional. When present, its port is the one tt probes.
- `error` becomes the error message if the boot fails.

```json
{"event":"step","key":"bundle","state":"start","label":"resolving the bundle"}
{"event":"step","key":"bundle","state":"done","label":"bundle resolved","detail":"qwen3-coder-30b-a3b @ p300x2"}
{"event":"step","key":"weights","state":"start"}
{"event":"progress","done":3221225472,"total":16000000000,"unit":"bytes"}
{"event":"step","key":"weights","state":"done"}
{"event":"container","id":"tt-model-qwen3-coder-30b-a3b-p300x2","endpoint":"http://127.0.0.1:20000/v1"}
```

### Step keys and labels

The keys `host`, `weights`, `image` and `container` are shared with tt-inference-server. For these keys tt supplies the wording and ignores `label`, so both backends show identical rows.

Any other key uses the event's own labels: the `start` label while the step runs, and the `done` label once it settles. A `done` without a label reuses the `start` label.

### Without events

A tt-model that predates `TT_MODEL_PROGRESS` still works. With `--verbose` it prints each step as `<label>…` and settles it as `✓ <label>  <detail>`, and the adapter reads those lines as rows. Once any event arrives these lines are ignored, so the two sources never double a row. A misread line only makes a row read oddly; it does not affect the serve.

Without a `container` event, tt finds the container after tt-model exits, by the `org.tenstorrent.tt-model` label matching the bundle name, and reads its published port with `docker port`.

## Weights progress

Apart from tt-model's `progress` events, neither backend reports download bytes through a pipe, so tt measures them.

| Download | Measured by | Shows |
|---|---|---|
| Host | Size of the repo in the Hugging Face cache | Bytes and total |
| Container (media server) | `du` inside the container | Bytes only |

For host downloads, tt measures every 1.5 seconds and asks the Hub once for the total size; without a total it shows bytes only. The repo comes from run.py's download line or, for tt-model without events, its `weights <repo>@<revision>` row.

For container downloads, tt starts measuring when the media server logs `Downloading weights for model:` or `Loading HuggingFace model:`, and reads the repo under the container's `HF_HOME`. It shows no total because the Hub total includes files the server does not download.

## Tests

`tests/fakes/data/tt-model-progress.ndjson` is the reference event stream. The fake tt-model replays it and the serve tests assert against the result.
