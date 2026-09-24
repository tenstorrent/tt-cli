



# The serve progress contract

`tt serve` renders one checklist, whichever backend is serving. This is what a
backend has to tell it to make that possible.

## The split

A serve has two halves, and they are owned by different things.


| Half                                                                   | Owner             | Share of the wall clock |
| ---------------------------------------------------------------------- | ----------------- | ----------------------- |
| **Preparation** — bundle, image, weights, container start              | the backend       | seconds to minutes      |
| **Container boot** — engine, device, weights, KV cache, warmup, server | *neither backend* | usually ~95%            |


The boot half is written by the engine, not by the tool that launched it, and
both backends launch the same stack (vLLM/a media inference engine on tt-metal). 
So tt reads the container's log itself — `docker logs --follow` — and classifies it with one tracker (`backends/serving/progress/phases.py`). Nothing a backend says about the boot is used.
That is what makes the two paths identical rather than merely similar: it is the
same code over the same bytes.

A backend therefore only has to answer two questions: the steps it took to prepare, and which container it started.

## tt-inference-server

Answers both in prose, which tt scrapes with two pinned regexes (see
`backends/serving/preparation.py`). `tt serve` also sets
`TT_SERVER_BOOT_ATTEMPTS=1` so run.py returns once the container is up instead
of waiting on the boot itself.

## tt-model

Answers structurally, when asked. `tt serve` runs it with `--detach` (so it does
not watch the boot either) and `TT_MODEL_PROGRESS=ndjson`.

### The events

One JSON object per line, on stderr. Unknown `event` values and unknown fields
are ignored, so the stream can grow without a flag day.

```jsonc
// a step starting, finishing, or failing. `key` is stable; `label` is only
// used for a key tt does not know, so renaming one cannot change the UI.
{"event":"step","key":"weights","state":"start","label":"fetching weights"}
{"event":"step","key":"weights","state":"done","detail":"16.0 GB"}
{"event":"step","key":"image","state":"fail","detail":"manifest unknown"}

// a `done` on its own is fine — "this was already true" draws as a settled row.
{"event":"step","key":"host","state":"done","detail":"docker 28.1 · hugepages"}

// progress for whichever step is active. unit is "bytes" or "count".
{"event":"progress","done":3221225472,"total":16000000000,"unit":"bytes"}

// the handoff. everything after this is read from the container's own log.
{"event":"container","id":"tt-model-qwen3-p300x2","endpoint":"http://127.0.0.1:20000/v1"}

// optional: a diagnosis, in diagnose_boot()'s existing shape.
{"event":"error","cause":"…","detail":"…","evidence":"…","actions":["…"]}
```

`key` should reuse these where they mean the same thing, so both backends draw
the same row: `host`, `image`, `weights`, `container`. For those, tt supplies
the wording and any `label` is ignored — that is what stops the two paths
drifting apart on a rename.

Anything else (e.g. `bundle`) is drawn with its own `label`: the one on `start`
while it runs, and the one on `done` once it has finished, so a step can read
"resolving the bundle" and then "bundle resolved". A `done` with no `label`
reuses the `start`'s.

### Without it

Any tt-model that predates the variable still works. `tt serve` runs it with
`--verbose` (a *global* option, so it precedes the subcommand and is never swept
into serve's passthrough to vLLM), which stops it capturing each step's output
into a buffer it then discards, and its own rows — `<label>…` then
`✓ <label>  <detail>  <duration>` — are read as steps. That is prose, and
parsing it is a stopgap; the failure mode is a row that reads oddly, never a
broken serve, and events win wherever both arrive.

One thing that fallback cannot recover is **byte progress**. tt-model computes
it and routes it to an activity row that is "TTY only; a no-op when piped"
(`console._Activity`), so through a pipe those bytes do not exist at all — a
50 GB bundle shows a spinner and a clock and nothing else. That is the single
biggest reason to emit `progress` events: the number is already there.

A canonical stream lives at `tests/fakes/data/tt-model-progress.ndjson` and is
what the tests assert against.