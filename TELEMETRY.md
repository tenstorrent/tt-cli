# Telemetry

`tt` can collect anonymous usage telemetry to help improve the CLI. It is **opt-in**:
nothing is collected, stored, or sent unless you explicitly say yes.

## Opting in and out

On the first interactive run, `tt` asks once:

```
Enable anonymous usage telemetry? [y/n]
```

The answer must be an explicit yes or no — a bare Enter re-asks rather than picking
one for you (Ctrl-C skips the question for now; it comes back next interactive run).

- **Yes** is persisted as `telemetry.enabled = true` in your config file
  (`tt config path`), so the consent is visible, commented, and revocable like any
  other setting.
- **No** keeps telemetry off, and you are never asked again.
- The question is only ever asked at a real terminal — never in scripts, pipelines,
  under `--json`/`--quiet`, `--offline`, or on CI, and never when no upload endpoint or
  key is configured (nothing could be sent, so there is nothing to consent to).
  Non-interactive installs simply stay opted out.

Change your mind any time:

```console
$ tt config set telemetry.enabled true    # opt in
$ tt config set telemetry.enabled false   # opt out
```

Opting out (or setting `DO_NOT_TRACK`) also **deletes** anything collected but not yet
uploaded, so nothing recorded before you opted out is ever sent.

## What is collected

If you opt in, `tt` records one [OpenTelemetry](https://opentelemetry.io/) span per
command:

- the command path (e.g. `tt device status`);
- **which** options were set — their names only, never their values;
- an argument value **only when it is already on a known list** the CLI holds in code:
  catalog model names, model-type/hardware filter values, device configuration
  names, config key names, and strict semantic versions. Anything else — a typo, a
  filesystem path, a private model name — is dropped entirely, never truncated or
  hashed;
- for `tt device status|info|reset`, **how many** device indices were given — the count,
  not the indices;
- the exit-code category (e.g. `OK`, `NO_DEVICES`) and command duration;
- whether the run happened on CI (a boolean derived from the *names* of well-known CI
  environment variables — never their values);
- coarse host facts: OS type (e.g. `Linux`), CPU architecture, Python version,
  tt version;
- a random per-install UUID. It is generated, not derived from hardware or accounts,
  so it identifies an install, not a person, and can be rotated by deleting
  `telemetry.toml` in the data directory.

Never collected: free-form argument or option values, file paths, config *values*
(`tt config set` records the key name only — values can hold secrets), environment
variable values, hostnames, usernames, or error messages/stack traces.

## See exactly what would be sent

You don't have to take the list above on faith. Set

```console
$ TT_TELEMETRY_LOG_FILE=/tmp/spans.jsonl tt device status
```

and read the file: every span is written there in the standard OTLP/JSON format. This
works **without opting in** and uploads nothing, so you can inspect the data before
deciding. Developers can also point `tt` at `scripts/otlp_sink.py`, a local OTLP
receiver that pretty-prints whatever arrives.

## Every switch that controls telemetry

| Control | Scope | Effect |
|---|---|---|
| `telemetry.enabled` (config) | durable | The opt-in itself. `false` (default) collects nothing; flipping to `false` also deletes any unsent data |
| `DO_NOT_TRACK` | durable | Cross-tool [convention](https://consoledonottrack.com/); same as opting out, including deleting unsent data |
| `TT_TELEMETRY_DISABLED=1` | this run | Kill switch: collects and sends nothing, keeps unsent data for a later run |
| `--offline` | this run | No network at all, telemetry included; keeps unsent data |
| `TT_TELEMETRY_LOG_FILE` | this run | Also write every span to this file (OTLP/JSON, local-only; works without opting in) |
| `telemetry.endpoint` / `TT_TELEMETRY_ENDPOINT` | config / this run | OTLP/HTTP traces endpoint. Empty = inert |
| `telemetry.posthog_project_key` / `TT_TELEMETRY_POSTHOG_KEY` | config / this run | Write-only ingest key. Empty = inert |
| `telemetry.flush_mode` / `TT_TELEMETRY_FLUSH_MODE` | config / this run | `async` (default) or `sync` — see below |

## Not telemetry: the update check

Separately from all of the above, `tt` looks up its own newest release on PyPI about
once a day (see "Keeping tt up to date" in docs/DEVELOPERS.md). That request carries `tt`'s
version in the User-Agent and nothing else, is never correlated with telemetry, and has
its own switches: `tt config set update.check false`, `TT_NO_UPDATE_CHECK=1`,
`--offline`. Opting out of telemetry does not turn it off, and vice versa.

## How delivery works

**No command ever waits on the network.** By default (`flush_mode = "async"`) each span
is appended to a local spool (`$TT_DATA_DIR/telemetry/spool.jsonl`, in the standard
[OTLP file](https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/) JSON Lines
format) and a detached background process uploads whole batches — roughly twice a day
for a typical user. `flush_mode = "sync"` exports in-process instead: slower per
command, but spans arrive immediately, which is what you want when developing against a
local collector. CI runs use sync automatically, since a detached uploader would be
reaped with the build.

`tt self send-telemetry` forces an upload right now and prints the result, and
`tt -v <any command>` reports what delivery decided to do — including whether a previous
upload failed and is waiting to be retried.

Uploads honour the standard `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`
environment variables. Note there is no implicit localhost exemption: when pointing
telemetry at a collector on the same host, add it to `NO_PROXY`.

The transport is plain OTLP/HTTP (the default endpoint is PostHog, authenticated with a
write-only ingest key), so the endpoint can be repointed at any OTLP collector — your
own included.
