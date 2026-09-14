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

If you opt in, `tt` records **one event per command**, named `tt_command`, carrying:

- the command path (e.g. `tt device status`);
- **which** options were set — their names only, never their values;
- an argument value **only when it is already on a known list** the CLI holds in code:
  catalog model names, model-type/hardware filter values, device configuration
  names, config key names, and strict semantic versions. Anything else — a typo, a
  filesystem path, a private model name — is dropped entirely, never truncated or
  hashed;
- for `tt device status|info|reset`, **how many** device indices were given — the count,
  not the indices;
- the outcome: the exit-code category (e.g. `OK`, `NO_DEVICES`), how long the command
  took, and — when it failed — one bounded fact about *why*: a short fixed label chosen
  by the code that raised the error (e.g. `model.unknown`, `tool.missing`), or, for an
  unexpected crash, the exception's class name (e.g. `FileNotFoundError`). Never the
  error message or a stack trace, which can contain paths;
- whether the run happened on CI (a boolean derived from the *names* of well-known CI
  environment variables — never their values);
- coarse host facts: OS type (e.g. `Linux`), CPU architecture, Python version,
  tt version;
- a random per-install UUID. It is generated, not derived from hardware or accounts,
  so it identifies an install, not a person, and can be rotated by deleting
  `telemetry.toml` in the data directory. It is the event's `distinct_id`, and the
  same coarse host facts are kept on the resulting PostHog profile — a profile of the
  *install*, which is what makes "how many installs ran `tt serve` this month, and did
  they come back" answerable.

Never collected: free-form argument or option values, file paths, config *values*
(`tt config set` records the key name only — values can hold secrets), environment
variable values, hostnames, usernames, error messages, or stack traces.

**Location.** PostHog derives a country/region from the address the upload comes from,
and the project is configured to **discard the IP address itself** at ingest, so no IP
is stored with the events. Country-level location is the only geographic fact kept.

The complete list of property names an event may carry is a closed set in the code
(`EVENT_PROPERTY_NAMES` in `src/tenstorrent/telemetry/attributes.py`), pinned by a
test: adding a property is a reviewed change, not something that can happen by accident.

## See exactly what would be sent

You don't have to take the list above on faith. Set

```console
$ TT_TELEMETRY_LOG_FILE=/tmp/tt-events.jsonl tt device status
```

and read the file: every event is written there as one JSON object per line — exactly
the object that would go into an upload. This works **without opting in** and uploads
nothing, so you can inspect the data before deciding. A typical line, pretty-printed:

```json
{
  "event": "tt_command",
  "uuid": "011745d5-a95c-4df5-ae24-32b851f33a2e",
  "distinct_id": "d2ef84fa-89bc-4e03-9f68-9f3db13f35e8",
  "timestamp": "2026-09-14T16:46:23.013190+00:00",
  "properties": {
    "command": "tt device status",
    "exit_code": 0,
    "exit_code_name": "OK",
    "duration_ms": 412,
    "tt_version": "1.0.1",
    "os_type": "Linux",
    "os_arch": "x86_64",
    "python_version": "3.12.12",
    "ci": false,
    "$lib": "tt-cli",
    "$lib_version": "1.0.1",
    "$set": {"tt_version": "1.0.1", "os_type": "Linux", "os_arch": "x86_64", "python_version": "3.12.12"},
    "$set_once": {"first_seen_version": "1.0.1", "first_seen_os_type": "Linux"}
  }
}
```

Developers can also point `tt` at `scripts/posthog_sink.py`, a local stand-in for the
PostHog endpoint that pretty-prints whatever arrives.

## Every switch that controls telemetry

| Control | Scope | Effect |
|---|---|---|
| `telemetry.enabled` (config) | durable | The opt-in itself. `false` (default) collects nothing; flipping to `false` also deletes any unsent data |
| `DO_NOT_TRACK` | durable | Cross-tool [convention](https://consoledonottrack.com/); same as opting out, including deleting unsent data |
| `TT_TELEMETRY_DISABLED=1` | this run | Kill switch: collects and sends nothing, keeps unsent data for a later run |
| `--offline` | this run | No network at all, telemetry included; keeps unsent data |
| `TT_TELEMETRY_LOG_FILE` | this run | Also write every event to this file (JSON lines, local-only; works without opting in) |
| `telemetry.endpoint` / `TT_TELEMETRY_ENDPOINT` | config / this run | PostHog batch capture URL (`https://us.i.posthog.com/batch/` by default; EU projects use `eu.i.posthog.com`). Empty = inert |
| `telemetry.posthog_project_key` / `TT_TELEMETRY_POSTHOG_KEY` | config / this run | Write-only ingest key. Empty = inert |
| `telemetry.flush_mode` / `TT_TELEMETRY_FLUSH_MODE` | config / this run | `async` (default) or `sync` — see below |

## Not telemetry: the update check

Separately from all of the above, `tt` looks up its own newest release on PyPI about
once a day (see "Keeping tt up to date" in docs/DEVELOPERS.md). That request carries `tt`'s
version in the User-Agent and nothing else, is never correlated with telemetry, and has
its own switches: `tt config set update.check false`, `TT_NO_UPDATE_CHECK=1`,
`--offline`. Opting out of telemetry does not turn it off, and vice versa.

## How delivery works

**No command ever waits on the network.** By default (`flush_mode = "async"`) each event
is appended to a local spool (`$TT_DATA_DIR/telemetry/events.jsonl`, one JSON object per
line) and a detached background process uploads whole batches — roughly twice a day for
a typical user. `flush_mode = "sync"` posts in-process instead: slower per command, but
events arrive immediately, which is what you want when developing against a local sink.
CI runs use sync automatically, since a detached uploader would be reaped with the build.

Each event carries the time it was *recorded* and its own random id, so a batch that is
uploaded hours later is dated correctly, and one that has to be retried after a lost
response is deduplicated rather than counted twice.

`tt self send-telemetry` forces an upload right now and prints the result, and
`tt -v <any command>` reports what delivery decided to do — including whether a previous
upload failed and is waiting to be retried.

Uploads honour the standard `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`
environment variables. Note there is no implicit localhost exemption: when pointing
telemetry at a sink on the same host, add it to `NO_PROXY`.

The transport is PostHog's [batch capture API](https://posthog.com/docs/api/capture):
one JSON POST authenticated with a write-only project key. The endpoint can be repointed
at PostHog's EU cloud or a self-hosted PostHog. Releases up to 1.0.1 sent OpenTelemetry
spans to PostHog's traces endpoint instead; a config file written by one of those still
works — the old endpoint URL is recognised and mapped to the new one — and any spans they
left spooled but unsent are deleted, not uploaded.
