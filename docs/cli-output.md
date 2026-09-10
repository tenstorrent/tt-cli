# tt CLI — terminal output design

How `tt` renders to the terminal. Read this before changing CLI output so the
experience stays consistent.

**Everything user-facing goes through `OutputManager` and `output.ui`.** No bare
`print()`, no hand-rolled ANSI, no `rich.Console()` constructed at a call site.

The design language is shared with TT-Studio's `python run.py`; the portable
distillation lives in `.claude/skills/cli-design/` (rules, plus a runnable
reference implementation).

---

## The rules that matter most

1. **Data on stdout, everything else on stderr.** This is a contract, not a
   preference — it is what keeps `tt --json … | jq` clean. `emit()` writes data;
   `status()`, `warn()`, `debug()`, and every `ui` method write status.
2. **`ui` is silent under `--json` and `--quiet`.** `Ui.enabled` is false in both,
   and every method becomes a no-op. Handles still work (timing, `.detail()`,
   `.fail()`) so call sites never need to branch on the mode.
3. **Motion requires a real TTY.** `Ui.live` gates every spinner, activity row,
   stepper, and phase rule. Piped output gets one collapsed line per step, no
   escape codes, and no elapsed suffix.
4. **Minimal by default; `-v` reveals everything.** Gate routine "done" output on
   `ui.show_detail()` (`verbose or not in_phase()`). Never gate a failure, a
   prompt, or an actionable warning.
5. **Keep progress, fold confirmations.** A live row earns its space. A second
   "✓ done" under a collapsed phase line does not.
6. **Raw tool output is evidence, not UI.** `Runner` captures it; `ui` presents a
   sentence we wrote. Nothing reaches the terminal because a subprocess happened
   to print it.
7. **One live display at a time.** A spinner and an activity row both own the
   cursor. `_ACTIVE_LIVE` enforces this; `ui.prompting()` suspends motion around
   `input()`/`getpass`/sudo, and `ui.handoff()` releases before a child takes the
   terminal.
8. **Never fake a percentage.** `progress_bar()` returns `""` for an unknown
   total. Pick a denominator you actually know and show a counter for the rest —
   `12/24 packages · 204 MB` beats an invented percent.

## The four zones

| Zone | Job | Never |
|---|---|---|
| Stepper | where am I in the whole run | per-tool detail |
| Phase body | milestones and notes, ~1 line each | raw child output |
| Activity row | proof it's alive, what's happening now | history |
| Cards | grouped state: ready, failure, interrupted | anything transient |

## Timings

Three clocks, all `time.monotonic()`, all formatted by `fmt_duration()`
(`420ms` / `5.0s` / `3m 34s`):

- **Per step** — appended to the collapsed line, but only past
  `ELAPSED_THRESHOLD_S` (0.8s) and never for a skipped step. A step the user never
  waited for stays clean.
- **Per phase** — on collapse: `✓ Phase 2/3 · Tools  4.1s`.
- **Per run** — the ready card's footer: `Ready in 3m 34s · 3 phases`.

Durations are **suppressed when piped** so CI output can't flap around the
threshold, and are carried in `--json` instead via `RunTimings.to_dict()`, which
makes install time diffable rather than something you eyeball.

## Phases

`ui.register_phases([...])` declares a **fixed** list. The count must never drift
with flags — that is what makes `k/N` trustworthy.

- A phase that doesn't apply is **skipped, not removed**: `ui.skip_phase(title, why)`.
  Say why out loud; a skip is a decision, not routine output.
- `ui.rename_phase(i, title)` exists for the case where the truth is genuinely
  unknowable until the phase is running (TT-Studio's Pull → Build). Anything
  decidable up front belongs in `register_phases()`; anything that simply doesn't
  apply is a skip.
- Prompts live **between** phases, never inside one.

## Failures: diagnose, don't dump

Printing the last N log lines is a worse version of the log file — no cause, no
next step, and it scrolls the useful part away. Instead:

1. Classify in a **pure** function (text in, dict out, unit-tested) →
   `{cause, detail, evidence, actions}`.
2. Render one `failure_card(...)`: cause in the title, one line of evidence, the
   **consequence** (is this fatal, or does the run continue?), then what to try.
3. Render it **after** the step collapses, never inside it.

`TTError(what=, why=, next_step=, details=)` is already this shape; prefer
enriching it over inventing a parallel path. `details["log_path"]` surfaces as
`Full output: …` in the error panel.

**Expected failures are notes, not errors.** One muted line with the accurate
cause and what happens instead: `○ Prebuilt image isn't published — building
locally`. Guessing the cause is worse than saying nothing.

**And check whether the behaviour is bad, not just the output.** Prettier
rendering of a self-inflicted error is the wrong fix; the best version of an error
message is the run that never needs it.

## Adding output

| You want | Use |
|---|---|
| a long operation as one line | `with ui.step("Installing tt-smi 3.0.30") as s:` |
| a version/count/size on that line | `s.detail("3.0.30")` |
| "nothing to do" | `s.skip("already up to date")` |
| a state worth explaining | `ui.note(...)` — states, not actions |
| a real milestone inside a phase | `ui.milestone(...)` |
| something the user must act on | `ui.alert(...)` — never folded |
| a long child process | `with ui.activity(label) as row:` + a pure parser |
| grouped end state | `ui.card(ready_panel(...))` |
| a failure | `ui.card(failure_card(...))` |
| a routine confirmation | `if ui.show_detail():` |

Labels are **actions** (`Installing tt-smi 3.0.30`); notes are **states**
(`tt-metal is managed outside tt`). Strip a trailing `…` from a label — the
spinner already says "in progress".

## Verifying changes

```bash
uv run pytest -q                                     # must stay green
python3 scripts/ui_demo.py                           # the whole output language, live
python3 scripts/ui_demo.py --fail                    # the failure paths
python3 scripts/ui_demo.py -v                        # folded detail returns
python3 scripts/ui_demo.py | cat -v | grep -c '\^\[' # non-TTY: expect 0
COLUMNS=40 python3 scripts/ui_demo.py                # and 80, 120
```

`tests/cli/test_cli_output.py` holds the guards: zero escape codes when piped, no
trailing whitespace, the step label appearing exactly once, and — under a real
PTY — the spinner advancing, the row being erased before each result, the stepper
filling in, and the cursor restored on exit.

Animated output is invisible to `CliRunner`, so anything with motion needs a PTY
test. And reproduce failures for real — occupy the port, unplug the network, point
at a nonexistent version. Mocked errors only teach a parser the wording you
imagined, not the wording the tool prints.
