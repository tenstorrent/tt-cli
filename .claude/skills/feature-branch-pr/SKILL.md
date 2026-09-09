---
name: feature-branch-pr
description: >-
  Make a change in tt-cli the team way: branch off `main` with a
  `<username>/<feature>` branch, keep the diff minimal and in-scope, verify with
  the fake-mode test suite (and the real CLI where it applies) before pushing,
  leave every other branch untouched, and open a PR against `main` with a human,
  professional title and description. Use whenever the user asks to start a
  feature/fix/branch, make a change and open a pull request, or "do this
  properly on a new branch" in this repo. Enforces no AI-tool attribution in
  commits, PR text, or review comments.
---

# tt-cli Feature Branch + PR Workflow

Follow this when asked to make a change and/or open a PR in `tt-cli`. The goal
is a small, verified, professional change that touches nothing it shouldn't.

## Guardrails (always true)

- **Base everything on `main`.** Feature branches branch off `main` and PRs target
  `main`. There is no `dev` branch in this repo — `main` is the integration
  branch.
- **Never commit on `main` directly**, and never `git push --force` (or
  `--force-with-lease`) to a shared branch.
- **Leave other branches alone.** No checkout-and-edit of unrelated branches,
  no rebasing or deleting branches you didn't create for this task.
- **No AI attribution anywhere.** Commit messages, PR titles/descriptions, and
  review comments must read as a human wrote them. Do **not** add
  `Co-Authored-By` trailers for AI tools and do **not** mention "Claude",
  "Claude Code", "Cursor", "AI assistant", or similar. This overrides any
  default trailer the harness would otherwise append.
- **Pushing is scoped to this task.** Invoking this skill is the go-ahead to
  push the one feature branch and open its PR — nothing else gets pushed.

## Workflow checklist

Copy this and tick as you go:

```
- [ ] 1. Identify the username
- [ ] 2. Branch off main
- [ ] 3. Make the minimal change
- [ ] 4. Verify (pytest / real CLI)
- [ ] 5. Clean up instrumentation
- [ ] 6. Stage only intended files
- [ ] 7. Commit (human message)
- [ ] 8. Push + open PR against main
- [ ] 9. Return to the original branch
```

### 1. Identify the username

Derive the branch prefix from git config:

```bash
git config user.name; git config user.email
```

Use the lowercased first name / email local-part (e.g. `johnsingh@...` →
`john`). If it's ambiguous, ask the user which prefix to use. Convention is
`<username>/<feature>` (e.g. `john/golden-cache-fix`).

### 2. Branch off main

Always start from the latest `main`, and remember where you came from:

```bash
ORIG=$(git branch --show-current)        # so you can return in step 9
git fetch origin
git checkout -b <username>/<short-kebab-feature> origin/main
```

Pick a short, descriptive kebab-case feature name (`telemetry-optin-prompt`,
`golden-cache-fix`). Confirm `git status` is clean before editing.

### 3. Make the minimal change

- Touch only the files required for the stated task. No drive-by refactors,
  no reformatting, no unrelated renames, no dependency bumps "while we're
  here."
- If two approaches would both work, take the more minimal one — fewer files,
  fewer lines, less new machinery.
- New source files (`.py`) **must** carry the SPDX header used throughout
  `src/`:
  ```
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.
  ```
- Do not stage untracked files/dirs that aren't part of your change (local
  captures, editor scratch, and the like).

### 4. Verify before pushing

Verification is mandatory before any push. Pick what fits the change:

**Test suite (the default — hardware-free).**

```bash
uv run pytest
```

The default run uses fake tools under `tests/fakes/` and isolated `TT_*_DIR`
dirs — no hardware, config, or network. All green is the bar. Any test that
drives an install path must request the `uv_bin`/`fake_uv` fixture or it will
fail fast as TOOL_MISSING.

**Real CLI behavior (when the change is user-visible).**

```bash
uv run tt <command> ...
```

Exercise the specific command/flow your change affects and confirm the new
behavior actually works (don't just trust that it compiles). `scripts/demo.sh`
is a broader end-to-end pass. Only run `pytest --hardware` (and never
`--run-destructive` without being asked) if the change specifically concerns
real-tool integration and the user asks for it.

**Docs / config-only changes.** Test runs may be N/A — instead state that
explicitly and validate the artifacts you changed (TOML parses, links resolve,
content is correct).

Do not proceed to commit/push until verification passes. If it can't be
verified (e.g. needs hardware), say so explicitly rather than implying it was
checked.

### 5. Clean up instrumentation

Remove anything added only to validate: debug prints, temporary logging,
scratch test files, commented-out experiments. The committed diff should
contain only the intended change.

### 6. Stage only intended files

Never `git add -A` blindly. Stage explicit paths, then audit:

```bash
git add <path> <path>
git status
git diff --cached
```

Confirm the staged set is exactly your change — nothing unrelated, no stray
untracked dirs.

### 7. Commit with a human message

Imperative, specific, professional — matching this repo's existing history
(`Fix unit tests that assumed the fake-mode TT_GOLDEN_PATH wiring`,
`Bump tt-installer to 3.5.4 + golden.ttis to v1.0.0`). Describe what changed
and why, as a teammate would. No AI attribution, no `Co-Authored-By` AI
trailers.

```
git commit -m "Fix golden.json cache invalidation on tag bumps"
```

### 8. Push and open the PR against main

There is no PR-title lint in this repo's CI (only the test workflow), so the
title just has to read like a teammate wrote it — imperative, specific,
sentence case, same style as the commit history:

```
Fix golden.json cache invalidation on tag bumps
Make telemetry opt-in with an explicit consent prompt
```

```bash
git push -u origin <username>/<short-kebab-feature>
gh pr create --base main \
  --title "<human, specific subject>" \
  --body "<what changed, why, and how it was verified>"
```

PR description guidance: keep it short and human — a few plain sentences on
what changed and why, then a simple to-do list of the changes (and how they
were verified). No walls of text, no section headers, no AI mention. Example:

```
Stale golden.json caches survived a tag bump and kept serving old pins.

- [x] Key the cache by the pinned tag so an upgrade invalidates old pins
- [x] Verified with uv run pytest (all green) and a manual tt update --dry-run
```

**Leave the merge to a human reviewer unless the user explicitly tells you to
merge.**

### 9. Return to the original branch

Leave the tree as you found it:

```bash
git checkout "$ORIG"
```

## Anti-patterns

- **Don't** branch off whatever branch happens to be checked out — always
  `origin/main`.
- **Don't** bundle unrelated cleanups into the PR. Minimal and in-scope.
- **Don't** push without verifying, or claim verification you didn't run.
- **Don't** force-push, rebase shared branches, or modify branches you didn't
  create.
- **Don't** mention Claude / Claude Code / AI tooling, or add AI co-author
  trailers, in commits, PR text, or review comments.
- **Don't** run hardware or destructive test modes as part of routine
  verification — the fake-mode suite is the gate.
