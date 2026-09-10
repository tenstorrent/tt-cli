This page is for maintainers. It walks through cutting a release of tt-cli end to end: refreshing the pinned upstream versions with `scripts/bump_pins.py`, testing and merging that PR, then bumping `__version__` and pushing a tag so `release.yml` publishes to PyPI. Each step says what happens, what you have to do, and where it bites.

A release is three stages, and only the last one is irreversible:

| Stage | What moves | Who does it | Undo |
|---|---|---|---|
| 1. Bump pins | `supplement.toml` and the files that travel with it, on branch `bot/bump-pins` | The Monday `bump-pins` workflow, or you from the Actions tab or locally | Close the PR |
| 2. Test and merge | The pins PR into `main` | You, after the checks and the review checklist | Revert the merge |
| 3. Tag | `__version__`, a `vX.Y.Z` tag, a GitHub release, a PyPI upload | You push the tag; `release.yml` does the rest | GitHub release yes, PyPI **no** |

Stage 1 and 2 are optional for a release that only ships tt-cli changes. Stage 3 is the release.

# 1. Bump the upstream pins

## What is pinned, and why a script owns it

`tt` delegates to four upstream projects. Their versions live in [`src/tenstorrent/tools/supplement.toml`](https://github.com/tenstorrent/tt-cli/blob/main/src/tenstorrent/tools/supplement.toml), and an old `tt` means old pins, which is why a pin bump is normally followed by a release. Dependabot cannot do this bump: two pins carry the sha256 of a downloaded asset, one pin is derived from another, and one drags a bundled data file along with it. So [`scripts/bump_pins.py`](https://github.com/tenstorrent/tt-cli/blob/main/scripts/bump_pins.py) does the whole thing in one go:

| Component | Moves to | Also moves |
|---|---|---|
| tt-installer | Latest GitHub release (`vX.Y.Z` tags only) | The `install.sh` sha256; the `tenstorrent/tt-installer@vX.Y.Z` step in `tests.yml` that provisions the hardware runner |
| tt-sw-manifest (`[golden]`) | Whatever tag the pinned `install.sh` converges to, never the latest manifest release | The `golden.json` sha256 |
| tt-inference-server | Latest GitHub release | The bundled `release_model_spec.json` (copied verbatim from that tag) and `model_support.json` (rebuilt from it plus `model_support_overrides.toml`) |
| tt-model-manager (`tt-model`) | Head commit of the default branch (upstream publishes no tags) | Nothing |

The `[golden]` pin *follows the installer* rather than tracking tt-sw-manifest directly because `tt update` refuses to run an `install.sh` whose baked-in golden tag differs from the pinned one. Bumping them independently would make `tt` display versions the installer will not install.

## How to run it

**Scheduled.** The `bump-pins` workflow runs every Monday at 08:00 UTC. If anything moved, it pushes to `bot/bump-pins` and opens a PR titled "Bump pinned upstream tool versions" with the `dependencies` label. If nothing moved, it opens nothing and leaves any existing PR alone.

**On demand.** Actions tab → `bump-pins` → "Run workflow". Same result as the scheduled run. Use this when an upstream release lands mid-week and you want to ship it.

**Locally**, from a checkout:

```console
$ export GITHUB_TOKEN=$(gh auth token)         # optional; avoids the anonymous API rate limit
$ uv run scripts/bump_pins.py --dry-run        # print the plan and the files it would write, touch nothing
$ uv run scripts/bump_pins.py --summary /tmp/body.md   # apply the bump, write the PR body to body.md
```

Run it locally when you want to see what a bump *would* do before Monday, or when you need to bump by hand and open the PR yourself (for example on a hotfix branch). The script needs network access to GitHub either way.

## Reading the PR body

The body the script writes is the review brief. It has up to three parts:

- **A table of what moved**, from → to, followed by per-component notes (how many models the new spec has, which `tests.yml` step moved, what the new `golden.json` pins for smi/flash/kmd/firmware).
- **Notes**: things the script saw but deliberately did not act on. Every refusal shows up here with the reason. So does "tt-sw-manifest vN is released but the installer still converges to vM" and "tt-model-manager now publishes tags, consider switching the pin".
- **A "Review before merging" checklist** of the things no test covers. Do not skip it; see [stage 2](#2-test-and-merge-the-pins-pr).

## Safeguards built into the script

The script would rather leave a pin alone than propose something that breaks `tt`. When it refuses, the reason is in the PR body under Notes, and the pin stays where it was.

- **Asset digests are cross-checked.** Every downloaded `install.sh` and `golden.json` is hashed and compared against the digest GitHub publishes for the release asset. A mismatch aborts the whole run rather than writing a wrong sha256.
- **The pinned installer is re-verified even when it did not move.** If the current release's `install.sh` no longer hashes to the recorded sha256, the script says so and touches nothing. A released asset changing underneath us is something a person should look at.
- **A tt-installer release that breaks `tt update` is refused.** The script downloads the new `install.sh` and checks that every flag `tt update` passes (`--mode-non-interactive`, `--versions`, `--reboot-option`, `--use-uv`, `--python-version`, `--update-firmware`) still appears in it, and that it still declares its golden tag. Either failing means the pin stays.
- **A tt-inference-server release with an unusable spec is refused.** The new `release_model_spec.json` has to parse, carry the schema version our catalog code expects, build cleanly with the overrides file, and declare a `release_version` equal to its tag. A tag with a stale `release_version` inside the spec would fail the catalog test, so it is not proposed.
- **Non-semver installer tags are ignored.** Only `vX.Y.Z` releases are candidates, so an oddly named release cannot become the pin.
- **`golden.json` is parsed before being pinned**, so a manifest that is not a flat component → version map is rejected.

## Footguns

- **Comments in `supplement.toml` are not rewritten.** The script preserves them but does not update them, so after a bump the hand-verification notes ("flag contract re-verified against v3.5.4 on 2026-08-18", "sha256 of the v3.5.4 asset") describe the *previous* release. The checklist tells you to fix them; the script cannot, because they record a human check it did not perform.
- **Override warnings mean something stopped working.** If the new spec drops a model that `model_support_overrides.toml` marks as broken, that mark is now dead and the PR body says so. Remove the entry (or decide the model's absence is fine) rather than merging past the warning.
- **A changed model `version` renames the server's volume directory.** Read the `model_support.json` diff: a bumped per-model version means tt-inference-server will look for a differently named volume, so a machine with a pre-seeded `~/data/tt-cache` needs a matching symlink.
- **tt-model is pinned to a commit, not a release.** A default-branch head is whatever upstream last merged. If the script notes that tt-model-manager has started publishing tags, switching the pin to one is a manual change to `supplement.toml`.
- **The bot branch is not yours.** Every run re-creates `bot/bump-pins` from `main` plus the new diff. Commits you push to that branch will be overwritten by the next scheduled run, so either merge the PR before Monday or do your follow-up work on a separate branch.
- **Rate limits.** Anonymous GitHub API calls are limited to 60/hour. A local run without `GITHUB_TOKEN` can fail with a 403 after a few invocations; the CI run always has a token.

# 2. Test and merge the pins PR

## What CI runs

Every PR runs the `tests` workflow: the fake-tool suite across four distro images (Ubuntu 22.04 and 24.04, Debian 13, Fedora 43), plus the `hardware` job on the self-hosted `tt-ubuntu-2204-n300-stable` runner. The hardware job first provisions the runner with `tenstorrent/tt-installer@<pinned version>` and then runs `pytest --hardware --run-destructive`, so a pins PR is the one place the *new* installer release gets exercised against a real board before anyone tags. Destructive here means `tt device reset` only; nothing in CI ever flashes firmware.

The hardware job is skipped for PRs from forks, and it is a smoke test. A pin counts as hardware-verified only after someone has run a real `tt update` and `tt serve` on a box with it.

## What you have to do

1. **Make sure the checks actually started.** See the first footgun below.
2. **Work through the "Review before merging" checklist** in the PR body. It varies by what moved, but the recurring items are:
   - tt-installer: re-read the new `install.sh` for the `tt update` flag contract (the script only checked the flag *names* still appear, not their semantics), then update the verification comments in `supplement.toml`.
   - tt-inference-server: diff upstream's board → device map against `_BOARDS_TO_DEVICE` in `backends/serving/inference_server.py`; re-check the container and volume naming `tt model stop` relies on; re-check the `run.py` argument contract; read the `model_support.json` diff; act on override warnings.
   - tt-sw-manifest: if the tt-smi version moved, re-capture the tt-smi parser fixtures after a real `tt update`.
3. **Push any fixes as commits on the PR** (or a fresh branch, if Monday is close). Comment updates in `supplement.toml`, override removals, and fixture refreshes all belong in the same PR so the pin and its verification land together.
4. **Merge into `main`.** The PR is never auto-merged; a human merging is the sign-off that the checklist was done.

## Footguns

- **A PR opened with the default token does not trigger CI.** GitHub refuses to start workflows on a PR created by a workflow using `GITHUB_TOKEN`, as a guard against recursive runs. If the `BUMP_PINS_TOKEN` repo secret is not set, the pins PR opens with no checks. Close and reopen it, or push an empty commit to it, and the `tests` workflow starts. To fix this permanently, set `BUMP_PINS_TOKEN` to a fine-grained PAT or GitHub App token with contents and pull-requests write, and keep "Allow GitHub Actions to create and approve pull requests" enabled under Settings → Actions → General.
- **Green checks are not the checklist.** Everything on the checklist is there precisely because the suite does not cover it. A green PR with an unread checklist is how a renamed container or a dropped flag ships.
- **The hardware runner can be down or wedged.** If the `hardware` job never schedules, the runner is offline; there is no fallback runner. You can merge on the fake-tool matrix alone if you accept that the new installer has not touched a board, but say so in the PR.
- **Concurrency cancels superseded PR runs.** Pushing to the PR cancels the previous run for that PR. That is intended, but a cancelled run looks like a failure in the checks list until the new one finishes.

# 3. Bump the version and push the tag

## How the release is triggered

The tag drives everything. Nothing in `release.yml` writes to the repo: no bot commits, no temp branches, no version rewriting. The version lives in exactly one place, `__version__` in [`src/tenstorrent/__init__.py`](https://github.com/tenstorrent/tt-cli/blob/main/src/tenstorrent/__init__.py); `pyproject.toml` reads it from there via `[tool.hatch.version]`, and `tt --version` prints it.

## What you have to do

1. **Pick the version.** Plain `1.2.3` for a release. `1.2.3rc1`, `1.2.3.dev0` and similar for a pre-release; these are flagged as pre-releases on GitHub and do not become the repo's "Latest release", and PyPI will not offer them to plain `pip install tenstorrent`. Write the normalised form (`1.2.3rc1`, not `1.2.3-rc1`).
2. **Bump `__version__`** in `src/tenstorrent/__init__.py` and commit it to `main`, either directly or via a PR. Nothing else needs to change for the version.
3. **Tag that commit and push the tag:**

   ```console
   $ git checkout main && git pull
   $ git tag v1.2.3
   $ git push origin v1.2.3
   ```

   The tag must be `v` plus `__version__` exactly.
4. **Watch the `release` workflow** in the Actions tab. It takes as long as the hardware suite does. When `publish-pypi` is green, confirm from a machine that does not have the checkout:

   ```console
   $ uv tool install --force tenstorrent==1.2.3 && tt --version
   ```

Installed users are told about the new version within a day: `tt` checks PyPI once daily and prints an upgrade notice on the next interactive run, and `tt self update` performs the upgrade.

## What the workflow does, in order

1. **Build** (`build` job): resolves `__version__`, rejects it if PyPI would reject it or if it is not in normalised form, and **fails immediately if the tag does not match it**. Then builds the sdist and wheel, runs `twine check --strict`, installs the wheel into a clean venv, and checks that `tt --version` prints the expected version, `tt --help` runs, and the bundled data files (`supplement.toml`, `model_support.json`) made it into the wheel. Nothing downstream runs if this fails, so a typo in the tag never occupies the hardware runner.
2. **Test** (`test` job): the full `tests` workflow, fake-tool matrix and hardware leg both, on the tagged commit.
3. **GitHub release** (`github-release`): creates the release for the tag with the sdist and wheel attached, and a changelog generated from the merged PRs and direct commits since the previous tag. Pre-release versions are flagged. This can be deleted and redone.
4. **PyPI** (`publish-pypi`): uploads the same artifacts to PyPI via trusted publishing. This runs last on purpose because it is the one step that cannot be undone or repeated.

There is also a disabled `publish-testpypi` rehearsal job. It stays off (`if: false`) because no TestPyPI trusted publisher is configured; re-enabling it means restoring its `if` and adding it back to the `needs` of the two publishing jobs.

## Safeguards built into the workflow

- **Tag and version must agree**, and the check runs before anything is built.
- **Unnormalised versions are rejected**, so the tag, the wheel filename and `tt --version` can never disagree about the same release.
- **The wheel is smoke-tested in a clean environment** before it is published, including the data files that a packaging slip could silently drop without breaking any import.
- **The full suite gates publishing.** A red test anywhere means no GitHub release and no PyPI upload.
- **Reversible before irreversible.** GitHub release first, PyPI last.
- **PyPI publishing only happens from `tenstorrent/tt-cli`** on a `v*` tag. A fork pushing a tag builds and tests, and creates a GitHub release on the fork, but never uploads.
- **Trusted publishing, no tokens.** PyPI verifies the workflow's OIDC identity, scoped to repository `tenstorrent/tt-cli`, workflow `release.yml`, environment `pypi`. There is nothing to rotate or leak. The publisher is configured at https://pypi.org/manage/account/publishing/.
- **Manual dispatch never publishes.** Running `release` from the Actions tab builds, validates and tests the current `main` without creating a release or uploading anywhere. Use it as a rehearsal before tagging.

## Footguns

- **A PyPI version can never be re-uploaded.** If `1.2.3` reaches PyPI with a bug, the fix is `1.2.4`. You can *yank* `1.2.3` on PyPI, which hides it from resolvers that have not pinned it, but you cannot replace it. Everything above the PyPI job exists to make sure you are happy with the artifacts before this point.
- **A wrong tag is fixed by retagging, not by force.** If the workflow fails the tag check, delete the tag locally and remotely, fix `__version__` (or the tag), and push again:

  ```console
  $ git tag -d v1.2.3 && git push origin :refs/tags/v1.2.3
  ```

  Do this *before* anything has been published. Moving a tag that already has a GitHub release or a PyPI upload behind it leaves the release pointing at a commit that is not the one people installed.
- **Tag the merge commit on `main`, not your branch.** The workflow only checks that the tag matches `__version__`; it does not check that the commit is on `main`. A tag pushed from a feature branch will release whatever that branch contains.
- **Re-running a failed workflow is fine up to the PyPI step.** Re-run failed jobs from the Actions tab; the build artifacts are re-used and `github-release` updates the existing release in place. If the failure was *in* the PyPI upload after some files went through, the re-run will fail on the files PyPI already has, and the release is effectively done; check https://pypi.org/p/tenstorrent.
- **The hardware runner is on the release path.** If the self-hosted runner is offline, the `test` job never finishes and the release does not ship. The `tests` workflow has a `hardware` input as an escape hatch, but `release.yml` does not pass it; to release with the runner down you have to temporarily add `with: hardware: false` to the `test` job in `release.yml`, commit, and retag. Make that a conscious decision, recorded in the commit.
- **Pushing the version bump and the tag in the wrong order.** If you tag before the `__version__` commit is on `main`, the tag points at a commit whose version does not match and fails fast. Harmless, but you have to delete and retag.
- **Pre-release strings must be normalised.** `1.2.3-rc1` builds locally but the workflow rejects it. Use `1.2.3rc1`.
- **The changelog is what the PR titles say.** Release notes are generated from merged PR titles and direct-commit subjects since the previous tag. A PR titled "fix" becomes a release-notes line that says "fix".

# Quick checklist

For a release that carries new pins:

- [ ] Pins PR open (Monday run, or "Run workflow", or local `--summary`), CI actually started on it
- [ ] Review checklist in the PR body done; `supplement.toml` comments updated; override warnings acted on
- [ ] Ideally a real `tt update` and `tt serve` on a box with the new pins
- [ ] Pins PR merged into `main`
- [ ] `__version__` bumped and on `main`
- [ ] Optional: `release` run from the Actions tab as a rehearsal
- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z` from `main` at that commit
- [ ] `release` workflow green through `publish-pypi`
- [ ] `uv tool install tenstorrent==X.Y.Z && tt --version` from a clean machine
