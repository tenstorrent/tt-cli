# Release branches

Releasing a new version of tt-cli starts with a short-lived release branch cut
from `main`. Stabilization happens on that branch, so `main` stays open for
ongoing feature work throughout the release.

## 1. Cut the branch

```console
git checkout -b release/1.2.x main
git push origin release/1.2.x
```

## 2. Stress-test it without publishing anything

Trigger `release.yml` manually (Actions tab → "release" → "Run workflow",
selecting `release/1.2.x`). It builds the sdist/wheel, checks the distribution
metadata, smoke-tests the installed wheel, and runs the full test suite
(fake-tool matrix + real hardware) — but publishes nothing, since neither the
`github-release` nor `publish-pypi` job runs without a `refs/tags/v*` ref.
Testers can install the wheel from that run's `dist` workflow artifact.

## 3. Hot fixes go directly on the release branch

A bug found during stress testing is fixed directly on the release branch
with a PR targeting `release/1.2.x`. There is no cherry-pick: the fixes reach
`main` when the release branch is merged back (step 5).

Re-test and repeat as needed.

## 4. Tag the release

Once stable, bump `__version__` in `src/tenstorrent/__init__.py`, commit to
the release branch, then tag and push:

```console
git tag v1.2.0
git push origin v1.2.0
```

This triggers `release.yml` for real: build → full test suite → GitHub
Release → PyPI.

## 5. Merge the branch into `main`

Open a PR from `release/1.2.x` into `main`. This brings the hot fixes and the
version bump onto `main` in one go.

## 6. Delete the branch

Delete `release/1.2.x` once it's tagged and merged into `main`. Keep it alive only
if you expect to ship a patch release (`v1.2.1`, `v1.2.2`, ...) against this
exact line later, after `main` has moved on to unrelated work.
