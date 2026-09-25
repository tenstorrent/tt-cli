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

## 3. Fix on `main`, cherry-pick onto the branch

A bug found during stress testing typically exists on `main` too. Fix it there
first with a normal PR, then bring it onto the release branch:

```console
git checkout release/1.2.x
git cherry-pick <sha-of-the-fix-on-main>
git push origin release/1.2.x
```

Re-test and repeat as needed. Never fix a general bug only on the release
branch — `main` would ship it again later otherwise.

## 4. Tag the release

Once stable, bump `__version__` in `src/tenstorrent/__init__.py`, commit to
the release branch, then tag and push:

```console
git tag v1.2.0
git push origin v1.2.0
```

This triggers `release.yml` for real: build → full test suite → GitHub
Release → PyPI.

## 5. Forward-port the version bump to `main`

Open a small, standalone PR against `main` that only bumps `__version__` to
`1.2.0`. Every actual fix already reached `main` in step 3 — **never merge the
release branch into `main` as a whole**; that reintroduces the divergent
history you cut the branch to avoid.

## 6. Delete the branch

Delete `release/1.2.x` once it's tagged and forward-ported. Keep it alive only
if you expect to ship a patch release (`v1.2.1`, `v1.2.2`, ...) against this
exact line later, after `main` has moved on to unrelated work.
