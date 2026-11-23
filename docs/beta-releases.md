# Beta releases

A feature that needs real users before it is ready for a stable release ships
as a beta from its own branch. Each beta feature gets a `beta/<feature>` branch
cut from `main`, beta versions are tagged on that branch and published to PyPI
as pre-releases, and the feature only reaches `main` once it graduates. Nothing
on a beta branch can end up in a stable release until it is merged into `main`.

```
main ──●────────●──────────────●──────── v1.2.3 ──── v1.3.0
        \        \ (sync)       \ (sync)                ▲
         beta/bio ●──●── v1.3.0b1 ──●── v1.3.0b3 ── PR ─┘ (graduate)
          \
           beta/studio-community ●── v1.3.0b2
```

## 1. Cut the branch

```console
git checkout -b beta/bio main
git push origin beta/bio
```

Feature PRs for the beta target `beta/bio`, not `main`, and are reviewed as
usual. Keep one feature per beta branch.

## 2. Keep it in sync with `main`

Merge `main` into the beta branch regularly, and always right before tagging a
beta, so beta users still get the fixes that shipped in stable patch releases.
Merge rather than rebase: published beta tags must stay reachable from the
branch.

```console
git checkout beta/bio && git pull
git merge origin/main
```

## 3. Pick the version

Beta versions use the **next minor** version with a `b` suffix, for example
`1.3.0b1` while the latest stable release is `1.2.x`. Once `1.3.0` ships, later
betas move to `1.4.0bN`, otherwise they would sort below the stable release.

The `bN` number is a **single counter shared by all beta branches**. Versions
cannot carry the feature name (PyPI rejects local versions like `1.3.0+bio`), so
`1.3.0b2` from `beta/studio-community` does not contain `1.3.0b1` from
`beta/bio`. Check which numbers are taken before picking one:

```console
git fetch --tags
git tag -l 'v1.3.0b*'
```

A collision fails at `git push` of the tag, since tags are unique.

## 4. Tag the beta

Bump `__version__` in `src/tenstorrent/__init__.py` on the beta branch, commit,
then tag that commit and push the tag:

```console
git tag v1.3.0b1
git push origin v1.3.0b1
```

`release.yml` builds, runs the full test suite, creates a GitHub pre-release
named after the feature, and publishes to PyPI. A PyPI upload cannot be undone,
so the same care applies as for a stable release.

## 5. Share it with testers

Testers install the exact beta version:

```console
uv tool install tenstorrent==1.3.0b1
# or
pipx install tenstorrent==1.3.0b1
```

Always pin the version. `pip install --pre tenstorrent` installs the highest
pre-release, which may be a different feature's beta.

Plain `pip install tenstorrent` and `tt self update` never pick up betas, so
stable users are unaffected. Beta users are not notified of newer betas either,
and a stable patch like `1.2.3` sorts below `1.3.0b1`, so announce each new beta
to its testers directly. Once `1.3.0` is released, `tt self update` moves beta
users onto it.

## 6. Graduate the feature

When the feature is ready for stable, open a PR from `beta/bio` into `main`.
Before merging, reset `__version__` to the value on `main`, since beta versions
must never land on `main`. The feature then ships in the next stable release,
cut the usual way (see [release-branches.md](release-branches.md)).

Delete `beta/bio` after the PR merges. The published beta tags and PyPI uploads
stay.

## 7. Or drop it

If the feature is abandoned, delete the branch. Yank its betas on PyPI if they
should no longer be installable.

## Workflow changes this needs

Not yet implemented; the steps above depend on them.

- **Tag origin:** `release.yml` only accepts tags on `main`. Also accept a tag
  whose commit is on `origin/beta/*` when the version is a beta (`bN`). Stable
  and `rc` versions must still come from `main`.
- **PyPI:** `publish-pypi` skips every pre-release today. Let beta versions
  through. `publish-compat-overrides` keeps skipping them.
- **Changelog:** stable releases should ignore pre-release tags when picking
  the previous tag, so they are not diffed against another feature's beta. Beta
  releases diff against the latest stable tag and name the release after the
  branch, e.g. "v1.3.0b3 (beta: bio)".
- **Guard on `main`:** fail any PR into `main` whose `__version__` is an alpha
  or beta version, so a graduation PR cannot carry the beta version over.
