# Beta releases

A feature that needs real users before it is ready for a stable release ships
as a beta from one shared branch. All beta features in flight at the same time
target the same `beta/<version>` branch, cut from `main` and named after the
minor version it is heading for. Betas are tagged from that branch's HEAD and
published to PyPI as pre-releases as features land on it, and nothing on it can
end up in a stable release until it is merged into `main`.

```
main ──●──────────────●────────────────●──────── v1.2.3 ──── v1.3.0
        \              \ (sync)         \ (sync)                ▲
         beta/1.3 ──●── v1.3.0b1 ──●──●── v1.3.0b3 ────── PR ────┘ (graduate)
                      (bio lands)   (deep-research lands)
```

## 1. Cut the branch

Cut it once, the first time a feature needs a beta after the previous beta
branch graduated or was retired:

```console
git checkout -b beta/1.3 main
git push origin beta/1.3
```

Name it after the minor version `main` is currently building toward. Feature
PRs for any beta target this one branch, not `main`, and are reviewed as
usual. Every beta feature in flight shares it until it graduates or the whole
branch does.

## 2. Keep it in sync with `main`

Merge `main` into the beta branch regularly, and always right before tagging a
beta, so beta users still get the fixes that shipped in stable patch releases.
Merge rather than rebase: published beta tags must stay reachable from the
branch.

```console
git checkout beta/1.3 && git pull
git merge origin/main
```

## 3. Pick the version

Beta versions use the **next minor** version with a `b` suffix, for example
`1.3.0b1` while the latest stable release is `1.2.x`. Once `1.3.0` ships, the
next beta branch moves to `1.4.0bN`, otherwise it would sort below the stable
release.

The `bN` number is a counter on the branch: it increments every time you cut a
new beta tag from the branch's current HEAD, typically because another
feature has landed on it since the last tag. Check which numbers are already
taken before picking one:

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
named after the branch, and publishes to PyPI. A PyPI upload cannot be undone,
so the same care applies as for a stable release.

## 5. Share it with testers

Testers install the exact beta version:

```console
uv tool install tenstorrent==1.3.0b1
# or
pipx install tenstorrent==1.3.0b1
```

Always pin the version. `pip install --pre tenstorrent` installs the highest
pre-release, which may carry more or fewer features than the one you meant.

Plain `pip install tenstorrent` and `tt self update` never pick up betas, so
stable users are unaffected. Beta users are not notified of newer betas either,
and a stable patch like `1.2.3` sorts below `1.3.0b1`, so announce each new beta
to its testers directly. Once `1.3.0` is released, `tt self update` moves beta
users onto it.

## 6. Graduate

Individual features normally ride the shared branch to graduation together:
when the accumulated work is ready for stable, open a PR from `beta/1.3` into
`main`. Before merging, reset `__version__` to the value on `main`, since beta
versions must never land on `main`. The branch's contents then ship in the
next stable release, cut the usual way (see
[release-branches.md](release-branches.md)).

Delete `beta/1.3` after the PR merges, then cut the next beta branch (for
example `beta/1.4`) off the new `main` the next time a feature needs one. The
published beta tags and PyPI uploads stay.

A feature that is ready before the rest of the branch, or that needs to ship
alone, can instead be cherry-picked (or merged from its own feature branch)
straight into `main` in its own PR, with its own version reset. This doesn't
require waiting on, or disturbing, the other features still on the beta
branch.

## 7. Dropping a feature

If one feature on the branch is abandoned, revert its commits on the beta
branch (or ask its author to) rather than deleting the branch — the other
features in flight stay on it. Yank that feature's betas on PyPI if they
should no longer be installable.

If every feature on the branch is abandoned, delete the branch and yank its
betas.

## What the workflows enforce

- **Tag origin.** `release.yml` accepts a stable or `rc` tag only on a commit on
  `main`. A beta tag (`X.Y.ZbN`) may also point at a commit on exactly one
  `beta/*` branch. Alpha and dev versions are held to the `main` rule.
- **PyPI.** Stable releases and betas are published. `rc`, alpha and dev
  versions get a GitHub pre-release only. `publish-compat-overrides` runs for
  stable releases only.
- **Release notes.** Every release is diffed against the latest stable tag it
  contains, never against another pre-release, so beta notes from one beta
  branch never pick up another's. A beta is named after its branch, e.g.
  `v1.3.0b3 (beta: 1.3)`.
- **No beta versions on `main`.** The `no beta version on main` check in
  `tests.yml` fails any PR into `main` whose `__version__` is an alpha or beta.

These rely on two repository settings, configured under Settings → Rules:

- `beta/*` branches have the same protection as `main` (changes through
  reviewed PRs only, no force pushes), since a beta tag on them reaches PyPI.
- `no beta version on main` is a required status check for `main`.
