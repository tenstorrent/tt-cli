---
name: draft-release-notes
description: >-
  Draft tt-cli GitHub release notes in the house style: diff the release tag or
  branch against the previous release tag, then group the changes into
  🚀 New Features / 🐛 Bug Fixes / 🔧 Technical Improvements with bold feature
  names and short descriptions. Use whenever someone asks to "draft a release
  note", "write release notes for vX.Y.Z", or "make release notes from the
  rc-vX.Y.Z branch". Delivers the result as a copyable markdown code block and
  never publishes or edits a release unless asked.
---

# tt-cli Release Notes Drafting

Follow this when asked to draft release notes for a new tt-cli version. The
notes are built from the real commits between two tags and read like a person
wrote them, not like a commit dump.

## Inputs to confirm

- **New version** (e.g. `v1.1.0`) and where it lives: the tag, or the release
  branch (e.g. `rc-v1.1.0`) if the tag doesn't exist yet.
- **Previous version** it ships after. If not given, it's the latest published
  release: `gh release list --repo tenstorrent/tt-cli --limit 3`.

Don't guess a version. Ask if unclear.

## Step 1: Gather the changes

```bash
git fetch origin --tags
git log <prev-tag>..<new-tag-or-branch> --oneline --no-merges
```

Subjects are squash-merged PR titles ending in `(#NNN)`, usually descriptive
enough. If one is cryptic, run `git show --stat <sha>` rather than guessing. If
you only read subjects, say so in the notes to the user.

Previous published releases are auto-generated commit lists, so they are not a
style reference. Use the template below. The published body also carries a
Contributors list and an Install block (see Step 4).

## Step 2: Categorize

- **🚀 New Features**: user-facing commands, flags, clients, models, output.
  Bold name plus a one-line description, with command names in backticks.
- **🐛 Bug Fixes**: broken behavior that now works, in past tense ("Fixed…").
  Doc and wording clarifications that help users can go here too.
- **🔧 Technical Improvements**: refactors, telemetry, dependency/pin bumps,
  CI, CODEOWNERS, docs about process, internal plumbing.
- Optional **🤖 Model Support**: only when model/catalog changes are too many
  for New Features.

Judgment calls to make deliberately, and mention afterward:

- Fold related commits into one bullet (e.g. all community-model changes, all
  terminal-output changes).
- Collapse repeated "Bump pinned upstream tool versions" commits into one line.
- Drop `Bump __version__`, merge commits and release-prep commits.
- Say if the release contains merges from `main` that were not originally meant
  for it.

## Step 3: Changelog link

Use `https://github.com/tenstorrent/tt-cli/compare/<prev>...<new>`, or the
release PR (`#NNN`) if the user has one. Say which you used.

## Step 4: Deliver

Hand back the notes inside a fenced `markdown` block so it can be pasted
verbatim. Below it, add a short plain-text list of judgment calls (what you
folded, dropped or moved, link choice, whether you read diffs or only
subjects).

Do not run `gh release edit/create` unless the user asks. If they do, keep the
Contributors and Install sections from the existing body:

```bash
gh release view <tag> --repo tenstorrent/tt-cli --json body -q .body > old.md
# merge by hand, then:
gh release edit <tag> --repo tenstorrent/tt-cli --notes-file new.md
```

Never add AI-tool attribution to the notes.

## Template

````
```markdown
## :rocket: TT CLI <version> is out!

A few highlights since <previous version>:

### 🚀 New Features
- **<Feature Name>**: <one-line description>

### 🐛 Bug Fixes
- Fixed <…>

### 🔧 Technical Improvements
- <…>

**Full changelog → <link>**
```
````
