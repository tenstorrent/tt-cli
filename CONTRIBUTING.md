# Contributing to the Tenstorrent CLI (tt)

Thanks for your interest in improving `tt`. This document explains how to report
problems and submit changes.

## Reporting Bugs

Report bugs via [GitHub Issues](../../issues). A good report includes:

- The `tt` version (`tt --version`) and how it was installed
- Your OS/distribution and, where relevant, the Tenstorrent hardware present
- The exact command run and its full output (re-run with `--verbose` if needed)
- What you expected to happen

Please do not report security vulnerabilities through public issues; see
[SECURITY.md](SECURITY.md).

## Submitting Changes

Bug fixes and new functionality are submitted via Pull Requests:

1. Fork the repository and create a branch from `main`.
2. Make your change, including tests for new behavior.
3. Ensure the checks below pass.
4. Open a Pull Request describing the change and the motivation for it.

Pull Requests are reviewed on a **weekly** cadence. Larger changes land more
smoothly when discussed in an issue first.

## Development

The project uses [uv](https://github.com/astral-sh/uv):

```bash
uv sync              # once, to create the environment
uv run pytest        # run the test suite (hardware-free by default)
uv run tt ...        # try the CLI locally
```

The default test run uses fake tools and requires no hardware, config, or
network.

### Upstream pins

The versions of the tools `tt` delegates to (tt-installer, tt-sw-manifest's
golden.json, tt-inference-server, tt-model-manager) are pinned in
`src/tenstorrent/tools/supplement.toml`. A scheduled workflow (`bump-pins`, Mondays;
also runnable from the Actions tab via "Run workflow") runs `scripts/bump_pins.py`
and opens one PR with every pin that moved, plus the files that must move with them
(asset sha256s, the bundled `release_model_spec.json`, the installer step in
`tests.yml`). The PR body carries a review checklist; it is never auto-merged. Run
`uv run scripts/bump_pins.py --dry-run` to see what a bump would do without writing.

## Releasing (maintainers)

Releases are cut by pushing a tag; `.github/workflows/release.yml` does the rest.

1. Bump `__version__` in `src/tenstorrent/__init__.py` — the only place the version
   lives, since `pyproject.toml` reads it via `[tool.hatch.version]`.
2. Commit that to `main`.
3. Tag it and push: `git tag v1.2.3 && git push origin v1.2.3`. The tag must match
   `__version__` exactly (bar the `v`), or the workflow stops before building.

The workflow then builds the sdist and wheel, checks the metadata, installs the wheel
into a clean environment and runs `tt`, gates on the full test suite (fake-tool matrix
plus real hardware), and publishes to TestPyPI → GitHub Releases → PyPI, in that order,
so the irreversible step happens last. Release notes are generated from the merged PRs
and commits since the previous tag; a version like `1.2.3rc1` is flagged as a
pre-release automatically.

Uploads use PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/), so
there are no API tokens to store — publishers must be registered on PyPI and TestPyPI
for repository `tenstorrent/tt-cli`, workflow `release.yml`, and environments `pypi`
and `testpypi`. Since the `tenstorrent` project does not exist on PyPI yet, register it
as a *pending* publisher, which claims the name and lets the first run create it.

To rehearse without publishing anything, run the workflow manually from the Actions
tab; it builds and validates the same artifacts, and optionally uploads them to
TestPyPI.

## Coding Standards

- New and modified code files must carry an SPDX header:

  ```python
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.
  ```

- Add tests alongside code changes; keep the suite green (`uv run pytest`).
- Match the style and conventions of the surrounding code.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), consistent with the rest of the project.
