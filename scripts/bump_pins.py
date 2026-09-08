# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Bump the upstream pins in supplement.toml, and everything that must move with them.

Dependabot and Renovate can see a version string in a TOML file but cannot finish the
bump: two of our pins carry a sha256 of a downloaded asset, the tt-sw-manifest tag has
to follow the tag baked into the pinned install.sh, and a tt-inference-server bump has
to refresh the bundled release_model_spec.json or the suite goes red. So this script
owns the whole bump and `.github/workflows/bump-pins.yml` runs it weekly (or on
demand) and opens one PR with the result.

What it does, per upstream:

* **tt-installer** — latest GitHub release. Downloads install.sh, records its sha256
  (cross-checked against GitHub's asset digest), and refuses the bump if the script
  has no `TTIS_GOLDEN_VERSIONS_TAG` line or has dropped a flag `tt update` passes.
  The `tenstorrent/tt-installer@vX` step in tests.yml moves with it.
* **tt-sw-manifest ([golden])** — follows the installer, never leads it: the tag is
  whatever the pinned install.sh converges to. A newer manifest release is only noted.
* **tt-inference-server** — latest release; the bundled release_model_spec.json is
  replaced verbatim from that tag, after checking it parses under our catalog code.
* **tt-model-manager** — head of the default branch (upstream publishes no tags).

Comments in supplement.toml are preserved but not rewritten — the ones describing the
last hand verification stay as they were, and the PR body says so.

    uv run scripts/bump_pins.py --dry-run          # show the plan, touch nothing
    uv run scripts/bump_pins.py --summary body.md  # apply + write the PR body

Set GITHUB_TOKEN to avoid the anonymous API rate limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import tomlkit

from tenstorrent.backends.installer import _TTIS_TAG_RE
from tenstorrent.tools.manifest import parse_golden

REPO_ROOT = Path(__file__).resolve().parents[1]
# build_model_support owns spec parsing and produces model_support.json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model_support as support_build  # noqa: E402

SUPPLEMENT = REPO_ROOT / "src" / "tenstorrent" / "tools" / "supplement.toml"
MODEL_SPEC = REPO_ROOT / "src" / "tenstorrent" / "modelhub" / "release_model_spec.json"
MODEL_SUPPORT = REPO_ROOT / "src" / "tenstorrent" / "modelhub" / "model_support.json"
TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

INSTALLER_REPO = "tenstorrent/tt-installer"
MANIFEST_REPO = "tenstorrent/tt-sw-manifest"
INFERENCE_REPO = "tenstorrent/tt-inference-server"
MODEL_MANAGER_REPO = "tenstorrent/tt-model-manager"

# Flags `tt update` passes to install.sh (backends/installer.py). A release that drops
# one would break `tt update`, so its bump is refused rather than proposed.
INSTALLER_FLAGS = (
    "--mode-non-interactive",
    "--versions",
    "--reboot-option",
    "--use-uv",
    "--python-version",
    "--update-firmware",
)

_SEMVER_TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")


# -- GitHub access ---------------------------------------------------------------------
@dataclass
class Release:
    tag: str
    # asset name → (download url, GitHub's "sha256:<hex>" digest or None)
    assets: dict[str, tuple[str, str | None]] = field(default_factory=dict)


class GitHub:
    """The handful of GitHub calls the bump needs. Tests substitute a fake."""

    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "tt-cli bump_pins"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(headers=headers, timeout=60, follow_redirects=True)

    def _json(self, path: str):
        resp = self._client.get(f"https://api.github.com/{path}")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _release(data: dict) -> Release:
        return Release(
            tag=data["tag_name"],
            assets={a["name"]: (a["browser_download_url"], a.get("digest")) for a in data["assets"]},
        )

    def latest_release(self, repo: str) -> Release:
        """The newest release that is neither a draft nor a pre-release."""
        return self._release(self._json(f"repos/{repo}/releases/latest"))

    def release(self, repo: str, tag: str) -> Release:
        return self._release(self._json(f"repos/{repo}/releases/tags/{tag}"))

    def default_branch_head(self, repo: str) -> tuple[str, str, str]:
        """(branch, commit sha, commit date) of the default branch."""
        branch = self._json(f"repos/{repo}")["default_branch"]
        commit = self._json(f"repos/{repo}/commits/{branch}")
        return branch, commit["sha"], commit["commit"]["committer"]["date"][:10]

    def tags(self, repo: str) -> list[str]:
        return [t["name"] for t in self._json(f"repos/{repo}/tags?per_page=5")]

    def download(self, url: str) -> bytes:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.content


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _check_digest(name: str, blob: bytes, digest: str | None) -> str:
    """sha256 of `blob`, verified against GitHub's published asset digest when present."""
    sha = _sha256(blob)
    if digest and digest != f"sha256:{sha}":
        raise RuntimeError(f"{name}: downloaded sha256 {sha} does not match GitHub's {digest}")
    return sha


# -- the plan ----------------------------------------------------------------------------
@dataclass
class Change:
    component: str
    old: str
    new: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    supplement: tomlkit.TOMLDocument
    changes: list[Change] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # refusals + informational
    files: dict[Path, bytes] = field(default_factory=dict)  # other files to rewrite

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def bump_installer_and_golden(plan: Plan, gh: GitHub, tests_workflow: str) -> str:
    """tt-installer to its latest release; [golden] to the tag that release converges to.

    Returns the (possibly rewritten) tests.yml text.
    """
    doc = plan.supplement
    tool = doc["tools"]["tt-installer"]
    golden = doc["golden"]
    old_version = str(tool["golden_version"])

    latest = gh.latest_release(INSTALLER_REPO)
    match = _SEMVER_TAG_RE.match(latest.tag)
    if not match:
        plan.notes.append(f"tt-installer: latest release tag `{latest.tag}` is not vX.Y.Z; left at {old_version}.")
        new_version = old_version
    else:
        new_version = match.group(1)

    # Always download the script we would end up pinned to — for a new release that
    # is how the sha256 and the golden tag are learned; for the current pin it is a
    # check that the asset still matches what supplement.toml records.
    url = str(tool["url_template"]).format(version=new_version)
    release = latest if new_version != old_version else gh.release(INSTALLER_REPO, f"v{new_version}")
    _, digest = release.assets.get("install.sh", (None, None))
    script = gh.download(url)
    sha = _check_digest(f"install.sh v{new_version}", script, digest)
    text = script.decode("utf-8", errors="replace")

    tag_match = _TTIS_TAG_RE.search(text)
    missing = [flag for flag in INSTALLER_FLAGS if flag not in text]
    if tag_match is None or missing:
        why = "has no TTIS_GOLDEN_VERSIONS_TAG line" if tag_match is None else f"dropped {', '.join(missing)}"
        plan.notes.append(f"tt-installer: refusing v{new_version} — install.sh {why}; left at {old_version}.")
        return tests_workflow
    script_tag = tag_match.group(1)

    if new_version != old_version:
        tool["golden_version"] = new_version
        tool["url"] = url
        tool["sha256"] = sha
        change = Change("tt-installer", old_version, new_version)
        old_step, new_step = f"{INSTALLER_REPO}@v{old_version}", f"{INSTALLER_REPO}@v{new_version}"
        if old_step in tests_workflow:
            tests_workflow = tests_workflow.replace(old_step, new_step)
            change.notes.append("tests.yml provisions the hardware runner with the same release.")
        else:
            change.notes.append(f"tests.yml has no `{old_step}` step to move — check it by hand.")
        plan.changes.append(change)
    elif sha != str(tool["sha256"]):
        plan.notes.append(
            f"tt-installer: the v{old_version} install.sh asset no longer matches the pinned "
            f"sha256 (now {sha}). Not touched — someone should look at why a released asset changed."
        )
        return tests_workflow

    old_tag = str(golden["tag"])
    if script_tag != old_tag:
        golden_url = str(golden["url_template"]).format(tag=script_tag)
        _, golden_digest = gh.release(MANIFEST_REPO, script_tag).assets.get("golden.json", (None, None))
        blob = gh.download(golden_url)
        golden_sha = _check_digest(f"golden.json {script_tag}", blob, golden_digest)
        versions = parse_golden(blob.decode("utf-8"), golden_url)  # raises if not a flat map
        golden["tag"] = script_tag
        golden["sha256"] = golden_sha
        shown = ", ".join(f"{k} {versions[k]}" for k in ("smi", "flash", "kmd", "firmware") if k in versions)
        plan.changes.append(
            Change("tt-sw-manifest (golden)", old_tag, script_tag,
                   [f"install.sh v{new_version} converges to this tag.", f"golden.json: {shown}."])
        )

    latest_manifest = gh.latest_release(MANIFEST_REPO).tag
    if latest_manifest != script_tag:
        plan.notes.append(
            f"tt-sw-manifest {latest_manifest} is released, but tt-installer v{new_version} still "
            f"converges to {script_tag}; [golden] follows the installer and stays there."
        )
    return tests_workflow


def bump_inference_server(plan: Plan, gh: GitHub) -> None:
    tool = plan.supplement["tools"]["tt-inference-server"]
    old = str(tool["golden_version"])
    tag = gh.latest_release(INFERENCE_REPO).tag
    if tag == old:
        return
    url = f"https://raw.githubusercontent.com/{INFERENCE_REPO}/{tag}/release_model_spec.json"
    spec = gh.download(url)
    # list describing the previous release. Build model_support.json here which is what the CLI reads.
    try:
        spec_doc = json.loads(spec.decode("utf-8"))
        if spec_doc.get("schema_version") != support_build.SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {spec_doc.get('schema_version')!r}, expected "
                f"{support_build.SPEC_SCHEMA_VERSION}"
            )
        overrides = support_build.load_overrides(support_build.OVERRIDES_PATH)
        document, warnings = support_build.build_document(spec_doc, overrides)
    except Exception as exc:  # bad JSON, or an override the new spec invalidates
        plan.notes.append(
            f"tt-inference-server: refusing {tag} — its release_model_spec.json is not "
            f"usable ({exc}); left at {old}."
        )
        return
    release_version = str(spec_doc.get("release_version", ""))
    if f"v{release_version}" != tag:
        plan.notes.append(
            f"tt-inference-server: refusing {tag} — release_model_spec.json says release_version "
            f"{release_version}, and the catalog test requires them to agree; left at {old}."
        )
        return
    old_models = json.loads(MODEL_SUPPORT.read_text())["models"]
    tool["golden_version"] = tag
    plan.files[MODEL_SPEC] = spec
    plan.files[MODEL_SUPPORT] = support_build.render(document).encode("utf-8")
    details = [
        f"release_model_spec.json re-bundled verbatim, and model_support.json rebuilt "
        f"from it: {len(document['models'])} models (was {len(old_models)})."
    ]
    # A mark or fallback whose model the new spec dropped stops doing anything;
    # say so in the PR rather than leaving it to be noticed later.
    details += [f"model_support_overrides.toml: {w}" for w in warnings]
    plan.changes.append(Change("tt-inference-server", old, tag, details))


def bump_model_manager(plan: Plan, gh: GitHub) -> None:
    tool = plan.supplement["tools"]["tt-model"]
    old = str(tool["golden_version"])
    branch, sha, date = gh.default_branch_head(MODEL_MANAGER_REPO)
    tags = gh.tags(MODEL_MANAGER_REPO)
    if tags:
        plan.notes.append(
            f"tt-model-manager now publishes tags ({', '.join(tags[:3])}); the pin is still a "
            f"`{branch}` commit — consider switching supplement.toml to a tagged release."
        )
    if sha == old:
        return
    tool["golden_version"] = sha
    plan.changes.append(Change("tt-model (tt-model-manager)", old[:12], sha[:12], [f"`{branch}` head as of {date}."]))


def build_plan(supplement_text: str, tests_workflow: str, gh: GitHub) -> Plan:
    plan = Plan(supplement=tomlkit.parse(supplement_text))
    new_workflow = bump_installer_and_golden(plan, gh, tests_workflow)
    if new_workflow != tests_workflow:
        plan.files[TESTS_WORKFLOW] = new_workflow.encode("utf-8")
    bump_inference_server(plan, gh)
    bump_model_manager(plan, gh)
    if plan.changed:
        plan.files[SUPPLEMENT] = tomlkit.dumps(plan.supplement).encode("utf-8")
    return plan


# -- reporting ---------------------------------------------------------------------------
_REVIEW_ITEMS = {
    "tt-installer": [
        "Re-read the new install.sh for the `tt update` flag contract (the script only checked "
        "the flag names still appear) and update the verification comments in supplement.toml, "
        "which still describe the previous release.",
    ],
    "tt-inference-server": [
        "Diff upstream `workflows/device_utils.py` BOARD_TYPE_COUNT_TO_DEVICE against "
        "`_BOARDS_TO_DEVICE` in backends/serving/inference_server.py.",
        "Re-check the container name prefix and mount/volume naming `tt model stop` and "
        "`tt model ps` identify containers by (backends/serving/inference_server.py, ps.py).",
        "Re-check the run.py argument contract (`--model`, `--workflow`, `--device`, `--service-port`, "
        "`--host-hf-cache`, `--host-volume`, `--vllm-override-args`, `--override-docker-image`, "
        "`--override-tt-config`).",
        "Read the model_support.json diff: a changed `version` renames the volume directory "
        "the server looks for, so a pre-seeded ~/data/tt-cache needs a matching symlink.",
        "Act on any `model_support_overrides.toml` warnings above — a mark or fallback whose "
        "model the new spec dropped no longer does anything.",
    ],
    "tt-model (tt-model-manager)": [
        "Re-check the `org.tenstorrent.tt-model` label family and `tt-model-<name>-<profile>` "
        "container naming `tt model ps` reads (upstream src/tt_kernel/container.py).",
    ],
    "tt-sw-manifest (golden)": [
        "Re-capture the tt-smi parser fixtures after a real `tt update` if the smi version moved.",
    ],
}


def render_summary(plan: Plan) -> str:
    lines = ["Weekly upstream pin bump, generated by `scripts/bump_pins.py`.", ""]
    if plan.changes:
        lines += ["| Component | From | To |", "|---|---|---|"]
        lines += [f"| {c.component} | `{c.old}` | `{c.new}` |" for c in plan.changes]
        lines.append("")
        for c in plan.changes:
            lines += [f"- **{c.component}**: {note}" for note in c.notes]
    else:
        lines.append("Everything is already at the latest upstream release. Nothing to change.")
    if plan.notes:
        lines += ["", "Notes:"] + [f"- {n}" for n in plan.notes]
    review = [item for c in plan.changes for item in _REVIEW_ITEMS.get(c.component, [])]
    if plan.changes:
        lines += ["", "Review before merging (none of this is checked by the suite):"]
        lines += [f"- [ ] {item}" for item in review]
        lines.append(
            "- [ ] The hardware leg of `tests` is a smoke test, not verification: a "
            "pin is hardware-verified only after a real `tt update` / `tt serve` on a box."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump the upstream pins in supplement.toml.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--summary", type=Path, help="write a Markdown summary (the PR body) here")
    args = parser.parse_args(argv)

    gh = GitHub(token=os.environ.get("GITHUB_TOKEN"))
    plan = build_plan(SUPPLEMENT.read_text(), TESTS_WORKFLOW.read_text(), gh)

    summary = render_summary(plan)
    sys.stdout.write(summary)
    if args.dry_run:
        if plan.files:
            print("Would write: " + ", ".join(str(p.relative_to(REPO_ROOT)) for p in plan.files))
        return 0
    for path, blob in plan.files.items():
        path.write_bytes(blob)
    if args.summary:
        args.summary.write_text(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
