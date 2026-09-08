# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report` — file issues and feedback with Tenstorrent.

`tt report issue` opens the browser at a prefilled GitHub new-issue URL for a
chosen repo, with an auto-collected environment section in the body. Everything
is client-side URL building: nothing is uploaded, and the user sees (and can
edit) every prefilled character before submitting on github.com.
"""

from __future__ import annotations

import dataclasses
import platform
import sys
import urllib.parse
import webbrowser

import typer

from .. import __version__
from .._compat import IntRange, prompt
from ..backends.device import get_device_backend
from ..cli import JsonFlag, QuietFlag, handle_tt_errors
from ..context import AppContext, get_app_context
from ..errors import ExitCode, TTError

report_app = typer.Typer(
    help="Report issues and feedback to Tenstorrent.", no_args_is_help=True
)


@dataclasses.dataclass(frozen=True)
class RepoTarget:
    key: str  # what the user types / picks
    slug: str  # GitHub org/repo
    description: str  # one line, shown in the picker and --help
    labels: tuple[str, ...] = ("bug",)


# Order matters: it is the picker numbering, and entry 1 is the picker default.
REPO_TARGETS: dict[str, RepoTarget] = {
    t.key: t
    for t in (
        RepoTarget("tt-cli", "tenstorrent/tt-cli", "this CLI (`tt`) itself"),
        RepoTarget("tt-metal", "tenstorrent/tt-metal", "TT-Metalium / TT-NN compute stack"),
        RepoTarget("tt-smi", "tenstorrent/tt-smi", "device status tool (`tt device`)"),
        RepoTarget("tt-flash", "tenstorrent/tt-flash", "firmware flashing"),
        RepoTarget("tt-kmd", "tenstorrent/tt-kmd", "kernel driver"),
        RepoTarget("tt-installer", "tenstorrent/tt-installer", "system stack installer (`tt update`)"),
        RepoTarget("tt-inference-server", "tenstorrent/tt-inference-server", "model serving (`tt serve`)"),
    )
}

_DEFAULT_TITLE = "[tt cli report] <short description>"
# GitHub 414s somewhere around ~8k of URL; urlencoding can triple the body, and
# tt-studio ships 8000 pre-encoding without trouble. 6000 leaves headroom — the
# real body is ~1-2k.
_MAX_BODY_CHARS = 6000


def _stdin_isatty() -> bool:  # test seam: CliRunner swaps sys.stdin during invoke
    return sys.stdin.isatty()


def _resolve_repo(value: str) -> RepoTarget:
    target = REPO_TARGETS.get(value.strip().lower())
    if target is None:
        raise TTError(
            f"Unknown repo {value!r}.",
            why="`tt report issue` files against a known Tenstorrent repo.",
            next_step=f"Pick one of: {', '.join(REPO_TARGETS)}.",
            exit_code=ExitCode.USAGE,
        )
    return target


def _pick_repo(appctx: AppContext) -> RepoTarget:
    targets = list(REPO_TARGETS.values())
    width = max(len(t.key) for t in targets)
    appctx.output.status("Which repo is the issue about?")
    for i, t in enumerate(targets, start=1):
        appctx.output.status(f"  {i}. {t.key:<{width}}  {t.description}")
    # err=True keeps the prompt on stderr: stdout stays pure data (the URL).
    choice = prompt("Repo", default=1, type=IntRange(1, len(targets)), err=True)
    return targets[choice - 1]


def _environment_lines(appctx: AppContext) -> list[str]:
    """Environment facts for the issue body. Every collector is fenced — a broken
    stack is exactly when people file issues, so this must never raise. No
    absolute paths or hostnames: the body lands in a public GitHub issue."""
    lines = [
        f"- tt CLI: {__version__}",
        f"- OS: {platform.system()} {platform.machine()}",
        f"- Python: {platform.python_version()}",
    ]
    try:
        for row in appctx.registry.status():
            # Deliberately not row.path: it is absolute and contains the username.
            if row.installed_version is None:
                lines.append(f"- {row.name}: golden {row.golden_version}, not installed")
            else:
                lines.append(
                    f"- {row.name}: golden {row.golden_version}, "
                    f"installed {row.installed_version} ({row.source})"
                )
    except Exception:
        lines.append("- tools: unavailable")
    try:
        snap = get_device_backend(appctx).snapshot()
        driver = snap.host.get("Driver")
        if driver:
            lines.append(f"- driver: {driver}")
        if not snap.devices:
            lines.append("- devices: none detected")
        else:
            lines.extend(_device_table(snap.devices))
    except TTError as err:
        lines.append(f"- devices: unavailable ({err.exit_code.name})")
    except Exception:
        lines.append("- devices: unavailable")
    try:
        # Imported inside the fence so a broken import degrades like everything else.
        from ..backends.serving.inference_server import InferenceServerBackend
        from ..backends.serving.ps import human_duration, list_served

        runtime = InferenceServerBackend(
            appctx.registry, appctx.runner, appctx.config, appctx.output
        ).container_runtime()
        served = list_served(appctx.runner, runtime)
        if not served:
            lines.append("- served models: none")
        for row in served:
            # name/backend/port/health only — ServedModel carries no mount paths.
            detail = f"{row.backend}, port {row.port}, {row.health}"
            if row.uptime_s is not None:
                detail += f", up {human_duration(row.uptime_s)}"
            lines.append(f"- served: {row.name} ({detail})")
    except TTError as err:
        lines.append(f"- served models: unavailable ({err.exit_code.name})")
    except Exception:
        lines.append("- served models: unavailable")
    return lines


def _device_table(devices) -> list[str]:
    """Devices as a markdown table: one row per device, one column per firmware
    field. tt_flash_version is dropped — a firmware field tt-flash reads but
    doesn't write, so it always reports N/A (June, 2026-08-18)."""
    fw_keys = sorted(
        {k for dev in devices for k in dev.firmware} - {"tt_flash_version"}
    )
    header = ["#", "board", *fw_keys]
    rows = [
        [str(dev.index), dev.board_type or "unknown", *(dev.firmware.get(k, "") for k in fw_keys)]
        for dev in devices
    ]
    return [
        "",  # blank line: markdown needs one between the list above and a table
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_issue_body(appctx: AppContext) -> str:
    environment = "\n".join(_environment_lines(appctx))
    body = (
        "<!-- Describe the issue: what you did, what you expected, what happened. -->\n"
        "\n"
        "\n"
        "\n"
        "---\n"
        "<details><summary>Environment (auto-collected by `tt report issue`)</summary>\n"
        "\n"
        f"{environment}\n"
        "\n"
        "</details>\n"
    )
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n\n(environment truncated)"
    return body


def build_issue_url(target: RepoTarget, *, title: str, body: str) -> str:
    # `labels` is GitHub's documented comma-separated new-issue parameter; it is
    # silently ignored for users without triage permission on the repo. Harmless.
    query = urllib.parse.urlencode(
        {"title": title, "body": body, "labels": ",".join(target.labels)}
    )
    return f"https://github.com/{target.slug}/issues/new?{query}"


@report_app.command("issue")
@handle_tt_errors
def report_issue(
    ctx: typer.Context,
    repo: str = typer.Argument(
        None,
        metavar="[REPO]",
        help="Repo to file against: "
        + ", ".join(REPO_TARGETS)
        + ". Omit to pick interactively.",
    ),
    title: str = typer.Option(
        _DEFAULT_TITLE, "--title", "-t", help="Issue title to prefill."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the issue URL without opening a browser."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Open a prefilled GitHub issue for a Tenstorrent repo, with environment details attached.

    The URL is printed on stdout; --json emits it as a JSON object and never opens
    a browser. --quiet suppresses the URL but still opens the browser.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)

    if repo is None:
        if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
            raise TTError(
                "Pick a repository to file the issue against.",
                why="The interactive picker needs a terminal and is disabled with --json/--quiet.",
                next_step=f"Re-run as `tt report issue <repo>` with one of: {', '.join(REPO_TARGETS)}.",
                exit_code=ExitCode.USAGE,
            )
        target = _pick_repo(appctx)
    else:
        target = _resolve_repo(repo)

    body = build_issue_body(appctx)
    url = build_issue_url(target, title=title, body=body)

    payload = {
        "repo": target.key,
        "github": target.slug,
        "labels": list(target.labels),
        "url": url,
    }
    # soft_wrap: the URL is one long line; Rich would crop it at terminal width.
    appctx.output.emit(payload, renderer=lambda d: d["url"], soft_wrap=True)

    if not appctx.output.json_mode and not no_browser:
        appctx.output.status(
            f"Opening a prefilled issue for {target.slug} in your browser …"
        )
        try:
            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            appctx.output.warn(
                "could not open a browser — copy the URL above into one by hand."
            )


@report_app.command("feedback")
@handle_tt_errors
def report_feedback(ctx: typer.Context) -> None:
    """[stub] Send open-ended feedback to the Tenstorrent product team."""
    raise TTError(
        "`tt report feedback` is not available yet.",
        why="It will route open-ended feedback to the product team, optionally with "
        "your email for follow-up research.",
        next_step="Meanwhile: https://discord.gg/tenstorrent or your TT contact.",
        exit_code=ExitCode.UNSUPPORTED,
    )
