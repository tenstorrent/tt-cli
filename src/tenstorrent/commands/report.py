# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report` — file issues and feedback with Tenstorrent.

`tt report issue` opens the browser at a prefilled GitHub new-issue URL for the
tt-cli repo, with an auto-collected environment section in the body. Everything
is client-side URL building: nothing is uploaded, and the user sees (and can
edit) every prefilled character before submitting on github.com.
"""

from __future__ import annotations

import dataclasses
import platform
import urllib.parse
import webbrowser

import typer

from .. import __version__
from ..backends.device import get_device_backend
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import AppContext, get_app_context
from ..errors import ExitCode, TTError

report_app = typer.Typer(
    help="Report issues and feedback to Tenstorrent.", no_args_is_help=True
)


@dataclasses.dataclass(frozen=True)
class RepoTarget:
    key: str  # short name, echoed in --json output
    slug: str  # GitHub org/repo
    labels: tuple[str, ...] = ("bug",)


# Every issue lands in the CLI's own tracker; the tt-cli team triages it onward.
ISSUE_TARGET = RepoTarget("tt-cli", "tenstorrent/tt-cli")

_DEFAULT_TITLE = "[tt cli report] <short description>"
# GitHub 414s somewhere around ~8k of URL; urlencoding can triple the body, and
# tt-studio ships 8000 pre-encoding without trouble. 6000 leaves headroom — the
# real body is ~1-2k.
_MAX_BODY_CHARS = 6000


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
    title: str = typer.Option(
        _DEFAULT_TITLE, "--title", "-t", help="Issue title to prefill."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the issue URL without opening a browser."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Open a prefilled GitHub issue on tenstorrent/tt-cli, with environment details attached.

    The URL is printed on stdout; --json emits it as a JSON object and never opens
    a browser. --quiet suppresses the URL but still opens the browser.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)

    target = ISSUE_TARGET
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
