# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt report` — file issues and feedback with Tenstorrent.

`tt report issue` opens the browser at a prefilled GitHub new-issue URL for the
tt-cli repo, with an auto-collected environment section in the body. Everything
is client-side URL building: nothing is uploaded, and the user sees (and can
edit) every prefilled character before submitting on github.com.

`tt report bundle` writes a redacted tar.gz of everything support usually asks for
(environment, tt-smi snapshot, config, tt and inference-server logs, container
logs) — see report_bundle.py — then drafts the support email around it: an `.eml`
with the archive attached that the desktop mail client opens as an editable draft,
falling back to a mailto: link (see report_email.py). Nothing is uploaded by tt
itself; the user reviews the draft and sends it.
"""

from __future__ import annotations

import dataclasses
import platform
import sys
import urllib.parse
import webbrowser
from pathlib import Path

import typer
from rich.text import Text

from .. import __version__
from ..backends.device import get_device_backend
from ..cli import JsonFlag, QuietFlag, handle_tt_errors
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
    title: str = typer.Option(
        _DEFAULT_TITLE, "--title", "-t", help="Issue title to prefill."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the issue URL without opening a browser."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Open a prefilled GitHub issue on tenstorrent/tt-cli, with environment details attached.

    The URL is printed on stdout; --json emits it as a JSON object and never opens
    a browser. --quiet suppresses the URL but still opens the browser.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)

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


def _stdin_isatty() -> bool:
    """Test seam: CliRunner replaces sys.stdin, so tests patch this, not isatty."""
    return sys.stdin.isatty()


def _render_bundle_summary(d: dict) -> Text:
    """Archive path first, so `tt report bundle | head -1` is still the file. A Text,
    not a str: the subject's [ttbr-…] would otherwise be read as Rich markup."""
    who = d["assignee"]
    return Text(
        "\n".join(
            [
                d["path"],
                f"email:     {d['eml']}",
                f"to:        {d['to']}",
                f"subject:   {d['subject']}",
                f"assignee:  {who['name']} <{who['email']}> (this week's DX triage)",
            ]
        )
    )


@report_app.command("bundle")
@handle_tt_errors
def report_bundle(
    ctx: typer.Context,
    title: str | None = typer.Option(
        None,
        "--title",
        "-t",
        help="Subject of the support email (default: asked for, or 'Bug report').",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the archive (default: ./tt-report-<reference>.tar.gz); "
        "the .eml is written beside it.",
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Write the archive and the .eml, but open nothing."
    ),
    mailto: bool = typer.Option(
        False,
        "--mailto",
        help="Skip the .eml and open a mailto: draft instead (webmail users; "
        "attach the archive by hand).",
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Collect a support bundle and open a pre-filled email to Tenstorrent support with it attached.

    The bundle holds environment, tt-smi snapshot, config, tt and inference-server
    logs and container logs; known secrets (HF tokens, JWT_SECRET, telemetry keys)
    are redacted. It keeps hostnames and local paths, so it is meant for
    support@tenstorrent.com, not a public issue. Every source that is missing or
    broken is noted in manifest.json instead of failing the command.

    The email is written as an .eml beside the archive and handed to the desktop's
    mail client as an editable draft; if that is not possible, a mailto: draft is
    opened, and failing that the To/Subject/body are printed to compose by hand.
    --json prints the details and never opens anything.
    """
    from . import report_email
    from .report_bundle import default_output_path, write_bundle

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet)
    out = appctx.output

    ref = report_email.make_ref()
    if title is None and not out.json_mode and not out.quiet and _stdin_isatty():
        title = typer.prompt(
            "Subject for the support email", default=report_email.DEFAULT_TITLE
        )
    title = report_email.clean_title(title)

    out.status("Collecting environment, config, logs and container output …")
    archive = output or default_output_path(ref)
    payload = write_bundle(appctx, archive, ref=ref)

    assignee = report_email.assignee_for_date()
    subject = report_email.build_subject(title, ref)
    body = report_email.build_body(
        ref=ref,
        assignee=assignee,
        title=title,
        environment_lines=_environment_lines(appctx),
        archive_name=archive.name,
    )
    mailto_url = report_email.build_mailto_url(subject, body, archive.name)
    eml = report_email.eml_path_for(archive)
    try:
        eml.write_bytes(
            report_email.build_eml(subject=subject, body=body, archive=archive, ref=ref)
        )
    except OSError as exc:
        raise TTError(
            f"Cannot write the support email to {eml}.",
            why=str(exc),
            next_step=f"The bundle is at {archive}; pass --output to a writable location.",
            exit_code=ExitCode.ERROR,
        ) from exc

    payload.update(
        {
            "eml": str(eml),
            "to": report_email.SUPPORT_EMAIL,
            "subject": subject,
            "assignee": {"name": assignee[0], "email": assignee[1]},
            "mailto": mailto_url,
        }
    )
    out.emit(payload, renderer=_render_bundle_summary, soft_wrap=True)
    if payload["size_bytes"] > report_email.ATTACHMENT_WARN_BYTES:
        out.warn(
            f"the bundle is {payload['size_bytes'] // (1024 * 1024)} MB; most mail "
            "servers reject attachments over ~25 MB. If the email bounces, share "
            "the archive another way and quote the reference."
        )

    if out.json_mode or no_open:
        return

    opened = False
    if not mailto:
        out.status("Opening the draft in your mail client … review it, then Send.")
        opened = report_email.open_file(eml)
    if not opened:
        out.status(f"Opening a mailto: draft … attach {archive.name} before sending.")
        opened = report_email.open_mailto(mailto_url)
    if not opened:
        out.warn(
            f"could not open a mail client — send {archive.name} to "
            f"{report_email.SUPPORT_EMAIL} by hand with this subject and body:"
        )
        # Raw print: the subject and body carry literal [brackets] that Rich would
        # otherwise read as markup.
        out.status_console.print(
            f"To:      {report_email.SUPPORT_EMAIL}\nSubject: {subject}\n\n{body}",
            markup=False,
            highlight=False,
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
