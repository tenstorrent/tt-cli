# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Support-email drafting for `tt report bundle`.

Bug reports reach Tenstorrent by emailing support@tenstorrent.com: the support inbox
turns the email into a Jira ticket and streams replies back to the sender. This
module builds that email around the support bundle, in two shapes:

- an `.eml` file (RFC 5322) with the bundle attached and an `X-Unsent: 1` header,
  which desktop mail clients (Thunderbird, Evolution, Outlook) open as an editable
  draft — the user reviews and hits Send;
- a `mailto:` URL as the fallback where no `.eml` handler exists (webmail-only
  machines, headless boxes). mailto: cannot carry attachments, so that body ends
  with a reminder to attach the bundle by hand.

The first body lines (`Assignee:` / `Reference:` / `Product:`) are machine-readable:
a Jira automation rule on the support project parses them to assign the ticket and
label it. The subject prefix, the `ttbr-` reference scheme, the body layout and the
weekly assignee rotation (ISO week % 3) are shared with tt-studio's bug reporter
(tt_setup/support_email.py) so both products land in the same triage queue.
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess
import uuid
import webbrowser
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import format_datetime, make_msgid
from pathlib import Path
from urllib.parse import quote

from .. import __version__

SUPPORT_EMAIL = "support@tenstorrent.com"
PRODUCT = "tt-cli"
SUBJECT_PREFIX = "[TT-CLI]"
DEFAULT_TITLE = "Bug report"

# Weekly DX triage rotation shared with tt-studio: ISO week number % 3 picks the
# assignee. Harmless discontinuity at ISO-year boundaries (week 52/53 -> week 1).
ROTATION: tuple[tuple[str, str], ...] = (
    ("Anirudh", "anirud@tenstorrent.com"),
    ("Jashan", "jashansingh@tenstorrent.com"),
    ("Raheem", "rnabeel@tenstorrent.com"),
)

# mailto: bodies beyond ~2000 chars get truncated by common mail clients and
# browsers; everything heavy lives in the attached bundle anyway.
MAX_MAILTO_BODY = 1800
MAX_SUBJECT_TITLE = 100
# Most mail servers reject attachments somewhere around 25 MB.
ATTACHMENT_WARN_BYTES = 20 * 1024 * 1024

_TRUNCATION_NOTICE = "\n[truncated — full details in the attached bundle]"
_PLACEHOLDER = "_fill in_"
# xdg-open can sit around while a client starts; it normally returns at once.
_OPEN_TIMEOUT_S = 15


def make_ref() -> str:
    """Reference matching a support ticket to one bundle (same shape as tt-studio)."""
    return f"ttbr-{uuid.uuid4().hex[:12]}"


def assignee_for_date(day: datetime.date | None = None) -> tuple[str, str]:
    """(name, email) of this week's triage assignee."""
    day = day or datetime.date.today()
    return ROTATION[day.isocalendar()[1] % len(ROTATION)]


def clean_title(title: str | None) -> str:
    title = (title or "").strip() or DEFAULT_TITLE
    if len(title) > MAX_SUBJECT_TITLE:
        title = title[: MAX_SUBJECT_TITLE - 1] + "…"
    return title


def build_subject(title: str | None, ref: str) -> str:
    """`[TT-CLI] <title> [ttbr-…]`."""
    return f"{SUBJECT_PREFIX} {clean_title(title)} [{ref}]"


def build_body(
    *,
    ref: str,
    assignee: tuple[str, str],
    title: str | None,
    environment_lines: list[str],
    archive_name: str,
) -> str:
    """Plain-text body. `environment_lines` is the short list `tt report issue`
    already builds; everything else is in the attached bundle."""
    name, email = assignee
    env_block = "\n".join(environment_lines) if environment_lines else "_unknown_"
    return f"""Assignee: {name} <{email}>
Reference: {ref}
Product: {PRODUCT}

{PRODUCT} bug report. Do not edit the Assignee/Reference lines — Jira
automation reads them.

## Summary
{clean_title(title)}

## Description
{_PLACEHOLDER}

## Steps to Reproduce
{_PLACEHOLDER}

## Expected / Actual
{_PLACEHOLDER} / {_PLACEHOLDER}

## Environment
{env_block}
(Full logs, tt-smi snapshot, config and container logs are in the attached {archive_name}.)

--
Sent from `tt report bundle` ({PRODUCT} {__version__}).
"""


def build_mailto_url(subject: str, body: str, archive_name: str) -> str:
    """mailto: draft to support. The attachment cannot ride along, so the body gets
    an explicit reminder. quote(safe="") — never quote_plus: `+` renders literally."""
    reminder = f"\nIMPORTANT: attach {archive_name} to this email before sending.\n"
    budget = MAX_MAILTO_BODY - len(reminder) - 1  # -1: the joining newline below
    if len(body) > budget:
        body = body[: budget - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE
    body = body.rstrip("\n") + "\n" + reminder
    return f"mailto:{SUPPORT_EMAIL}?subject={quote(subject, safe='')}&body={quote(body, safe='')}"


def build_eml(
    *,
    subject: str,
    body: str,
    archive: Path,
    ref: str,
    now: datetime.datetime | None = None,
) -> bytes:
    """An RFC 5322 message with the bundle attached. `X-Unsent: 1` makes desktop
    clients open it as an editable draft. No `From:` on purpose: the client fills
    in the sender's own account."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    msg = EmailMessage(policy=SMTP)
    msg["To"] = SUPPORT_EMAIL
    msg["Subject"] = subject
    msg["Date"] = format_datetime(now)
    msg["Message-ID"] = make_msgid(domain="tt-cli.local")
    msg["X-Unsent"] = "1"
    msg["X-TT-Reference"] = ref
    msg["X-Mailer"] = f"{PRODUCT}/{__version__}"
    msg.set_content(body)
    msg.add_attachment(
        archive.read_bytes(),
        maintype="application",
        subtype="gzip",
        filename=archive.name,
    )
    return msg.as_bytes()


def eml_path_for(archive: Path) -> Path:
    """`tt-report-<ref>.tar.gz` -> `tt-report-<ref>.eml`, beside the archive."""
    name = archive.name
    for suffix in (".tar.gz", ".tgz"):
        if name.endswith(suffix):
            return archive.with_name(name[: -len(suffix)] + ".eml")
    return archive.with_name(archive.stem + ".eml")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_file(path: Path) -> bool:
    """Hand `path` to the desktop's default handler. True only when the launcher
    reports success; False for headless hosts, a missing launcher, or a launcher
    that found no handler (xdg-open exits 3/4). Module-level on purpose: tests
    patch it."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        if system == "Darwin":
            argv = ["open", str(path)]
        else:
            if not _has_display():
                return False
            launcher = shutil.which("xdg-open")
            if launcher is None:
                return False
            argv = [launcher, str(path)]
        result = subprocess.run(
            argv, capture_output=True, check=False, timeout=_OPEN_TIMEOUT_S
        )
        return result.returncode == 0
    except Exception:
        return False


def open_mailto(url: str) -> bool:
    """Open a mailto: draft through the default browser/handler. Test seam.

    Headless Linux is checked explicitly: webbrowser.open happily reports success
    after spawning a browser nobody can see, which would hide the print-by-hand
    fallback from SSH users."""
    if platform.system() == "Linux" and not _has_display():
        return False
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
