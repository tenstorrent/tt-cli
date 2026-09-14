# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Cards are pure builders, so they render into a fixed-width Console in a test."""

import io

from rich.console import Console

from tenstorrent.ui.cards import (
    failure_card,
    interrupted_panel,
    kept_panel,
    notice_panel,
    ready_panel,
)
from tenstorrent.ui.theme import THEME


def render(renderable, width=80):
    console = Console(file=io.StringIO(), width=width, theme=THEME, highlight=False)
    console.print(renderable)
    return console.file.getvalue()


def test_notice_panel_shows_title_and_body():
    out = render(notice_panel("Heads up", ["line one", "line two"]))
    assert "Heads up" in out
    assert "line one" in out and "line two" in out


def test_ready_panel_lays_out_rows_and_footer_hints():
    out = render(
        ready_panel(
            "tt is ready",
            [("URL", "http://localhost:8000", "up"), ("Mode", "Local")],
            ["Ready in 3m 34s · 3 phases", "Stop · tt model stop"],
        )
    )
    assert "tt is ready" in out
    assert "URL" in out and "localhost:8000" in out
    assert "up" in out
    assert "Ready in 3m 34s" in out
    assert "tt model stop" in out


def test_failure_card_puts_the_cause_in_the_title_and_lists_actions():
    card = failure_card(
        "Docker Control service",
        {
            "cause": "port 8002 is still taken",
            "detail": "Another process was holding port 8002.",
            "evidence": "ERROR: [Errno 98] Address already in use",
            "actions": ["lsof -i :8002", "tt update, then re-run"],
        },
        consequence="Startup continues — the backend falls back to the Docker SDK.",
    )
    out = render(card, width=100)
    assert "port 8002 is still taken" in out
    assert "Another process was holding" in out
    assert "Errno 98" in out
    assert "Startup continues" in out
    assert "Try:" in out
    assert "lsof -i :8002" in out


def test_failure_card_appends_the_log_path_when_no_action_mentions_it():
    card = failure_card(
        "tt-installer",
        {"cause": "it exited non-zero", "detail": "The system installer failed.", "actions": []},
        log_path="/tmp/logs/tt-installer.log",
    )
    out = render(card, width=100)
    assert "tail -50 /tmp/logs/tt-installer.log" in out


def test_failure_card_does_not_duplicate_a_log_path_an_action_already_names():
    card = failure_card(
        "tt-installer",
        {
            "cause": "it exited non-zero",
            "detail": "The system installer failed.",
            "actions": ["tail -50 /tmp/logs/x.log"],
        },
        log_path="/tmp/logs/x.log",
    )
    out = render(card, width=100)
    assert out.count("/tmp/logs/x.log") == 1


def test_failure_card_evidence_is_one_elided_line_not_a_log_dump():
    card = failure_card(
        "thing",
        {
            "cause": "c",
            "detail": "d",
            "evidence": "first error line\nsecond line\nthird line",
            "actions": [],
        },
    )
    out = render(card, width=100)
    assert "first error line" in out
    assert "second line" not in out


def test_kept_panel_states_what_survived():
    out = render(kept_panel("Preserved", ["Config · ~/.config/tenstorrent"], ["Remove · tt config reset"]))
    assert "Preserved" in out
    assert "Config" in out
    assert "tt config reset" in out


def test_interrupted_panel_offers_resume_and_cleanup():
    out = render(interrupted_panel("tt update", cleanup="tt update --dry-run"))
    assert "Interrupted" in out
    assert "Resume · tt update" in out
    assert "Clean up" in out


def test_cards_render_at_a_narrow_width_without_crashing():
    out = render(notice_panel("Narrow", ["a fairly long line of body text that must wrap"]), width=40)
    assert "Narrow" in out
