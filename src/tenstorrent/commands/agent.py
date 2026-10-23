# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt agent` — set up Claude Code with the Tenstorrent skills for what you are doing today.

The flow is: make sure Claude Code is installed (offering the documented installer
if not), ask what the user wants to do, register the `tenstorrent/skills` plugin
marketplace with Claude Code, install the plugins for that goal, and hand the
terminal over to `claude` with those plugins loaded.

tt never installs plugins the user did not ask for: choosing a goal is the consent
for exactly the plugins that goal lists, and `--dry-run` shows the whole plan
without touching anything.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import typer
from rich.table import Table

from .._compat import IntRange, confirm, prompt
from ..cli import JsonFlag, NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import AppContext, get_app_context
from ..errors import ExitCode, TTError

# The Claude Code plugin marketplace. MARKETPLACE_NAME is the `name` field of the
# repo's .claude-plugin/marketplace.json — it is what `claude plugin install
# <plugin>@<name>` keys on, and it is fixed by the repo, not by tt.
MARKETPLACE_NAME = "tenstorrent-skills"
DEFAULT_MARKETPLACE_SOURCE = "tenstorrent/skills"
MARKETPLACE_SOURCE_ENV = "TT_AGENT_MARKETPLACE"

CLAUDE_BINARY = "claude"
# The native installer from https://code.claude.com/docs/en/setup (macOS, Linux, WSL).
CLAUDE_INSTALL_COMMAND = "curl -fsSL https://claude.ai/install.sh | bash"
CLAUDE_INSTALL_DOCS = "https://code.claude.com/docs/en/setup"
# Where the native installer puts the launcher; a fresh install is not on PATH until
# the user opens a new shell, so tt looks here too.
_NATIVE_INSTALL_BIN = Path("~/.local/bin/claude")


@dataclasses.dataclass(frozen=True)
class Goal:
    key: str  # what the user types: `tt agent deploy`
    title: str  # the picker line
    plugins: tuple[str, ...]  # install order matters: dependencies first
    why: str  # one line on what the plugins bring


# Order matters: it is the picker numbering.
GOALS: dict[str, Goal] = {
    g.key: g
    for g in (
        Goal(
            "deploy",
            "Deploy a model on your Tenstorrent hardware",
            ("tt-serve-model",),
            "host discovery and deployment readiness for inference on this machine",
        ),
        Goal(
            "bringup",
            "Bring up a new model",
            # tt-model-bringup declares tt-autodebug as a required dependency.
            ("tt-autodebug", "tt-model-bringup"),
            "staged TTNN model bring-up, plus the AutoDebug skills it requires",
        ),
        Goal(
            "develop",
            "Actively develop",
            ("tt-skills", "tt-review-skills", "tt-autodebug"),
            "the skill finder, TT code review, and AutoDebug/AutoTriage/AutoFix",
        ),
    )
}
# Letters as shown in the picker, so `tt agent a` works like answering "1".
_LETTERS = {letter: key for letter, key in zip("abc", GOALS)}


def _stdin_isatty() -> bool:  # test seam: CliRunner swaps sys.stdin during invoke
    return sys.stdin.isatty()


def _resolve_goal(value: str) -> Goal:
    text = value.strip().lower()
    if text in GOALS:
        return GOALS[text]
    if text in _LETTERS:
        return GOALS[_LETTERS[text]]
    if text.isdigit() and 1 <= int(text) <= len(GOALS):
        return list(GOALS.values())[int(text) - 1]
    raise TTError(
        f"Unknown goal {value!r}.",
        why="`tt agent` sets up Claude Code for one of a fixed set of goals.",
        next_step=f"Pick one of: {', '.join(GOALS)} (or run `tt agent` to choose interactively).",
        exit_code=ExitCode.USAGE,
    )


def _pick_goal(appctx: AppContext) -> Goal:
    goals = list(GOALS.values())
    appctx.output.status("What are you looking to do today?")
    for letter, goal in zip("abc", goals):
        appctx.output.status(f"  {letter}. {goal.title}")
    # err=True keeps the prompt on stderr: stdout stays pure data.
    choice = prompt("Choice", default=1, type=IntRange(1, len(goals)), err=True)
    return goals[choice - 1]


def marketplace_source(appctx: AppContext) -> str:
    """Where the marketplace comes from: env, then config, then the GitHub repo.

    A local checkout path is what you want while developing plugins; the default is
    the published repo (`owner/repo`, which `claude plugin marketplace add` resolves
    to GitHub).
    """
    override = os.environ.get(MARKETPLACE_SOURCE_ENV)
    if override:
        return override
    configured = appctx.config.get("agent.marketplace_source")
    return str(configured) if configured else DEFAULT_MARKETPLACE_SOURCE


# -- Claude Code -------------------------------------------------------------------
def find_claude(appctx: AppContext) -> str | None:
    """An installed Claude Code: TT_TOOL_BIN_CLAUDE → tools.override.claude → PATH →
    the native installer's launcher. None if there is none."""
    override = os.environ.get("TT_TOOL_BIN_CLAUDE") or appctx.config.get(
        "tools.override.claude"
    )
    if override:
        return str(override)
    found = shutil.which(CLAUDE_BINARY)
    if found:
        return found
    native = _NATIVE_INSTALL_BIN.expanduser()
    if native.exists() and os.access(native, os.X_OK):
        return str(native)
    return None


def _missing_claude_error(next_step: str) -> TTError:
    return TTError(
        "Claude Code is not installed.",
        why="`tt agent` sets up and launches Claude Code; it needs the `claude` binary.",
        next_step=next_step,
        exit_code=ExitCode.TOOL_MISSING,
        details={"tool": CLAUDE_BINARY, "install_command": CLAUDE_INSTALL_COMMAND},
    )


def ensure_claude(appctx: AppContext, *, yes: bool, dry_run: bool) -> str:
    """The `claude` executable, installing Claude Code first if the user agrees."""
    found = find_claude(appctx)
    if found:
        return found
    manual = (
        f"Install it with `{CLAUDE_INSTALL_COMMAND}` (see {CLAUDE_INSTALL_DOCS}), "
        "then re-run `tt agent`."
    )
    if dry_run:
        appctx.output.status("Claude Code is not installed; `tt agent` would offer to run:")
        appctx.output.status(f"  {CLAUDE_INSTALL_COMMAND}", soft_wrap=True)
        return CLAUDE_BINARY
    if appctx.offline:
        raise TTError(
            "Claude Code is not installed, and --offline forbids downloading it.",
            next_step=manual,
            exit_code=ExitCode.OFFLINE,
        )
    interactive = _stdin_isatty() and not appctx.output.json_mode and not appctx.output.quiet
    if not yes:
        if not interactive:
            raise _missing_claude_error(manual)
        appctx.output.status(
            "Claude Code is not installed. The recommended installer from the Claude Code "
            f"documentation ({CLAUDE_INSTALL_DOCS}) is:"
        )
        appctx.output.status(f"  {CLAUDE_INSTALL_COMMAND}", style="bold", soft_wrap=True)
        appctx.output.status(
            "It downloads the `claude` launcher into ~/.local/bin (no sudo needed)."
        )
        if not confirm("Run it now?"):
            raise _missing_claude_error(manual)
    appctx.output.status("Installing Claude Code …")
    appctx.runner.stream(["bash", "-c", CLAUDE_INSTALL_COMMAND], tool="claude-installer")
    found = find_claude(appctx)
    if not found:
        raise TTError(
            "The installer finished, but `claude` was not found.",
            why="Look at the installer's output above for what went wrong.",
            next_step=f"Open a new shell and run `claude --version`; see {CLAUDE_INSTALL_DOCS}.",
            exit_code=ExitCode.TOOL_MISSING,
            details={"tool": CLAUDE_BINARY},
        )
    return found


# -- plugins --------------------------------------------------------------------------
def _configured_marketplaces(appctx: AppContext, claude: str) -> dict[str, dict]:
    """Marketplaces Claude Code already knows, by name. Empty if the listing fails —
    the add below then reports the real problem."""
    result = appctx.runner.capture(
        [claude, "plugin", "marketplace", "list", "--json"], check=False, tool=CLAUDE_BINARY
    )
    if result.returncode != 0:
        appctx.output.debug(f"claude plugin marketplace list failed: {result.stderr.strip()}")
        return {}
    try:
        entries = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return {}
    return {e["name"]: e for e in entries if isinstance(e, dict) and "name" in e}


def _marketplace_location(entry: dict) -> str | None:
    """The source a listed marketplace was added from, in the form the user gave."""
    return entry.get("repo") or entry.get("path") or entry.get("url")


def _plan_steps(claude: str, source: str, registered: dict | None, plugins: tuple[str, ...]) -> list[list[str]]:
    """The claude commands `tt agent` would run, in order."""
    if registered is None:
        steps = [[claude, "plugin", "marketplace", "add", source]]
    else:
        steps = [[claude, "plugin", "marketplace", "update", MARKETPLACE_NAME]]
    steps.extend([claude, "plugin", "install", f"{name}@{MARKETPLACE_NAME}"] for name in plugins)
    return steps


def install_plugins(appctx: AppContext, claude: str, goal: Goal) -> list[list[str]]:
    """Register the marketplace (or refresh it) and install the goal's plugins.
    Returns the commands that ran."""
    source = marketplace_source(appctx)
    registered = _configured_marketplaces(appctx, claude).get(MARKETPLACE_NAME)
    if registered is not None:
        have = _marketplace_location(registered)
        if have and have != source:
            appctx.output.warn(
                f"Claude Code already has a marketplace named {MARKETPLACE_NAME!r} from "
                f"{have}, not {source}; using the existing one. To switch: "
                f"`claude plugin marketplace remove {MARKETPLACE_NAME}` and re-run."
            )
    steps = _plan_steps(claude, source, registered, goal.plugins)
    if appctx.offline and registered is None and not Path(source).expanduser().is_dir():
        raise TTError(
            f"The {MARKETPLACE_NAME} marketplace is not registered with Claude Code, and "
            "--offline forbids fetching it.",
            next_step="Re-run without --offline, or point agent.marketplace_source at a local checkout.",
            exit_code=ExitCode.OFFLINE,
        )
    for argv in steps:
        appctx.output.status(f"$ {shlex.join(argv)}", style="dim", soft_wrap=True)
        # `marketplace update` is a refresh, not a requirement: offline or flaky
        # network must not stop an install that the local copy can satisfy.
        refresh = argv[2:4] == ["marketplace", "update"]
        if refresh and appctx.offline:
            appctx.output.status("  (skipped: --offline)")
            continue
        code = appctx.runner.stream(argv, check=not refresh, tool=CLAUDE_BINARY)
        if refresh and code != 0:
            appctx.output.warn("could not refresh the marketplace; installing from the local copy.")
    return steps


# -- command -----------------------------------------------------------------------------
def _render_plan(payload: dict) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Goal", f"{payload['goal']}: {payload['title']}")
    table.add_row("Marketplace", f"{payload['marketplace']} ({payload['marketplace_source']})")
    table.add_row("Plugins", ", ".join(payload["plugins"]))
    table.add_row("Claude Code", payload["claude"] or "not installed")
    return table


@handle_tt_errors
def agent(
    ctx: typer.Context,
    goal: str = typer.Argument(
        None,
        metavar="[GOAL]",
        help="What you want to do: " + ", ".join(GOALS) + ". Omit to pick interactively.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be installed and launched; change nothing."
    ),
    no_launch: bool = typer.Option(
        False, "--no-launch", help="Install the plugins but do not start Claude Code."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Install Claude Code without asking if it is missing."
    ),
    json_mode: JsonFlag = False,
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Set up Claude Code with Tenstorrent skills for today's task, then launch it.

    Checks that Claude Code is installed, offering the documented installer if not.
    Asks what you are looking to do, installs the matching plugins from the
    tenstorrent/skills marketplace, and hands this terminal to `claude`.
    Anything after `--` is passed to `claude` unchanged.
    --json prints the plan and the commands run, and never launches.
    """
    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)
    extra = list(ctx.args)

    claude = ensure_claude(appctx, yes=yes, dry_run=dry_run)

    if goal is None:
        if appctx.output.json_mode or appctx.output.quiet or not _stdin_isatty():
            raise TTError(
                "Pick what you want to do.",
                why="The interactive picker needs a terminal and is disabled with --json/--quiet.",
                next_step=f"Re-run as `tt agent <goal>` with one of: {', '.join(GOALS)}.",
                exit_code=ExitCode.USAGE,
            )
        chosen = _pick_goal(appctx)
    else:
        chosen = _resolve_goal(goal)
    # Normalize to the canonical key (deploy/bringup/develop) regardless of how it was
    # supplied -- letter, number, or the interactive picker, none of which leave the
    # raw `goal` param holding a telemetry-safe value otherwise. Telemetry reads
    # ctx.params after this function returns (see telemetry/attributes.py's "agent"
    # entry), so this is what it sees.
    ctx.params["goal"] = chosen.key

    source = marketplace_source(appctx)
    payload = {
        "goal": chosen.key,
        "title": chosen.title,
        "plugins": list(chosen.plugins),
        "marketplace": MARKETPLACE_NAME,
        "marketplace_source": source,
        "claude": claude if find_claude(appctx) else None,
        "dry_run": dry_run,
        # The hand-off command; None when tt does not start claude itself.
        "launch": None,
    }

    appctx.output.status(f"\n{chosen.title}: loading {chosen.why}.")
    if dry_run:
        registered = _configured_marketplaces(appctx, claude).get(MARKETPLACE_NAME) if payload["claude"] else None
        payload["steps"] = [shlex.join(s) for s in _plan_steps(claude, source, registered, chosen.plugins)]
        payload["launch"] = shlex.join([claude, *extra])
        appctx.output.emit(payload, renderer=_render_plan)
        appctx.output.status("\nWould run:")
        for step in payload["steps"]:
            appctx.output.status(f"  $ {step}", soft_wrap=True)
        appctx.output.status(f"  $ {payload['launch']}", soft_wrap=True)
        return

    steps = install_plugins(appctx, claude, chosen)
    payload["steps"] = [shlex.join(s) for s in steps]

    if appctx.output.json_mode or no_launch:
        appctx.output.emit(payload, renderer=_render_plan)
        if no_launch:
            appctx.output.status(
                f"\nPlugins are installed. Start Claude Code with `{shlex.join([CLAUDE_BINARY, *extra])}`.",
                soft_wrap=True,
            )
        return

    appctx.output.status(f"\nStarting Claude Code with {', '.join(chosen.plugins)} …")
    appctx.runner.exec_tty([claude, *extra])
