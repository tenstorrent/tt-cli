# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""`tt bio` — biomolecular prediction, design and embeddings, on chips nothing else holds.

tt-bio is its own CLI with its own live display, and this does not reimplement it:
every run is handed the terminal with `exec_tty`, and flags tt does not itself read
are forwarded verbatim, so a tt-bio release that adds an option works here the day
it ships.

What tt adds is the one thing tt-bio cannot see for itself — **who else is using the
card**. tt-bio arbitrates device opens with an flock on a per-card lease file
(`tt_bio/device_lease.py`); tt-cli tracks chips through the devices a *container*
mounts (`backends/serving/chips.py`). Those are disjoint mechanisms: a model served
by `tt serve` holds chip 2 without ever taking tt-bio's lease, so an unguarded
`tt-bio predict` opens that chip underneath a running server.

So before handing over, tt computes the chips no container holds and passes them
both ways:

* ``--devices <free>`` — the advice tt-bio's scheduler places work with.
* ``TT_BIO_LEASE_CARDS=<free>`` — the *grant*. tt-bio refuses an open outside that
  set at its own device-open choke point, so a run that selects a card by discovery
  (which is how a Galaxy box went down upstream) is stopped by tt-bio itself rather
  than trusted to behave.

A chip tt could not check is never handed out: when the container runtime cannot be
reached the grant is simply not set and tt-bio behaves exactly as it does standalone.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from ..backends.serving import chips
from ..cli import NoColorFlag, QuietFlag, VerboseFlag, handle_tt_errors
from ..context import AppContext, get_app_context
from ..errors import ExitCode, TTError

TOOL = "tt-bio"

# Unknown options belong to tt-bio, not to tt: the wrapper must not go stale every
# time upstream adds a flag. Same contract `tt serve` uses for bundle ids.
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}

bio_app = typer.Typer(
    help="Protein structure prediction, binder design, and embeddings (tt-bio).",
    no_args_is_help=True,
)

# The commands that open a card, and so need a chip grant. tt-bio's other commands
# (weights, preflight, score, msa) never open one and are forwarded untouched.
_DEVICE_COMMANDS = frozenset(
    {"predict", "design", "embed", "affinity", "saprot", "controller", "worker", "train", "warmup"}
)

# How tt-bio names the two spellings of the same thing. `--devices` is the short
# alias upstream added for `tt-bio embed`; `--device_ids` is the original.
_DEVICE_FLAGS = ("--devices", "--device_ids", "--num_devices")

# tt-bio reads its grant from this: an open outside the set is refused inside tt-bio,
# at the one point every device open passes through.
GRANT_ENV = "TT_BIO_LEASE_CARDS"


def _runtime() -> str | None:
    """docker or podman, or None. Unlike the serving backends this never raises:
    a host with no container runtime has no containers holding chips either, which
    is a perfectly good answer, not an error."""
    return shutil.which("docker") or shutil.which("podman")


def _free_chips(appctx: AppContext) -> list[int] | None:
    """Chips no running container holds. None means 'could not determine'.

    None and [] are different answers and the callers treat them differently: None
    leaves tt-bio unconstrained (we have nothing trustworthy to say), [] is a
    definite 'every chip is busy' and stops the run.
    """
    ids = chips.all_ids()
    if not ids:
        return None
    runtime = _runtime()
    if runtime is None:
        return None
    return chips.free(appctx.runner, runtime, ids)


def _requested_devices(args: list[str]) -> str | None:
    """The value of a device flag the user passed, if any.

    Both `--devices 0,1` and `--devices=0,1` are honoured, because tt-bio's click
    parser takes either and a wrapper that only understood one would silently drop
    the user's choice and substitute its own.
    """
    for index, arg in enumerate(args):
        for flag in _DEVICE_FLAGS:
            if arg == flag and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
    return None


def _parse_ids(value: str) -> list[int] | None:
    """`0,2` → [0, 2]. None when it is not a plain id list — tt-bio also accepts PCI
    BDFs, and tt has no business second-guessing those."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def _busy_error(held: list[int], free_now: list[int]) -> TTError:
    held_text = ", ".join(map(str, sorted(held)))
    return TTError(
        f"Chip(s) {held_text} are in use by a running model server or another tt-bio run.",
        why="tt-bio opens a card directly, and a container holding one does not take "
        "tt-bio's lease — so this run would collide with a server rather than queue "
        "behind it.",
        next_step="`tt model ps` to see what is running, `tt model stop <model>` to free "
        + (
            f"them, or re-run restricted to the free chips: --devices {','.join(map(str, free_now))}"
            if free_now
            else "them, then re-run."
        ),
        exit_code=ExitCode.TOOL_FAILED,
        reason="bio.device.in_use",
        details={"held": sorted(held), "free": free_now},
    )


def _grant(
    appctx: AppContext, args: list[str], *, require_free: bool = True
) -> tuple[list[str], dict[str, str]]:
    """Resolve the chip grant for a device-opening run.

    Returns the argv to forward and the environment to add. Either may be empty:
    an unconstrained run is a legitimate outcome (no cards, no runtime), not a
    failure, and tt-bio is perfectly usable without tt's help.
    """
    free_now = _free_chips(appctx)
    if free_now is None:
        return args, {}

    requested = _requested_devices(args)
    if requested is not None:
        wanted = _parse_ids(requested)
        if wanted is None:
            # A BDF or some other spelling tt does not parse. Forward it untouched
            # rather than guess; tt-bio's own lease still protects the open.
            return args, {}
        held = [chip for chip in wanted if chip not in free_now]
        if held:
            raise _busy_error(held, free_now)
        # The user's choice is already free: grant exactly that, no wider.
        return args, {GRANT_ENV: ",".join(str(chip) for chip in wanted)}

    if not free_now:
        if require_free:
            raise _busy_error(sorted(chips.all_ids()), [])
        # A long-lived session (the notebook) is not a single job: it is worth opening
        # with nothing free, to read it, edit a sequence or look at an earlier result,
        # and a chip may come free while it is open. The fold cell fails honestly on
        # tt-bio's own lease if one never does.
        appctx.output.warn(
            "Every chip is in use, so a fold will not run yet — "
            "`tt model ps` shows what is holding them."
        )
        # Deliberately no grant: tt-bio reads an *empty* TT_BIO_LEASE_CARDS as
        # unbounded, so writing "" here would hand over the whole box.
        return args, {}

    csv = ",".join(str(chip) for chip in free_now)
    appctx.output.ui.note(
        f"Running on chip(s) {csv} — the chips no served model is holding."
        if len(free_now) < len(chips.all_ids())
        else f"Running on chip(s) {csv}."
    )
    return [*args, "--devices", csv], {GRANT_ENV: csv}


def _jupyter_beside(appctx: AppContext, binary: Path) -> Path | None:
    """Where the notebook server is, given where tt-bio is. None if it is not there.

    Two places, because `uv tool install --with` puts a package *in* the tool's venv
    without linking its entry points into UV_TOOL_BIN_DIR: only `tt-bio` is symlinked
    there, while `jupyter` stays in the venv's own bin. So the managed layout is
    checked explicitly, and `beside the binary` covers a tt-bio resolved through
    TT_TOOL_BIN_TT_BIO or a config override, where the user owns the environment.
    """
    candidates = [
        Path(binary).parent / "jupyter",
        appctx.paths.tools_dir / TOOL / "bin" / "jupyter",
    ]
    return next((c for c in candidates if c.exists()), None)


def _hand_off(ctx: typer.Context, command: str, extra: list[str] | None = None) -> None:
    """Resolve tt-bio (installing it at the pin on first use) and replace this process."""
    appctx = get_app_context(ctx)
    args = [*(extra or []), *ctx.args]
    env: dict[str, str] = {}
    if command in _DEVICE_COMMANDS:
        args, env = _grant(appctx, args)
    binary = appctx.registry.ensure(TOOL, offline=appctx.offline)
    appctx.runner.exec_tty([str(binary), command, *args], env=env)


def _passthrough(name: str, summary: str):
    """Register one forwarding subcommand. Declared in a loop because every one of
    them is the same four lines, and a typo in the eleventh is a silent wrong flag."""

    @bio_app.command(
        name,
        context_settings=PASSTHROUGH,
        no_args_is_help=False,
        help=f"{summary}\n\nArguments and options are passed to `tt-bio {name}`.",
    )
    @handle_tt_errors
    def _command(ctx: typer.Context) -> None:
        _hand_off(ctx, name)

    return _command


for _name, _summary in (
    ("predict", "Fold a structure: protein, DNA, RNA and ligand complexes."),
    ("design", "Design binders against a target."),
    ("embed", "Protein language-model embeddings (ESMC)."),
    ("affinity", "Protein-ligand binding affinity without folding (Nesso-1)."),
    ("saprot", "Structure-aware embeddings and variant-effect scoring (SaProt)."),
    ("controller", "Run a persistent controller so models stay resident between calls."),
    ("weights", "List, download or prune model weights."),
    ("preflight", "Can this machine run a model right now?"),
    ("score", "Interface scores of a fold you already have."),
    ("msa", "Download the offline MSA database."),
):
    _passthrough(_name, _summary)


@bio_app.command("run", context_settings=PASSTHROUGH, no_args_is_help=True)
@handle_tt_errors
def run(ctx: typer.Context) -> None:
    """Run any other `tt-bio` command, forwarded verbatim.

    The escape hatch: a tt-bio subcommand this version of tt does not list by name
    still reaches it, so a new upstream command is never gated on a tt release.
    """
    # no_args_is_help means bare `tt bio run` has already printed help and exited,
    # so ctx.args is never empty here (same contract as the other leaf commands).
    appctx = get_app_context(ctx)
    command, *rest = ctx.args
    args, env = (rest, {})
    if command in _DEVICE_COMMANDS:
        args, env = _grant(appctx, rest)
    binary = appctx.registry.ensure(TOOL, offline=appctx.offline)
    appctx.runner.exec_tty([str(binary), command, *args], env=env)


@bio_app.command("notebook")
@handle_tt_errors
def notebook(
    ctx: typer.Context,
    directory: str = typer.Option(
        None,
        "--dir",
        "-d",
        help="Where to keep notebooks (default: ~/tt-bio-notebooks).",
    ),
    port: int = typer.Option(8888, "--port", help="Port for the notebook server."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open a browser; just print the URL."
    ),
    json_mode: bool = typer.Option(
        False,
        "--json",
        help="Report the notebook, chips and port as JSON and exit WITHOUT opening "
        "the server (handing over the terminal and emitting clean JSON on it are "
        "mutually exclusive).",
    ),
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Open a notebook that folds a protein, on chips nothing else is holding.

    The first thing a new user should run. `tt bio predict` assumes you already know
    what you want to fold; this hands you a worked example you can edit and re-run,
    which is how people actually explore these models.

    JupyterLab and py3Dmol live in tt-bio's own environment (see supplement.toml), so
    the notebook can import tt_bio and render a structure from one interpreter.
    """
    from .bio_notebook import STARTER_NAME, write_starter

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)

    workdir = Path(directory).expanduser() if directory else Path.home() / "tt-bio-notebooks"

    # The grant first: it is written into the notebook, so it has to be settled before
    # the file is created, and a box with nothing free should fail here rather than
    # after starting a server.
    _args, env = _grant(appctx, [], require_free=False)
    chips = [int(c) for c in env[GRANT_ENV].split(",")] if env.get(GRANT_ENV) else None

    binary = appctx.registry.ensure(TOOL, offline=appctx.offline)
    jupyter = _jupyter_beside(appctx, binary)
    if jupyter is None:
        # tt-bio installed before the notebook extras were pinned. The version did not
        # change, so ensure() had nothing to re-resolve and returned the old venv —
        # install() re-runs uv at the same pin, which does pick the new `deps` up.
        # Self-healing on purpose: the alternative is telling someone to run a command
        # that, for this exact reason, would not fix it either.
        appctx.output.ui.note("Adding the notebook environment to tt-bio (one time).")
        binary = appctx.registry.install(appctx.registry.spec(TOOL), offline=appctx.offline).path
        jupyter = _jupyter_beside(appctx, binary)
    if jupyter is None:
        raise TTError(
            "tt-bio is installed, but the notebook server is not beside it.",
            why="The `deps` on the tt-bio tool in supplement.toml should have put "
            "jupyterlab in its environment.",
            next_step="Report this with `tt report issue` — the install reported success.",
            exit_code=ExitCode.TOOL_MISSING,
            details={"tool": TOOL},
        )

    starter = workdir / STARTER_NAME
    created = write_starter(starter, chips)

    if json_mode:
        appctx.output.emit(
            {
                "directory": str(workdir),
                "notebook": str(starter),
                "created": created,
                "chips": chips,
                "port": port,
            }
        )
        return

    appctx.output.ui.note(
        f"{'Wrote' if created else 'Keeping'} {starter}"
        + ("" if created else " (already there, left as you had it)")
    )

    argv = [
        str(jupyter),
        "lab",
        f"--notebook-dir={workdir}",
        f"--port={port}",
        f"--ServerApp.default_url=/lab/tree/{STARTER_NAME}",
    ]
    if no_browser:
        argv.append("--no-browser")
    appctx.runner.exec_tty(argv, env=env)


@bio_app.command("status")
@handle_tt_errors
def status(
    ctx: typer.Context,
    json_mode: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    quiet: QuietFlag = False,
    verbose: VerboseFlag = False,
    no_color: NoColorFlag = False,
) -> None:
    """Which chips a tt-bio run would get, and whether tt-bio is installed."""
    from rich.table import Table

    appctx = get_app_context(ctx)
    appctx.output.apply_flags(json_mode=json_mode, quiet=quiet, verbose=verbose, no_color=no_color)

    spec = appctx.registry.spec(TOOL)
    try:
        installed: str | None = str(appctx.registry.resolve(TOOL))
    except TTError:
        installed = None

    ids = chips.all_ids()
    free_now = _free_chips(appctx)
    payload = {
        "tool": {
            "name": TOOL,
            "pinned_version": spec.golden_version,
            "path": installed,
            "installed": installed is not None,
        },
        "chips": {
            "present": ids,
            "free": free_now,
            "held": None if free_now is None else [c for c in ids if c not in free_now],
        },
        "grant_env": GRANT_ENV,
    }

    def render(data: dict) -> Table:
        table = Table(title="tt bio")
        table.add_column("")
        table.add_column("")
        tool = data["tool"]
        table.add_row("tt-bio", tool["path"] or f"not installed (pinned {tool['pinned_version']})")
        chip = data["chips"]
        table.add_row("chips present", ", ".join(map(str, chip["present"])) or "none")
        if chip["free"] is None:
            table.add_row("chips free", "unknown (no container runtime to ask)")
        else:
            table.add_row("chips free", ", ".join(map(str, chip["free"])) or "none")
            table.add_row("chips held", ", ".join(map(str, chip["held"])) or "none")
        return table

    appctx.output.emit(payload, renderer=render)
