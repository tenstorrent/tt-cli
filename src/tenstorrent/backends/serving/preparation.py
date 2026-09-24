# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""What each backend does before the container exists, and how it says so.

The watcher (`boot.py`) is backend-agnostic: it runs a preparation, draws
whatever rows it reports, and takes over the moment it names a container. Only
these adapters know that tt-inference-server says it in prose and tt-model says
it in JSON. See `docs/serve-progress-contract.md`.
"""

from __future__ import annotations

import json
import re
import shutil
from typing import Sequence

from ...progress import HOST_PHASES, Event, PhaseTracker

#: Shared row wording, keyed by step. Taken from the host phase template rather
#: than restated, so the two backends cannot drift apart on a rename.
_SHARED = {phase.key: phase for phase in HOST_PHASES}


class Preparation:
    """A backend's run-up to the container.

    `feed` turns one of its output lines into checklist events; `container` is
    the id or name it started, once known. Implementations must never raise on
    a line they do not understand — a backend is allowed to print anything.
    """

    #: the placeholder row shown until the backend reports something of its own
    label = "preparing"

    def __init__(self) -> None:
        self.container: str | None = None
        #: the port the container actually published, when the backend picks it
        #: rather than being told. None means the caller's own value stands.
        self.endpoint_port: int | None = None
        #: the Hugging Face repo whose weights are coming down *now*, if any.
        #: Neither backend reports bytes through a pipe, so this is what lets
        #: the watcher measure the download itself (see progress/weights.py).
        self.weights_repo: str | None = None

    def feed(self, line: str) -> list[Event]:
        raise NotImplementedError

    def finish(self) -> list[Event]:
        return []

    def evidence(self) -> list[str]:
        return []

    def resolve_container(self, runner, runtime: str) -> str | None:
        """A last look once the backend has exited, for one that never said."""
        return self.container

    def engines(self) -> Sequence[str] | None:
        """Which stack the container runs, if the backend knows better than the
        caller did before it started. None leaves the caller's answer alone."""
        return None


class RunPyPreparation(Preparation):
    """tt-inference-server: prose, scraped with two pinned regexes.

    run.py echoes the `docker run` command it is about to execute (the only
    place the container's name appears) and logs the id on its way out. Both are
    implementation details rather than a published contract; when either changes
    the handoff degrades to "no container to follow" and the serve is otherwise
    unaffected. Re-check on a pin bump, alongside `_BOARDS_TO_DEVICE`.
    """

    label = "starting tt-inference-server"

    # The name comes first and is the friendlier of the two, so it wins.
    _NAME_RE = re.compile(r"--name\s+(tt-inference-server-\S+)")
    _ID_RE = re.compile(r"Created Docker container ID:\s*(\S+)")
    _FETCH_RE = re.compile(r"Downloading model to host (?:HF cache|volume):\s*(\S+)")

    def __init__(self) -> None:
        super().__init__()
        self._tracker = PhaseTracker(HOST_PHASES)

    def feed(self, line: str) -> list[Event]:
        if self.container is None:
            match = self._NAME_RE.search(line) or self._ID_RE.search(line)
            if match:
                self.container = match.group(1)
        fetching = self._FETCH_RE.search(line)
        if fetching:
            self.weights_repo = fetching.group(1)
        events = self._tracker.feed(line)
        current = self._tracker.current
        if current is None or current.key != "weights":
            self.weights_repo = None  # that step is over, whatever came next
        return events

    def finish(self) -> list[Event]:
        return self._tracker.finish()

    def evidence(self) -> list[str]:
        return self._tracker.evidence()


class ModelManagerPreparation(Preparation):
    """tt-model: NDJSON when it offers it, its own step lines when it does not.

    `TT_MODEL_PROGRESS=ndjson` is an ask, not a requirement, so there has to be
    something to show without it. tt-model run with `--verbose` announces each
    step as `<label>…` and settles it as `✓ <label>  <detail>  <duration>`, and
    those become rows here. It is prose, and parsing it is a stopgap — but the
    alternative is one motionless row for the ten minutes a 50 GB bundle takes,
    and the failure mode is a row that reads oddly, never a broken serve.
    Events win wherever both arrive.

    What this cannot recover is byte progress. tt-model computes it and routes
    it to an activity row that is "TTY only; a no-op when piped"
    (console._Activity), so through a pipe the bytes do not exist. That is what
    the contract's `progress` event is for.
    """

    label = "preparing the bundle"

    #: every container tt-model starts carries this label family
    LABEL = "org.tenstorrent.tt-model"

    def __init__(self, repo_id: str) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._tail: list[str] = []
        self._active: str | None = None
        self._labels: dict[str, str] = {}
        self._events = False
        self.diagnosis: dict | None = None

    def feed(self, line: str) -> list[Event]:
        event = _as_event(line)
        if event is not None:
            self._events = True
            return self._consume(event)
        if line.strip():
            self._tail.append(line.strip())
            del self._tail[:-40]
        # Only until the real thing shows up: a stream that has spoken once is
        # authoritative, and reading its rows out of prose as well would double
        # every step.
        return [] if self._events else self._from_prose(line)

    def engines(self) -> Sequence[str] | None:
        """Read from the manifest, which a first-ever serve only has on disk
        once tt-model has pulled the bundle — after we were first asked."""
        from ...modelhub import bundles

        engine = (bundles.serve_details(self._repo_id) or {}).get("engine")
        return [engine] if engine else None

    def _from_prose(self, line: str) -> list[Event]:
        """tt-model's own `--verbose` rows, for a build without the events."""
        text = line.rstrip()
        settled = _RESULT_RE.match(text)
        if settled:
            started, self._active = self._active is not None, None
            self.weights_repo = None
            detail = (settled.group("detail") or "").strip()
            if _DURATION_RE.match(detail):
                detail = ""  # the row already carries tt's own timer
            return _settled(
                settled.group("label").strip(),
                settled.group("mark") != "\u2717",
                detail or None,
                started=started,
            )
        starting = _STARTING_RE.match(text)
        if starting:
            self._active = starting.group("label").strip()
            fetching = _WEIGHTS_ROW_RE.match(self._active)
            self.weights_repo = fetching.group(1) if fetching else None
            return [Event("start", self._active)]
        return []

    def _consume(self, event: dict) -> list[Event]:
        kind = event.get("event")
        if kind == "container":
            self.container = _text(event.get("id")) or self.container
            self.endpoint_port = _port_of(_text(event.get("endpoint"))) or self.endpoint_port
            return []
        if kind == "error":
            self.diagnosis = event
            return []
        if kind == "progress":
            total = _number(event.get("total"))
            if total <= 0:
                return []
            return [
                Event(
                    "progress",
                    done=_number(event.get("done")),
                    total=total,
                    is_bytes=event.get("unit") == "bytes",
                )
            ]
        if kind != "step":
            return []  # forward compatibility: an event we do not know yet
        key = _text(event.get("key"))
        state = event.get("state")
        shared = _SHARED.get(key)
        detail = _text(event.get("detail")) or None
        if state == "start":
            self._active = key
            label = shared.label if shared else _label(event, key)
            self._labels[key] = label
            return [Event("start", label)]
        if state not in ("done", "fail"):
            return []
        # A done label of its own wins, then the shared wording, then whatever
        # the step called itself when it started.
        label = (
            shared.done_label if shared
            else _text(event.get("label")) or self._labels.get(key) or key or "done"
        )
        started, self._active = self._active == key, None
        return _settled(label, state == "done", detail, started=started)

    def finish(self) -> list[Event]:
        """Close a step the backend left open (it exited, or it never said)."""
        if self._active is None:
            return []
        if not self._events:  # a prose row: its label is all we have
            label, self._active = self._active, None
            return [Event("done", label)]
        shared = _SHARED.get(self._active)
        label = shared.done_label if shared else self._labels.get(self._active, self._active)
        self._active = None
        return [Event("done", label)]

    def evidence(self) -> list[str]:
        return list(self._tail)

    def resolve_container(self, runner, runtime: str) -> str | None:
        """Ask docker which container this bundle left running.

        By label, never by parsing tt-model's output — but *not* by the repo
        label: that records the bundle's canonical home, and a bundle pulled
        from a fork or a personal namespace carries the original. The bare
        `org.tenstorrent.tt-model` label is the bundle name, which is the last
        segment of either, so that is what matches.

        `ps -a`, not `ps`: a container that started and died at once is exactly
        the case worth reporting, and following its log is how we find out why.
        """
        if self.container is not None:
            return self.container
        docker = shutil.which("docker")
        if docker is None:
            return None
        listed = runner.capture(
            [docker, "ps", "-a", "--filter", f"label={self.LABEL}", "--format",
             '{{.ID}}\t{{.Label "%s"}}\t{{.Label "%s.repo"}}' % (self.LABEL, self.LABEL)],
            tool="docker", check=False,
        )
        wanted = self._repo_id.rsplit("/", 1)[-1].lower()
        for row in listed.stdout.splitlines():  # docker lists newest first
            fields = row.split("\t")
            if len(fields) < 3:
                continue
            container, name, repo = (field.strip() for field in fields[:3])
            if name.lower() == wanted or repo.lower() == self._repo_id.lower():
                self.container = container
                if self.endpoint_port is None:
                    # --port is honoured exactly, but left to itself tt-model
                    # walks up from 20000 past whatever is busy — so the
                    # container, not the default, says where it listens.
                    self.endpoint_port = _published_port(runner, docker, container)
                return self.container
        return None


#: tt-model's `--verbose` rows (console.step / _render_result). `--verbose` is a
#: global option, so it sits before the subcommand and is never swept into
#: `tt-model serve`'s passthrough to vLLM.
#: Only ✓ and ✗. tt-model marks a *skipped* step with ○ and also uses ○ as the
#: marker for ordinary notes ("… is loaded but is a different image — reloading"),
#: so a ○ row cannot be told from prose.
_RESULT_RE = re.compile(
    r"^\s*(?P<mark>[\u2713\u2717])\s+(?P<label>.+?)(?:\s{2,}(?P<detail>.*))?$"
)
_STARTING_RE = re.compile(r"^\s*(?P<label>[^\s\u2713\u2717\u25cb!\u21bb].*?)\u2026\s*$")
#: tt-model times its own rows too; showing both reads as two different numbers.
#: tt-model labels its weights step `weights <repo>@<revision>`.
_WEIGHTS_ROW_RE = re.compile(r"^weights\s+(\S+?)(?:@\S*)?$")
_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?\s*[smh]$|^\d+\s*[hm]\s*\d+\s*[ms]$")


def _published_port(runner, docker: str, container: str) -> int | None:
    """The lowest host port the container publishes, or None."""
    listed = runner.capture([docker, "port", container], tool="docker", check=False)
    ports = {
        int(match.group(1))
        for match in re.finditer(r":(\d+)\s*$", listed.stdout, re.MULTILINE)
    }
    return min(ports) if ports else None


def _port_of(url: str) -> int | None:
    match = re.search(r"://[^/:]+:(\d+)", url)
    return int(match.group(1)) if match else None


def _settled(label: str, ok: bool, detail: str | None, *, started: bool) -> list[Event]:
    """A finished step, whether or not it was ever announced as starting.

    "This was already true" is a normal thing for a backend to report, and a
    settled row with nothing to settle would otherwise draw nothing at all.
    The row opens and closes in the same breath, so it carries no timer.
    """
    end = Event("done" if ok else "fail", label, detail=detail)
    return [end] if started else [Event("start", label), end]


def _as_event(line: str) -> dict | None:
    """The line as a progress event, or None for ordinary output.

    Cheap guard before json.loads: most lines are not JSON, and a backend is
    free to print a JSON payload of its own that is none of our business.
    """
    text = line.strip()
    if not text.startswith("{") or '"event"' not in text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _label(event: dict, key: str) -> str:
    return _text(event.get("label")) or key or "working"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


__all__: Sequence[str] = ("Preparation", "RunPyPreparation", "ModelManagerPreparation")
