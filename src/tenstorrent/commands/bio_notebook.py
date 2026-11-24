# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""The notebook `tt bio notebook` seeds a fresh working directory with.

Built in Python rather than shipped as a .ipynb so the chips the run was actually
granted can be written into it, and so it is testable as data instead of a blob.

It drives the `tt-bio` CLI rather than importing a fold function: tt-bio's Python
surface is `esmc`, `saprot`, `rfd4` and friends, and there is no public `predict`
entry point to depend on. Shelling out is also the honest demo — it is the command
the reader will use again tomorrow, and the notebook shows it rather than hiding it.
"""

from __future__ import annotations

import json

STARTER_NAME = "start-here.ipynb"

# A 20-residue Trp-cage with `msa: empty`: the smallest thing that is still a real
# fold. No MSA server is contacted, so the first cell a new user runs cannot fail
# on network egress.
TRPCAGE_YAML = """version: 1
sequences:
  - protein:
      id: A
      sequence: NLYIQWLKDGGPSSGRPPPS
      msa: empty
"""


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def starter_cells(chips: list[int] | None) -> list[dict]:
    """The cells of the seeded notebook, in order."""
    if chips:
        granted = (
            f"This notebook was started with **chip(s) {', '.join(map(str, chips))}** "
            "reserved for it — the chips no served model was holding."
        )
    else:
        granted = (
            "No chip grant was recorded for this notebook, so tt-bio will choose for "
            "itself. If something else on this box is serving a model, stop it first "
            "or pass `--devices` below."
        )

    return [
        _md(
            "# Fold a protein on Tenstorrent\n"
            "\n"
            f"{granted}\n"
            "\n"
            "Run the cells in order. The first fold compiles kernels, so it takes a few\n"
            "minutes; later folds in this session are much faster.\n"
        ),
        _md("## 1. What am I running on?\n"),
        _code(
            "import os, subprocess\n"
            "\n"
            'grant = os.environ.get("TT_BIO_LEASE_CARDS") or "(none recorded)"\n'
            'print("chips reserved for this notebook:", grant)\n'
            'print(subprocess.run(["tt-bio", "--version"], capture_output=True, text=True).stdout.strip())\n'
        ),
        _md(
            "## 2. The thing to fold\n"
            "\n"
            "Trp-cage: 20 residues, one of the smallest proteins that folds on its own.\n"
            "**Change the sequence and re-run from here** — that is the point of a notebook.\n"
        ),
        _code(
            "from pathlib import Path\n"
            "\n"
            'target = Path("target.yaml")\n'
            f"target.write_text('''{TRPCAGE_YAML}''')\n"
            "print(target.read_text())\n"
        ),
        _md(
            "## 3. Fold it\n"
            "\n"
            "This is exactly what `tt bio predict` runs for you from a terminal.\n"
        ),
        _code(
            "import subprocess, sys\n"
            "\n"
            "proc = subprocess.Popen(\n"
            '    ["tt-bio", "predict", "target.yaml", "--model", "boltz2",\n'
            '     "--out_dir", "out", "--override"],\n'
            "    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,\n"
            ")\n"
            "for line in proc.stdout:\n"
            "    print(line, end='')\n"
            "rc = proc.wait()\n"
            'print("\\nexit", rc)\n'
        ),
        _md("## 4. How confident is it?\n"),
        _code(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            'results = sorted(Path("out").glob("*/results.json"))\n'
            "if not results:\n"
            '    print("No results.json yet — did the cell above finish?")\n'
            "else:\n"
            "    data = json.loads(results[0].read_text())\n"
            "    print(json.dumps(data, indent=2)[:1200])\n"
        ),
        _md(
            "## 5. Look at it\n"
            "\n"
            "Colour runs blue (confident) to red (not). Drag to rotate.\n"
        ),
        _code(
            "import py3Dmol\n"
            "from pathlib import Path\n"
            "\n"
            'cifs = sorted(Path("out").glob("*/structures/*.cif"))\n'
            "if not cifs:\n"
            '    print("No structure written yet.")\n'
            "else:\n"
            "    view = py3Dmol.view(width=700, height=500)\n"
            '    view.addModel(cifs[0].read_text(), "cif")\n'
            "    # b-factor carries pLDDT, so this colours by per-residue confidence.\n"
            '    view.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",\n'
            '                                               "min": 0, "max": 1}}})\n'
            "    view.zoomTo()\n"
            "    display(view.show())\n"
        ),
        _md(
            "## Where to go next\n"
            "\n"
            "- `tt-bio predict --help` — every option (samples, recycles, `--fast`, PAE output)\n"
            "- Other models: `--model esmfold2` needs no MSA, `--model protenix-v1` is\n"
            "  AlphaFold3-family. `tt bio preflight <model>` says whether one is ready here.\n"
            "- Binding affinity: add a ligand and `properties: affinity`, or try\n"
            "  `tt-bio affinity` for a score without folding.\n"
            "- Design a binder against a target: `tt-bio design --model boltzgen`.\n"
            "- Back in a terminal, `tt bio status` shows which chips are free right now.\n"
        ),
    ]


def starter_notebook(chips: list[int] | None) -> dict:
    """A complete .ipynb document, ready to json.dump."""
    # nbformat 4.5 and up require a cell id, and warn today that the absence will
    # become a hard error. Assigned here, positionally and deterministically, so two
    # runs produce the same file and a diff of it stays readable.
    cells = starter_cells(chips)
    for index, cell in enumerate(cells):
        cell["id"] = f"tt-bio-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_starter(path, chips: list[int] | None) -> bool:
    """Write the starter notebook unless something is already there.

    Returns whether it wrote. Never overwrites: the file is the user's the moment
    they open it, and clobbering an edited notebook on every launch would be the
    one unforgivable thing this command could do.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(starter_notebook(chips), indent=1) + "\n")
    return True
