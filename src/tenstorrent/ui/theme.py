# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""One palette and one glyph set, used by name everywhere in the UI layer.

Styles are referenced as markup (`[success]✓[/success]`) or `style=` so a theme
change is one edit here, not a grep across the tree.
"""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
        # Tenstorrent purple. "accent.bold" exists because Rich cannot resolve a
        # modifier stacked on a theme name ("[bold accent]").
        "accent": "color(99)",
        "accent.bold": "bold color(99)",
    }
)

# Braille spinner: one frame per tick, same set TT-Studio uses.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

GLYPH_DONE = "✓"
GLYPH_FAIL = "✗"
GLYPH_SKIP = "○"
GLYPH_ACTIVE = "◉"
GLYPH_PENDING = "○"
GLYPH_ALERT = "!"

# Pinned so a terminal resize cannot reflow a card mid-render.
PANEL_WIDTH = 78

# Steps faster than this don't get an elapsed suffix — it's noise on an
# operation the user never waited for.
ELAPSED_THRESHOLD_S = 0.8
