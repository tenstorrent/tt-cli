# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Keeping `tt` itself current.

Two separate problems, deliberately kept apart:

* **Noticing** that a newer tt exists (check.py) is safe everywhere: a once-a-day
  version lookup in a detached process, and a one-line stderr notice on the next
  interactive run. Modeled on gh / update-notifier, never on an in-process fetch.
* **Performing** the upgrade (update.py) is only safe where the evidence proves how tt
  was installed *and* that nothing else lives in its environment (layout.py). In an
  isolated layout (uv tool, pipx, a venv whose only tenant is tt) tt delegates to the
  tool that owns the environment. In a shared venv it prints the exact command and
  leaves the decision to the user: pip and uv resolve only the requested package, so
  an upgrade there can silently break the other tenants' pins.
"""
