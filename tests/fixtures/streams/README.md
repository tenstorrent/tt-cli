# Captured tool output

Real output from real tools, kept verbatim as parser fixtures. The wording a
parser has to survive is the wording the tool actually prints — a mocked stream
only teaches it the wording we imagined.

| File | Captured from |
|---|---|
| `uv_pip_install.txt` | `uv pip install --no-cache numpy httpx` (uv 0.9, 2026-09-10) — the cold path, with `Downloading`/`Downloaded` lines |
| `uv_pip_cached.txt` | the same install with a warm cache — no download lines at all |
| `git_clone.txt` | `git clone --progress --depth 1` of this repo (git 2.43, 2026-09-10), carriage returns converted to newlines |

Re-capture with `tee` if a tool's output format changes; don't hand-edit these.
