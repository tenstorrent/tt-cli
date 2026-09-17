# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

#!/usr/bin/env python3
"""Fake tt-studio run.py: accepts exactly the shapes tt drives (`run MODEL` and
`--stop-model MODEL`, so an argv drift fails loudly), records argv, cwd and the
HF_TOKEN it inherited."""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-model", default=None)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run")
    run.add_argument("model")
    args = parser.parse_args()
    if args.command is None and args.stop_model is None:
        parser.error("expected `run MODEL` or --stop-model MODEL")
    log = os.environ.get("FAKE_STUDIO_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\n")
    cwd_log = os.environ.get("FAKE_STUDIO_CWD_LOG")
    if cwd_log:
        with open(cwd_log, "a") as fh:
            fh.write(os.getcwd() + "\n")
    env_log = os.environ.get("FAKE_STUDIO_ENV_LOG")
    if env_log:
        with open(env_log, "a") as fh:
            fh.write(json.dumps({"HF_TOKEN": os.environ.get("HF_TOKEN")}) + "\n")
    if os.environ.get("FAKE_STUDIO_FAIL"):
        print("fake tt-studio: deploy failed", file=sys.stderr)
        return 1
    if args.stop_model:
        print(f"fake tt-studio: stopped {args.stop_model}")
    else:
        print(f"fake tt-studio: deployed {args.model} at http://localhost:7001/v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
