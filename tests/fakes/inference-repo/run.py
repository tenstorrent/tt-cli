# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

#!/usr/bin/env python3
"""Fake tt-inference-server run.py: records argv, prints a listening banner."""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow", default="server")
    parser.add_argument("--device", default=None)
    parser.add_argument("--docker-server", action="store_true")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--host-hf-cache", default=None)
    # SERVICE_PORT in the real run.py; the docker path publishes it as
    # <bind_host>:<service_port>:8000.
    parser.add_argument("--service-port", default=None)
    # Launch settings tt forwards from the model support list. Only accepted here
    # so argparse does not reject them — the assertions read the recorded argv.
    parser.add_argument("--vllm-override-args", default=None)
    parser.add_argument("--override-docker-image", default=None)
    parser.add_argument("--override-tt-config", default=None)
    args = parser.parse_args()
    log = os.environ.get("FAKE_INFERENCE_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\n")
    cwd_log = os.environ.get("FAKE_INFERENCE_CWD_LOG")
    if cwd_log:
        with open(cwd_log, "a") as fh:
            fh.write(os.getcwd() + "\n")
    if os.environ.get("FAKE_INFERENCE_FAIL"):
        print("fake inference server: model load failed", file=sys.stderr)
        return 1
    print(f"fake inference server: {args.model} workflow={args.workflow}")
    print("listening on http://0.0.0.0:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
