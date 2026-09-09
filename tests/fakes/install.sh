#!/bin/sh
# Fake tt-installer install.sh: records argv, never sudo's, exits 0.
# FAKE_INSTALLER_FAIL=1 forces a failure.
if [ -n "$FAKE_INSTALLER_LOG" ]; then
    printf '%s\n' "$*" >> "$FAKE_INSTALLER_LOG"
    # Sidecar log of the working directory tt ran us from: the real installer leaks
    # a wget-log into its CWD, so tt must not run it in the user's directory.
    pwd >> "$FAKE_INSTALLER_LOG.cwd"
fi
if [ -n "$FAKE_INSTALLER_FAIL" ]; then
    echo "fake install.sh: forced failure" >&2
    exit 1
fi
echo "fake install.sh: converged system stack"
exit 0
