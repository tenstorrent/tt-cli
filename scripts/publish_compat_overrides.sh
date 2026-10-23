#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.
#
# Upload overrides.json (build_model_support.py --compat-overrides) to the
# compatibility matrix bucket's SFTP endpoint, at
# model-spec-overrides/<release>/overrides.json, where the data pipeline reads it.
#
# COMPAT_SFTP_PORT is optional (default 22).

set -euo pipefail

file="${1:?usage: $0 overrides.json}"
: "${COMPAT_SFTP_HOST:?}" "${COMPAT_SFTP_USER:?}"
: "${COMPAT_SFTP_KEY_FILE:?}" "${COMPAT_SFTP_KNOWN_HOSTS_FILE:?}"

release="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["release"])' "${file}")"
if [[ ! "${release}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: ${file} has release '${release}', expected vX.Y.Z" >&2
  exit 1
fi
dest="model-spec-overrides/${release}"

# A leading '-' lets mkdir fail when the directory already exists.
sftp -b - -P "${COMPAT_SFTP_PORT:-22}" -i "${COMPAT_SFTP_KEY_FILE}" \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="${COMPAT_SFTP_KNOWN_HOSTS_FILE}" \
  "${COMPAT_SFTP_USER}@${COMPAT_SFTP_HOST}" <<EOF
-mkdir model-spec-overrides
-mkdir ${dest}
put ${file} ${dest}/overrides.json
EOF
echo "uploaded ${file} to ${dest}/overrides.json"
