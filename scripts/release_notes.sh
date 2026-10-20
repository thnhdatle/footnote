#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints the CHANGELOG.md section of a release: release_notes.sh v0.1.0
set -euo pipefail

version="${1:?usage: release_notes.sh <tag>}"
version="${version#v}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
notes=$(awk -v v="${version}" '
  index($0, "## [" v "]") == 1 { found = 1; next }
  found && /^## \[/ { exit }
  found && /^\[[^]]+\]: / { exit }
  found { print }
' CHANGELOG.md)
if [ -z "$(printf '%s' "${notes}" | tr -d '[:space:]')" ]; then
  echo "release_notes: no CHANGELOG.md section for ${version}" >&2
  exit 1
fi
printf '%s\n' "${notes}"
