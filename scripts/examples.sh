#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs the check.sh of every directory in examples/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
status=0
for dir in examples/*/; do
  name="$(basename "${dir}")"
  if [ ! -x "${dir}check.sh" ]; then
    echo "examples/${name}: no executable check.sh" >&2
    status=1
    continue
  fi
  echo "== examples/${name}"
  "${dir}check.sh" || status=1
done
exit "${status}"
