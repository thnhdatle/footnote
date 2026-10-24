#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs a command and prints a one-line PASS/FAIL summary.
set -uo pipefail
name="$1"
shift
if "$@"; then
  echo "PASS: ${name}"
else
  rc=$?
  echo "FAIL: ${name} (exit ${rc})"
  exit "${rc}"
fi
