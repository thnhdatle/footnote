#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the image and runs the README quickstart.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
make --no-print-directory docker
trap 'make --no-print-directory down' EXIT
examples/quickstart/run.sh
