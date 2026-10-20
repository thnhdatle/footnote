#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# make quickstart-test: build the image, then run the quickstart criterion of the acceptance
# suite (the offline quickstart completes in under 5 minutes with a valid citation).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
make --no-print-directory docker
exec make --no-print-directory acceptance ONLY=FNT-11
