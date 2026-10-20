#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Coverage gates: whole repo >= 80%, domain+app >= 90%.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run coverage report --fail-under=80
uv run coverage report --fail-under=90 --include='src/footnote/domain/*,src/footnote/app/*'
