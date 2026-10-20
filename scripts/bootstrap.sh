#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Checks tool versions, syncs the Python environment, pulls compose images and sets the hooks path.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
missing=0
for tool in uv docker git gitleaks hadolint shellcheck; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing tool: $tool"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1
uv sync --all-extras --frozen 2>/dev/null || uv sync --all-extras
git config core.hooksPath .githooks
if [ -f deploy/docker-compose.yml ]; then
  docker compose -f deploy/docker-compose.yml pull --ignore-buildable --quiet || true
fi
echo "bootstrap done"
