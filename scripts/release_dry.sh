#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the wheel, sdist and multi-arch image without pushing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
TAG="${TAG:-ghcr.io/thnhdatle/footnote:v0.1.0-dry}"
DEFAULT_PLATFORMS="linux/amd64,linux/arm64"

echo "== uv build"
rm -rf dist
uv build

echo "== docker buildx build --platform ${DEFAULT_PLATFORMS} (no push)"
platforms="${DEFAULT_PLATFORMS}"
if ! docker buildx build --platform "${DEFAULT_PLATFORMS}" -t "${TAG}" . ; then
  host="linux/$(docker version --format '{{.Server.Arch}}')"
  echo "WARN: multi-arch build failed; falling back to ${host}" >&2
  platforms="${host}"
  docker buildx build --platform "${platforms}" --load -t "${TAG}" .
else
  host="linux/$(docker version --format '{{.Server.Arch}}')"
  docker buildx build --platform "${host}" --load -t "${TAG}" .
fi

echo "== image user"
user="$(docker image inspect --format '{{.Config.User}}' "${TAG}")"
case "${user}" in
  "" | 0 | 0:* | root | root:*)
    echo "image runs as root (User='${user}')" >&2
    exit 1
    ;;
  *) echo "non-root: ${user} (platforms built: ${platforms})" ;;
esac
