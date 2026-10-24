#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs pip-audit, the license gate and trivy on the local image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
IMAGE="${IMAGE:-thnhdatle/footnote:dev}"
EXCEPTIONS="licenses/exceptions.yaml"

ALLOWED="Apache-2.0;MIT;MIT-CMU;BSD-2-Clause;BSD-3-Clause;ISC;MPL-2.0;0BSD;Zlib;PSF-2.0;Unlicense;CC0-1.0"
ALLOWED+=";MIT License;BSD License;Apache Software License;ISC License (ISCL)"
ALLOWED+=";Mozilla Public License 2.0 (MPL 2.0);Python Software Foundation License"

runtime_requirements() {
  uv export --no-dev --no-hashes --no-emit-project --frozen 2>/dev/null
}

echo "== pip-audit: locked runtime dependencies"
runtime_requirements | uv run --frozen pip-audit --strict --disable-pip --no-deps -r /dev/stdin

echo "== licenses: pip-licenses --allow-only (exceptions in ${EXCEPTIONS})"
packages="$(runtime_requirements | grep -v '^ *#' | sed 's/[ =;].*//' | tr '\n' ' ')"
# shellcheck disable=SC2086 # the package list is intentionally word-split
inventory="$(uv run --frozen pip-licenses --from=mixed --format=json --packages ${packages})"
inv_file="$(mktemp)"
trap 'rm -f "${inv_file}"' EXIT
printf '%s' "${inventory}" >"${inv_file}"
exempt="$(uv run --frozen python -I - "${EXCEPTIONS}" "${inv_file}" <<'PY'
import json
import re
import sys

import yaml


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed = {norm(p["Name"]): p for p in json.load(open(sys.argv[2]))}
entries = yaml.safe_load(open(sys.argv[1]))["exceptions"]
bad = []
for e in entries:
    p = installed.get(norm(e["name"]))
    if p is None:
        bad.append(f"{e['name']}: listed but not a runtime dependency")
    elif p["Version"] != str(e["version"]) or p["License"] != e["license"]:
        bad.append(
            f"{e['name']}: file says {e['version']} / {e['license']}, "
            f"installed {p['Version']} / {p['License']}"
        )
    elif not e.get("reason"):
        bad.append(f"{e['name']}: missing reason")
for line in bad:
    print(line, file=sys.stderr)
if bad:
    sys.exit(1)
print("\n".join(e["name"] for e in entries))
PY
)"
# shellcheck disable=SC2086 # the package lists are intentionally word-split
uv run --frozen pip-licenses --from=mixed --format=plain \
  --packages ${packages} --ignore-packages ${exempt} \
  --allow-only="${ALLOWED}"

echo "== trivy: ${IMAGE}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "image ${IMAGE} not found; run 'make docker' first" >&2
  exit 1
fi
trivy image --quiet --scanners vuln --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 "${IMAGE}"
