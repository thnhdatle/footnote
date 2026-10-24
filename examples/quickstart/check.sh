#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static check of the quickstart example.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "${here}/../.."

for f in README.md config.yaml compose.yaml run.sh handbook.pdf make_sample_pdf.py; do
  [ -f "examples/quickstart/${f}" ] || { echo "missing examples/quickstart/${f}" >&2; exit 1; }
done

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
uv run --frozen python examples/quickstart/make_sample_pdf.py "${tmp}/handbook.pdf" >/dev/null
cmp "${tmp}/handbook.pdf" examples/quickstart/handbook.pdf \
  || { echo "handbook.pdf differs from make_sample_pdf.py output" >&2; exit 1; }

FOOTNOTE_DATABASE_URL=postgresql://x:x@localhost/x \
  FOOTNOTE_ADMIN_DATABASE_URL=postgresql://x:x@localhost/x \
  FOOTNOTE_AUTH_ADMIN_KEY=x FOOTNOTE_CONFIG=examples/quickstart/config.yaml \
  uv run --frozen footnote config validate >/dev/null

docker compose -f deploy/docker-compose.yml -f examples/quickstart/compose.yaml -p footnote \
  --profile fakellm config -q
echo "quickstart: ok"
