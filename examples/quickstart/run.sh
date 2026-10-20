#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quickstart: offline smoke test with the fakellm stub (README in this directory).
# Needs the image (`make docker`), Docker Compose v2, curl and jq. Leaves the stack running;
# stop it with `make down`.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${here}/../.." && pwd)"
cd "${root}"

url="${FOOTNOTE_URL:-http://127.0.0.1:8000}"
admin_url="${FOOTNOTE_ADMIN_URL:-http://127.0.0.1:9094}"
admin_key="${ADMIN_KEY:-dev-admin-key-change-me}"
question="${QUESTION:-How many vacation days do full-time employees get per year?}"

compose=(docker compose -f deploy/docker-compose.yml -f examples/quickstart/compose.yaml -p footnote)

echo "== start the stack"
"${compose[@]}" --profile fakellm up -d --wait
curl -fsS "${admin_url}/readyz" >/dev/null

echo "== create a tenant and a key"
tenant=$(curl -fsS -X POST "${url}/v1/admin/tenants" \
  -H "Authorization: Bearer ${admin_key}" -H "Content-Type: application/json" \
  -d "{\"name\": \"quickstart-$(date +%s)\"}" | jq -r .id)
key=$(curl -fsS -X POST "${url}/v1/admin/tenants/${tenant}/keys" \
  -H "Authorization: Bearer ${admin_key}" -H "Content-Type: application/json" \
  -d '{"name": "quickstart"}' | jq -r .key)

echo "== create a collection (shipped thresholds)"
collection=$(curl -fsS -X POST "${url}/v1/collections" \
  -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
  -d '{"name": "handbook"}' | jq -r .id)

echo "== upload the sample PDF"
job=$(curl -fsS -X POST "${url}/v1/collections/${collection}/documents" \
  -H "Authorization: Bearer ${key}" -F "file=@examples/quickstart/handbook.pdf" | jq -r .job_id)

echo "== wait for the ingest job"
for _ in $(seq 1 300); do
  status=$(curl -fsS -H "Authorization: Bearer ${key}" "${url}/v1/jobs/${job}" | jq -r .status)
  case "${status}" in
    done) break ;;
    failed) echo "ingest job failed" >&2; exit 1 ;;
  esac
  sleep 1
done
[ "${status}" = "done" ] || { echo "ingest job not done: ${status}" >&2; exit 1; }
echo "job ${job}: ${status}"

echo "== ask: ${question}"
out="$(mktemp)"
trap 'rm -f "${out}"' EXIT
curl -fsS -X POST "${url}/v1/collections/${collection}/answers" \
  -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
  -d "$(jq -n --arg q "${question}" '{question: $q}')" \
  | tee "${out}" | jq '{answer, abstained, evidence_score, citations}'
jq -e '(.abstained | not) and (.citations | length >= 1)' "${out}" >/dev/null \
  || { echo "no cited answer" >&2; exit 1; }
