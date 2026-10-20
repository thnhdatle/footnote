#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# make release-check: pre-release checks, run before tagging.
# Usage: scripts/release_check.sh   Prints every finding and exits 1 if there is any.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
fail=0
finding() {
  echo "FINDING: $*" >&2
  fail=1
}

echo "== only project files tracked"
# Hidden paths and top-level Markdown files outside these lists do not belong in the repository.
while IFS= read -r path; do
  finding "unexpected hidden path: ${path}"
done < <(git ls-files | grep -E '(^|/)\.' \
  | grep -vE '^\.(github|githooks)/|^\.(dockerignore|editorconfig)$|(^|/)\.gitignore$' || true)
while IFS= read -r path; do
  finding "unexpected top-level Markdown file: ${path}"
done < <(git ls-files '*.md' | grep -v / \
  | grep -vE '^(README|CHANGELOG|CONTRIBUTING|CODE_OF_CONDUCT|SECURITY|ROADMAP)\.md$' || true)

echo "== SPDX header in every source file"
while IFS= read -r path; do
  [ -f "${path}" ] || continue
  if ! head -n 5 "${path}" | grep -q 'SPDX-License-Identifier: Apache-2.0'; then
    finding "missing SPDX header: ${path}"
  fi
done < <(git ls-files 'src/*.py' 'tests/*.py' 'acceptance/*.py' 'bench/*.py' 'examples/*.py' \
  'migrations/*.py' 'scripts/*.sh' '.githooks/*' 'Makefile' 'Dockerfile')

echo "== TODO only as TODO(roadmap) with a ROADMAP.md entry"
while IFS=: read -r path line text; do
  [ "${path}" = "scripts/release_check.sh" ] && continue
  if ! grep -q 'TODO(roadmap)' <<<"${text}"; then
    finding "TODO without (roadmap): ${path}:${line}"
  fi
done < <(git grep -n -I 'TODO' -- . ':!*.md' || true)
if git grep -q -I 'TODO(roadmap)' -- . ':!*.md' \
  ':!scripts/release_check.sh'; then
  [ -f ROADMAP.md ] || finding "TODO(roadmap) present but ROADMAP.md missing"
  while IFS=: read -r path line text; do
    key="$(sed -E 's/.*TODO\(roadmap\):? *//' <<<"${text}" | cut -c1-30)"
    grep -qiF -- "${key}" ROADMAP.md || finding "no ROADMAP.md entry for ${path}:${line}: ${key}"
  done < <(git grep -n -I 'TODO(roadmap)' -- . ':!*.md' \
    ':!scripts/release_check.sh' || true)
fi

echo "== community files"
for f in CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md LICENSE ROADMAP.md CHANGELOG.md \
  .github/CODEOWNERS .github/dependabot.yml .github/PULL_REQUEST_TEMPLATE.md \
  .github/ISSUE_TEMPLATE/bug_report.yml .github/ISSUE_TEMPLATE/feature_request.yml; do
  [ -f "${f}" ] || finding "missing community file: ${f}"
done

echo "== config.example.yaml validates"
if [ -f deploy/config.example.yaml ]; then
  if ! out="$(uv run --frozen footnote --config deploy/config.example.yaml config validate 2>&1)"; then
    finding "config validate failed on deploy/config.example.yaml: ${out}"
  fi
else
  finding "missing deploy/config.example.yaml"
fi

echo "== CHANGELOG 0.1.0 section is dated and has entries"
if ! grep -Eq '^## \[0\.1\.0\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$' CHANGELOG.md; then
  finding "CHANGELOG.md has no dated '## [0.1.0] - YYYY-MM-DD' section"
fi
if ! awk '/^## \[0\.1\.0\]/{f=1; next} /^## \[/{f=0} f && /^- /{n++} END{exit n?0:1}' CHANGELOG.md; then
  finding "CHANGELOG.md has no entries under [0.1.0]"
fi
scripts/release_notes.sh v0.1.0 >/dev/null || finding "release notes for v0.1.0 are empty"

echo "== bench/RESULTS.md"
[ -f bench/RESULTS.md ] || finding "missing bench/RESULTS.md"

echo "== commit messages: one line, <= 72 characters"
while IFS= read -r sha; do
  msg="$(git log -1 --format=%B "${sha}")"
  msg="${msg%$'\n'}"
  if [ "$(wc -l <<<"${msg}")" -ne 1 ]; then
    finding "multi-line commit message: ${sha}"
  elif [ "${#msg}" -gt 72 ]; then
    finding "commit subject over 72 characters (${#msg}): ${sha}"
  fi
done < <(git rev-list HEAD)

if [ "${fail}" -ne 0 ]; then
  exit 1
fi
