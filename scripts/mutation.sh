#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# make mutation: mutmut on the core modules listed in pyproject.toml, using unit tests only.
#
# Score = detected / (detected + undetected), per module and overall, where
#   detected   = killed + timeout + segfault (a hang or crash is a detected behaviour change),
#   undetected = survived + no tests + suspicious (no test caught it; "no tests" is counted
#                against us so untested code cannot inflate the score).
# Skipped mutants are ignored. Exits nonzero when the overall score is below THRESHOLD.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
threshold="${MUTATION_THRESHOLD:-60}"

rm -rf mutants
uv run --frozen mutmut run || echo "mutmut run exited nonzero; scoring what was recorded" >&2
uv run --frozen mutmut results --all true >mutants/results.txt

awk -v threshold="${threshold}" '
  /: / {
    name = $1; status = substr($0, index($0, ": ") + 2)
    sub(/:$/, "", name)
    mod = name; sub(/\.x_.*$/, "", mod); sub(/\.x.*__mutmut_.*$/, "", mod)
    if (name ~ /\.xǁ/) { mod = name; sub(/\.xǁ.*$/, "", mod) }
    if (status ~ /^(killed|timeout|segfault)/) det[mod]++
    else if (status ~ /^(survived|no tests|suspicious)/) und[mod]++
    else next
    seen[mod] = 1
  }
  END {
    for (m in seen) {
      d = det[m] + 0; u = und[m] + 0
      printf "%-45s detected=%d undetected=%d score=%.1f%%\n", m, d, u, 100 * d / (d + u)
      td += d; tu += u
    }
    if (td + tu == 0) { print "no mutants scored"; exit 1 }
    score = 100 * td / (td + tu)
    printf "TOTAL detected=%d undetected=%d score=%.1f%% (threshold %s%%)\n", td, tu, score, threshold
    exit (score >= threshold) ? 0 : 1
  }
' mutants/results.txt
