# Benchmarks

`make bench` reproduces every published quality and latency number with one command. Services under
test run natively on the host (`footnote serve`); only Postgres runs in compose.

## What it runs

1. `acceptance/runner.py --bench`: the quality and performance acceptance criteria (FNT-01 SciFact quality,
   FNT-03 ablation, FNT-07 gated latency, FNT-08 abstention; the compose stack is kept up).
   Raw output: `acceptance/logs/` (gitignored).
2. `python -m bench.latency.run`: FNT-07 "published, not gated" latency on the same
   100,000 TREC-COVID chunks, loaded once and sharing one tenant:
   - per-stage p95 (embed_query, lexical, vector, rerank) from `debug: true` searches
     over 1,000 queries after 50 warm-up, plus client p95 in debug mode;
   - `rerank_n` 50: 3 runs x 1,000 queries (p50/p95/p99, median and min-max);
   - fp32 reranker (`FOOTNOTE_RERANK_VARIANT=fp32`, serve restarted): 3 runs;
   - k6 open-model run (`bench/latency/k6.js`): constant 2 req/s for 5 min against hybrid +
     rerank with shipped defaults; p50/p95/p99, dropped iterations, error rate.
   Output: `out/bench/latency.json` (gitignored).
3. `python -m bench.results`: renders `bench/RESULTS.md` from
   `acceptance/logs/FNT-01|07|08.json` and `out/bench/latency.json`.

Each section of `RESULTS.md` names the machine and the Footnote version (`RELEASE` in the
`Makefile`); the committed results were measured on a v0.1.0 pre-release build.

## Prerequisites

- Docker running; compose Postgres up (`make up`, or the acceptance runner starts it).
- Warm eval cache: `uv run footnote eval cache status --require scifact,trec-covid`
  must exit 0 (if the trec-covid marker is missing but vectors are cached:
  `uv run python -m acceptance.fnt07_data --warm`).
- `k6` on PATH (`brew install k6`); models under `~/.cache/rayld/footnote/models`
  (INT8 and fp32 reranker files).
- Run on an otherwise idle Mac (reference: M3 Pro 36 GB, Docker VM 8 CPU / 10 GB).

## Runtime

About 2 h with a warm cache: the acceptance criteria about 1 h (SciFact runs dominate), FNT-07 data
prep about 3 min plus about 1 min load, bench/latency about 25 min (1,000 queries x
(1 debug + 3 + 3 runs) at roughly 0.35-1.3 s each, plus 5 min k6). A cold embedding cache
adds about 90 min.

## Just the latency scenario

```
uv run --frozen python -m bench.latency.run
uv run --frozen python -m bench.results
```

Smoke overrides (not for publishing): `BENCH_CHUNKS`, `BENCH_QUERIES`, `BENCH_WARMUP`,
`BENCH_RUNS`, `BENCH_K6_RATE`, `BENCH_K6_DURATION`, `BENCH_OUT`, and `BENCH_SKIP`
(comma list of `stages,rerank50,k6,fp32`).
