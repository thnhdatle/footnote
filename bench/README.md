# Benchmarks

Reproduce every published quality and latency number with `make bench`. Footnote runs
natively on the host (`footnote serve`); only Postgres runs in compose.

## Prerequisites

```sh
make up                                                          # compose Postgres (or let the runner start it)
uv run footnote eval cache status --require scifact,trec-covid   # must exit 0
uv run python -m acceptance.fnt07_data --warm                    # only if the trec-covid marker is missing but vectors are cached
brew install k6                                                  # k6 on PATH
```

- Docker running.
- Models under `~/.cache/rayld/footnote/models`, INT8 and fp32 reranker files.
- An otherwise idle Mac (reference: M3 Pro 36 GB, Docker VM 8 CPU / 10 GB).

## What it runs

```sh
make bench    # all three steps below
```

| Step | Output |
|---|---|
| `acceptance/runner.py --bench`: FNT-01, FNT-03, FNT-07, FNT-08 | `acceptance/logs/` (gitignored) |
| `python -m bench.latency.run`: FNT-07 published latency, not gated | `out/bench/latency.json` (gitignored) |
| `python -m bench.results`: render results | `bench/RESULTS.md` |

The latency step uses the same 100,000 TREC-COVID chunks:

- per-stage p95 over 1,000 debug searches after 50 warm-up
- `rerank_n` 50: 3 runs x 1,000 queries
- fp32 reranker (`FOOTNOTE_RERANK_VARIANT=fp32`): 3 runs
- k6 (`bench/latency/k6.js`): constant 2 req/s for 5 min, hybrid + rerank, shipped defaults

Each section of `RESULTS.md` names the machine and the Footnote version (`RELEASE` in the
`Makefile`). The committed results were measured on a v0.1.0 pre-release build.

## Runtime

About 2 h with a warm cache. A cold embedding cache adds about 90 min.

## Just the latency scenario

```
uv run --frozen python -m bench.latency.run
uv run --frozen python -m bench.results
```

Smoke overrides, not for publishing:

| Variable | Overrides |
|---|---|
| `BENCH_CHUNKS` | chunk count |
| `BENCH_QUERIES` | query count |
| `BENCH_WARMUP` | warm-up queries |
| `BENCH_RUNS` | runs |
| `BENCH_K6_RATE` | k6 rate |
| `BENCH_K6_DURATION` | k6 duration |
| `BENCH_OUT` | output path |
| `BENCH_SKIP` | comma list of `stages,rerank50,k6,fp32` |
