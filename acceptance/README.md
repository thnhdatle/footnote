# Acceptance suite

Run the end-to-end checks of what Footnote promises and read their results. The latest results
are in [`report.md`](report.md). Published numbers come from `make bench`
([`bench/RESULTS.md`](../bench/RESULTS.md)) and the evaluation runs
([`eval/ablation.md`](../eval/ablation.md)).

## Running

```sh
make acceptance                     # every criterion, reference-machine thresholds
make acceptance CI=1                # CI thresholds; criteria marked `ci: skip` are skipped
make acceptance ONLY=FNT-04,FNT-05  # a subset (writes report.partial.md, not committed)
```

The runner builds the image and starts the stack from
[`deploy/docker-compose.yml`](../deploy/docker-compose.yml). It writes `acceptance/report.md`
and stops the stack afterwards.

Requirements:

- Docker with Compose v2, VM of 8 CPUs and 10 GB memory (reference machine: see `machine` in
  [`criteria.yaml`](criteria.yaml))
- [uv](https://docs.astral.sh/uv/) and a synced environment (`make bootstrap`)
- the fakellm image `ghcr.io/thnhdatle/fakellm:v0.1.0`, or a local build tagged
  `thnhdatle/fakellm:dev` (citation, incremental-indexing, swap and quickstart criteria)
- network access once to huggingface.co (models) and the BEIR host (SciFact, TREC-COVID)
- about 5 GB of disk, including about 1 GB under `~/.cache/rayld/footnote/`
- answer-faithfulness criterion only: LM Studio serving on `http://localhost:1234/v1` and
  at least 18 GiB of free memory
- time: a few minutes per functional criterion; about 2 hours for a full run with a warm
  evaluation cache, about 90 minutes more with a cold one

Environment variables (all optional):

| Variable | Effect |
|---|---|
| `CI` | use CI thresholds (`pass.ci`) |
| `FOOTNOTE_NIGHTLY=1` | run `nightly` criteria under `CI=1` |
| `RAYLD_FAKELLM_IMAGE` | fakellm image instead of the released one |
| `RAYLD_FOOTNOTE_IMAGE` | Footnote image for compose (default `thnhdatle/footnote:dev`) |
| `LMSTUDIO_URL` | `lmstudio` probe endpoint (default `http://localhost:1234/v1`) |
| `FOOTNOTE_ACC_LMSTUDIO_URL`, `FOOTNOTE_ACC_LLM_MODEL` | answer generation and judging endpoint and model (default model `gemma-4-26b-a4b-it-qat-mlx`) |
| `FOOTNOTE_KEEP_STACK=1` | keep the stack running afterwards |
| `FOOTNOTE_PG_PORT` | host port of the compose Postgres |

## Criteria

Each criterion has one test named after it (`test_acc_fnt01_…`). Thresholds are in
[`criteria.yaml`](criteria.yaml): `pass.mac` for the reference machine, `pass.ci` for CI.

| ID | Kind | What it checks | Requires |
|---|---|---|---|
| FNT-01 | quality | SciFact retrieval quality, shipped defaults | nightly, network |
| FNT-02 | quality | faithfulness and citation precision of 100 answers | lmstudio, network |
| FNT-03 | quality | full ablation report, committed as `eval/ablation.md` | nightly, network |
| FNT-04 | functional | every citation points at a retrieved chunk | fakellm, docker |
| FNT-05 | functional | no cross-tenant leak, also with RLS alone | docker |
| FNT-06 | functional | re-indexing embeds only changed chunks | fakellm, docker |
| FNT-07 | performance | search latency on 100,000 TREC-COVID chunks | nightly, network, mac |
| FNT-08 | quality | abstention gate and calibration drift | nightly, network |
| FNT-09 | functional | parser fidelity on generated fixtures | — |
| FNT-10 | functional | no mixed-version results during version swaps | fakellm, docker |
| FNT-11 | release | offline quickstart under 5 minutes with a valid citation | docker, fakellm |

Capabilities:

| Capability | Met when |
|---|---|
| `docker` | a Docker daemon is running |
| `network` | datasets cached or the dataset host reachable |
| `nightly` | `CI` unset, `FOOTNOTE_NIGHTLY=1`, or CI with a warm evaluation cache |
| `mac` | Apple Silicon (the reference machine) |
| `lmstudio` | the LM Studio endpoint answers |
| `fakellm` | a fakellm image is present |

Where criteria run:

- FNT-02 has `ci: skip`. It runs only on a developer machine.
- FNT-07 needs `mac`. Latency is measured on the reference machine only.
- FNT-01, FNT-03 and FNT-08 run on push CI only with a warm evaluation cache. Otherwise
  they run nightly.
- With `CI=1`, a missing fakellm image is a failure, not a skip.

## The report

`report.md` has one row per criterion: result, measured values and thresholds. Logs go to
`acceptance/logs/<ID>.txt` and `.json` and are not committed.

| Result | Meaning |
|---|---|
| **PASS** | every value meets its threshold |
| **FAIL** | a value misses its threshold or the test failed; also listed under "Not met" |
| **Skipped (environment)** | a required capability is missing, or no thresholds for this environment |
| **Blocked (environment)** | reference thresholds (no `CI`) and the first missing capability is Docker, Apple Silicon or the local model (or its free memory) |

The runner, not the test, decides skips. A criterion without exactly one matching test, or an
acceptance source that skips itself, is listed under "Criteria check" and fails the run.
"Sibling images used" names the fakellm image of the run.
