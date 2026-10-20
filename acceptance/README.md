# Acceptance suite

End-to-end checks of what Footnote promises: retrieval and answer quality on public
datasets, valid citations, tenant isolation, incremental indexing, consistent reads during
version swaps, parser fidelity, query latency at 100,000 chunks, the abstention gate and the
quickstart. The numbers in the README, in [`bench/RESULTS.md`](../bench/RESULTS.md) and in
[`eval/ablation.md`](../eval/ablation.md) come from `make bench` and the evaluation runs; the latest
`make acceptance` results, from a separate run, are in [`report.md`](report.md).

## Running

```sh
make acceptance                     # every criterion, reference-machine thresholds
make acceptance CI=1                # CI thresholds; criteria marked `ci: skip` are skipped
make acceptance ONLY=FNT-04,FNT-05  # a subset (writes report.partial.md, not committed)
```

The runner (`acceptance/runner.py`) builds the image (`make docker`), starts Postgres,
`fakellm` and Footnote from [`deploy/docker-compose.yml`](../deploy/docker-compose.yml),
probes the capabilities each criterion requires, runs each criterion's test, writes
`acceptance/report.md` and stops the stack. Quality and latency criteria run against a
native `footnote serve` started by the test; the compose `footnote` service is stopped
while they run. The answer-faithfulness criterion runs last, with the stack reduced to
Postgres, so the local model has the memory it needs.

Requirements:

- Docker with Compose v2 and a VM of 8 CPUs and 10 GB of memory (the reference machine is a
  MacBook Pro M3 Pro 36 GB; see `machine` in [`criteria.yaml`](criteria.yaml));
- [uv](https://docs.astral.sh/uv/) and a synced environment (`make bootstrap`);
- the `fakellm` image `ghcr.io/thnhdatle/fakellm:v0.1.0` (or a local build tagged
  `thnhdatle/fakellm:dev`) for the citation, incremental-indexing, swap and quickstart
  criteria;
- network access to `huggingface.co` once for the pinned models, and to the BEIR dataset
  host for SciFact and TREC-COVID;
- about 5 GB of disk: models, datasets and the evaluation embedding cache under
  `~/.cache/rayld/footnote/` (about 1 GB), plus the Docker images;
- for the answer-faithfulness criterion only: LM Studio serving the model on
  `http://localhost:1234/v1` and at least 18 GiB of free memory;
- time: the functional criteria take a few minutes each; a full run takes about 2 hours
  with a warm evaluation cache and about 90 minutes more with a cold one.

Environment variables (all optional):

| Variable | Effect |
|---|---|
| `CI` | selects the CI thresholds (`pass.ci`) |
| `FOOTNOTE_NIGHTLY=1` | lets the `nightly` criteria run under `CI=1` (set by the nightly workflow) |
| `RAYLD_FAKELLM_IMAGE` | `fakellm` image to use instead of the released one |
| `RAYLD_FOOTNOTE_IMAGE` | Footnote image for the compose stack (default `thnhdatle/footnote:dev`) |
| `LMSTUDIO_URL` | endpoint probed for the `lmstudio` capability (default `http://localhost:1234/v1`) |
| `FOOTNOTE_ACC_LMSTUDIO_URL`, `FOOTNOTE_ACC_LLM_MODEL` | endpoint and model used to generate and judge answers (default model `gemma-4-26b-a4b-it-qat-mlx`) |
| `FOOTNOTE_KEEP_STACK=1` | leave the compose stack running after the run |
| `FOOTNOTE_PG_PORT` | host port of the compose Postgres |

## Criteria

Each criterion has an id, one test named after it (`test_acc_fnt01_…`), a kind, the
capabilities it requires and thresholds for two environments in
[`criteria.yaml`](criteria.yaml): `pass.mac` for the reference machine and `pass.ci` for CI
runners. Tests read the thresholds at run time.

| ID | Kind | What it checks | Requires |
|---|---|---|---|
| FNT-01 | quality | Retrieval quality on BEIR SciFact test (300 queries) with the shipped defaults: nDCG@10 floors per mode, R@10 for hybrid + rerank | nightly, network |
| FNT-02 | quality | Faithfulness and citation precision of 100 generated answers, judged by the same local model that generated them | lmstudio, network |
| FNT-03 | quality | One command writes the full ablation report (every row with R@10, nDCG@10 and p95) and it is committed as `eval/ablation.md` | nightly, network |
| FNT-04 | functional | Every citation in at least 1,000 scripted answers points at a retrieved chunk; invalid markers are removed and unsupported sentences flagged | fakellm, docker |
| FNT-05 | functional | No cross-tenant leak in 10,000 API calls, also with the application tenant predicates disabled (RLS alone) | docker |
| FNT-06 | functional | Re-indexing embeds exactly the changed chunks; an identical re-upload embeds nothing; delete leaves no rows | fakellm, docker |
| FNT-07 | performance | Hybrid + rerank search latency on 100,000 TREC-COVID chunks (p95 over 1,000 queries, median of 3 runs) | nightly, network, mac |
| FNT-08 | quality | Abstention gate on 100 answerable and 100 unanswerable SciFact claims (AUC, answer and abstain rates) and calibration drift | nightly, network |
| FNT-09 | functional | Parser fidelity (headings, tables, paragraph order, PDF pages) on generated fixtures | — |
| FNT-10 | functional | No mixed-version results during 50 version swaps and an embedding-space change under concurrent queries | fakellm, docker |
| FNT-11 | release | The offline quickstart completes in under 5 minutes with a valid citation | docker, fakellm |

Capabilities: `docker` (a running Docker daemon), `network` (datasets cached or the dataset
host reachable), `nightly` (satisfied locally when `CI` is unset, with
`FOOTNOTE_NIGHTLY=1` in the nightly workflow, or in CI when the evaluation cache is warm),
`mac` (Apple Silicon, the reference machine), `lmstudio` (the LM Studio endpoint answers)
and `fakellm` (the `sibling:fakellm` entry: a `fakellm` image is present).

Not every criterion runs everywhere:

- FNT-02 has `ci: skip`: it needs a local model and runs only on a developer machine.
- FNT-07 requires `mac`, so the Linux nightly workflow skips it and its CI threshold is not
  used there; latency is measured on the reference machine only.
- FNT-01, FNT-03 and FNT-08 run on push CI only when the evaluation cache is warm, otherwise
  in the nightly workflow.
- With `CI=1`, a missing `fakellm` image is a failure, not a skip: CI pulls the released
  image before the suite runs.

## The report

`report.md` starts with the Footnote version and the date, then has one row per criterion
with the result, the measured values and the thresholds. Per-criterion logs
(`acceptance/logs/<ID>.txt` and `.json`) are written locally during the run and are not
committed. Results are:

- **PASS**: every measured value meets its threshold;
- **FAIL**: a measured value misses its threshold, or the test failed; listed again under
  "Not met";
- **Skipped (environment)**: a required capability is missing on this host, or the
  criterion has no thresholds for this environment;
- **Blocked (environment)**: with the reference-machine thresholds (no `CI`), the first
  missing capability is Docker, Apple Silicon or the local model (including too little
  free memory for it).

The runner, not the test, decides when a criterion is skipped: acceptance sources contain no
skip constructs. Before running, the runner checks that every criterion has exactly one test
with its name and that no acceptance source skips itself; any problem is listed under
"Criteria check" and fails the run. "Sibling images used" names the `fakellm` image of the
run.
