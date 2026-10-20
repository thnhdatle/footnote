# Evaluation

This guide explains how to measure Footnote's retrieval quality, abstention behaviour and
answer faithfulness on public benchmarks with `footnote eval`. The settings it mentions are
in [configuration.md](configuration.md); the data terms are in
[models-and-data.md](models-and-data.md); the gate and thresholds are in
[citations.md](citations.md).

Every command runs the application in-process against your configured database, indexes a
dataset into a temporary tenant and deletes that tenant afterwards unless you pass `--keep`.
The harness is a developer tool: nothing in it runs in production.

## 1. Get the data

```sh
scripts/fetch_eval_data.sh            # every dataset in eval/datasets.yaml
scripts/fetch_eval_data.sh scifact    # one dataset
make eval-data                        # the same as the first line
```

The script downloads the pinned BEIR archives (SciFact and TREC-COVID), checks their SHA-256
against `eval/datasets.yaml` and unpacks them into `~/.cache/rayld/footnote/datasets`
(`FOOTNOTE_EVAL_DATA_DIR` overrides it). It is safe to run again. Datasets are never
committed. A custom dataset in BEIR layout (`corpus.jsonl`, `queries.jsonl`,
`qrels/*.tsv`) in that directory is accepted too.

## 2. The embedding cache

Evaluation runs embed the whole corpus. To make repeated runs fast, document embeddings are
cached in SQLite files under `eval.cache_dir` (`~/.cache/rayld/footnote/eval`), keyed by the
embedding model and the text. Results are identical with or without the cache. Query
embeddings are not cached, so latency numbers include them.

```sh
footnote eval cache status                      # warm or cold, data present or absent
footnote eval cache status --require scifact    # exit 1 unless scifact is warm
```

A dataset becomes warm after one full `eval retrieval` run. Pass `--no-cache` to a command to
bypass the cache.

## 3. Retrieval

```sh
footnote eval retrieval --dataset scifact
footnote eval retrieval --dataset scifact --modes vector,hybrid --chunking heading,recursive
footnote eval retrieval --dataset scifact --ablation
footnote eval retrieval --dataset scifact --split train --keep
make eval-quick
```

| Option | Effect |
|---|---|
| `--modes` | comma-separated `vector`, `lexical`, `hybrid`, `hybrid+rerank` (default: all four) |
| `--chunking` | comma-separated `heading`, `recursive` (default: the shipped strategy) |
| `--ablation` | the fixed row set of the retrieval ablation: lexical match and rank, RRF weights, reranker file, `rerank_n` |
| `--split` | qrels split, `test` (default) or `train` |
| `--keep` | keep the temporary tenant so you can inspect it |
| `--quick` | the first 50 queries per split; `make eval-quick` runs this with a warm cache |
| `--limit N` | the first `N` queries per split |

Each configuration reports recall@1/5/10/100, nDCG@10, MRR@10 and the p50 and p95 search
latency. Searches run with depth 100, and chunks are pooled to documents by first occurrence
before the metrics are cut. Two baselines are added: `bm25s`, computed in-process with English
stopwords and a stemmer, and the published BEIR references (BM25 0.665, BM25 with a
cross-encoder 0.688 nDCG@10 on SciFact).

## 4. Abstention

```sh
footnote eval abstention --dataset scifact
```

This computes the evidence-gate score of every claim without calling the LLM, using the same
gate as `/answers`. It writes `abstention_curve.csv`: for each threshold from 0.00 to 1.00 in
steps of 0.01, the share of claims that would be answered and abstained. The report also gives
the ROC AUC and the same curve for `eval/sets/unanswerable.jsonl`, 200 questions that are
outside the corpus (cooking, sport, geography, software, history, music). These numbers are
published for information; they are not a pass or fail gate.

## 5. Calibration

```sh
footnote eval calibrate --dataset scifact --split train
footnote eval calibrate --dataset scifact --split train --apply
```

Calibration indexes the SciFact corpus without the gold documents of the unanswerable group,
uses the train claims whose gold documents are all indexed, and picks the largest threshold
that still answers at least `eval.calibration_target_answer_rate` (0.85) of them. It does this
for the reranker gate (per reranker file) and for the vector gate (per embedding space).

Without `--apply` the values are only printed. With `--apply` the reranker file's manifest
value and the space's `calibrated_threshold` are written, and the value to put in your
configuration is printed. Thresholds resolve as described in [citations.md](citations.md).

## 6. Answers

```sh
footnote eval answers --dataset scifact --limit 100 --judge-model <name>
```

This needs a running LLM: set `FOOTNOTE_LLM_BASE_URL` to an OpenAI-compatible server (for
example LM Studio) and `llm.model` to the model that generates the answers. The command:

1. takes the first `--limit` distinct query ids of `qrels/test.tsv` and asks each one as
   "Is the following claim supported by the documents? Explain with citations: *claim*";
2. generates all answers first through the same code as `/answers`, so the evidence gate and
   the citation verifier apply;
3. then judges each answered item with `--judge-model`, one judge call per item, using the
   rubric in `eval/prompts/faithfulness.md`. Running in two passes means LM Studio loads one
   model at a time instead of swapping on every item.

The answer is split into sentences, and the judge labels each one `supported`, `unsupported`
or `non_factual` and says, for each citation, whether that chunk supports the sentence.

| Metric | Meaning |
|---|---|
| faithfulness | sentences supported by their cited chunks / sentences not labelled non-factual; a sentence with no citation is unsupported unless it is non-factual |
| citation precision | citations whose chunk supports their sentence / all citations |
| answer rate | answered items / items; abstained and unsupported counts are reported with it |

Metrics cover answered items only, so a model that abstains more can look more faithful: read
them together with the answer rate. If the judge's reply is not valid JSON it is retried once;
after a second failure every sentence of that item counts as unsupported and
`judge_parse_failures` goes up. A high count means the judge model is too weak for the
rubric.

> **Judge = generator.** If `--judge-model` equals `llm.model`, the model grades its own work
> and scores are inflated. `summary.md` then says "Judge model = generator model (same-model
> judging inflates scores)". Use a different, preferably stronger, model for numbers you
> intend to publish.

## 7. Outputs

Every run writes to a new directory `out/eval/<run_id>` (ignored by Git), where `<run_id>` is
the UTC time and the short commit (for example `20261020T084100Z-1a2b3c4`).

| File | From | Contents |
|---|---|---|
| `retrieval.csv` | `eval retrieval` | one row per configuration and split |
| `ablation.md` | `eval retrieval` | the same rows as a table with the BEIR references and the run's provenance (models, file hashes, dataset hash) |
| `abstention_curve.csv` | `eval abstention` | answer and abstain rate per threshold, ROC AUC |
| `answers.csv` | `eval answers` | per item: query id, abstained, unsupported, sentences, supported, non-factual, citations, citations supported, parse failure |
| `summary.md` | `eval answers` | the metrics, the generator and judge model ids and the same-model note |

Use `--output-dir` to write elsewhere.
