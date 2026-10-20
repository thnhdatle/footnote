# Evaluation

Measure retrieval quality, abstention and answer faithfulness on public benchmarks with
`footnote eval`.

Each command indexes a dataset into a temporary tenant and deletes it afterwards. Pass `--keep`
to keep it.

## 1. Get the data

```sh
make eval-data                        # every dataset (SciFact, TREC-COVID)
scripts/fetch_eval_data.sh scifact    # one dataset
```

Data goes to `~/.cache/rayld/footnote/datasets`. Set `FOOTNOTE_EVAL_DATA_DIR` to change it.
Any dataset in BEIR layout in that directory also works.

## 2. Embedding cache

Corpus embeddings are cached, so repeated runs are fast. Results are the same with or without
the cache.

```sh
footnote eval cache status                      # warm or cold
footnote eval cache status --require scifact    # exit 1 unless scifact is warm
```

Pass `--no-cache` to skip it.

## 3. Retrieval

```sh
footnote eval retrieval --dataset scifact               # all modes
footnote eval retrieval --dataset scifact --ablation    # the full ablation table
make eval-quick                                         # first 50 queries, warm cache
```

| Option | Effect |
|---|---|
| `--modes` | `vector`, `lexical`, `hybrid`, `hybrid+rerank` (default: all) |
| `--chunking` | `heading`, `recursive` (default: shipped strategy) |
| `--ablation` | run every ablation row |
| `--split` | `test` (default) or `train` |
| `--limit N` | first N queries |
| `--keep` | keep the temporary tenant |

Reports recall@k, nDCG@10, MRR@10 and search latency, next to a BM25 baseline.

## 4. Abstention

```sh
footnote eval abstention --dataset scifact
```

Shows how often the evidence gate would answer or abstain at each threshold. Writes
`abstention_curve.csv`.

## 5. Calibration

```sh
footnote eval calibrate --dataset scifact --split train            # print the threshold
footnote eval calibrate --dataset scifact --split train --apply    # save it
```

Picks the highest gate threshold that still answers 85% of answerable questions.

## 6. Answers

```sh
footnote eval answers --dataset scifact --limit 100 --judge-model <name>
```

Needs a running LLM (`FOOTNOTE_LLM_BASE_URL` and `llm.model`). Footnote answers each claim, then
the judge model checks every sentence and citation.

| Metric | Meaning |
|---|---|
| faithfulness | share of sentences supported by their citations |
| citation precision | share of citations that support their sentence |
| answer rate | share of questions answered instead of abstained |

Use a judge model different from `llm.model`: a model grading itself inflates the scores.

## 7. Outputs

Each run writes to `out/eval/<run_id>`. Use `--output-dir` to change it.

| File | Contents |
|---|---|
| `retrieval.csv`, `ablation.md` | retrieval metrics per configuration |
| `abstention_curve.csv` | answer and abstain rate per threshold |
| `answers.csv`, `summary.md` | answer metrics per item and in total |
