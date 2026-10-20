# Models and data

This guide lists the models Footnote runs, how to download and verify them, how to use
other embedding models, and the licenses of the models and of the benchmark data used to
evaluate retrieval. It is not legal advice: you are responsible for checking that the
terms fit your use.

## 1. Local models

Footnote runs two small models itself, with ONNX Runtime on the CPU. They are pinned in
[`models/manifest.yaml`](../../models/manifest.yaml) by repository, revision and the SHA-256
of every file.

| Id | Role | Source and revision | License | Size |
|---|---|---|---|---|
| `bge-small-en-v1.5` | embedder (384 dimensions, CLS pooling, normalised, max 512 tokens); also the tokenizer used to count chunk tokens | `BAAI/bge-small-en-v1.5` at `5c38ec7c405e` | MIT | 133 MB |
| `ms-marco-MiniLM-L6-v2` | reranker (22.7M parameters, max 256 tokens) | `cross-encoder/ms-marco-MiniLM-L6-v2` at `233902d25c44` | Apache-2.0 | `int8` 23 MB, `fp32` 91 MB |

The reranker has an `int8` variant per CPU architecture (`arm64` and `amd64`), which is
the default, and an `fp32` variant. Choose with `rerank.variant`. The manifest is the
source of truth for the full revisions and hashes.

The answer-generating language model is not a local model. Footnote never downloads it:
you point `llm.base_url` at an OpenAI-compatible endpoint you run or pay for.

### Training data caveat

Both licenses allow commercial use, but a license on model weights says nothing about the
data the model was trained on.

- The reranker card lists MS MARCO passage ranking as training data. The MS MARCO terms are
  non-commercial. Whether using a model trained on it fits your use is your assessment.
  If it does not, set `rerank.provider: noop` (no reranking; the evidence gate then uses
  vector similarity) or use a reranker you have cleared.
- The embedder's card describes pre-training and contrastive fine-tuning and lists no
  per-dataset training data.

Footnote does not train or fine-tune any model.

## 2. Download and verify

```sh
footnote models pull          # models the configuration needs
footnote models pull --all    # every manifest model and variant
footnote models verify        # size and SHA-256 of every needed file
```

- Files go to `models.dir` (`~/.cache/rayld/footnote/models` by default, `/models` in the
  image and in compose).
- `pull` downloads from `huggingface.co` at the pinned revision, checks every file against
  the manifest hash, and keeps files that already verify. It is safe to run again.
- `verify` prints `status`, model id, file key and path per file and exits 1 if any file is
  missing or wrong.
- `/readyz` has a `models` check. It turns ready after `pull` without a restart.
- A deployment without internet access at run time: pull on a connected host, copy the
  `models.dir` directory to the volume, and run `verify`.

`footnote config validate` also checks that `embedding.spaces`, `rerank.*` and
`chunking.tokenizer` refer to manifest entries that exist.

## 3. Embedding spaces

An embedding space is a named model with a fixed number of dimensions. Vectors from
different models are not comparable, so each space has its own vector index. A collection
has one active space.

The default space is `bge-small-en-v1.5-onnx` (provider `onnx_local`, model
`bge-small-en-v1.5`). It is registered at startup from `embedding.spaces`. Add more:

```sh
footnote spaces register bge-m3 --provider openai_compat \
  --model text-embedding-bge-m3 --dimensions 1024
footnote spaces list
```

| Option | Meaning |
|---|---|
| `--provider onnx_local` | a manifest model run in process; dimensions come from the manifest |
| `--provider openai_compat` | an OpenAI-compatible `/embeddings` endpoint at `embedding.base_url`; `--dimensions` is required |
| `--send-dimensions` | send the `dimensions` parameter to the endpoint (for models that support shortening) |
| `--calibrated-threshold` | vector evidence threshold for this space (0-1), used when reranking is off |

- Names use lowercase letters, digits, `.`, `_` and `-`, up to 63 characters. Registration
  is idempotent by name.
- Up to 4000 dimensions are supported: `vector` index up to 2000 and `halfvec` from 2001.
  More is `422 embedding_dimension_unsupported`.
- `embedding.max_spaces` (default 8) caps the number of spaces.
- Config-registered spaces and `footnote spaces register` are equivalent. The same
  operations exist as `POST/GET /v1/admin/embedding-spaces`.
- A space cannot be deleted while a collection uses it.

## 4. Changing a collection's model

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"embedding_space": "bge-m3"}'
```

This starts a `reembed` job and returns `202`. Searches keep using the old space until
every current chunk has a vector in the new one, then switch once. New uploads are
embedded in both spaces meanwhile. Progress is in the collection's `reembed` field.
Re-embedding sends every chunk to the embedder, so it costs time and, with a hosted
endpoint, money. Setting the old space again cancels it. Old vectors are removed by
garbage collection `ingest.gc_grace` after the switch; storage holds two sets of vectors
until then.

Evidence thresholds are per space. Check them after a switch
([citations.md](citations.md)).

## 5. Benchmark datasets

Footnote's retrieval quality is measured on public benchmarks. The datasets are downloaded
on demand into `~/.cache/rayld/footnote` and are never committed to the repository or
shipped in the image. They are used by the evaluation harness
([evaluation.md](evaluation.md)); this section records the terms of the data.

| Dataset | Used for | License and terms |
|---|---|---|
| BEIR SciFact claims | retrieval, answers, abstention | CC BY 4.0 |
| BEIR SciFact abstracts (the corpus) | the same | ODC-By 1.0 |
| BEIR TREC-COVID (CORD-19 papers) | latency at 100k chunks | mixed per-paper licenses; used for benchmarking only, not redistributed |

Attribution follows the dataset licenses: credit the SciFact authors (Wadden et al.), the
BEIR benchmark (Thakur et al.) and, for TREC-COVID, the CORD-19 and TREC-COVID organisers
if you publish results. Do not republish CORD-19 text.

## 6. Your own data

Footnote stores what you upload. Originals are kept in the database while
`ingest.retain_originals` is true, and are included in backups
([operations.md](operations.md)). The tenant export includes originals on request
([multi-tenancy.md](multi-tenancy.md)).
