# Models and data

Download the local models, add embedding models, and check the licenses.

This is not legal advice. Check that the terms fit your use.

## 1. Local models

Two models run on the CPU, pinned in [`models/manifest.yaml`](../../models/manifest.yaml).

| Id | Role | Source and revision | License | Size |
|---|---|---|---|---|
| `bge-small-en-v1.5` | embedder (384 dimensions), tokenizer | `BAAI/bge-small-en-v1.5` at `5c38ec7c405e` | MIT | 133 MB |
| `ms-marco-MiniLM-L6-v2` | reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` at `233902d25c44` | Apache-2.0 | `int8` 23 MB, `fp32` 91 MB |

`rerank.variant` picks `int8` (default) or `fp32`. The answer model is not local: set
`llm.base_url`.

**Training data.** The reranker was trained on MS MARCO, whose terms are non-commercial.
If that does not fit your use, set `rerank.provider: noop` or use another reranker.

## 2. Download and verify

```sh
footnote models pull          # models the configuration needs
footnote models pull --all    # every model and variant
footnote models verify        # check hashes; exit 1 on a bad file
```

Files go to `models.dir`: `~/.cache/rayld/footnote/models` by default, `/models` in the
image. For an offline host, pull elsewhere, copy `models.dir`, then run `verify`.

## 3. Embedding spaces

A space is a named embedding model. Each collection uses one. The default is
`bge-small-en-v1.5-onnx`.

```sh
footnote spaces register bge-m3 --provider openai_compat \
  --model text-embedding-bge-m3 --dimensions 1024    # add a space
footnote spaces list                                 # list spaces
```

| Option | Meaning |
|---|---|
| `--provider onnx_local` | manifest model; dimensions from the manifest |
| `--provider openai_compat` | endpoint at `embedding.base_url`; needs `--dimensions` |
| `--send-dimensions` | send `dimensions` to the endpoint |
| `--calibrated-threshold` | vector evidence threshold (0-1) |

- Names: lowercase letters, digits, `.`, `_`, `-`, up to 63 characters.
- Up to 4000 dimensions.
- At most `embedding.max_spaces` (default 8) spaces.
- A space in use cannot be deleted.

## 4. Changing a collection's model

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"embedding_space": "bge-m3"}'    # starts a reembed job
```

Searches switch to the new space once every chunk is embedded. Setting the old space again
cancels it. Re-embedding costs time, and money with a hosted endpoint. Check the evidence
threshold after a switch ([citations.md](citations.md)).

## 5. Benchmark datasets

[evaluation.md](evaluation.md) downloads these on demand. They are never shipped.

| Dataset | Used for | License and terms |
|---|---|---|
| BEIR SciFact claims | retrieval, answers, abstention | CC BY 4.0 |
| BEIR SciFact abstracts (the corpus) | the same | ODC-By 1.0 |
| BEIR TREC-COVID (CORD-19 papers) | latency at 100k chunks | mixed per-paper licenses; benchmarking only |

If you publish results, credit SciFact (Wadden et al.), BEIR (Thakur et al.) and the
CORD-19 and TREC-COVID organisers. Do not republish CORD-19 text.

## 6. Your own data

Uploaded files stay in the database while `ingest.retain_originals` is true, so they are
in backups ([operations.md](operations.md)) and can be exported
([multi-tenancy.md](multi-tenancy.md)).
