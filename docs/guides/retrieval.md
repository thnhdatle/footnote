# Retrieval

This guide explains how `POST /v1/collections/{id}/search` finds chunks, which settings
change the result, and how to inspect a search. The request and response fields are in
[api.md](api.md); the option list is in [configuration.md](configuration.md).

## 1. How a search runs

1. One read-only snapshot is opened. It fixes, for the whole search, which version of each
   document is current and which embedding space the collection uses. Concurrent uploads and
   re-embeds cannot mix versions or spaces into one result.
2. The query is embedded with the collection's active space.
3. Two legs run: a lexical leg (Postgres full-text search) and a vector leg (pgvector
   cosine similarity). Each returns its best `k_lexical` or `k_vector` chunks, 50 by default.
4. The legs are fused with weighted reciprocal rank fusion (RRF).
5. The best `rerank_n` fused chunks (30 by default) are rescored by a cross-encoder
   reranker, and the best `k` are returned.

Only chunks of documents' current versions can appear. Filters apply inside both legs.

## 2. Modes

| `mode` | Legs | Use it for |
|---|---|---|
| `hybrid` (default) | lexical and vector, fused | most queries |
| `vector` | vector only | paraphrased or conceptual questions |
| `lexical` | lexical only | exact names, identifiers, error codes |

`retrieval.default_mode` sets the default. A request can override the collection's
`rerank` and `k`:

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "mode": "hybrid", "rerank": true, "k": 5}'
```

## 3. Filters

```json
{"query": "parental leave",
 "filters": {"metadata": {"team": {"in": ["hr", "legal"]}, "lang": "en"},
             "document_ids": ["c91b..."],
             "created_after": "2026-01-01T00:00:00Z"}}
```

- `metadata`: every key must match. A scalar means equality; `{"in": [...]}` means any of
  the values.
- `document_ids`: restrict to these documents.
- `created_after`, `created_before`: ISO 8601 timestamps on the document's creation time.

Metadata changes made with `PATCH` apply to filters at once.

## 4. Collection settings that change results

Set them on create or `PATCH /v1/collections/{id}`. Retrieval settings need no
re-indexing and take effect on the next search.

| Setting | Default | Effect |
|---|---|---|
| `retrieval.lexical_match` | `any` | `any`: a chunk matches when it has any query term (OR). `all`: every term (AND, `plainto_tsquery`). `websearch`: web-search syntax (quoted phrases, `or`, `-term`; terms are otherwise ANDed). |
| `retrieval.lexical_rank` | `ts_rank` | `ts_rank` or `ts_rank_cd` (also rewards term proximity) |
| `retrieval.rrf_weights` | lexical 0.2, vector 1.0 | relative weight of each leg in fusion |
| `retrieval.rrf_k` | `60` | RRF constant; larger flattens rank differences |
| `retrieval.k_lexical`, `k_vector` | `50` | candidates per leg |
| `retrieval.rerank` | `true` | run the reranker |
| `retrieval.rerank_n` | `30` | candidates reranked; at least `k` |
| `retrieval.k` | `8` | results returned |
| `language` | `simple` | `simple` or `english` text search configuration; `english` stems words. Changing it starts an `fts_reconfig` job. |

`any` is the default because requiring every term of a natural-language question matches
almost nothing. If lexical results look noisy, lower the lexical weight rather than
switching to `all`.

The lexical weight 0.2 was chosen on the SciFact train split: a sweep over 1.0, 0.5, 0.3
and 0.2 picked the highest train nDCG@10 (0.7503 at 0.2, 0.7157 at equal weighting). The
grid stops at 0.2, so it is the best value measured, not a proven optimum. `rrf_k`,
`k_lexical` and `k_vector` were not tuned. The evidence-gate thresholds are calibrated with
the 0.2 weight, so recalibrate ([evaluation.md](evaluation.md)) if you change it.

Reranking improves the order of the top candidates and costs time on every search. Set
`rerank: false` for the lowest latency. With `rerank.provider: noop` in the configuration
no reranker runs at all.

## 5. Vector search tuning

Server-wide options (not per collection):

| Option | Default | Effect |
|---|---|---|
| `retrieval.ef_search` | `64` | HNSW candidates examined; higher is more accurate and slower |
| `retrieval.iterative_scan` | `relaxed_order` | lets pgvector keep scanning when filters drop rows |
| `retrieval.max_scan_tuples` | `20000` | cap on tuples an iterative scan visits |
| `retrieval.exact_scan_max` | `20000` | collections with at most this many chunks use an exact scan instead of the index |

Exact scans give perfect recall on small collections and are fast there. If the index scan
returns fewer than `k_vector` rows (for example under a selective filter), Footnote repeats
the vector leg as an exact scan. Both cases are visible in debug output and in
`footnote_search_exact_scan_total{reason}` (`small_collection`, `fallback`).

The lexical leg reads every chunk of the collection that the tenant can see, because
row-level security prevents Postgres from using the full-text index. Its cost grows with
the collection size. Keep very large corpora in separate collections.

## 6. Debugging a result

Send `"debug": true`:

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "debug": true}' | jq '.debug, .hits[0].debug'
```

- Each hit has `rrf_score`, `rerank_score`, `ranks` (rank per leg) and `scores` (lexical
  rank score, cosine similarity).
- The response has `space` (the embedding space used), `reranked`, `exact_scan_reason` and
  `stage_seconds` (`embed_query`, `lexical`, `vector`, `rerank`).

A chunk that ranks high in one leg and is absent from the other is normal. A hit with a
high `rrf_score` and a low `rerank_score` was surfaced by the legs and judged weak by the
reranker.

## 7. Embedding spaces and re-embedding

Each collection searches one active embedding space. Changing the space re-embeds the
collection in the background while searches keep using the old space; results switch once
at the end. See [models-and-data.md](models-and-data.md) and [operations.md](operations.md).

## 8. Limits

- `query` is 1 to 4000 characters and must not be blank.
- `k` is 1 to 100.
- A query with no lexical terms in common with the collection still returns vector hits in
  `hybrid` and `vector` modes.
- Search needs the collection's embedding model: a missing local model file or an
  unreachable `openai_compat` endpoint fails vector and hybrid searches with `503` or `502`
  ([errors.md](errors.md)). Lexical mode needs no embedder.
