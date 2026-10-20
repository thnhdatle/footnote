# Retrieval

Search a collection, then tune or debug the results.

All options are in [configuration.md](configuration.md).

## 1. Modes

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "mode": "hybrid", "rerank": true, "k": 5}'
```

| `mode` | Searches | Use it for |
|---|---|---|
| `hybrid` (default) | full-text and vector | most queries |
| `vector` | vector only | paraphrased questions |
| `lexical` | full-text only | exact names and codes |

Only current document versions are searched.

## 2. Filters

```json
{"query": "parental leave",
 "filters": {"metadata": {"team": {"in": ["hr", "legal"]}, "lang": "en"},
             "document_ids": ["c91b..."],
             "created_after": "2026-01-01T00:00:00Z"}}
```

Every `metadata` key must match. `{"in": [...]}` matches any listed value.

## 3. Collection settings

Set them with `PATCH /v1/collections/{id}`. They apply on the next search.

| Setting | Default | Effect |
|---|---|---|
| `retrieval.lexical_match` | `any` | `any`, `all` or `websearch` terms |
| `retrieval.lexical_rank` | `ts_rank` | `ts_rank` or `ts_rank_cd` |
| `retrieval.rrf_weights` | lexical 0.2, vector 1.0 | weight of each search |
| `retrieval.rrf_k` | `60` | larger flattens ranks |
| `retrieval.k_lexical`, `k_vector` | `50` | candidates per search |
| `retrieval.rerank` | `true` | run the reranker |
| `retrieval.rerank_n` | `30` | candidates reranked; at least `k` |
| `retrieval.k` | `8` | results returned |
| `language` | `simple` | `simple` or `english` (stems words) |

For noisy full-text results, lower the lexical weight rather than using `all`. After
changing weights, recalibrate the evidence threshold ([evaluation.md](evaluation.md)).
Set `rerank: false` for lower latency.

## 4. Vector search tuning

Server-wide options:

| Option | Default | Effect |
|---|---|---|
| `retrieval.ef_search` | `64` | higher is more accurate, slower |
| `retrieval.iterative_scan` | `relaxed_order` | keeps scanning under filters |
| `retrieval.max_scan_tuples` | `20000` | iterative scan row cap |
| `retrieval.exact_scan_max` | `20000` | exact scan up to this many chunks |

Full-text cost grows with collection size. Split very large corpora into collections.

## 5. Debugging a result

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "debug": true}' | jq '.debug, .hits[0].debug'
```

Each hit shows `rrf_score`, `rerank_score`, `ranks` and `scores`. The response shows the
space, whether reranking ran, and `stage_seconds`.

## 6. Limits

- `query`: 1 to 4000 characters.
- `k`: 1 to 100.
- Vector and hybrid search need the embedding model, else `503` or `502`
  ([errors.md](errors.md)). Lexical mode does not.
