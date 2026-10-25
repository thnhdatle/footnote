# HTTP API guide

Call every Footnote API resource with `curl`.

The full schema is the OpenAPI document. Error codes are in [errors.md](errors.md).

## 1. Overview

| Listener | Default | Serves |
|---|---|---|
| `server.listen` | `127.0.0.1:8000` | tenant API, admin API (`/v1/admin`), `/healthz`, `/readyz`, `/v1/version`, `/openapi.json` |
| `admin.listen` | `127.0.0.1:9094` | `/healthz`, `/readyz`, `/metrics` |

The admin listener has no authentication. Keep it on loopback.

```sh
uv run python -m footnote.api.openapi    # regenerate docs/api/openapi.json after an API change
```

The examples use these variables:

```sh
export FOOTNOTE_URL=http://127.0.0.1:8000
export ADMIN_KEY=...      # the value of FOOTNOTE_AUTH_ADMIN_KEY
export KEY=fn_...         # a tenant API key (section 10)
export COLLECTION=...     # a collection id (section 5)
```

## 2. Authentication

```sh
curl -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/collections          # tenant routes
curl -H "Authorization: Bearer $ADMIN_KEY" $FOOTNOTE_URL/v1/admin/tenants  # admin routes
```

- A bad, revoked or expired key is `401 unauthenticated`.
- A key on the wrong kind of route is `401`.
- Without a configured admin key, all `/v1/admin` requests are `401`.
- Another tenant's ids are `404`.

## 3. Conventions

- JSON bodies with snake_case fields. Unknown fields are `422 validation_failed`.
- Errors use one envelope ([errors.md](errors.md)).
- `X-Request-Id` matching `[A-Za-z0-9._-]{1,128}` is kept, otherwise generated.
- Lists take `?limit=` (1-500, default 50) and `?cursor=`. Pass `next_cursor` back as
  `?cursor=`; it is `null` on the last page.
- Keys, embedding spaces and webhook deliveries are not paginated.
- JSON bodies: at most 1 MiB. Uploads: `ingest.max_file_bytes` (default 50 MiB).
- Background work returns `202` with a `job_id` (section 9).
- Timestamps are ISO 8601. Ids are UUIDs.

## 4. Quick start

[getting-started.md](getting-started.md) creates a tenant, a key and a collection, then
uploads, searches and asks.

## 5. Collections

| Route | Does |
|---|---|
| `POST /v1/collections` | create with `name` (1-256 characters) and optional settings (`201`) |
| `GET /v1/collections` | list |
| `GET /v1/collections/{id}` | read, with running `jobs` |
| `PATCH /v1/collections/{id}` | change settings |
| `DELETE /v1/collections/{id}` | delete (`202` with `job_id`) |

Omitted settings use the configuration defaults. A duplicate name is `409`.

### Settings groups

| Group | Field | Bounds |
|---|---|---|
| `chunking` | `strategy` | `heading`, `recursive`, `fixed` |
| | `max_tokens` | 32-2048 |
| | `overlap_tokens` | 0 to `max_tokens`/2 |
| | `min_tokens` | 0 to `max_tokens` |
| | `tokenizer` | manifest id with a tokenizer |
| `retrieval` | `lexical_match` | `any`, `all`, `websearch` |
| | `lexical_rank` | `ts_rank`, `ts_rank_cd` |
| | `k` | 1-100 |
| | `k_lexical`, `k_vector` | 1-1000 each |
| | `rrf_k` | 1-1000 |
| | `rrf_weights` | `{"lexical", "vector"}`, 0-10, not both 0 |
| | `rerank` | boolean |
| | `rerank_n` | 1-200, at least `k` |
| `answer` | `evidence_threshold_rerank`, `evidence_threshold_vector` | 0-1, or `null` for the default |
| | `prompt_template` | template name |
| | `unsupported_as_abstain` | boolean |
| top level | `language` | `simple`, `english` |
| | `embedding_space` | registered space name |
| | `webhook` | see below |

### Updating settings

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"chunking": {"max_tokens": 256, "overlap_tokens": 32}}'    # starts a reindex job
```

Retrieval, answer and webhook changes return `200`. These return `202` with a job:

| Change | Job |
|---|---|
| `chunking` | `reindex` |
| `language` | `fts_reconfig` |
| `embedding_space` | `reembed` (progress in `reembed`) |

### Webhook

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"webhook": {"url": "https://example.com/hooks/footnote", "events": ["document.ready"]}}'
```

`webhook` takes `url`, `events` and `rotate_secret`. The secret is returned once, as
`webhook_secret`. Deliveries are at `GET /v1/collections/{id}/webhook-deliveries`. See
[webhooks.md](webhooks.md).

## 6. Documents

Routes are under `/v1/collections/{collection_id}/documents`.

### Upload

`POST` with `multipart/form-data`:

| Part | Required | Notes |
|---|---|---|
| `file` | yes | one part only |
| `external_id` | no | up to 512 characters; reuse for a new version |
| `title` | no | up to 512 characters |
| `metadata` | no | JSON object, at most 16 KiB |

It returns `202` with `document_id`, `version` and `job_id`. An identical upload returns
`200` with `"unchanged": true`. See [ingestion.md](ingestion.md) for limits and
rejections.

### Other document routes

| Route | Does |
|---|---|
| `GET .../documents` | list |
| `GET .../documents/{doc_id}` | read, with `status`, `error` and `versions` |
| `PATCH .../documents/{doc_id}` | change `metadata` or `title` |
| `DELETE .../documents/{doc_id}` | delete all versions (`204`) |
| `GET .../documents/{doc_id}/text` | text as `text/plain` |
| `GET .../text?format=elements` | elements with pages and headings |
| `GET .../documents/{doc_id}/original` | current original |
| `GET .../versions/{version}/original` | any version's original |

Text routes return `409 document_not_ready` before indexing. Original routes return
`404 original_not_retained` with `ingest.retain_originals: false`.

`?format=elements` returns the elements in reading order:

```json
{"data": [
  {"kind": "heading", "text": "Financial overview", "level": 2,
   "heading_path": ["Annual report 2025", "Financial overview"], "page": 2, "page_to": 2, "order": 5},
  {"kind": "image", "text": "A bar chart of revenue by quarter...", "level": null,
   "heading_path": ["Annual report 2025", "Financial overview"], "page": 2, "page_to": 2, "order": 6}
]}
```

`kind` is `heading`, `paragraph`, `list_item`, `table`, `code` or `image`. `level` is set on
headings only. `heading_path` lists the enclosing headings. `page` is null for formats without
pages.

### Chunks

`GET /v1/collections/{id}/chunks/{chunk_id}` returns a chunk's full text. A chunk not in a
current version is `404 chunk_not_found`.

## 7. Search

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "mode": "hybrid", "rerank": true, "k": 5,
       "filters": {"metadata": {"team": {"in": ["hr", "legal"]}},
                   "created_after": "2026-01-01T00:00:00Z"},
       "debug": true}'
```

| Field | Notes |
|---|---|
| `query` | required, 1-4000 characters |
| `filters.metadata` | equality or `{"in": [..]}`; at most 100 values, 32 keys |
| `filters.document_ids` | at most 1000 ids |
| `filters.created_after`, `filters.created_before` | ISO 8601 |
| `mode` | `hybrid` (default), `vector`, `lexical` |
| `rerank` | default from the collection |
| `k` | 1-100; default from the collection |
| `debug` | default false |

The response is `{"hits": [...]}`. See [retrieval.md](retrieval.md).

## 8. Answers

`POST /v1/collections/{id}/answers` takes `question` (required), `filters`, `mode`,
`rerank` and `stream` (default false). It needs `llm.base_url`.

### JSON response

| Field | Meaning |
|---|---|
| `answer` | text with `[n]` markers, or `null` |
| `citations` | `{n, chunk_id, document_id, title, page_from, page_to, quote_start, quote_end}` |
| `abstained`, `abstention_reason` | `insufficient_evidence` or `unsupported` |
| `unsupported` | no valid citation |
| `evidence_score` | gate score (may be null) |
| `usage` | `{input_tokens, output_tokens}` |
| `trace_id` | trace id, or null |

See [citations.md](citations.md).

### Errors

| Status | Cause |
|---|---|
| `503 llm_not_configured` | empty `llm.base_url` |
| `409 evidence_threshold_uncalibrated` | no evidence threshold |
| `502` | the LLM failed |

### Streaming

```sh
curl -N -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/answers \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?", "stream": true}'    # -N: no buffering
```

| Event | Data |
|---|---|
| `retrieval` | `{"hits": [...]}` |
| `delta` | `{"text": "..."}`, zero or more |
| `citations` | `{"citations": [...], "answer": "...", "unsupported": false}` |
| `done` | `usage`, `abstained`, `abstention_reason`, `unsupported`, `trace_id` |
| `error` | `{"code", "message"}`, replaces `citations` and `done` |

Show `delta` text only as a preview. Use `citations.answer` as the final text.

## 9. Jobs

```sh
curl -s -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/jobs/$JOB    # poll until done or failed
```

- `status`: `queued`, `running`, `done`, `failed`.
- `kind`: `ingest`, `reindex`, `fts_reconfig`, `reembed`, `gc`, `webhook`,
  `collection_delete`, `tenant_delete`.
- A failed job has `last_error`.

## 10. Admin API

Routes need the admin key.

### Tenants

| Route | Does |
|---|---|
| `POST /v1/admin/tenants` | create with `name` and optional `quotas` (`201`) |
| `GET /v1/admin/tenants`, `GET /v1/admin/tenants/{id}` | list, read |
| `PATCH /v1/admin/tenants/{id}` | change `name` or `quotas` |
| `DELETE /v1/admin/tenants/{id}` | revoke keys, delete data (`202` with `job_id`) |
| `GET /v1/admin/tenants/{id}/usage` | `collections`, `documents`, `chunks`, `vectors`, `stored_bytes` |
| `GET /v1/admin/tenants/{id}/export` | NDJSON export; `?include_vectors=true`, `?include_originals=true` |

Quotas: `max_collections`, `max_documents`, `max_chunks`, `max_upload_bytes_per_day`.
`0` means unlimited. See [multi-tenancy.md](multi-tenancy.md).

### Keys

| Route | Does |
|---|---|
| `POST /v1/admin/tenants/{id}/keys` | create with `name` (default `default`) and `expires_at`; key shown once (`201`) |
| `GET /v1/admin/tenants/{id}/keys` | list |
| `DELETE /v1/admin/tenants/{id}/keys/{key_id}` | revoke at once (`204`) |
| `POST /v1/admin/tenants/{id}/keys/{key_id}/rotate` | new key (`201`); old one valid for `auth.key_rotation_grace` (default 24h) |

### Embedding spaces

| Route | Does |
|---|---|
| `POST /v1/admin/embedding-spaces` | register (`201`) |
| `GET /v1/admin/embedding-spaces` | list |
| `DELETE /v1/admin/embedding-spaces/{name}` | drop (`204`); `409 space_in_use` while used |

| Field | Notes |
|---|---|
| `name` | lowercase letters, digits, `.`, `_`, `-`; up to 63 |
| `provider` | `onnx_local` or `openai_compat` |
| `model` | model name |
| `dimensions` | required unless the manifest knows the model; up to 4000 |
| `send_dimensions` | default false |
| `normalise` | default true |
| `calibrated_threshold` | 0-1 or null |

## 11. Operations endpoints

| Route | Returns |
|---|---|
| `GET /healthz` | `{"status": "ok"}`, or `503` while shutting down |
| `GET /readyz` | `{"status": "ready", ...}`, or `503` with problems |
| `GET /metrics` | Prometheus metrics, admin listener only |
| `GET /v1/version` | `version`, `commit`, `build_date`, `python_version` |

## 12. Programmatic embedding

Build one `ApiDeps` and pass it to both factories. The contract is in
`src/footnote/api/deps.py`.

```python
from footnote.api import ApiDeps, ApiSettings, ReadinessCheck, VersionInfo
from footnote.api import create_admin_app, create_app

deps = ApiDeps(
    collections=...,
    documents=...,
    admin=...,
    search=...,
    answers=...,  # app-layer services
    jobs=...,
    keys=...,
    metrics=...,
    logger=...,  # ports, metrics, log
    settings=ApiSettings(admin_key="...", version=VersionInfo("0.1.0")),
    readiness=(ReadinessCheck("database", check_database),),
)
app = create_app(deps)  # serve with uvicorn on server.listen
admin = create_admin_app(deps)  # serve with uvicorn on admin.listen
```

The factories open no connections. You own pools, the worker and the server lifecycle.
