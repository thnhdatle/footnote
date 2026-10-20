# HTTP API guide

This guide shows how to use Footnote's HTTP API: authentication, conventions and each
resource, with `curl` examples. The authoritative list of paths, fields and statuses is
the OpenAPI document; this guide explains how the pieces fit together. Error codes are
registered in [errors.md](errors.md).

## 1. Overview

Footnote exposes two HTTP listeners.

- Data plane (`server.listen`, default `127.0.0.1:8000`): the tenant API
  (`/v1/collections...`, `/v1/jobs/...`), the admin API (`/v1/admin/...`), `/healthz`,
  `/readyz`, `/v1/version` and `/openapi.json`.
- Admin listener (`admin.listen`, default `127.0.0.1:9094`): only `/healthz`, `/readyz`
  and `/metrics`. It has no authentication, carries no tenant data and is meant to stay on
  loopback (Prometheus scrapes and `footnote healthcheck` call it). Do not confuse it with
  the admin API, which lives on the data plane under `/v1/admin`.

The API is described by an OpenAPI 3.1 document, committed at `docs/api/openapi.json` and
served by a running data plane at `GET /openapi.json`. It covers the data plane only. To
regenerate the committed file after changing a route or schema:

```sh
uv run python -m footnote.api.openapi
```

A unit test fails when the committed document drifts from the code.

The examples below assume these shell variables:

```sh
export FOOTNOTE_URL=http://127.0.0.1:8000
export ADMIN_KEY=...      # the value of FOOTNOTE_AUTH_ADMIN_KEY
export KEY=fn_...         # a tenant API key (section 4)
export COLLECTION=...     # a collection id (section 4)
```

## 2. Authentication

Tenant routes take a tenant API key:

```sh
curl -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/collections
```

- Keys start with `fn_`. An unknown, revoked or expired key, a missing header and a
  malformed header are all `401 unauthenticated`.
- Every request runs under the resolved tenant's row-level security. Ids of another
  tenant's resources (collections, documents, chunks, jobs) are `404`, never `403`.
- Create and revoke keys with the admin API (section 10).

Admin routes (`/v1/admin/...`) take the admin key instead:

```sh
curl -H "Authorization: Bearer $ADMIN_KEY" $FOOTNOTE_URL/v1/admin/tenants
```

- The key comes from `FOOTNOTE_AUTH_ADMIN_KEY` or the file named by `auth.admin_key_file`.
  See [configuration.md](configuration.md).
- When no admin key is configured the admin API is disabled: every `/v1/admin` request is
  `401`.
- Tenant keys are `401` on `/v1/admin`, and the admin key is `401` on tenant routes.

## 3. Conventions

- Bodies are JSON with snake_case field names. Request bodies reject unknown fields with
  `422 validation_failed`.
- Every non-2xx response uses one envelope (see [errors.md](errors.md) for all codes):

```json
{"error": {"code": "validation_failed", "message": "invalid request",
           "request_id": "01JA...", "details": [{"field": "k", "message": "must be <= 100"}]}}
```

  `details` is present for validation errors. Common codes: `unauthenticated` (401),
  `not_found` (404), `validation_failed` (422), `request_too_large` (413),
  `quota_exceeded` (429).
- `X-Request-Id`: a request header is accepted if it matches `[A-Za-z0-9._-]{1,128}`;
  otherwise a ULID is generated. The id in use is always echoed in the `X-Request-Id`
  response header and in the error envelope.
- Cursor pagination on list endpoints: `?limit=` (1-500, default 50) and `?cursor=`
  (opaque). The response is `{"data": [...], "next_cursor": "..."}`; pass `next_cursor`
  as `?cursor=` for the next page. `next_cursor` is `null` on the last page. A malformed
  cursor is `422`. Lists are ordered by id.
  - Three lists return everything in one response and always have `next_cursor: null`:
    keys of a tenant, embedding spaces, and webhook deliveries (deliveries accept `?limit=`
    only and return the newest first).
- Request size limits: JSON bodies are limited to 1 MiB; a larger body is
  `413 request_too_large`. Uploads are limited by `ingest.max_file_bytes` (default 50 MiB;
  `413 file_too_large`).
- Background work answers `202 Accepted` with a `job_id` and is polled with
  `GET /v1/jobs/{job_id}` (section 9).
- Timestamps are ISO 8601; ids are UUIDs.

## 4. Quick start

Create a tenant and a key (admin):

```sh
curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "acme"}'
```

```json
{"id": "0b6e...", "name": "acme", "status": "active",
 "quotas": {"max_collections": 0, "max_documents": 0, "max_chunks": 0,
            "max_upload_bytes_per_day": 0}}
```

```sh
curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants/0b6e.../keys \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "ci"}'
```

```json
{"id": "5c1d...", "tenant_id": "0b6e...", "prefix": "fn_AB12", "name": "ci",
 "created_at": "2026-10-20T09:00:00Z", "last_used_at": null, "revoked_at": null,
 "expires_at": null, "revoked": false, "key": "fn_AB12..."}
```

`key` is shown once; store it as `$KEY`. Create a collection (settings default from
config):

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"name": "handbook"}'
```

```json
{"id": "7a3f...", "name": "handbook", "status": "active", "chunk_count": 0,
 "active_space": "bge-small-en-v1.5-onnx", "embedding_space": "bge-small-en-v1.5-onnx",
 "language": "simple", "reembed": null,
 "chunking": {"strategy": "heading", "max_tokens": 400, "overlap_tokens": 40,
              "min_tokens": 50, "tokenizer": "bge-small-en-v1.5"},
 "retrieval": {"k": 8, "rerank": true, "...": "..."},
 "answer": {"evidence_threshold_rerank": null, "evidence_threshold_vector": null,
            "prompt_template": "answer_v1", "unsupported_as_abstain": false},
 "webhook": {"url": null, "events": ["document.ready", "document.failed"],
             "has_secret": false},
 "jobs": [], "created_at": "...", "updated_at": "...",
 "job_id": null, "job_ids": [], "webhook_secret": null}
```

Upload a document and poll its job:

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/documents \
  -H "Authorization: Bearer $KEY" \
  -F file=@handbook.pdf -F external_id=handbook-2026 \
  -F 'metadata={"team": "hr"}'
```

```json
{"document_id": "c91b...", "unchanged": false, "version": 1, "job_id": "e2d4..."}
```

```sh
curl -s -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/jobs/e2d4...
```

```json
{"id": "e2d4...", "kind": "ingest", "status": "done", "attempts": 1, "last_error": null,
 "progress": {"done": 12, "total": 12}, "run_at": "...", "created_at": "...",
 "started_at": "...", "finished_at": "...", "updated_at": "..."}
```

Search:

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "how many vacation days do I get?", "k": 3}'
```

```json
{"hits": [{"chunk_id": "4f0a...", "document_id": "c91b...", "title": "handbook.pdf",
           "heading_path": ["Benefits", "Time off"], "page_from": 14, "page_to": 14,
           "text": "Full-time employees receive 25 vacation days per year...",
           "debug": null}],
 "debug": null}
```

Answer (JSON):

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/answers \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
```

```json
{"answer": "Full-time employees receive 25 vacation days per year [1].",
 "citations": [{"n": 1, "chunk_id": "4f0a...", "document_id": "c91b...",
                "title": "handbook.pdf", "page_from": 14, "page_to": 14,
                "quote_start": 0, "quote_end": 54}],
 "abstained": false, "abstention_reason": null, "unsupported": false,
 "evidence_score": 0.93, "usage": {"input_tokens": 612, "output_tokens": 18},
 "trace_id": null}
```

Answer streamed (`-N` disables curl's buffering):

```sh
curl -N -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/answers \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?", "stream": true}'
```

The event sequence is described in section 8.

## 5. Collections

`POST /v1/collections` takes `name` (1-256 characters) and optional settings. Every
omitted setting takes its default from the configuration (`chunking`, `retrieval`,
`answer`, `collections.default_language`, `embedding.default_space`). Settings may be sent
partially: omitted fields inside a group keep their default. Returns `201` with the
collection. A duplicate name is `409`; a tenant over `max_collections` is
`429 quota_exceeded`.

### Settings groups

| Group | Field | Bounds |
|---|---|---|
| `chunking` | `strategy` | `heading`, `recursive`, `fixed` |
| | `max_tokens` | 32-2048 |
| | `overlap_tokens` | 0 to `max_tokens`/2 |
| | `min_tokens` | 0 to `max_tokens` |
| | `tokenizer` | a manifest id that has a tokenizer file |
| `retrieval` | `lexical_match` | `any`, `all`, `websearch` |
| | `lexical_rank` | `ts_rank`, `ts_rank_cd` |
| | `k` | 1-100 |
| | `k_lexical`, `k_vector` | 1-1000 each |
| | `rrf_k` | 1-1000 |
| | `rrf_weights` | `{"lexical", "vector"}`, 0-10 each, not both 0 |
| | `rerank` | boolean |
| | `rerank_n` | 1-200 and at least `k` |
| `answer` | `evidence_threshold_rerank`, `evidence_threshold_vector` | 0-1, or `null` to fall back to config or the calibrated value |
| | `prompt_template` | template name |
| | `unsupported_as_abstain` | boolean |
| top level | `language` | `simple`, `english` |
| | `embedding_space` | a registered space name |
| | `webhook` | see below |

Out-of-range values are `422 validation_failed` with `details` naming the field.

The collection resource also shows `status`, `chunk_count`, `active_space`,
`embedding_space` (the target while a re-embed runs, otherwise the active space),
`reembed`, and, on single-collection reads, `jobs` in progress. `GET /v1/collections`
lists collections (paginated), `GET /v1/collections/{id}` reads one.

### Updating settings

`PATCH /v1/collections/{id}` takes the same settings (all optional, partial per group).

- `200` when nothing needs background work (retrieval, answer, webhook changes).
- `202` with `job_id` (and `job_ids`) when a job starts:
  - changing `chunking` starts a `reindex` job (each document is re-chunked from its
    stored elements);
  - changing `language` starts an `fts_reconfig` job;
  - changing `embedding_space` starts a `reembed` job.
- The response body is always the collection plus `job_id`, `job_ids` and
  `webhook_secret`. Poll jobs with `GET /v1/jobs/{job_id}`.
- Queries keep working during a re-embed and keep using the active space. Progress is in
  the collection's `reembed` field: `{"target_space": "...", "done": 120, "total": 400,
  "state": "running"}`. When the job finishes, `active_space` switches to the target.
  Setting `embedding_space` back to the active space cancels the re-embed.

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"chunking": {"max_tokens": 256, "overlap_tokens": 32}}'
```

### Webhook

`webhook` is `{"url", "events", "rotate_secret"}`.

- `events` is any of `document.ready`, `document.failed` (default both).
- Setting a URL generates a signing secret, returned once as `webhook_secret` in the
  create or patch response; later reads show only `webhook.has_secret`. A changed URL gets
  a new secret.
- `"rotate_secret": true` replaces the secret and returns the new one (`422` when the
  collection has no URL).
- `"url": null` removes the webhook.
- A URL that fails the SSRF policy is `422 webhook_url_rejected` (HTTPS only and public
  addresses only, unless `webhooks.allow_http` / `webhooks.allow_private` are set).
- Deliveries are POSTs with `X-Footnote-Timestamp` and
  `X-Footnote-Signature: sha256=<hex HMAC-SHA256(secret, "<timestamp>.<body>")>`.

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"webhook": {"url": "https://example.com/hooks/footnote", "events": ["document.ready"]}}'
```

`GET /v1/collections/{id}/webhook-deliveries?limit=` lists delivery attempts, newest
first: `id`, `event`, `url`, `status`, `attempts`, `response_code`, `error`, and
timestamps.

### Deleting

`DELETE /v1/collections/{id}` returns `202` with `{"job_id": ...}`. The collection
disappears from reads at once; a `collection_delete` job removes its rows.

## 6. Documents

All routes are under `/v1/collections/{collection_id}/documents`.

### Upload

`POST` with `multipart/form-data`:

| Part | Required | Notes |
|---|---|---|
| `file` | yes | the file; one part only |
| `external_id` | no | up to 512 characters; reusing one creates a new version of that document |
| `title` | no | up to 512 characters |
| `metadata` | no | JSON object, at most 16 KiB |

- The file type is sniffed from the content (PDF, DOCX, HTML, Markdown, plain text);
  the `Content-Type` of the part is ignored.
- `202` returns `{"document_id", "unchanged": false, "version", "job_id"}`.
- `200` with `"unchanged": true` (no `version`, no `job_id`) means the file, title and
  chunking settings equal the current version's; no version is created, but `metadata` is
  still updated.
- Synchronous rejections:
  - `413 file_too_large` (over `ingest.max_file_bytes`) or `request_too_large`;
  - `415 unsupported_media_type`;
  - `422` for `pdf_encrypted`, `too_many_pages`, `archive_too_large`, or an invalid form
    or `metadata`;
  - `429 quota_exceeded` (documents or daily upload bytes);
  - `404` for an unknown collection.

### Status

`GET /v1/collections/{id}/documents` lists documents (paginated);
`GET .../documents/{doc_id}` returns one with `chunk_count`, `current_version` and
`versions`.

- `status` is `queued`, `parsing`, `embedding`, `ready` or `failed`. A failed document has
  `error: {"code", "message"}` (for example `parse_failed`, `parse_timeout`,
  `quota_exceeded`).
- Each version has `version`, `status` (`processing`, `ready`, `failed`, `superseded`),
  `media_type`, `bytes`, `pages`, `sha256`, `indexed_title`, `error`, `has_original` and
  `created_at`.
- Queries see the old or the new version of a document, never a mix.

### Change and delete

- `PATCH .../documents/{doc_id}` with `metadata` and/or `title` (up to 512) needs no
  re-upload. Metadata filters see the new metadata at once; the indexed title changes
  with the next version.
- `DELETE .../documents/{doc_id}` removes all versions, chunks, vectors and stored
  originals. Returns `204`.

### Text, elements and originals

- `GET .../documents/{doc_id}/text` returns the current version's text as `text/plain`
  (elements joined by blank lines).
- `GET .../text?format=elements` returns `{"data": [{"kind", "text", "level",
  "heading_path", "page", "page_to", "order"}, ...]}`.
- Both are `409 document_not_ready` before the document's first version is indexed.
- `GET .../documents/{doc_id}/original` returns the current version's original bytes with
  the sniffed media type; `GET .../versions/{version}/original` returns any version,
  including superseded ones. `404 original_not_retained` when originals are not kept
  (`ingest.retain_originals: false`).

### Chunks

`GET /v1/collections/{id}/chunks/{chunk_id}` returns `id`, `collection_id`,
`document_id`, `document_title`, `text`, `heading_path`, `token_count` and
`memberships` (`version`, `ordinal`, `page_from`, `page_to`). A chunk that is not in a
current version is `404 chunk_not_found`. Use it to read the full text behind a search hit
or citation.

## 7. Search

`POST /v1/collections/{id}/search`:

| Field | Notes |
|---|---|
| `query` | required, 1-4000 characters |
| `filters.metadata` | object; a scalar means equality, `{"in": [..]}` means any of the values (at most 100 values, 32 keys) |
| `filters.document_ids` | list of document ids (at most 1000) |
| `filters.created_after`, `filters.created_before` | ISO 8601 timestamps |
| `mode` | `hybrid` (default, `retrieval.default_mode`), `vector`, `lexical` |
| `rerank` | boolean; default the collection's `retrieval.rerank` |
| `k` | 1-100; default the collection's `k` |
| `debug` | boolean, default false |

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "parental leave", "mode": "hybrid", "rerank": true, "k": 5,
       "filters": {"metadata": {"team": {"in": ["hr", "legal"]}},
                   "created_after": "2026-01-01T00:00:00Z"},
       "debug": true}'
```

The response is `{"hits": [...], "debug": ...}`. Each hit has `chunk_id`, `document_id`,
`title`, `heading_path`, `page_from`, `page_to` (the first current-version membership) and
`text`. With `debug: true`:

- each hit has `debug: {"rrf_score", "rerank_score", "ranks", "scores"}`;
- the response has `debug: {"space", "reranked", "exact_scan_reason", "stage_seconds"}`
  (the space name, whether the reranker ran, why an exact scan was used if one was, and
  per-stage timings).

## 8. Answers

`POST /v1/collections/{id}/answers` takes `question` (required), `filters`, `mode`,
`rerank` (as in search) and `stream` (default false). It requires a configured LLM
(`llm.base_url`).

### JSON response

- `answer`: the text with verified `[n]` markers, or `null` when abstained.
- `citations`: `{n, chunk_id, document_id, title, page_from, page_to, quote_start,
  quote_end}`; `quote_start` and `quote_end` are character offsets into the cited chunk's
  text (see the chunk endpoint).
- `abstained`, `abstention_reason` (`insufficient_evidence` or `unsupported`).
- `unsupported`: the model produced an answer with no valid citation. With
  `answer.unsupported_as_abstain` it is reported as an abstention instead.
- `evidence_score`: the score the evidence gate compared with the threshold (may be null).
- `usage`: `{input_tokens, output_tokens}`; `trace_id`: the W3C trace id when tracing is on,
  else null.

### Abstention

When the best evidence is below the threshold, no LLM call is made: `answer` is `null`,
`abstained` is `true` and `abstention_reason` is `insufficient_evidence`.

### Errors

- `503 llm_not_configured`: `llm.base_url` is empty.
- `409 evidence_threshold_uncalibrated`: no threshold is set and none is calibrated for the
  collection's reranker or space.
- `502`: the LLM provider failed (before any output).
- `404`, `422` as usual.

### Streaming

With `"stream": true` the response is `text/event-stream`. Each event is
`event: <name>` and `data: <JSON>`. The sequence:

1. `retrieval`: `{"hits": [...]}` (same hit shape as search).
2. `delta`: `{"text": "..."}`, zero or more, raw model text.
3. `citations`: `{"citations": [...], "answer": "<cleaned text>", "unsupported": false}`.
4. `done`: `usage`, `abstained`, `abstention_reason`, `unsupported` and `trace_id`
   (no `evidence_score`).

An abstention streams `retrieval`, `citations` (empty list, `answer: null`) and `done`.
If the LLM fails mid-stream the last event is `error` with `{"code", "message"}` in
place of `citations` and `done`.

The first event is produced before the response starts, so validation errors, `404`,
`503 llm_not_configured` and `409 evidence_threshold_uncalibrated` are normal JSON error
responses with the usual envelope.

```text
event: retrieval
data: {"hits": [{"chunk_id": "4f0a...", "title": "handbook.pdf", "text": "Full-time employees receive 25 vacation days...", "...": "..."}]}

event: delta
data: {"text": "Full-time employees "}

event: delta
data: {"text": "receive 25 vacation days per year [1]."}

event: citations
data: {"citations": [{"n": 1, "chunk_id": "4f0a...", "quote_start": 0, "quote_end": 54, "...": "..."}], "answer": "Full-time employees receive 25 vacation days per year [1].", "unsupported": false}

event: done
data: {"abstained": false, "unsupported": false, "usage": {"input_tokens": 612, "output_tokens": 18}, "trace_id": null, "...": "..."}
```

The cleaned answer and citations in the `citations` event equal those of the
non-streamed request; markers can be split across `delta` events, so render `delta` text
only as a preview and use `citations.answer` as the final text.

## 9. Jobs

`GET /v1/jobs/{job_id}` is tenant-scoped (another tenant's id is `404`). Fields: `id`,
`kind`, `status`, `attempts`, `last_error`, `progress` (`{"done", "total"}` where the kind
has one, otherwise null), `run_at`, `created_at`, `started_at`, `finished_at`,
`updated_at`.

- `status`: `queued`, `running`, `done`, `failed`. A job is retried with backoff up to
  `ingest.job_max_attempts`, then `failed` with `last_error`.
- `kind`: `ingest`, `reindex`, `fts_reconfig`, `reembed`, `gc`, `webhook`,
  `collection_delete`, `tenant_delete`. The ones the API returns in a `job_id` are
  `ingest` (upload), `reindex`, `fts_reconfig`, `reembed`, `collection_delete` and
  `tenant_delete`.
- A collection read lists its queued or running jobs in `jobs` (`id`, `kind`, `status`,
  `progress`).

Poll with a short sleep until `status` is `done` or `failed`.

## 10. Admin API

All routes need the admin key (section 2) and live under `/v1/admin`.

### Tenants

- `POST /v1/admin/tenants` with `name` and optional `quotas` creates a tenant (`201`).
  Quotas default to `tenancy.default_quotas`; `0` means unlimited.
- `GET /v1/admin/tenants` lists tenants (paginated); `GET /v1/admin/tenants/{id}` reads one.
- `PATCH /v1/admin/tenants/{id}` changes `name` and/or `quotas`; omitted quota fields keep
  their value:

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/admin/tenants/$TENANT \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"quotas": {"max_documents": 1000}}'
```

  Quotas: `max_collections`, `max_documents`, `max_chunks`, `max_upload_bytes_per_day`.
- `DELETE /v1/admin/tenants/{id}` returns `202` with `job_id`. All keys are revoked
  immediately; data, keys and the tenant row are removed by a `tenant_delete` job.

### Usage and export

- `GET /v1/admin/tenants/{id}/usage` returns `collections`, `documents`, `chunks`,
  `vectors` and `stored_bytes`.
- `GET /v1/admin/tenants/{id}/export` streams `application/x-ndjson`: one
  `{"type": "...", "data": {...}}` object per line (collections, documents, versions,
  elements, chunks, memberships). Webhook secrets are never exported.
  - `?include_vectors=true` adds vectors.
  - `?include_originals=true` adds original files, base64 encoded.

```sh
curl -s -H "Authorization: Bearer $ADMIN_KEY" \
  "$FOOTNOTE_URL/v1/admin/tenants/$TENANT/export?include_originals=true" > acme.ndjson
```

### Keys

- `POST /v1/admin/tenants/{id}/keys` with optional `name` (default `default`) and
  `expires_at` returns `201` with the key record and `key`, shown once and never stored in
  clear.
- `GET /v1/admin/tenants/{id}/keys` lists all keys: `id`, `tenant_id`, `prefix`, `name`,
  `created_at`, `last_used_at`, `revoked_at`, `expires_at`, `revoked` (never the key).
- `DELETE /v1/admin/tenants/{id}/keys/{key_id}` revokes at once (`204`).
- `POST /v1/admin/tenants/{id}/keys/{key_id}/rotate` returns `201` with a replacement key
  (shown once). The old key stays valid for `auth.key_rotation_grace` (default 24h).

### Embedding spaces

- `POST /v1/admin/embedding-spaces` registers a space (`201`), idempotent by `name`:
  `name` (lowercase letters, digits, `.`, `_`, `-`; up to 63), `provider` (`onnx_local` or
  `openai_compat`), `model`, `dimensions` (required unless the model manifest knows the
  model), `send_dimensions` (default false), `normalise` (default true) and
  `calibrated_threshold` (0-1 or null).
- The response includes `id`, `index_type` and `dimensions`. `index_type` is `vector` up to
  2000 dimensions and `halfvec` from 2001 to 4000. More than 4000 dimensions is rejected
  with `422 embedding_dimension_unsupported`.
- Registering more spaces than `embedding.max_spaces` is `409 space_limit_reached`.
- `GET /v1/admin/embedding-spaces` lists every space (no pagination).
- `DELETE /v1/admin/embedding-spaces/{name}` drops a space and its index (`204`);
  `409 space_in_use` while a collection uses it.

## 11. Operations endpoints

- `GET /healthz`: `{"status": "ok"}`, or `503 {"status": "draining"}` during shutdown.
- `GET /readyz`: `{"status": "ready", "checks": {...}}`. When any check reports a problem
  the response is `503` with `{"status": "not_ready", "checks": {"database": ["..."]}}`
  listing the problems per check. Both are on the data plane and on the admin listener.
- `GET /metrics`: Prometheus exposition, on the admin listener only (default
  `127.0.0.1:9094`).
- `GET /v1/version` (data plane): `{"version", "commit", "build_date", "python_version"}`.

## 12. Programmatic embedding

Integrators and authors of a custom bootstrap build one `ApiDeps` and hand it to the two
application factories. See the docstring of `src/footnote/api/deps.py` for the contract.

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

- `ReadinessCheck(name, probe)` takes an async callable that returns a list of problems
  (empty means ready).
- `ApiSettings` also carries `access_log`, `max_json_bytes` (default 1 MiB),
  `multipart_overhead_bytes` and `readiness_timeout`.
- The factories open no connections and start no tasks; the caller owns pools, the worker
  and the server lifecycle.
