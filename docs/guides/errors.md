# Error codes

Look up what an error code means.

Every non-2xx response has this body:

```json
{"error": {"code": "quota_exceeded", "message": "max_documents 100 would be exceeded (100 + 1)", "request_id": "01JA…"}}
```

- Match on `code`, never on `message`.
- Quote `request_id` when you report a problem.
- Field errors add `details: [{"field": "retrieval.k", "message": "…"}]`.

## Request errors

| Code | Status | Meaning |
|---|---|---|
| `validation_failed` | 422 | invalid request or setting; `details` names the field |
| `bad_request` | 400 | body could not be read |
| `unauthenticated` | 401 | missing, unknown, revoked or expired key; wrong key type for the route; no admin key configured |
| `not_found` | 404 | not found for this tenant (another tenant's ids look missing too) |
| `chunk_not_found` | 404 | chunk not in any current document version |
| `original_not_retained` | 404 | original removed (`ingest.retain_originals: false`) |
| `method_not_allowed` | 405 | wrong HTTP method for the route |
| `conflict` | 409 | conflicts with the current state |
| `collection_exists` | 409 | collection name already used |
| `document_not_ready` | 409 | no indexed version yet; retry after the ingest job is `done` |
| `evidence_threshold_uncalibrated` | 409 | no evidence threshold set ([citations.md](citations.md)) |
| `space_in_use` | 409 | a collection uses the space |
| `space_limit_reached` | 409 | over `embedding.max_spaces` |
| `file_too_large` | 413 | over `ingest.max_file_bytes` |
| `request_too_large` | 413 | body too large (1 MiB for JSON) |
| `unsupported_media_type` | 415 | not PDF, DOCX or UTF-8 text; detected from content |
| `pdf_encrypted` | 422 | encrypted PDF |
| `too_many_pages` | 422 | over `ingest.max_pages` |
| `archive_too_large` | 422 | DOCX over `ingest.max_uncompressed_bytes` or `ingest.max_archive_members` |
| `parse_failed` | 422 | corrupt PDF or DOCX |
| `embedding_dimension_unsupported` | 422 | space over 4000 dimensions |
| `webhook_url_rejected` | 422 | URL not HTTPS, malformed, unresolvable or not public ([webhooks.md](webhooks.md)) |
| `quota_exceeded` | 429 | tenant quota reached: `max_collections`, `max_documents`, `max_upload_bytes_per_day` |
| `internal` | 500 | unexpected error; logged under the `request_id` |
| `upstream_failed` | 502 | LLM, embedder or reranker failed after retries |
| `embedding_upstream_failed` | 502 | embedding endpoint failed after `embedding.max_retries` |
| `embedding_request_rejected` | 502 | embedding endpoint rejected the request (4xx other than 429) |
| `embedding_dimension_mismatch` | 502 | embedding endpoint returned the wrong dimension |
| `llm_not_configured` | 503 | `/answers` with empty `llm.base_url` |
| `embedding_not_configured` | 503 | `openai_compat` space with empty `embedding.base_url` |
| `unavailable` | 503 | a dependency is missing or unreachable |

## Streaming answer errors

After a stream starts, errors arrive as a last event:

```text
event: error
data: {"code": "upstream_failed", "message": "llm stream ended before [DONE]"}
```

| Code | Meaning |
|---|---|
| `upstream_failed` | LLM stream failed or was cut |
| `internal` | unexpected error |

## Document errors

A failed document shows `error: {"code", "message"}`. Codes marked *retried* retry up to
`ingest.job_max_attempts` first.

| Code | Meaning |
|---|---|
| `parse_failed` | file could not be read |
| `parse_timeout` | parsing took longer than `ingest.parse_timeout` |
| `pdf_encrypted` | encrypted PDF |
| `too_many_pages` | over `ingest.max_pages` |
| `archive_too_large` | DOCX too large or too many members |
| `unsupported_media_type` | no parser for the stored file |
| `quota_exceeded` | over the tenant's `max_chunks` |
| `embedding_dimension_mismatch` | embedder returned the wrong dimension (*retried* for `openai_compat`) |
| `upstream_failed` | *Retried.* a provider kept failing |
| `embedding_upstream_failed` | *Retried.* embedding endpoint kept failing |
| `embedding_request_rejected` | *Retried.* embedding endpoint rejected the request |
| `embedding_not_configured` | *Retried.* the space needs `embedding.base_url` |
| `unavailable` | *Retried.* the space's model is missing |
| `vectors_missing` | *Retried.* a re-embed started during ingest |
| `internal` | *Retried.* unexpected error |

## Job errors

A failed job shows `last_error` in `GET /v1/jobs/{job_id}`.

| Code in `last_error` | Job kind | Meaning |
|---|---|---|
| `reembed_incomplete` | `reembed` | documents kept changing; the job retries |
| any document error code | `ingest` | the version failed; same error as the document |
| `ingest_failed` | `ingest` | permanent ingest failure, no specific code |
| `upstream_failed` and other provider codes | `ingest`, `reembed`, `webhook` | provider failed after retries |
| (policy message) | `webhook` | URL failed the webhook network policy at delivery |
