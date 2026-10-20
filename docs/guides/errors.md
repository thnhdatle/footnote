# Error codes

Every non-2xx response of the HTTP API has the same body:

```json
{"error": {"code": "quota_exceeded", "message": "max_documents 100 would be exceeded (100 + 1)", "request_id": "01JA…"}}
```

- `code` is stable, `snake_case` and part of the public API. Match on it, never on
  `message`.
- `request_id` equals the `X-Request-Id` response header (yours if you sent a valid
  one, otherwise a generated ULID). Quote it when reporting a problem.
- `422 validation_failed` (and the other 422 codes raised for one field) add
  `details: [{"field": "retrieval.k", "message": "…"}]`.
- `401` responses carry `WWW-Authenticate: Bearer realm="footnote"`.
- `5xx` responses never contain exception text; the server logs it under the same
  `request_id`.

Codes are listed by where they appear. A code can appear in more than one place (for
example `quota_exceeded` is a synchronous 429 for uploads and a document error when the
chunk quota is exceeded during indexing).

## Request errors

| Code | Status | Meaning |
|---|---|---|
| `validation_failed` | 422 | The request or a setting is invalid: wrong type, unknown field, value out of bounds, malformed cursor, bad multipart body, empty file, metadata not a JSON object or over 16 KiB, title over 512 characters, unknown `embedding_space` or prompt template, `expires_at` in the past. `details` names the field. |
| `bad_request` | 400 | The body could not be read (for example a broken chunked transfer). |
| `unauthenticated` | 401 | Missing or malformed `Authorization: Bearer` header; unknown, revoked or expired API key; a tenant key on an admin route; any token on the admin API while no admin key is configured. |
| `not_found` | 404 | The resource does not exist **for this tenant**. Other tenants' ids look exactly like missing ones (never 403). Also unknown routes, deleted or `deleting` collections, unknown tenants, keys and embedding spaces on the admin API. |
| `chunk_not_found` | 404 | `GET …/chunks/{chunk_id}`: the chunk is not part of any current document version (unknown, superseded or another tenant's). |
| `original_not_retained` | 404 | `GET …/original`: the original bytes of that version were removed (`ingest.retain_originals: false`). |
| `method_not_allowed` | 405 | The route exists but not with this HTTP method. |
| `conflict` | 409 | The request conflicts with the current state (generic). |
| `collection_exists` | 409 | `POST /v1/collections`: the tenant already has a collection with this name. |
| `document_not_ready` | 409 | `GET …/text`: the document has no indexed version yet. Retry after its ingest job is `done`. |
| `evidence_threshold_uncalibrated` | 409 | `/answers`: no evidence threshold is set for the active gate (collection setting, config `answer.*`, or a calibrated value for the reranker file / embedding space). Never happens with the shipped models. |
| `space_in_use` | 409 | `DELETE /v1/admin/embedding-spaces/{name}`: a collection uses the space (active or re-embed target). |
| `space_limit_reached` | 409 | Registering a new space would exceed `embedding.max_spaces`. |
| `file_too_large` | 413 | The uploaded file exceeds `ingest.max_file_bytes`. |
| `request_too_large` | 413 | The request body exceeds its limit (1 MiB for JSON bodies; `ingest.max_file_bytes` plus multipart overhead for uploads). |
| `unsupported_media_type` | 415 | The upload is not PDF, DOCX or UTF-8 text (HTML, Markdown, TXT). The type is sniffed from the content; the client's `Content-Type` is ignored. |
| `pdf_encrypted` | 422 | The uploaded PDF is encrypted. |
| `too_many_pages` | 422 | The PDF has more than `ingest.max_pages` pages. |
| `archive_too_large` | 422 | The DOCX declares more than `ingest.max_uncompressed_bytes` or more than `ingest.max_archive_members` members (checked again while reading). |
| `parse_failed` | 422 | The upload looks like a PDF or DOCX but cannot be read (corrupt file). |
| `embedding_dimension_unsupported` | 422 | Registering a space with more than 4000 dimensions. |
| `webhook_url_rejected` | 422 | The webhook URL is not HTTPS (unless `webhooks.allow_http`), is malformed, does not resolve, or resolves to a non-public address (loopback, RFC 1918, CGNAT, link-local, ULA, multicast…) unless `webhooks.allow_private` ([webhooks.md](webhooks.md)). |
| `quota_exceeded` | 429 | A tenant quota would be exceeded: `max_collections`, `max_documents`, `max_upload_bytes_per_day`. |
| `internal` | 500 | Unexpected server error. Logged with its stack trace under the `request_id`. |
| `upstream_failed` | 502 | A provider (LLM, embedding endpoint, reranker) failed after its retry budget. |
| `embedding_upstream_failed` | 502 | The `openai_compat` embedding endpoint failed after `embedding.max_retries` (for example while embedding a search query). |
| `embedding_request_rejected` | 502 | The embedding endpoint rejected the request (4xx other than 429). |
| `embedding_dimension_mismatch` | 502 | The embedding endpoint returned vectors whose dimension differs from the space's. |
| `llm_not_configured` | 503 | `/answers` while `llm.base_url` is empty. |
| `embedding_not_configured` | 503 | An `openai_compat` space is used while `embedding.base_url` is empty. |
| `unavailable` | 503 | A required dependency is not configured or not reachable (for example a model that is not in the manifest). |

## Streaming answer errors

With `"stream": true`, errors that happen before the first event (validation, 404,
`llm_not_configured`, `evidence_threshold_uncalibrated`, retrieval failures) are ordinary
JSON error responses. Errors after the stream started arrive as a final SSE event:

```text
event: error
data: {"code": "upstream_failed", "message": "llm stream ended before [DONE]"}
```

| Code | Meaning |
|---|---|
| `upstream_failed` | The LLM stream failed, was cut or sent a malformed event. |
| `internal` | Unexpected error while streaming (logged with its stack). |

Any other code of the tables above can appear here when the LLM client raises it.

## Document errors

A document whose ingest failed has `status: "failed"` and `error: {"code", "message"}`
(`GET …/documents/{doc_id}`; also each version's `error`, and the `document.failed`
webhook payload). Codes that are a property of the file fail the version at once;
transient ones (marked *retried*) are retried with backoff up to
`ingest.job_max_attempts` and only then shown here.

| Code | Meaning |
|---|---|
| `parse_failed` | The parser could not read the file (corrupt PDF/DOCX, invalid UTF-8, …). |
| `parse_timeout` | Parsing took longer than `ingest.parse_timeout`; the parser child process was killed. |
| `pdf_encrypted` | The PDF is encrypted. |
| `too_many_pages` | The PDF has more than `ingest.max_pages` pages. |
| `archive_too_large` | The DOCX expands beyond `ingest.max_uncompressed_bytes` or has too many members (enforced again while reading). |
| `unsupported_media_type` | The stored original has a media type no parser handles. |
| `quota_exceeded` | Indexing the document would exceed the tenant's `max_chunks`. |
| `embedding_dimension_mismatch` | The embedder returned vectors of the wrong dimension for the space (*retried* when the `openai_compat` provider reports it; immediate when such a vector reaches storage). |
| `upstream_failed` | *Retried.* An embedder or other provider kept failing. |
| `embedding_upstream_failed` | *Retried.* The `openai_compat` embedding endpoint kept failing. |
| `embedding_request_rejected` | *Retried.* The embedding endpoint rejected the request (4xx other than 429). |
| `embedding_not_configured` | *Retried.* The collection's space needs `embedding.base_url`. |
| `unavailable` | *Retried.* The space's embedder cannot be served (for example its model is missing). |
| `vectors_missing` | *Retried.* The version swap found a chunk without a vector for a needed space (a re-embed started during the ingest and the dual-write could not catch up within the swap attempts). |
| `internal` | *Retried.* Unexpected error; the message names the exception type. |

## Job errors

`GET /v1/jobs/{job_id}` shows `status: "failed"` and `last_error` (free text, prefixed
with a code where one exists) after `ingest.job_max_attempts` attempts, or at once for
errors a retry cannot fix (bad job arguments, a collection or tenant not in the state
the job expects).

| Code in `last_error` | Job kind | Meaning |
|---|---|---|
| `reembed_incomplete` | `reembed` | The switch to the new space did not converge because documents kept changing; the job is retried. |
| any document error code | `ingest` | The version failed for good; `last_error` is `<code>: <message>` and the document shows the same error (see above). |
| `ingest_failed` | `ingest` | Default code of a permanent ingest failure when no more specific code applies. |
| `upstream_failed` and other provider codes | `ingest`, `reembed`, `webhook` | The provider kept failing after the retry budget. |
| (policy message) | `webhook` | At delivery time the URL no longer passes the webhook SSRF policy (checked again at every delivery); the attempt is logged as failed. |

Webhook deliveries record each attempt (HTTP status or error text) in
`GET /v1/collections/{id}/webhook-deliveries`.
