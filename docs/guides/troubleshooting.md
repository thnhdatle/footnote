# Troubleshooting

Start with these three checks. They answer most questions.

```sh
curl -s http://127.0.0.1:9094/readyz          # what is not ready, per check
footnote config validate                      # configuration and manifest problems (exit 2)
footnote models verify                        # model files present and intact
```

Every error response carries a `request_id`; the server logs the cause under the same id.
Error codes are listed in [errors.md](errors.md). Operating details are in
[operations.md](operations.md).

## 1. The service does not start

| Symptom | Cause and fix |
|---|---|
| exit code 2 and a list of problems | the configuration is invalid; all problems are listed at once. Unknown keys, secrets in the YAML file and out-of-range values are errors ([configuration.md](configuration.md)) |
| `database.url: required` | set `FOOTNOTE_DATABASE_URL` or `database.url_file`; the admin URL is separate (`FOOTNOTE_ADMIN_DATABASE_URL`) |
| connection refused or authentication failed | check the host, the role passwords set by `bootstrap.sql`, and that `footnote_app` can `CONNECT` |
| `migration failed` | `footnote migrate` runs as `footnote_admin`. Check the admin URL and that `bootstrap.sql` ran, so the schema `footnote` exists and is owned by that role |
| port in use | change `server.listen` or `admin.listen` |
| `manifest` or `rerank.model` errors | a configured model id is not in `models/manifest.yaml`; run `footnote config validate` |

## 2. `/readyz` returns 503

The body lists the problems per check ([operations.md](operations.md) section 8).

| Check | Typical fix |
|---|---|
| `database`, `admin_database` | the URL, the network or the password is wrong, or the pool is exhausted |
| `migrations` | run `footnote migrate` with the same image version as the service |
| `postgres_versions` | use PostgreSQL 15+ and pgvector 0.8.0+; run `CREATE EXTENSION vector` (bootstrap does) |
| `app_role` | `footnote_app` must not be a superuser, have `BYPASSRLS`, own objects in schema `footnote` or be a member of `footnote_admin`. Do not use the owner's URL as `FOOTNOTE_DATABASE_URL` |
| `models` | run `footnote models pull`; a file with the wrong hash is downloaded again |
| `llm` | `llm.base_url` is set but `GET <base_url>/models` fails: wrong URL, server down, or a container reaching `localhost` (use `host.docker.internal` or the service name) |
| `embedding` | the same for `embedding.base_url` when an `openai_compat` space exists |

If you only need search and have no language model, leave `llm.base_url` empty and the
`llm` check is skipped.

## 3. Documents

| Symptom | Cause and fix |
|---|---|
| a document stays `queued` | no worker is running. `footnote serve` alone does not run jobs: use `--with-worker` or start `footnote worker`. Check `footnote_jobs_queue_depth` |
| `parsing` or `embedding` for a long time | look at `GET /v1/jobs/{id}` (`attempts`, `last_error`). Large PDFs are slow; a down embedding endpoint is retried with backoff |
| `failed` with `parse_failed` | the file is corrupt or not valid UTF-8 text |
| `failed` with `parse_timeout` | raise `ingest.parse_timeout`, or split the file |
| `failed` with `quota_exceeded` | the tenant's `max_chunks` quota; raise it ([multi-tenancy.md](multi-tenancy.md)) |
| `failed` with `embedding_*` or `upstream_failed` | the embedding endpoint failed after retries; fix it and upload again |
| upload `415` | the file is not PDF, DOCX or UTF-8 text; the sniffed content decides, not the extension or `Content-Type` |
| upload `413` | the file exceeds `ingest.max_file_bytes` |
| `200 unchanged: true` after an upload | the file, title and chunking settings equal the current version's; nothing to index |
| extracted text is empty or garbled | a scanned PDF has no text layer (no OCR); a PDF with odd fonts can lose structure. Read `.../text?format=elements` to see what the parser produced |
| headings are not detected in a PDF | headings are inferred from font size relative to body text; a document with one font size has none |

## 4. Search

| Symptom | Cause and fix |
|---|---|
| no results | the collection has no `ready` documents, a filter is too narrow, or `mode: "lexical"` found no shared words. Retry with `mode: "vector"` and no filters |
| a document you just uploaded is missing | its job is not `done` yet |
| results look wrong | send `"debug": true`: compare `ranks`, `scores` and `rerank_score` ([retrieval.md](retrieval.md)) |
| exact names are not found | use `mode: "hybrid"` or `lexical`; keep `lexical_match: any` |
| `503 unavailable` or `embedding_not_configured` | the collection's space needs a model file or `embedding.base_url` |
| `502 embedding_*` | the embedding endpoint failed or returned the wrong dimension |
| search is slow | check `stage_seconds`. The lexical leg reads the whole collection, so split very large corpora; reranking costs about as much as `rerank_n` model passes, so lower `rerank_n` or set `rerank: false` |
| a different result right after a model change | a re-embed switches results once, when it finishes |

## 5. Answers

| Symptom | Cause and fix |
|---|---|
| `503 llm_not_configured` | set `llm.base_url` |
| `409 evidence_threshold_uncalibrated` | the space or reranker has no calibrated value: set `answer.evidence_threshold_*` in the collection or configuration ([citations.md](citations.md)) |
| always `insufficient_evidence` | the gate score is below the threshold. Check `evidence_score`, lower the threshold, and check the retrieval first |
| answers although the documents lack the answer | raise the threshold; check `unsupported` |
| `unsupported: true` | the model gave no valid `[n]` marker: a weak or reasoning model, or one that ignores the prompt. Try another model or `llm.extra_body` |
| `502 upstream_failed` | the LLM failed after retries; check `llm.timeout` and the endpoint |
| text cut off | raise `llm.max_tokens` (default 512) |
| streamed text shows markers that are not in the final answer | expected; use `citations.answer` |

## 6. Authentication and quotas

- `401 unauthenticated` on a tenant route: the key is missing, malformed, revoked or expired,
  or it is the admin key. On `/v1/admin`: a tenant key, or no admin key is configured.
- `404` for an id that exists: it belongs to another tenant, or its collection was deleted.
- `429 quota_exceeded`: the message names the quota; raise it with the admin API.
- A rotated key stops working after `auth.key_rotation_grace`.

## 7. Webhooks

| Symptom | Cause and fix |
|---|---|
| `422 webhook_url_rejected` | the URL is not HTTPS, does not resolve, or resolves to a private address; see [webhooks.md](webhooks.md) for `webhooks.allow_http` and `allow_private` |
| no delivery appears | the collection has no URL or secret, the event is not subscribed, or the version was superseded before it became current |
| deliveries `pending` | retries remain; read `error` and `response_code` in `webhook-deliveries` |
| deliveries `failed` | your receiver returned non-2xx or timed out (`webhooks.timeout`, 5s), or the policy rejected the URL at delivery time |
| signature mismatch | sign the raw body bytes, with the stored secret, as `"<timestamp>.<body>"` |

## 8. Database

| Symptom | Cause and fix |
|---|---|
| `could not resize shared memory segment` while indexing or registering a space | the container's `/dev/shm` is too small; set `shm_size: 1gb` ([operations.md](operations.md)) |
| `prepared statement ... does not exist` or `already exists` | a transaction pooler is in use; set `database.prepare_threshold: null` |
| `permission denied for table tenants` | normal for `footnote_app`; admin work uses the admin URL or API |
| `too many connections` | processes times (`database.pool_max` + `admin_database.pool_max`) exceed `max_connections`; lower the pools or use a pooler |
| statement timeout | raise `database.statement_timeout` only after you find the slow query; search on a very large collection is the usual cause |
| disk grows after many uploads | originals of every version are kept (`ingest.retain_originals`) and `gc` has not run; check the `gc` queue ([operations.md](operations.md)) |

## 9. Re-embedding and jobs

- A collection's `reembed` field shows `done` and `total`. If it does not move, check the job
  (`GET /v1/jobs/{id}`) and the embedding endpoint.
- `reembed_incomplete` in a job's `last_error`: documents kept changing while the switch
  was attempted. The job retries; reduce concurrent uploads.
- A job with `status: failed` and `attempts` equal to `ingest.job_max_attempts` will not run
  again. Fix the cause, then re-upload the document or change the setting again.
- A killed worker's jobs return to the queue after `ingest.job_lease` (default 60s).

## 10. Reporting a problem

Include the version (`GET /v1/version`), the `request_id`, the output of `/readyz` and
`footnote config print --redact`, and the relevant log lines. Do not include API keys,
database URLs or document text.
