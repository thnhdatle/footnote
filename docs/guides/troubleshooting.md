# Troubleshooting

Find the cause of a problem and fix it.

```sh
curl -s http://127.0.0.1:9094/readyz    # what is not ready, per check
footnote config validate                # configuration problems (exit 2)
footnote models verify                  # model files intact
```

Every error has a `request_id`. The server logs the cause under it. Codes are in
[errors.md](errors.md).

## 1. The service does not start

| Symptom | Fix |
|---|---|
| exit code 2 with a problem list | fix the configuration ([configuration.md](configuration.md)) |
| `database.url: required` | set `FOOTNOTE_DATABASE_URL` |
| connection refused or authentication failed | check host and passwords |
| `migration failed` | check the admin URL; run `bootstrap.sql` |
| port in use | change `server.listen` or `admin.listen` |
| `manifest` or `rerank.model` errors | unknown model id; run `footnote config validate` |

## 2. `/readyz` returns 503

| Check | Fix |
|---|---|
| `database`, `admin_database` | check URL, network, password |
| `migrations` | run `footnote migrate` with the same image |
| `postgres_versions` | PostgreSQL 15+, pgvector 0.8.0+ |
| `app_role` | do not use the owner's URL as `FOOTNOTE_DATABASE_URL` |
| `models` | run `footnote models pull` |
| `llm` | fix `llm.base_url`; in a container use `host.docker.internal`, not `localhost` |
| `embedding` | fix `embedding.base_url` |

See [operations.md](operations.md) section 8.

## 3. Documents

| Symptom | Fix |
|---|---|
| stays `queued` | start a worker (`--with-worker` or `footnote worker`) |
| stuck in `parsing` or `embedding` | check `last_error` in `GET /v1/jobs/{id}` |
| `parse_failed` | corrupt file or invalid UTF-8 |
| `parse_timeout` | raise `ingest.parse_timeout` or split the file |
| `quota_exceeded` | raise `max_chunks` ([multi-tenancy.md](multi-tenancy.md)) |
| `embedding_*` or `upstream_failed` | fix the embedding endpoint, upload again |
| upload `415` | not PDF, DOCX or UTF-8 text |
| upload `413` | over `ingest.max_file_bytes` |
| `200 unchanged: true` | nothing changed |
| empty or garbled text | scanned PDF (no OCR); check `.../text?format=elements` |
| no PDF headings | the PDF uses one font size |

## 4. Search

| Symptom | Fix |
|---|---|
| no results | retry with `mode: "vector"` and no filters |
| new document missing | its job is not `done` |
| wrong results | send `"debug": true` ([retrieval.md](retrieval.md)) |
| exact names not found | use `hybrid` or `lexical` with `lexical_match: any` |
| `503 unavailable` or `embedding_not_configured` | add the model file or `embedding.base_url` |
| `502 embedding_*` | embedding endpoint failed or wrong dimension |
| slow | check `stage_seconds`; lower `rerank_n` or set `rerank: false` |
| results change after a model change | expected after a re-embed |

## 5. Answers

| Symptom | Fix |
|---|---|
| `503 llm_not_configured` | set `llm.base_url` |
| `409 evidence_threshold_uncalibrated` | set `answer.evidence_threshold_*` ([citations.md](citations.md)) |
| always `insufficient_evidence` | check retrieval; lower the threshold |
| answers without evidence | raise the threshold |
| `unsupported: true` | try another model or `llm.extra_body` |
| `502 upstream_failed` | check the LLM endpoint and `llm.timeout` |
| text cut off | raise `llm.max_tokens` (default 512) |
| streamed markers missing at the end | expected; use `citations.answer` |

## 6. Authentication and quotas

| Symptom | Fix |
|---|---|
| `401` on a tenant route | use a valid tenant key, not the admin key |
| `401` on `/v1/admin` | use the admin key; configure one |
| `404` for an existing id | another tenant's, or a deleted collection |
| `429 quota_exceeded` | raise the named quota |
| rotated key stopped | `auth.key_rotation_grace` ended |

## 7. Webhooks

| Symptom | Fix |
|---|---|
| `422 webhook_url_rejected` | use a public HTTPS URL ([webhooks.md](webhooks.md)) |
| no delivery | check the URL and `events` |
| deliveries `pending` | retries remain; read `error` and `response_code` |
| deliveries `failed` | receiver returned non-2xx or exceeded `webhooks.timeout` (5s) |
| signature mismatch | sign the raw body as `"<timestamp>.<body>"` |

## 8. Database

| Symptom | Fix |
|---|---|
| `could not resize shared memory segment` | set `shm_size: 1gb` ([operations.md](operations.md)) |
| `prepared statement ... does not exist` or `already exists` | set `database.prepare_threshold: null` |
| `permission denied for table tenants` | normal for `footnote_app` |
| `too many connections` | lower `database.pool_max` and `admin_database.pool_max` |
| statement timeout | find the slow query before raising `database.statement_timeout` |
| disk keeps growing | check `ingest.retain_originals` and the `gc` queue |

## 9. Re-embedding and jobs

- Stuck `reembed`: check the job and the embedding endpoint.
- `reembed_incomplete`: documents kept changing. It retries. Reduce uploads.
- A job at `ingest.job_max_attempts` will not run again. Fix the cause and redo the action.
- A killed worker's jobs return after `ingest.job_lease` (default 60s).

## 10. Reporting a problem

Include `GET /v1/version`, the `request_id`, `/readyz` output,
`footnote config print --redact` and the logs. Never include API keys, database URLs or
document text.
