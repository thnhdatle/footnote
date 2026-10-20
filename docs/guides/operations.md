# Operations

Set up the database, deploy, upgrade, back up and monitor Footnote.

Settings are in [configuration.md](configuration.md). Problems are in
[troubleshooting.md](troubleshooting.md).

## 1. Processes

| Command | Runs |
|---|---|
| `footnote serve` | API (default `127.0.0.1:8000`) and admin listener (default `127.0.0.1:9094`) |
| `footnote serve --with-worker` | the same, plus the job worker |
| `footnote worker` | the job worker and admin listener |
| `footnote migrate` | schema migrations, then exits |

Uploads, cleanup, webhooks and deletions are jobs, so run at least one worker. The admin
listener has no authentication: keep it on a private network.

## 2. PostgreSQL

You need PostgreSQL 15+ and pgvector 0.8.0+.

**Shared memory.** Set `shm_size: 1gb` on the Postgres container, or a memory-backed
`emptyDir` at `/dev/shm` in Kubernetes. Docker's 64 MB default makes index builds fail.

### Roles and schema

```sh
psql "$PRIVILEGED_URL" \
  -v app_password="'<app password>'" -v admin_password="'<admin password>'" \
  -f deploy/sql/bootstrap.sql    # once, as a privileged user; safe to re-run

export FOOTNOTE_DATABASE_URL=postgresql://footnote_app:...@db.example.com:5432/footnote
export FOOTNOTE_ADMIN_DATABASE_URL=postgresql://footnote_admin:...@db.example.com:5432/footnote
```

[`deploy/sql/bootstrap.sql`](../../deploy/sql/bootstrap.sql) creates the extension, the
schema and two roles.

| Role | Used by | Properties |
|---|---|---|
| `footnote_admin` | `footnote migrate`, the admin API, `footnote tenants\|keys\|spaces`, tenant deletion | owns schema `footnote`; no superuser, no `BYPASSRLS` |
| `footnote_app` | tenant requests and the worker | no superuser, no `BYPASSRLS`, owns nothing, not a member of `footnote_admin` |

Never use the `footnote_admin` URL as `FOOTNOTE_DATABASE_URL`.

### Managed services

Run the bootstrap as `rds_superuser` (RDS, Aurora), `cloudsqlsuperuser` (Cloud SQL) or
`postgres` (Supabase). If roles must be created with the provider's tools, use
`LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE`. Never make `footnote_app` a member
of `footnote_admin`.

### Poolers

```sh
export FOOTNOTE_DATABASE_PREPARE_THRESHOLD=null    # required behind a transaction pooler
```

Transaction-mode poolers are safe for the app URL. Use a direct or session-mode connection
for the admin URL. Each process opens up to `database.pool_max` (default 20) plus
`admin_database.pool_max` (default 4) connections. Keep the total below `max_connections`.

## 3. Deploy

The image is `ghcr.io/thnhdatle/footnote:v0.1.0`. It runs as user 65532 on a read-only
filesystem.

- Mount a writable `/tmp` and a volume at `/models`. Run `footnote models pull` once.
- Pass secrets by environment or `*_file` keys.
- In a container, set `FOOTNOTE_SERVER_LISTEN` and `FOOTNOTE_ADMIN_LISTEN` to `0.0.0.0:...`.
- In Kubernetes, use `/readyz` for readiness and `/healthz` for liveness.

The compose passwords and admin key are for development only. Footnote speaks plain HTTP:
put TLS in front of it.

## 4. Upgrade

1. Read the changelog and take a backup.
2. Run `footnote migrate` with the new image.
3. Start the new image. Check `GET /v1/version` and `/readyz`.
4. Roll the other processes.

Old processes turn `not_ready` after the migration, so plan a short maintenance window. Skip it only if the release notes say a rolling
update is safe.
To roll back, restore the backup. Never downgrade the schema by hand.

## 5. Backups

All data is in one database. Back it up with `pg_dump`, physical backups or snapshots.

**Uploaded files live in table `footnote.document_blobs`.** A backup that skips it loses
them.

```sql
SELECT pg_size_pretty(pg_total_relation_size('footnote.document_blobs'));  -- stored file size
```

- With `ingest.retain_originals: true` (default), every version's file is kept.
- Back up vectors too. Rebuilding them re-embeds every chunk.
- Restore the `/models` volume with `footnote models pull`.

Restore into a database that already ran `bootstrap.sql`, owned by `footnote_admin`. Then
run `footnote migrate` and check `/readyz`. Test a restore before you need it.

## 6. Garbage collection

Replaced data is removed `ingest.gc_grace` (default `10m`) after a swap. `0s` removes it
at once, but running searches may miss rows. Any value above `database.statement_timeout`
(default `30s`) is safe. A growing `footnote_jobs_queue_depth{kind="gc"}` means workers are
behind.

## 7. Embedding spaces

Register a space before moving collections to it. Registration needs `footnote_admin`.
Expect double vector storage during a re-embed ([models-and-data.md](models-and-data.md)).

## 8. Readiness

`GET /readyz` returns 503 with the problems per check.

| Check | Fails when |
|---|---|
| `database` | `footnote_app` pool cannot query |
| `admin_database` | `footnote_admin` pool cannot query |
| `migrations` | schema behind; run `footnote migrate` |
| `postgres_versions` | PostgreSQL < 15, or pgvector missing or < 0.8.0 |
| `app_role` | `footnote_app` is superuser, has `BYPASSRLS`, owns objects, or is in `footnote_admin` |
| `models` | a model file is missing or corrupt |
| `llm` | `llm.base_url` set, its `GET /models` fails |
| `embedding` | `embedding.base_url` set, an `openai_compat` space exists, its `GET /models` fails |

## 9. Metrics

Metrics are at `/metrics` on the admin listener. There are no tenant labels.

| Metric | Type | Labels |
|---|---|---|
| `footnote_http_requests_total` | counter | `route`, `method`, `status` |
| `footnote_http_request_duration_seconds` | histogram | `route` |
| `footnote_jobs_total`, `footnote_ingest_jobs_total` | counter | `kind`, `status` (`succeeded`, `retried`, `failed`); ingest has only `status` |
| `footnote_jobs_queue_depth` | gauge | `kind` |
| `footnote_ingest_duration_seconds` | histogram | `stage` (`parse`, `chunk`, `embed`, `swap`) |
| `footnote_search_duration_seconds` | histogram | `stage` (`embed_query`, `lexical`, `vector`, `rerank`) |
| `footnote_answers_total` | counter | `outcome` (`answered`, `abstained`, `unsupported`, `error`) |
| `footnote_embedding_requests_total`, `footnote_embedding_texts_total` | counter | `provider` |
| `footnote_embedding_truncated_total` | counter | none |
| `footnote_search_exact_scan_total` | counter | `reason` (`small_collection`, `fallback`) |
| `footnote_webhook_deliveries_total` | counter | `status` |
| `footnote_build_info` | gauge | `version`, `commit`, `python_version` |

Scrape `footnote:9094` with job name `footnote`. Logs are JSON on stdout.

## 10. Alerts

Starting points. Tune them to your traffic.

```yaml
groups:
  - name: footnote
    rules:
      - alert: FootnoteNotReady
        expr: up{job="footnote"} == 0
        for: 2m
      - alert: FootnoteServerErrors
        expr: sum(rate(footnote_http_requests_total{status=~"5.."}[5m]))
              / sum(rate(footnote_http_requests_total[5m])) > 0.02
        for: 10m
      - alert: FootnoteJobBacklog
        expr: sum(footnote_jobs_queue_depth) > 100
        for: 15m
      - alert: FootnoteJobsFailing
        expr: sum(increase(footnote_jobs_total{status="failed"}[15m])) > 0
```

Also alert on Postgres disk and connections, answer errors and failed webhooks.

## 11. CLI reference

Pass `--config/-c <file>` before the subcommand, or set `FOOTNOTE_CONFIG`. Exit codes: 0
success, 1 failure, 2 bad configuration or usage.

| Command | Purpose |
|---|---|
| `footnote serve [--with-worker] [--listen ADDR] [--admin-listen ADDR]` | run the service |
| `footnote worker [--admin-listen ADDR]` | run the worker |
| `footnote migrate [--revision REV]` | apply migrations |
| `footnote healthcheck [--timeout S]` | exit 0 if `/readyz` is 200 |
| `footnote config validate` | check configuration |
| `footnote config print [--redact/--no-redact]` | print configuration |
| `footnote models pull\|verify [--all]` | download or check models |
| `footnote tenants create\|list\|delete\|export` | manage tenants |
| `footnote keys create\|list\|revoke\|rotate` | manage keys |
| `footnote spaces register\|list` | manage spaces |
| `footnote version` | print the version |

`tenants`, `keys` and `spaces` use `FOOTNOTE_ADMIN_DATABASE_URL`, not the admin key.
