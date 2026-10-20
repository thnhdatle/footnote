# Operations

This guide covers database setup, deployment, upgrades, backups, garbage collection,
monitoring and the CLI. Settings are in [configuration.md](configuration.md); problems are
in [troubleshooting.md](troubleshooting.md).

## 1. Processes

| Command | Runs |
|---|---|
| `footnote serve` | data plane (`server.listen`, default `127.0.0.1:8000`) and admin listener (`admin.listen`, default `127.0.0.1:9094`) |
| `footnote serve --with-worker` | the same, plus the job worker in the process |
| `footnote worker` | the job worker and the admin listener only |
| `footnote migrate` | schema migrations, then exits |

Uploads, re-indexing, re-embedding, garbage collection, webhooks and deletions are jobs.
Only a worker runs them, so a deployment needs at least one `serve --with-worker` or one
`worker` process. You can run many of either; workers claim jobs without double-processing.
The admin listener serves `/healthz`, `/readyz` and `/metrics` without authentication. Keep
it on a private network.

On SIGTERM a process drains requests and jobs for up to `shutdown_timeout` (default `30s`)
and exits. Unfinished jobs are picked up by another worker when their lease expires.

## 2. PostgreSQL

Requirements: PostgreSQL 15 or newer and pgvector 0.8.0 or newer. `/readyz` checks both.

**Shared memory.** Parallel HNSW index builds use `/dev/shm`, and the Docker default of
64 MB is too small. The compose `postgres` service sets `shm_size: 1gb`; do the same in every
container deployment (`shm_size`, or a memory-backed `emptyDir` at `/dev/shm` in Kubernetes).
Otherwise index builds can fail with "could not resize shared memory segment". Managed
services size this themselves.

### Roles and schema

Run [`deploy/sql/bootstrap.sql`](../../deploy/sql/bootstrap.sql) once as a privileged user.
It is idempotent and creates the `vector` extension, two roles and the schema:

```sh
psql "$PRIVILEGED_URL" \
  -v app_password="'<app password>'" -v admin_password="'<admin password>'" \
  -f deploy/sql/bootstrap.sql
```

| Role | Used by | Properties |
|---|---|---|
| `footnote_admin` | `footnote migrate`, the admin API, `footnote tenants\|keys\|spaces`, the `tenant_delete` job | owns schema `footnote` and everything in it; no superuser, no `BYPASSRLS` |
| `footnote_app` | tenant requests and the worker | no superuser, no `BYPASSRLS`, owns nothing, not a member of `footnote_admin` |

Neither role needs superuser or `BYPASSRLS`, which managed Postgres does not allow
([multi-tenancy.md](multi-tenancy.md)). Set the URLs:

```sh
export FOOTNOTE_DATABASE_URL=postgresql://footnote_app:...@db.example.com:5432/footnote
export FOOTNOTE_ADMIN_DATABASE_URL=postgresql://footnote_admin:...@db.example.com:5432/footnote
```

### Managed services

The bootstrap needs a user who can `CREATE EXTENSION vector`, create roles and create a
schema owned by another role: a member of `rds_superuser` on Amazon RDS and Aurora,
`cloudsqlsuperuser` on Google Cloud SQL, `postgres` on Supabase. Use an engine version with
pgvector 0.8.0 or newer. If your provider restricts role creation, create the roles with
its tools using `LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE`, then run the rest
of the file. Creating a schema owned by `footnote_admin` may require the executing user to
be a member of `footnote_admin`; grant that to the administrator only, never to
`footnote_app`.

### Poolers

The tenant is set with a transaction-local setting, so a transaction-mode pooler
(PgBouncer, RDS Proxy, Supavisor) is safe. Disable psycopg's server-side prepared
statements, which transaction poolers cannot track:

```sh
export FOOTNOTE_DATABASE_PREPARE_THRESHOLD=null
```

Use a direct or session-mode connection for `FOOTNOTE_ADMIN_DATABASE_URL`.

Connections: each process opens up to `database.pool_max` (default 20) plus
`admin_database.pool_max` (default 4). Keep the sum over all processes below the server's
`max_connections`. `database.statement_timeout` (default `30s`) bounds every statement.

## 3. Deploy

The image (`ghcr.io/thnhdatle/footnote:v0.1.0`; `make docker` builds `thnhdatle/footnote:dev`
locally) runs as user 65532 on a read-only root filesystem, with
`footnote` as its entry point and `serve` as its default command.

- Mount a writable `/tmp` (uploads spool there) and a volume at `/models`, then run
  `footnote models pull` once ([models-and-data.md](models-and-data.md)).
- Pass secrets by environment or `*_file` keys ([configuration.md](configuration.md)). Inside
  a container set `FOOTNOTE_SERVER_LISTEN` and `FOOTNOTE_ADMIN_LISTEN` to `0.0.0.0:...`.
- The image `HEALTHCHECK` runs `footnote healthcheck` (exit 0 when the admin listener's
  `/readyz` answers 200). In Kubernetes use `/readyz` for readiness and `/healthz` for liveness.

`deploy/docker-compose.yml` runs the whole chain; its passwords and admin key are for
development only. Footnote speaks plain HTTP: terminate TLS in front of the data plane.

## 4. Upgrade

1. Read the changelog, then take a backup (section 5).
2. Run `footnote migrate` with the new image as `footnote_admin`; it prints
   `migrated to head`.
3. Start the new image. `/readyz` reports `migrations` until the schema matches the code.
4. Check `GET /v1/version` and `/readyz`, then roll the remaining processes.

Processes still on the old image report `not_ready` once the schema has moved on, so plan
a short maintenance window unless the release notes say a rolling update is safe. Rolling
back after a migration means restoring the backup; do not downgrade the schema by hand.

## 5. Backups

Everything Footnote stores is in one PostgreSQL database, so a database backup is the
backup: `pg_dump`/`pg_restore`, physical backups with point-in-time recovery, or the
managed service's snapshots.

**Stored originals are in the database.** Uploaded files are kept zstd-compressed in the
table `footnote.document_blobs`, one row per document version, with parsed elements in
`footnote.document_versions`. A whole-database backup includes them. A backup that skips
that table, for example one that excludes large tables, loses the originals and
`GET .../original`.

- With `ingest.retain_originals: true` (the default) every version's original is kept until
  the document is deleted, so the table grows with each re-upload. Size it:

  ```sql
  SELECT pg_size_pretty(pg_total_relation_size('footnote.document_blobs'));
  ```

- With `false` the original is deleted when the version becomes `ready`; re-chunking still
  works because elements are stored.
- Vectors are derived data but costly to rebuild (every chunk is re-embedded). Back them up.
- The `/models` volume is not in the database; restore it with `footnote models pull`.
- `footnote tenants export <id> --include-originals` gives a per-tenant logical copy
  ([multi-tenancy.md](multi-tenancy.md)); it does not replace database backups.

Restore into a database that already went through `bootstrap.sql` (roles and extension
exist), as `footnote_admin` or with owner mapping so objects stay owned by
`footnote_admin`. Then run `footnote migrate` (a no-op at the same version) and check
`/readyz`. HNSW indexes are rebuilt during a logical restore and need the shared memory of section 2.
Test a restore before you need it.

## 6. Garbage collection

Replacing a document version or switching embedding spaces leaves rows that must stay
readable for searches that started earlier. A `gc` job removes them `ingest.gc_grace`
later (default `10m`), never inside the swap:

- memberships of superseded or failed versions, chunks no current version uses and their
  vectors, and superseded originals when `ingest.retain_originals` is false;
- vectors of a space that is neither active nor the re-embed target.

Set `ingest.gc_grace: 0s` to collect immediately; a search that started before a swap may
then miss rows. Any grace above `database.statement_timeout` (default `30s`) is safe.
Storage includes the superseded rows until then. A growing `footnote_jobs_queue_depth{kind="gc"}` means
workers are not keeping up or are down.

## 7. Embedding spaces

Spaces are global and registered by operators: from `embedding.spaces` at startup, with
`footnote spaces register`, or through the admin API. Registration creates the space's
vector index, so it needs `footnote_admin`. Register a new space before moving collections
to it, and plan for double vector storage while a collection re-embeds
([models-and-data.md](models-and-data.md)).

## 8. Readiness

`GET /readyz` is on the data plane and the admin listener. It returns 503 with the problems
per check.

| Check | Fails when |
|---|---|
| `database` | the `footnote_app` pool cannot run a query |
| `admin_database` | the `footnote_admin` pool cannot run a query |
| `migrations` | the schema revision is not the code's head; run `footnote migrate` |
| `postgres_versions` | PostgreSQL is older than 15 or pgvector is missing or older than 0.8.0 |
| `app_role` | `footnote_app` is a superuser, has `BYPASSRLS`, owns objects in schema `footnote` or is a member of `footnote_admin` |
| `models` | a local model file the configuration needs is missing or fails its SHA-256 |
| `llm` | `llm.base_url` is set and its `GET /models` fails (only listed, never a completion) |
| `embedding` | `embedding.base_url` is set, an `openai_compat` space exists, and its `GET /models` fails |

## 9. Metrics

Prometheus metrics are at `/metrics` on the admin listener. There are no tenant or
collection labels.

| Metric | Type | Labels |
|---|---|---|
| `footnote_http_requests_total` | counter | `route`, `method`, `status` |
| `footnote_http_request_duration_seconds` | histogram | `route` |
| `footnote_jobs_total`, `footnote_ingest_jobs_total` | counter | `kind`, `status` (`succeeded`, `retried`, `failed`); the ingest series has only `status` |
| `footnote_jobs_queue_depth` | gauge | `kind`; refreshed by worker processes |
| `footnote_ingest_duration_seconds` | histogram | `stage` (`parse`, `chunk`, `embed`, `swap`) |
| `footnote_search_duration_seconds` | histogram | `stage` (`embed_query`, `lexical`, `vector`, `rerank`) |
| `footnote_answers_total` | counter | `outcome` (`answered`, `abstained`, `unsupported`, `error`) |
| `footnote_embedding_requests_total`, `footnote_embedding_texts_total` | counter | `provider` |
| `footnote_embedding_truncated_total` | counter | none |
| `footnote_search_exact_scan_total` | counter | `reason` (`small_collection`, `fallback`) |
| `footnote_webhook_deliveries_total` | counter | `status` |
| `footnote_build_info` | gauge | `version`, `commit`, `python_version` |

Scrape `footnote:9094` with job name `footnote` (the alerts below assume it).

Logs are JSON on stdout (`log.format`, `log.level`). Traces are exported over OTLP when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set.

## 10. Alerts

Starting points; tune thresholds to your traffic.

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

Also alert on disk use, replication and connection saturation in Postgres itself, and on
`footnote_answers_total{outcome="error"}` and `footnote_webhook_deliveries_total{status="failed"}`.

## 11. CLI reference

Every command accepts `--config/-c <file>` before the subcommand (or `FOOTNOTE_CONFIG`).
Exit codes: 0 success, 1 failure, 2 invalid configuration or usage.

| Command | Purpose |
|---|---|
| `footnote serve [--with-worker] [--listen ADDR] [--admin-listen ADDR]` | run the service |
| `footnote worker [--admin-listen ADDR]` | run the job worker |
| `footnote migrate [--revision REV]` | apply migrations as `footnote_admin` |
| `footnote healthcheck [--timeout S]` | exit 0 if the local `/readyz` answers 200 |
| `footnote config validate` | validate the configuration and manifest references |
| `footnote config print [--redact/--no-redact]` | print the effective configuration |
| `footnote models pull\|verify [--all]` | download or check model files |
| `footnote tenants create\|list\|delete\|export` | manage tenants |
| `footnote keys create\|list\|revoke\|rotate` | manage tenant keys |
| `footnote spaces register\|list` | manage embedding spaces |
| `footnote version` | print the version |

`tenants`, `keys` and `spaces` use `FOOTNOTE_ADMIN_DATABASE_URL`: they need database access,
not the admin key.
