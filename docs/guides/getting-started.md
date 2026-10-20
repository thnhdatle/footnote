# Getting started

This guide takes you from nothing to a first cited answer on one machine with Docker
Compose. It uses the stack in [`deploy/docker-compose.yml`](../../deploy/docker-compose.yml):
Postgres with pgvector, Footnote, and `fakellm`, a deterministic stub that stands in for
the language model.

For production deployment see [operations.md](operations.md). For every request and
field see [api.md](api.md).

## 1. Prerequisites

- Docker with Compose v2 and about 10 GB of VM memory for the whole stack.
- Network access to `huggingface.co` once, to download the pinned local models.
- `curl`. The examples use `jq` to pick fields out of JSON; it is optional.

## 2. Build the image and start the stack

From the repository root:

```sh
make docker   # builds thnhdatle/footnote:dev
make up       # compose up with the fakellm profile, waits until healthy
```

To use the released image instead of a local build, skip `make docker` and run
`RAYLD_FOOTNOTE_IMAGE=ghcr.io/thnhdatle/footnote:v0.1.0 make up`.

`make up` runs these services in order:

| Service | What it does |
|---|---|
| `postgres` | `pgvector/pgvector:0.8.1-pg17`, with `shm_size: 1gb` (see [operations.md](operations.md)) |
| `bootstrap` | one-shot: `deploy/sql/bootstrap.sql` creates the extension, the roles and the schema |
| `migrate` | one-shot: `footnote migrate` as `footnote_admin` |
| `models` | one-shot: `footnote models pull` downloads the pinned models into the `models` volume |
| `fakellm` | stub LLM and embeddings endpoint on `127.0.0.1:8090` |
| `footnote` | `footnote serve --with-worker`: data plane on `:8000`, admin listener on `:9094` |

Check that the service is ready:

```sh
curl -s http://127.0.0.1:9094/readyz
```

```json
{"status": "ready", "checks": {"...": "..."}}
```

A `503` lists what is missing per check. See [troubleshooting.md](troubleshooting.md).

## 3. Create a tenant and a key

Everything a client stores belongs to a tenant. The compose file sets the admin key to
`dev-admin-key-change-me`; change it for anything but a laptop.

```sh
export FOOTNOTE_URL=http://127.0.0.1:8000
export ADMIN_KEY=dev-admin-key-change-me

TENANT=$(curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "demo"}' | jq -r .id)

export KEY=$(curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants/$TENANT/keys \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "quickstart"}' | jq -r .key)
```

The key starts with `fn_` and is shown once. See [multi-tenancy.md](multi-tenancy.md).

## 4. Create a collection and upload a document

```sh
export COLLECTION=$(curl -s -X POST $FOOTNOTE_URL/v1/collections \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"name": "handbook"}' | jq -r .id)

JOB=$(curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/documents \
  -H "Authorization: Bearer $KEY" -F file=@handbook.pdf | jq -r .job_id)
```

Poll the job until `status` is `done`:

```sh
curl -s -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/jobs/$JOB | jq '{status, progress}'
```

Parsing, chunking and embedding happen in the worker. `footnote serve` without
`--with-worker` does not process uploads; run `footnote worker` next to it. See
[ingestion.md](ingestion.md).

## 5. Search and ask

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "how many vacation days do I get?", "k": 3}'

curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/answers \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
```

The answer has `answer`, `citations` (each with `chunk_id`, pages and a quote span),
`abstained` and `unsupported`. With `fakellm` the text is scripted, so use a real model
to judge answer quality. See [retrieval.md](retrieval.md) and [citations.md](citations.md).

## 6. Use a real language model

Point Footnote at any OpenAI-compatible chat endpoint. For LM Studio on the host:

```sh
export FOOTNOTE_LLM_BASE_URL=http://host.docker.internal:1234/v1
export FOOTNOTE_LLM_MODEL=<model name your endpoint serves>
docker compose -f deploy/docker-compose.yml -p footnote up -d footnote
```

Set `FOOTNOTE_LLM_MODEL` to the model name the endpoint serves. With an empty
`llm.base_url`, search works and `/answers` returns `503 llm_not_configured`. Options are
in [configuration.md](configuration.md).

## 7. Run without Docker

You need Postgres 15 or newer with pgvector 0.8.0 or newer and `uv`.

```sh
psql -h localhost -U postgres -d footnote \
  -v app_password="'change-me'" -v admin_password="'change-me-too'" \
  -f deploy/sql/bootstrap.sql

export FOOTNOTE_DATABASE_URL=postgresql://footnote_app:change-me@localhost/footnote
export FOOTNOTE_ADMIN_DATABASE_URL=postgresql://footnote_admin:change-me-too@localhost/footnote
export FOOTNOTE_AUTH_ADMIN_KEY=...

uv run footnote migrate
uv run footnote models pull
uv run footnote serve --with-worker
```

Models go to `~/.cache/rayld/footnote/models` unless `models.dir` says otherwise.

## 8. Clean up

```sh
make down   # stops the stack and removes its volumes, including the models
```

## Next steps

- [ingestion.md](ingestion.md): supported formats, versions, limits.
- [retrieval.md](retrieval.md): modes, filters, tuning.
- [citations.md](citations.md): how answers are verified.
- [operations.md](operations.md): roles, poolers, backups, upgrades, metrics.
