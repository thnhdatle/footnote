# Getting started

Go from nothing to a first cited answer on one machine with
[Docker Compose](../../deploy/docker-compose.yml).

## 1. Prerequisites

- Docker with Compose v2 and about 10 GB of VM memory.
- Access to `huggingface.co` once, for the models.
- `curl`, and optionally `jq`.

## 2. Build the image and start the stack

```sh
make docker   # builds thnhdatle/footnote:dev
make up       # starts the stack with fakellm, waits until healthy
```

```sh
RAYLD_FOOTNOTE_IMAGE=ghcr.io/thnhdatle/footnote:v0.1.0 make up    # or use the released image
```

| Service | What it does |
|---|---|
| `postgres` | Postgres with pgvector |
| `bootstrap` | one-shot: roles and schema |
| `migrate` | one-shot: migrations |
| `models` | one-shot: downloads models |
| `fakellm` | stub LLM on `127.0.0.1:8090` |
| `footnote` | API on `:8000`, admin on `:9094` |

```sh
curl -s http://127.0.0.1:9094/readyz    # {"status": "ready", ...}
```

A `503` lists what is missing. See [troubleshooting.md](troubleshooting.md).

## 3. Create a tenant and a key

```sh
export FOOTNOTE_URL=http://127.0.0.1:8000
export ADMIN_KEY=dev-admin-key-change-me    # set in the compose file

TENANT=$(curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "demo"}' | jq -r .id)

export KEY=$(curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants/$TENANT/keys \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "quickstart"}' | jq -r .key)
```

Change the admin key for anything but a laptop. The tenant key is shown once.

## 4. Create a collection and upload a document

```sh
export COLLECTION=$(curl -s -X POST $FOOTNOTE_URL/v1/collections \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"name": "handbook"}' | jq -r .id)

JOB=$(curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/documents \
  -H "Authorization: Bearer $KEY" -F file=@handbook.pdf | jq -r .job_id)

curl -s -H "Authorization: Bearer $KEY" $FOOTNOTE_URL/v1/jobs/$JOB | jq '{status, progress}'
```

Poll the job until `status` is `done`.

## 5. Search and ask

```sh
curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query": "how many vacation days do I get?", "k": 3}'

curl -s -X POST $FOOTNOTE_URL/v1/collections/$COLLECTION/answers \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
```

fakellm returns scripted text. Use a real model to judge quality.

## 6. Use a real language model

```sh
export FOOTNOTE_LLM_BASE_URL=http://host.docker.internal:1234/v1    # LM Studio on the host
export FOOTNOTE_LLM_MODEL=<model name your endpoint serves>
docker compose -f deploy/docker-compose.yml -p footnote up -d footnote
```

Any OpenAI-compatible chat endpoint works. The compose file always sets these two variables
(fakellm by default), so they override `llm.base_url` and `llm.model` in a mounted config file.

## 7. Run without Docker

You need Postgres 15+ with pgvector 0.8.0+, and `uv`.

```sh
psql -h localhost -U postgres -d footnote \
  -v app_password="'change-me'" -v admin_password="'change-me-too'" \
  -f deploy/sql/bootstrap.sql                  # roles and schema

export FOOTNOTE_DATABASE_URL=postgresql://footnote_app:change-me@localhost/footnote
export FOOTNOTE_ADMIN_DATABASE_URL=postgresql://footnote_admin:change-me-too@localhost/footnote
export FOOTNOTE_AUTH_ADMIN_KEY=...

uv run footnote migrate                        # create the schema
uv run footnote models pull                    # download models
uv run footnote serve --with-worker            # API and worker
```


## 8. Clean up

```sh
make down   # stops the stack and deletes its volumes, including the models
```

## Next steps

- [ingestion.md](ingestion.md): formats, versions, limits.
- [retrieval.md](retrieval.md): modes, filters, tuning.
- [citations.md](citations.md): how answers are verified.
- [operations.md](operations.md): deploy, backups, upgrades, metrics.
