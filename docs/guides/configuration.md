# Configuration

Configure Footnote with one YAML file plus environment variables.

[`deploy/config.example.yaml`](../../deploy/config.example.yaml) lists every option with its
default.

## Sources and precedence

```sh
footnote --config config.yaml serve    # or set FOOTNOTE_CONFIG
footnote config validate               # check the configuration
footnote config print --redact         # show the effective values, secrets as ***
```

Flags beat environment variables, which beat the file, which beats defaults. Each option's
variable is in the tables below. Mapping and list options take JSON.

## Secrets

Secrets are rejected in the YAML file. Set the variable, or point the `*_file` key at a
file.

| Environment variable | Purpose | File key |
|---|---|---|
| `FOOTNOTE_DATABASE_URL` | app role connection URL | `database.url_file` |
| `FOOTNOTE_ADMIN_DATABASE_URL` | owner role connection URL | `admin_database.url_file` |
| `FOOTNOTE_AUTH_ADMIN_KEY` | admin API key | `auth.admin_key_file` |
| `FOOTNOTE_LLM_API_KEY` | LLM provider key | `llm.api_key_file` |
| `FOOTNOTE_EMBEDDING_API_KEY` | embedding provider key | `embedding.api_key_file` |

## Units

- Durations: `ms`, `s`, `m`, `h`, `d`, chainable (`30s`, `24h`, `1h30m`).
- Sizes: bytes with an optional suffix. `KiB`, `MiB`, `GiB` are powers of 1024; `KB`,
  `MB`, `GB` of 1000.

## Validation

Invalid configuration exits with code 2 and lists every problem. Unknown keys are errors.
The `chunking`, `retrieval` and `answer` sections are defaults for new collections
([api.md](api.md)).

## Observability

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=<collector URL>          # turns tracing on
export OTEL_SDK_DISABLED=true                               # keeps it off
```

`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` also works. Prompts are logged only with
`log.capture_content: true`.

## Test faults

`testing.disable_app_tenant_predicates` is for the tenant-isolation fault test. Never set it in
production.

## Options

### `server`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `server.listen` | `127.0.0.1:8000` | `FOOTNOTE_SERVER_LISTEN` | API listener; `0.0.0.0:8000` in compose |

### `admin`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `admin.listen` | `127.0.0.1:9094` | `FOOTNOTE_ADMIN_LISTEN` | health and metrics listener |

### Top level

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `shutdown_timeout` | `30s` | `FOOTNOTE_SHUTDOWN_TIMEOUT` | drain time on SIGTERM |

### `log`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `log.format` | `json` | `FOOTNOTE_LOG_FORMAT` | `json` or `text` |
| `log.level` | `info` | `FOOTNOTE_LOG_LEVEL` | `debug`, `info`, `warn`, `error` |
| `log.capture_content` | `false` | `FOOTNOTE_LOG_CAPTURE_CONTENT` | log prompts and completions (256 chars) |
| `log.access_log` | `false` | `FOOTNOTE_LOG_ACCESS_LOG` | one log line per request |

### `database`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `database.url` | none | `FOOTNOTE_DATABASE_URL` | secret: env var or `_file` only |
| `database.url_file` | `""` | `FOOTNOTE_DATABASE_URL_FILE` | file with the URL |
| `database.pool_max` | `20` | `FOOTNOTE_DATABASE_POOL_MAX` | maximum pooled connections |
| `database.prepare_threshold` | `5` | `FOOTNOTE_DATABASE_PREPARE_THRESHOLD` | `null` behind a transaction pooler |
| `database.statement_timeout` | `30s` | `FOOTNOTE_DATABASE_STATEMENT_TIMEOUT` | statement timeout |

### `admin_database`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `admin_database.url` | none | `FOOTNOTE_ADMIN_DATABASE_URL` | secret: env var or `_file` only |
| `admin_database.url_file` | `""` | `FOOTNOTE_ADMIN_DATABASE_URL_FILE` | file with the URL |
| `admin_database.pool_max` | `4` | `FOOTNOTE_ADMIN_DATABASE_POOL_MAX` | maximum pooled connections |

### `auth`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `auth.admin_key` | none | `FOOTNOTE_AUTH_ADMIN_KEY` | secret: env var or `_file` only |
| `auth.admin_key_file` | `""` | `FOOTNOTE_AUTH_ADMIN_KEY_FILE` | file with the admin key |
| `auth.key_rotation_grace` | `24h` | `FOOTNOTE_AUTH_KEY_ROTATION_GRACE` | old key validity after rotation |

### `tenancy`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `tenancy.default_quotas.max_collections` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_COLLECTIONS` | `0` = unlimited |
| `tenancy.default_quotas.max_documents` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_DOCUMENTS` | `0` = unlimited |
| `tenancy.default_quotas.max_chunks` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_CHUNKS` | `0` = unlimited |
| `tenancy.default_quotas.max_upload_bytes_per_day` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_UPLOAD_BYTES_PER_DAY` | `0` = unlimited |

### `collections`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `collections.default_language` | `simple` | `FOOTNOTE_COLLECTIONS_DEFAULT_LANGUAGE` | `simple` or `english` |

### `llm`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `llm.base_url` | `""` | `FOOTNOTE_LLM_BASE_URL` | chat endpoint; empty disables answers |
| `llm.model` | `""` | `FOOTNOTE_LLM_MODEL` | model name; required with `llm.base_url` |
| `llm.api_key` | none | `FOOTNOTE_LLM_API_KEY` | secret: env var or `_file` only |
| `llm.api_key_file` | `""` | `FOOTNOTE_LLM_API_KEY_FILE` | file with the key |
| `llm.extra_headers` | `{}` | `FOOTNOTE_LLM_EXTRA_HEADERS` | sent on every request |
| `llm.extra_body` | `{}` | `FOOTNOTE_LLM_EXTRA_BODY` | merged into chat requests |
| `llm.temperature` | `0` | `FOOTNOTE_LLM_TEMPERATURE` | 0..2 |
| `llm.max_tokens` | `512` | `FOOTNOTE_LLM_MAX_TOKENS` | completion cap |
| `llm.timeout` | `120s` | `FOOTNOTE_LLM_TIMEOUT` | per request |
| `llm.max_retries` | `2` | `FOOTNOTE_LLM_MAX_RETRIES` | retries on 429, 5xx, network errors |

### `embedding`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `embedding.default_space` | `bge-small-en-v1.5-onnx` | `FOOTNOTE_EMBEDDING_DEFAULT_SPACE` | one of `embedding.spaces` |
| `embedding.spaces` | `[{"name": "bge-small-en-v1.5-onnx", "provider": "onnx_local", "model": "bge-small-en-v1.5", "calibrated_threshold": null}]` | `FOOTNOTE_EMBEDDING_SPACES` | registered at startup |
| `embedding.base_url` | `""` | `FOOTNOTE_EMBEDDING_BASE_URL` | `openai_compat` endpoint |
| `embedding.api_key` | none | `FOOTNOTE_EMBEDDING_API_KEY` | secret: env var or `_file` only |
| `embedding.api_key_file` | `""` | `FOOTNOTE_EMBEDDING_API_KEY_FILE` | file with the key |
| `embedding.extra_headers` | `{}` | `FOOTNOTE_EMBEDDING_EXTRA_HEADERS` | sent on every request |
| `embedding.batch_size` | `64` | `FOOTNOTE_EMBEDDING_BATCH_SIZE` | texts per request |
| `embedding.max_concurrency` | `4` | `FOOTNOTE_EMBEDDING_MAX_CONCURRENCY` | parallel requests |
| `embedding.timeout` | `30s` | `FOOTNOTE_EMBEDDING_TIMEOUT` | per request |
| `embedding.max_retries` | `3` | `FOOTNOTE_EMBEDDING_MAX_RETRIES` | retries on 429, 5xx, network errors |
| `embedding.max_spaces` | `8` | `FOOTNOTE_EMBEDDING_MAX_SPACES` | max spaces |
| `embedding.hnsw.m` | `16` | `FOOTNOTE_EMBEDDING_HNSW_M` | index build; `ef_construction` >= 2 × `m` |
| `embedding.hnsw.ef_construction` | `64` | `FOOTNOTE_EMBEDDING_HNSW_EF_CONSTRUCTION` | index build; `ef_construction` >= 2 × `m` |

### `models`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `models.dir` | `~/.cache/rayld/footnote/models` | `FOOTNOTE_MODELS_DIR` | `/models` in the image |
| `models.manifest` | `models/manifest.yaml` | `FOOTNOTE_MODELS_MANIFEST` | pinned local models |
| `models.intra_op_threads` | `0` | `FOOTNOTE_MODELS_INTRA_OP_THREADS` | `0` = ONNX Runtime default |

### `rerank`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `rerank.provider` | `onnx_cross_encoder` | `FOOTNOTE_RERANK_PROVIDER` | `onnx_cross_encoder` or `noop` |
| `rerank.model` | `ms-marco-MiniLM-L6-v2` | `FOOTNOTE_RERANK_MODEL` | manifest model id |
| `rerank.variant` | `int8` | `FOOTNOTE_RERANK_VARIANT` | `int8` or `fp32` |
| `rerank.max_tokens` | `256` | `FOOTNOTE_RERANK_MAX_TOKENS` | token cap per pair |
| `rerank.batch_size` | `32` | `FOOTNOTE_RERANK_BATCH_SIZE` | pairs per batch |

### `chunking`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `chunking.strategy` | `heading` | `FOOTNOTE_CHUNKING_STRATEGY` | `heading`, `recursive`, `fixed` |
| `chunking.tokenizer` | `bge-small-en-v1.5` | `FOOTNOTE_CHUNKING_TOKENIZER` | manifest model with a tokenizer |
| `chunking.max_tokens` | `400` | `FOOTNOTE_CHUNKING_MAX_TOKENS` | 32..2048 |
| `chunking.overlap_tokens` | `40` | `FOOTNOTE_CHUNKING_OVERLAP_TOKENS` | 0..`max_tokens`/2 |
| `chunking.min_tokens` | `50` | `FOOTNOTE_CHUNKING_MIN_TOKENS` | 0..`max_tokens` |

### `retrieval`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `retrieval.default_mode` | `hybrid` | `FOOTNOTE_RETRIEVAL_DEFAULT_MODE` | `vector`, `lexical`, `hybrid` |
| `retrieval.lexical_match` | `any` | `FOOTNOTE_RETRIEVAL_LEXICAL_MATCH` | `any`, `all`, `websearch` |
| `retrieval.lexical_rank` | `ts_rank` | `FOOTNOTE_RETRIEVAL_LEXICAL_RANK` | `ts_rank` or `ts_rank_cd` |
| `retrieval.k_lexical` | `50` | `FOOTNOTE_RETRIEVAL_K_LEXICAL` | full-text candidates, 1..1000 |
| `retrieval.k_vector` | `50` | `FOOTNOTE_RETRIEVAL_K_VECTOR` | vector candidates, 1..1000 |
| `retrieval.rrf_k` | `60` | `FOOTNOTE_RETRIEVAL_RRF_K` | 1..1000 |
| `retrieval.rrf_weights.lexical` | `0.2` | `FOOTNOTE_RETRIEVAL_RRF_WEIGHTS_LEXICAL` | 0..10, not both 0 |
| `retrieval.rrf_weights.vector` | `1.0` | `FOOTNOTE_RETRIEVAL_RRF_WEIGHTS_VECTOR` | 0..10, not both 0 |
| `retrieval.rerank` | `true` | `FOOTNOTE_RETRIEVAL_RERANK` | run the reranker |
| `retrieval.rerank_n` | `30` | `FOOTNOTE_RETRIEVAL_RERANK_N` | candidates reranked, k..200 |
| `retrieval.k` | `8` | `FOOTNOTE_RETRIEVAL_K` | results returned, 1..100 |
| `retrieval.ef_search` | `64` | `FOOTNOTE_RETRIEVAL_EF_SEARCH` | HNSW search width, 1..1000 |
| `retrieval.iterative_scan` | `relaxed_order` | `FOOTNOTE_RETRIEVAL_ITERATIVE_SCAN` | `off`, `relaxed_order`, `strict_order` |
| `retrieval.max_scan_tuples` | `20000` | `FOOTNOTE_RETRIEVAL_MAX_SCAN_TUPLES` | iterative scan row cap |
| `retrieval.exact_scan_max` | `20000` | `FOOTNOTE_RETRIEVAL_EXACT_SCAN_MAX` | exact scan up to this many chunks |

### `answer`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `answer.evidence_threshold_rerank` | `null` | `FOOTNOTE_ANSWER_EVIDENCE_THRESHOLD_RERANK` | `null` = reranker's calibrated value |
| `answer.evidence_threshold_vector` | `null` | `FOOTNOTE_ANSWER_EVIDENCE_THRESHOLD_VECTOR` | `null` = space's calibrated value |
| `answer.unsupported_as_abstain` | `false` | `FOOTNOTE_ANSWER_UNSUPPORTED_AS_ABSTAIN` | treat uncited answers as abstentions |
| `answer.prompt_template` | `answer_v1` | `FOOTNOTE_ANSWER_PROMPT_TEMPLATE` | shipped prompt name |

### `ingest`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `ingest.workers` | `4` | `FOOTNOTE_INGEST_WORKERS` | jobs per worker process |
| `ingest.max_file_bytes` | `50MiB` | `FOOTNOTE_INGEST_MAX_FILE_BYTES` | upload size cap |
| `ingest.max_pages` | `2000` | `FOOTNOTE_INGEST_MAX_PAGES` | PDF page cap |
| `ingest.max_uncompressed_bytes` | `200MiB` | `FOOTNOTE_INGEST_MAX_UNCOMPRESSED_BYTES` | DOCX unpacked size cap |
| `ingest.max_archive_members` | `10000` | `FOOTNOTE_INGEST_MAX_ARCHIVE_MEMBERS` | DOCX member cap |
| `ingest.parse_timeout` | `300s` | `FOOTNOTE_INGEST_PARSE_TIMEOUT` | parser time limit |
| `ingest.retain_originals` | `true` | `FOOTNOTE_INGEST_RETAIN_ORIGINALS` | keep uploaded files |
| `ingest.gc_grace` | `10m` | `FOOTNOTE_INGEST_GC_GRACE` | delay before cleanup; `0s` = at once |
| `ingest.job_lease` | `60s` | `FOOTNOTE_INGEST_JOB_LEASE` | job lease |
| `ingest.job_max_attempts` | `5` | `FOOTNOTE_INGEST_JOB_MAX_ATTEMPTS` | attempts before a job fails |
| `ingest.poll_interval` | `1s` | `FOOTNOTE_INGEST_POLL_INTERVAL` | idle poll interval |

### `webhooks`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `webhooks.allow_http` | `false` | `FOOTNOTE_WEBHOOKS_ALLOW_HTTP` | allow `http://` URLs |
| `webhooks.allow_private` | `false` | `FOOTNOTE_WEBHOOKS_ALLOW_PRIVATE` | allow private addresses |
| `webhooks.timeout` | `5s` | `FOOTNOTE_WEBHOOKS_TIMEOUT` | per attempt |
| `webhooks.max_attempts` | `5` | `FOOTNOTE_WEBHOOKS_MAX_ATTEMPTS` | delivery attempts |

### `eval`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `eval.cache_dir` | `~/.cache/rayld/footnote/eval` | `FOOTNOTE_EVAL_CACHE_DIR` | eval embedding cache |
| `eval.calibration_target_answer_rate` | `0.85` | `FOOTNOTE_EVAL_CALIBRATION_TARGET_ANSWER_RATE` | (0, 1] |

### `testing`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `testing.disable_app_tenant_predicates` | `false` | `FOOTNOTE_TESTING_DISABLE_APP_TENANT_PREDICATES` | test only; needs `FOOTNOTE_ENABLE_TEST_FAULTS=1` |
