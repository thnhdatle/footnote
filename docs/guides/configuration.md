# Configuration

Footnote reads one YAML file plus environment variables. The annotated reference file
is [`deploy/config.example.yaml`](../../deploy/config.example.yaml); it contains every
option with its default, and a test keeps it in sync with the code.

## Sources and precedence

From highest to lowest: command line flags, environment variables, the config file,
built-in defaults. The file path comes from `--config` or `FOOTNOTE_CONFIG`; without
either, defaults plus environment are used.

Every option has the environment variable `FOOTNOTE_<PATH>`: the dotted path with dots
replaced by underscores, upper-cased (`database.pool_max` is `FOOTNOTE_DATABASE_POOL_MAX`).
Mapping and list options (`llm.extra_headers`, `llm.extra_body`, `embedding.extra_headers`,
`embedding.spaces`) take JSON. For options that allow `null`, the value `null` clears them.

## Secrets

Secrets are never read from the YAML file (a secret key there is a validation error) and
never appear in the example file. Set the environment variable, or point the sibling
`*_file` key at a file (Docker or Kubernetes secrets); the file content is trimmed. If both
are set the environment variable wins. `footnote config print --redact` shows secrets as `***`.

| Environment variable | Purpose | File key |
|---|---|---|
| `FOOTNOTE_DATABASE_URL` | app role connection URL | `database.url_file` |
| `FOOTNOTE_ADMIN_DATABASE_URL` | owner role connection URL | `admin_database.url_file` |
| `FOOTNOTE_AUTH_ADMIN_KEY` | admin API key | `auth.admin_key_file` |
| `FOOTNOTE_LLM_API_KEY` | LLM provider key | `llm.api_key_file` |
| `FOOTNOTE_EMBEDDING_API_KEY` | embedding provider key | `embedding.api_key_file` |

## Units

Durations are a number followed by `ms`, `s`, `m`, `h` or `d`, and can be chained
(`30s`, `10m`, `24h`, `14d`, `1h30m`). Sizes are bytes with an optional suffix
(`64MiB`, `1GB`; `KiB`, `MiB`, `GiB` are powers of 1024, `KB`, `MB`, `GB` of 1000).

## Validation

Validation runs at startup and reports every problem at once, then exits with code 2.
Unknown keys are errors. The `chunking`, `retrieval` and `answer` sections are the
defaults of new collections and obey the same bounds as the collection settings API
([api.md](api.md)): for example `retrieval.k` 1..100, `retrieval.k_lexical` and `retrieval.k_vector`
1..1000, `retrieval.rerank_n` between `k` and 200, `chunking.max_tokens` 32..2048,
`chunking.overlap_tokens` at most half of `max_tokens`. Embedding spaces need unique names,
`embedding.default_space` must be one of them, `openai_compat` spaces need `dimensions`
and `embedding.base_url`, and `embedding.hnsw.ef_construction` must be at least twice
`embedding.hnsw.m`.

## Observability environment

Logging is configured by the `log.*` options. Prometheus metrics are served by the admin
listener at `/metrics`. Tracing is off by default and follows the standard OpenTelemetry
environment: it turns on when `OTEL_EXPORTER_OTLP_ENDPOINT` (or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, unless `OTEL_SDK_DISABLED=true`; the other
`OTEL_EXPORTER_OTLP_*` variables apply to the exporter. Prompt and completion content is
logged and attached to spans only with `log.capture_content: true`, truncated to 256 characters.

## Test faults

`testing.disable_app_tenant_predicates` (used by the tenant-isolation fault test of the
[acceptance suite](../../acceptance/README.md)) is honoured only
when the process also has `FOOTNOTE_ENABLE_TEST_FAULTS=1`, and logs a warning when active.

## Options

### `server`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `server.listen` | `127.0.0.1:8000` | `FOOTNOTE_SERVER_LISTEN` | data plane; compose: 0.0.0.0:8000 |

### `admin`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `admin.listen` | `127.0.0.1:9094` | `FOOTNOTE_ADMIN_LISTEN` | /healthz, /readyz, /metrics |

### Top level

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `shutdown_timeout` | `30s` | `FOOTNOTE_SHUTDOWN_TIMEOUT` | grace for in-flight requests and jobs on SIGTERM |

### `log`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `log.format` | `json` | `FOOTNOTE_LOG_FORMAT` | json \| text |
| `log.level` | `info` | `FOOTNOTE_LOG_LEVEL` | debug \| info \| warn \| error |
| `log.capture_content` | `false` | `FOOTNOTE_LOG_CAPTURE_CONTENT` | prompts/completions in logs and spans, truncated to 256 chars |
| `log.access_log` | `false` | `FOOTNOTE_LOG_ACCESS_LOG` | one info line per request when true |

### `database`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `database.url` | none | `FOOTNOTE_DATABASE_URL` | Secret: set only by env var or the sibling `_file` key; rejected in the YAML file. |
| `database.url_file` | `""` | `FOOTNOTE_DATABASE_URL_FILE` | file with the URL; or FOOTNOTE_DATABASE_URL |
| `database.pool_max` | `20` | `FOOTNOTE_DATABASE_POOL_MAX` | maximum pooled connections |
| `database.prepare_threshold` | `5` | `FOOTNOTE_DATABASE_PREPARE_THRESHOLD` | null behind a transaction pooler |
| `database.statement_timeout` | `30s` | `FOOTNOTE_DATABASE_STATEMENT_TIMEOUT` | per-transaction statement timeout |

### `admin_database`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `admin_database.url` | none | `FOOTNOTE_ADMIN_DATABASE_URL` | Secret: set only by env var or the sibling `_file` key; rejected in the YAML file. |
| `admin_database.url_file` | `""` | `FOOTNOTE_ADMIN_DATABASE_URL_FILE` | file with the URL; or FOOTNOTE_ADMIN_DATABASE_URL |
| `admin_database.pool_max` | `4` | `FOOTNOTE_ADMIN_DATABASE_POOL_MAX` | maximum pooled connections |

### `auth`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `auth.admin_key` | none | `FOOTNOTE_AUTH_ADMIN_KEY` | Secret: set only by env var or the sibling `_file` key; rejected in the YAML file. |
| `auth.admin_key_file` | `""` | `FOOTNOTE_AUTH_ADMIN_KEY_FILE` | file with the admin key; or FOOTNOTE_AUTH_ADMIN_KEY |
| `auth.key_rotation_grace` | `24h` | `FOOTNOTE_AUTH_KEY_ROTATION_GRACE` | old tenant key stays valid this long after rotation |

### `tenancy`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `tenancy.default_quotas.max_collections` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_COLLECTIONS` | 0 = unlimited |
| `tenancy.default_quotas.max_documents` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_DOCUMENTS` | 0 = unlimited |
| `tenancy.default_quotas.max_chunks` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_CHUNKS` | 0 = unlimited |
| `tenancy.default_quotas.max_upload_bytes_per_day` | `0` | `FOOTNOTE_TENANCY_DEFAULT_QUOTAS_MAX_UPLOAD_BYTES_PER_DAY` | 0 = unlimited |

### `collections`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `collections.default_language` | `simple` | `FOOTNOTE_COLLECTIONS_DEFAULT_LANGUAGE` | simple \| english |

### `llm`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `llm.base_url` | `""` | `FOOTNOTE_LLM_BASE_URL` | FOOTNOTE_LLM_BASE_URL; "" disables answers; LM Studio: http://localhost:1234/v1, Triage: http://localhost:8080/v1 |
| `llm.model` | `""` | `FOOTNOTE_LLM_MODEL` | chat model name sent to the provider; required when `llm.base_url` is set |
| `llm.api_key` | none | `FOOTNOTE_LLM_API_KEY` | Secret: set only by env var or the sibling `_file` key; rejected in the YAML file. |
| `llm.api_key_file` | `""` | `FOOTNOTE_LLM_API_KEY_FILE` | file with the key; or FOOTNOTE_LLM_API_KEY |
| `llm.extra_headers` | `{}` | `FOOTNOTE_LLM_EXTRA_HEADERS` | forwarded on every request |
| `llm.extra_body` | `{}` | `FOOTNOTE_LLM_EXTRA_BODY` | merged into chat requests (e.g. reasoning switches) |
| `llm.temperature` | `0` | `FOOTNOTE_LLM_TEMPERATURE` | 0..2 |
| `llm.max_tokens` | `512` | `FOOTNOTE_LLM_MAX_TOKENS` | completion cap |
| `llm.timeout` | `120s` | `FOOTNOTE_LLM_TIMEOUT` | per request |
| `llm.max_retries` | `2` | `FOOTNOTE_LLM_MAX_RETRIES` | on 429, 5xx and transport errors |

### `embedding`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `embedding.default_space` | `bge-small-en-v1.5-onnx` | `FOOTNOTE_EMBEDDING_DEFAULT_SPACE` | must be one of spaces |
| `embedding.spaces` | `[{"name": "bge-small-en-v1.5-onnx", "provider": "onnx_local", "model": "bge-small-en-v1.5", "calibrated_threshold": null}]` | `FOOTNOTE_EMBEDDING_SPACES` | registered at startup via footnote_register_space |
| `embedding.base_url` | `""` | `FOOTNOTE_EMBEDDING_BASE_URL` | FOOTNOTE_EMBEDDING_BASE_URL (openai_compat) |
| `embedding.api_key` | none | `FOOTNOTE_EMBEDDING_API_KEY` | Secret: set only by env var or the sibling `_file` key; rejected in the YAML file. |
| `embedding.api_key_file` | `""` | `FOOTNOTE_EMBEDDING_API_KEY_FILE` | file with the key; or FOOTNOTE_EMBEDDING_API_KEY |
| `embedding.extra_headers` | `{}` | `FOOTNOTE_EMBEDDING_EXTRA_HEADERS` | forwarded on every embedding request |
| `embedding.batch_size` | `64` | `FOOTNOTE_EMBEDDING_BATCH_SIZE` | texts per request |
| `embedding.max_concurrency` | `4` | `FOOTNOTE_EMBEDDING_MAX_CONCURRENCY` | parallel requests |
| `embedding.timeout` | `30s` | `FOOTNOTE_EMBEDDING_TIMEOUT` | per request |
| `embedding.max_retries` | `3` | `FOOTNOTE_EMBEDDING_MAX_RETRIES` | on 429, 5xx and transport errors |
| `embedding.max_spaces` | `8` | `FOOTNOTE_EMBEDDING_MAX_SPACES` | registered spaces cap |
| `embedding.hnsw.m` | `16` | `FOOTNOTE_EMBEDDING_HNSW_M` | pgvector index build; ef_construction >= 2*m |
| `embedding.hnsw.ef_construction` | `64` | `FOOTNOTE_EMBEDDING_HNSW_EF_CONSTRUCTION` | pgvector index build; ef_construction >= 2*m |

### `models`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `models.dir` | `~/.cache/rayld/footnote/models` | `FOOTNOTE_MODELS_DIR` | /models in the image (volume) |
| `models.manifest` | `models/manifest.yaml` | `FOOTNOTE_MODELS_MANIFEST` | pinned local models |
| `models.intra_op_threads` | `0` | `FOOTNOTE_MODELS_INTRA_OP_THREADS` | 0 = ONNX Runtime default |

### `rerank`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `rerank.provider` | `onnx_cross_encoder` | `FOOTNOTE_RERANK_PROVIDER` | onnx_cross_encoder \| noop |
| `rerank.model` | `ms-marco-MiniLM-L6-v2` | `FOOTNOTE_RERANK_MODEL` | models/manifest.yaml id |
| `rerank.variant` | `int8` | `FOOTNOTE_RERANK_VARIANT` | int8 (per-arch file) \| fp32 |
| `rerank.max_tokens` | `256` | `FOOTNOTE_RERANK_MAX_TOKENS` | query + passage token cap |
| `rerank.batch_size` | `32` | `FOOTNOTE_RERANK_BATCH_SIZE` | pairs per inference batch |

### `chunking`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `chunking.strategy` | `heading` | `FOOTNOTE_CHUNKING_STRATEGY` | heading \| recursive \| fixed |
| `chunking.tokenizer` | `bge-small-en-v1.5` | `FOOTNOTE_CHUNKING_TOKENIZER` | models/manifest.yaml id with a tokenizer file |
| `chunking.max_tokens` | `400` | `FOOTNOTE_CHUNKING_MAX_TOKENS` | 32..2048 |
| `chunking.overlap_tokens` | `40` | `FOOTNOTE_CHUNKING_OVERLAP_TOKENS` | 0..max_tokens/2 |
| `chunking.min_tokens` | `50` | `FOOTNOTE_CHUNKING_MIN_TOKENS` | 0..max_tokens |

### `retrieval`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `retrieval.default_mode` | `hybrid` | `FOOTNOTE_RETRIEVAL_DEFAULT_MODE` | vector \| lexical \| hybrid |
| `retrieval.lexical_match` | `any` | `FOOTNOTE_RETRIEVAL_LEXICAL_MATCH` | any \| all \| websearch |
| `retrieval.lexical_rank` | `ts_rank` | `FOOTNOTE_RETRIEVAL_LEXICAL_RANK` | ts_rank \| ts_rank_cd |
| `retrieval.k_lexical` | `50` | `FOOTNOTE_RETRIEVAL_K_LEXICAL` | lexical leg candidates, 1..1000 |
| `retrieval.k_vector` | `50` | `FOOTNOTE_RETRIEVAL_K_VECTOR` | vector leg candidates, 1..1000 |
| `retrieval.rrf_k` | `60` | `FOOTNOTE_RETRIEVAL_RRF_K` | 1..1000 |
| `retrieval.rrf_weights.lexical` | `0.2` | `FOOTNOTE_RETRIEVAL_RRF_WEIGHTS_LEXICAL` | 0..10 each, not both 0; lexical: SciFact train-split choice |
| `retrieval.rrf_weights.vector` | `1.0` | `FOOTNOTE_RETRIEVAL_RRF_WEIGHTS_VECTOR` | 0..10 each, not both 0 |
| `retrieval.rerank` | `true` | `FOOTNOTE_RETRIEVAL_RERANK` | rerank the fused list |
| `retrieval.rerank_n` | `30` | `FOOTNOTE_RETRIEVAL_RERANK_N` | candidates reranked, k..200 |
| `retrieval.k` | `8` | `FOOTNOTE_RETRIEVAL_K` | results returned, 1..100 |
| `retrieval.ef_search` | `64` | `FOOTNOTE_RETRIEVAL_EF_SEARCH` | hnsw.ef_search, 1..1000 |
| `retrieval.iterative_scan` | `relaxed_order` | `FOOTNOTE_RETRIEVAL_ITERATIVE_SCAN` | off \| relaxed_order \| strict_order |
| `retrieval.max_scan_tuples` | `20000` | `FOOTNOTE_RETRIEVAL_MAX_SCAN_TUPLES` | hnsw.max_scan_tuples |
| `retrieval.exact_scan_max` | `20000` | `FOOTNOTE_RETRIEVAL_EXACT_SCAN_MAX` | collections at or below this many chunks use exact scan |

### `answer`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `answer.evidence_threshold_rerank` | `null` | `FOOTNOTE_ANSWER_EVIDENCE_THRESHOLD_RERANK` | null = manifest calibrated_threshold of the active reranker file |
| `answer.evidence_threshold_vector` | `null` | `FOOTNOTE_ANSWER_EVIDENCE_THRESHOLD_VECTOR` | null = calibrated_threshold of the collection's space |
| `answer.unsupported_as_abstain` | `false` | `FOOTNOTE_ANSWER_UNSUPPORTED_AS_ABSTAIN` | report unsupported answers as abstentions |
| `answer.prompt_template` | `answer_v1` | `FOOTNOTE_ANSWER_PROMPT_TEMPLATE` | app/prompts/<name>.md |

### `ingest`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `ingest.workers` | `4` | `FOOTNOTE_INGEST_WORKERS` | concurrent jobs per worker process |
| `ingest.max_file_bytes` | `50MiB` | `FOOTNOTE_INGEST_MAX_FILE_BYTES` | upload size cap |
| `ingest.max_pages` | `2000` | `FOOTNOTE_INGEST_MAX_PAGES` | PDF page cap |
| `ingest.max_uncompressed_bytes` | `200MiB` | `FOOTNOTE_INGEST_MAX_UNCOMPRESSED_BYTES` | decompression cap (zip bombs) |
| `ingest.max_archive_members` | `10000` | `FOOTNOTE_INGEST_MAX_ARCHIVE_MEMBERS` | DOCX/zip member cap |
| `ingest.parse_timeout` | `300s` | `FOOTNOTE_INGEST_PARSE_TIMEOUT` | sandboxed parser wall clock |
| `ingest.retain_originals` | `true` | `FOOTNOTE_INGEST_RETAIN_ORIGINALS` | keep uploaded files after parsing |
| `ingest.gc_grace` | `10m` | `FOOTNOTE_INGEST_GC_GRACE` | delay before garbage collecting replaced chunks; `0s` collects at once |
| `ingest.job_lease` | `60s` | `FOOTNOTE_INGEST_JOB_LEASE` | job lease; renewed every third of it |
| `ingest.job_max_attempts` | `5` | `FOOTNOTE_INGEST_JOB_MAX_ATTEMPTS` | attempts before a job fails permanently |
| `ingest.poll_interval` | `1s` | `FOOTNOTE_INGEST_POLL_INTERVAL` | idle queue poll |

### `webhooks`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `webhooks.allow_http` | `false` | `FOOTNOTE_WEBHOOKS_ALLOW_HTTP` | allow http:// targets |
| `webhooks.allow_private` | `false` | `FOOTNOTE_WEBHOOKS_ALLOW_PRIVATE` | allow private and loopback addresses |
| `webhooks.timeout` | `5s` | `FOOTNOTE_WEBHOOKS_TIMEOUT` | per delivery attempt |
| `webhooks.max_attempts` | `5` | `FOOTNOTE_WEBHOOKS_MAX_ATTEMPTS` | delivery attempts |

### `eval`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `eval.cache_dir` | `~/.cache/rayld/footnote/eval` | `FOOTNOTE_EVAL_CACHE_DIR` | embedding cache of the evaluation harness |
| `eval.calibration_target_answer_rate` | `0.85` | `FOOTNOTE_EVAL_CALIBRATION_TARGET_ANSWER_RATE` | (0, 1] |

### `testing`

| Option | Default | Environment variable | Description |
|---|---|---|---|
| `testing.disable_app_tenant_predicates` | `false` | `FOOTNOTE_TESTING_DISABLE_APP_TENANT_PREDICATES` | honoured only with env FOOTNOTE_ENABLE_TEST_FAULTS=1 (tenant-isolation fault test); logs a warning |
