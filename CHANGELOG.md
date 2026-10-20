# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-21

First public release, published as the container image `ghcr.io/thnhdatle/footnote:v0.1.0`
(`linux/amd64`, `linux/arm64`).

### Added

- Multi-tenancy on Postgres row-level security with separate `footnote_app` and
  `footnote_admin` roles, tenant API keys (rotation, expiry), quotas, usage, export and an
  admin API.
- Collections with per-collection chunking, retrieval, answer and webhook settings.
- Document upload with content-sniffed parsing of PDF, DOCX, HTML, Markdown and plain text in
  a sandboxed child process with size, page, archive and time limits.
- Heading, recursive and fixed chunking with token budgets, and stored originals and parsed
  elements per document version.
- Incremental indexing: content-hash chunk identity, versioned documents, atomic version
  swaps and delayed garbage collection (`ingest.gc_grace`).
- Postgres job queue with leases, retries and workers (ingest, reindex, FTS reconfiguration,
  re-embedding, garbage collection, webhooks, deletions).
- Embedding spaces with `onnx_local` (pinned BGE-small ONNX model) and `openai_compat`
  providers, per-space HNSW indexes and `halfvec` for large dimensions.
- Re-embedding on space change with dual-write and an atomic switch.
- Hybrid retrieval: Postgres full-text and pgvector legs, weighted rank fusion, metadata and
  date filters, optional cross-encoder reranking, exact-scan fallback and debug output.
- Answers with an evidence gate, abstention, verified `[n]` citations with quote spans and
  server-sent-event streaming.
- Signed ingest webhooks with an SSRF policy, retries and a delivery log.
- Observability: Prometheus metrics, structured logs, OpenTelemetry traces, `/healthz` and
  `/readyz` with database, role, model and endpoint checks.
- Command line: `serve`, `worker`, `migrate`, `healthcheck`, `config`, `models`, `tenants`,
  `keys`, `spaces`, `eval` and `version`.
- Evaluation harness (`footnote eval`, from a clone with the `eval` extra): retrieval
  metrics and ablations on BEIR datasets, abstention curves, threshold calibration and
  answer faithfulness with a local judge.
- Multi-architecture container image (distroless, non-root, read-only root filesystem), a
  compose stack with Postgres, migrations, model download and an optional stub LLM, and
  Kubernetes manifests.
- User guides for getting started, ingestion, retrieval, citations, multi-tenancy,
  webhooks, models and data, evaluation, configuration, errors, operations and
  troubleshooting.

[Unreleased]: https://github.com/thnhdatle/footnote/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thnhdatle/footnote/releases/tag/v0.1.0
