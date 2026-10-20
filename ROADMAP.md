# Roadmap

Features that are not in v0.1 and planned follow-ups. Items marked `[gfi]` are suitable as
good first issues. Every `TODO(roadmap)` in the code has an entry here.

## Not in v0.1

- [ ] OCR of scanned images
- [ ] Connectors (S3, Git, Drive, Confluence)
- [ ] Chat UI
- [ ] Agent builder
- [ ] GraphRAG
- [ ] Query rewriting, multi-query and HyDE
- [ ] Semantic answer cache
- [ ] Conversation memory
- [ ] File types beyond PDF, DOCX, HTML, Markdown and TXT [gfi]
- [ ] Full-text search languages beyond `simple` and `english` [gfi]
- [ ] Per-request model override [gfi]

## Planned follow-ups

- [ ] Python package on PyPI: ship `migrations/`, `alembic.ini` and `models/manifest.yaml`
      inside the wheel (read through `importlib.resources`) so that `footnote migrate` and
      `footnote models pull` work outside the image; v0.1 ships as a container image only
- [ ] Ingestion jobs on [Foreman](https://github.com/thnhdatle/foreman), a Postgres job queue
- [ ] Lexical search through a `SECURITY DEFINER` function, as API key lookup and job claims do
- [ ] MCP server
- [ ] Python client
- [ ] Per-key rate limits
- [ ] Air-gapped model bundle
- [ ] BM25 extension (for example `pg_search`)
- [ ] Helm chart
