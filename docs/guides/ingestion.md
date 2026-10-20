# Ingestion

Upload files, update them as new versions, and fix failed ones.

Request shapes are in [api.md](api.md). Error codes are in [errors.md](errors.md).

## 1. Pipeline

```sh
footnote serve --with-worker    # API and worker in one process
footnote worker                 # or a separate worker
```

An upload queues a job. A worker parses, chunks and embeds the file, then makes the new
version searchable at once. Without a worker, uploads stay `queued`.

## 2. Supported formats

The type is detected from the content, not the `Content-Type`.

| Format | Detected by | Notes |
|---|---|---|
| PDF | starts with `%PDF-` | has page numbers |
| DOCX | ZIP with `word/document.xml` | no page numbers |
| HTML | `<!doctype html`, `<html`, or a `.html`/`.htm` name | navigation and scripts dropped |
| Markdown | UTF-8 text named `.md` or `.markdown` | front matter ignored |
| Text | any other UTF-8 text | split on blank lines |

Anything else is `415 unsupported_media_type`. There is no OCR, so scanned PDFs give no
text.

## 3. Limits

| Option | Default | Over the limit |
|---|---|---|
| `ingest.max_file_bytes` | `50MiB` | `413 file_too_large` |
| `ingest.max_pages` | `2000` | `422 too_many_pages` |
| `ingest.max_uncompressed_bytes` | `200MiB` | `422 archive_too_large` |
| `ingest.max_archive_members` | `10000` | `422 archive_too_large` |
| `ingest.parse_timeout` | `300s` | fails with `parse_timeout` |

Encrypted PDFs are `422 pdf_encrypted`.

## 4. Chunking

Collection settings. Defaults come from the `chunking` configuration.

| Setting | Default | Meaning |
|---|---|---|
| `strategy` | `heading` | `heading`, `recursive` or `fixed` |
| `max_tokens` | `400` | tokens per chunk (32-2048) |
| `overlap_tokens` | `40` | overlap (at most half of `max_tokens`) |
| `min_tokens` | `50` | smaller chunks merge |
| `tokenizer` | `bge-small-en-v1.5` | model that counts tokens |

The title is the file name unless you send `title`.

## 5. Versions

- Same `external_id`, changed file: a new version.
- Same file, title and chunking: `200` with `"unchanged": true`. Only `metadata` updates.
- No `external_id`: always a new document.
- `PATCH` of `metadata` applies at once; of `title`, with the next version.
- `DELETE` removes the document and every version.

## 6. Changing chunking or language

A `PATCH` of `chunking` starts a `reindex` job. A `PATCH` of `language` (`simple` or
`english`) starts an `fts_reconfig` job. Search keeps working. For the embedding model see
[models-and-data.md](models-and-data.md).

## 7. Failures

- Transient errors retry up to `ingest.job_max_attempts` (default 5). Then the document is
  `failed` and `document.failed` is sent ([webhooks.md](webhooks.md)).
- Bad files fail at once.
- To retry, upload again with the same `external_id`.

Failed documents do not affect search.

## 8. Originals and text

| Route | Returns |
|---|---|
| `GET .../documents/{id}/text` | parsed text |
| `GET .../documents/{id}/text?format=elements` | elements with pages and headings |
| `GET .../documents/{id}/original` | stored file |
| `GET .../versions/{n}/original` | any version's file |

With `ingest.retain_originals: false`, files are deleted once indexed, and the original
routes return `404 original_not_retained`. Keeping them makes backups larger
([operations.md](operations.md)).
