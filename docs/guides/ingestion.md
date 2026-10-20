# Ingestion

This guide explains what happens to a file between `POST .../documents` and the moment
it is searchable: parsing, chunking, embedding, versions, limits and failures. Request
and response shapes are in [api.md](api.md); error codes are in [errors.md](errors.md).

## 1. Pipeline

An upload does two things in one transaction: it stores the original file (compressed
with zstd, in the `document_blobs` table) and queues an `ingest` job. A worker then runs
the job:

1. Parse the original in a sandboxed child process into elements (headings, paragraphs,
   list items, tables, code).
2. Chunk the elements with the collection's chunking settings.
3. Compare the chunks with the document's existing chunks by content hash.
4. Embed only the chunks that have no vector yet.
5. Swap the new version in with one transaction.

Searches see the old version of a document until the swap and the new version after it,
never a mix. Only a worker runs jobs: start `footnote serve --with-worker` or a separate
`footnote worker`. `ingest.workers` (default 4) sets how many jobs one worker process runs
at once.

## 2. Supported formats

The type is detected from the content. The `Content-Type` of the upload is ignored.

| Format | Detected by | Elements |
|---|---|---|
| PDF | starts with `%PDF-` | headings by font size, paragraphs, lists, tables as Markdown, code; `page` and `page_to` |
| DOCX | ZIP with `word/document.xml` | `Heading 1..6` styles, lists, tables; no page numbers |
| HTML | starts with `<!doctype html` or `<html`, or filename ends `.html`/`.htm` | `h1`-`h6`, `p`, `li`, `table`, `pre`; navigation, footer, scripts and styles are dropped |
| Markdown | UTF-8 text with a `.md` or `.markdown` filename | CommonMark; front matter is ignored |
| Text | any other UTF-8 text | paragraphs split on blank lines |

Anything else is `415 unsupported_media_type`. Scanned PDFs without a text layer produce no
text; Footnote does not run OCR.

## 3. Limits and sandbox

| Option | Default | Effect |
|---|---|---|
| `ingest.max_file_bytes` | `50MiB` | larger uploads are `413 file_too_large` |
| `ingest.max_pages` | `2000` | more PDF pages is `422 too_many_pages` |
| `ingest.max_uncompressed_bytes` | `200MiB` | DOCX that expands further is `422 archive_too_large` |
| `ingest.max_archive_members` | `10000` | DOCX with more members is `422 archive_too_large` |
| `ingest.parse_timeout` | `300s` | the parser process is killed; the document fails with `parse_timeout` |

Encrypted PDFs are `422 pdf_encrypted`. The parser runs in a freshly spawned child
process with a CPU limit, no core dumps, no file creation and, on Linux, an address-space
limit. XML is read without entity resolution or network access. A parser crash or a
timeout fails one document and never the worker.

## 4. Chunking

Each collection has chunking settings (defaults from the `chunking` configuration):

| Setting | Default | Meaning |
|---|---|---|
| `strategy` | `heading` | `heading`: one chunk per section under the deepest heading, split by token budget at paragraph boundaries. `recursive`: paragraph, then sentence, then word splits. `fixed`: token windows with overlap. |
| `max_tokens` | `400` | budget per chunk (32-2048) |
| `overlap_tokens` | `40` | tokens repeated from the previous chunk (at most half of `max_tokens`) |
| `min_tokens` | `50` | smaller chunks merge with neighbours |
| `tokenizer` | `bge-small-en-v1.5` | a manifest model with a tokenizer file; counts tokens |

Tables are never split inside a row and code is never split inside a line. A single
element larger than the budget stays whole.

The text that is embedded, reranked and indexed for lexical search is the title, a
newline, the heading path joined with ` > `, a blank line and the chunk text. The default
title is the file name (or the document's current title when you upload a new version); send `title` to set it.

## 5. Versions and incremental indexing

A chunk is identified by the SHA-256 of its embedded text (whitespace and Unicode
normalised) within its document. This gives the following behaviour.

- Upload with an existing `external_id` and a changed file: a new version. Chunks whose text
  did not change keep their vectors; only new chunks are embedded.
- Upload of the same file, title and chunking settings: `200` with `"unchanged": true`.
  Nothing is indexed, but `metadata` is updated.
- Without `external_id`, every upload is a new document.
- `PATCH` of `metadata` applies at once; `PATCH` of `title` changes what is indexed with the
  next version.
- `DELETE` of a document removes all its versions, chunks, vectors and originals.

Version status is `processing`, `ready`, `failed` or `superseded`. Document status is
`queued`, `parsing`, `embedding`, `ready` or `failed`.

## 6. Changing chunking or language

`PATCH /v1/collections/{id}` with new `chunking` settings starts a `reindex` job. It
re-chunks every document from its stored elements, so the original files are not read
again, and queues one ingest job per document. Changing `language` (`simple` or `english`)
starts an `fts_reconfig` job that rebuilds the lexical index. Search keeps working while
either runs. To change the embedding model see [models-and-data.md](models-and-data.md).

## 7. Retries, failures and cleanup

- A transient failure (embedding endpoint down, a conflict with a re-embed) is retried with
  backoff up to `ingest.job_max_attempts` (default 5). After the last attempt the version
  and document are `failed` with an `error` and the `document.failed` webhook is sent
  ([webhooks.md](webhooks.md)).
- A permanent failure (corrupt file, encrypted PDF, chunk quota) fails at once.
- A worker that dies loses its lease after `ingest.job_lease` (default 60s) and another
  worker takes the job.
- Superseded rows (old memberships, chunks no version uses, old vectors) are removed by a
  `gc` job `ingest.gc_grace` (default 10m) after the swap. Set `0s` to collect at once.
  See [operations.md](operations.md).

Upload the file again with the same `external_id` to retry a failed document. Failed documents do not affect search.

## 8. Originals and text

- `GET .../documents/{id}/text` returns the parsed text; `?format=elements` returns the
  elements with their pages and heading paths.
- `GET .../documents/{id}/original` returns the stored file, and
  `GET .../versions/{n}/original` returns any version.
- With `ingest.retain_originals: false` the original is deleted once the version is
  `ready` and these routes answer `404 original_not_retained`. Re-chunking never needs the
  original, because elements are stored with each version.
- Retaining every version's original makes the database, and so backups, larger. See
  [operations.md](operations.md).

## 9. Monitoring

Job counts, queue depth and stage timings (`parse`, `chunk`, `embed`, `swap`) are exported
as Prometheus metrics. See [operations.md](operations.md).
