# SPDX-License-Identifier: Apache-2.0
"""`/v1/collections/{id}/documents`: upload, read, patch, delete, text, originals.

The upload body is parsed as it streams: the middleware cuts it off past
`ingest.max_file_bytes` plus a fixed multipart overhead, the file part is spooled to a
temporary file beyond 8 MiB, and the exact file size is checked again by the use case
(413 `file_too_large`). Content sniffing and the synchronous PDF/DOCX checks run behind
the `UploadInspector` port, never on the client's Content-Type.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, Path, Query, Request, Response
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from footnote.api.auth import Tenant
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.pagination import DEFAULT_LIMIT, Cursor, Limit, decode_cursor, next_cursor
from footnote.api.schemas.common import Page, error_responses
from footnote.api.schemas.documents import (
    DocumentDetailOut,
    DocumentOut,
    DocumentPatch,
    ElementOut,
    ElementsOut,
    UploadOut,
)
from footnote.app.documents import METADATA_MAX_BYTES, UploadRequest
from footnote.domain.documents import CollectionId, DocumentId
from footnote.domain.errors import TooLargeError, ValidationError

router = APIRouter(prefix="/v1/collections/{collection_id}/documents", tags=["documents"])
Deps = Annotated[ApiDeps, Depends(get_deps)]

SPOOL_MAX_BYTES = 8 * 1024 * 1024
_FIELD_MAX_BYTES = METADATA_MAX_BYTES + 1024
_FIELDS = ("external_id", "title", "metadata")

_UPLOAD_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "external_id": {"type": "string", "maxLength": 512},
                        "title": {"type": "string", "maxLength": 512},
                        "metadata": {
                            "type": "string",
                            "description": "JSON object, at most 16 KiB",
                        },
                    },
                }
            }
        },
    }
}


class _UploadParser(MultiPartParser):
    """Keeps up to 8 MiB of the file in memory, then spools to `TMPDIR`."""

    spool_max_size = SPOOL_MAX_BYTES


async def _read_upload(request: Request, max_file_bytes: int) -> UploadRequest:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise ValidationError("file", "expected a multipart/form-data body with a `file` part")
    parser = _UploadParser(
        request.headers,
        request.stream(),
        max_files=1,
        max_fields=len(_FIELDS),
        max_part_size=_FIELD_MAX_BYTES,
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise ValidationError("file", exc.message) from exc
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise ValidationError("file", "a `file` part is required")
        if upload.size is not None and upload.size > max_file_bytes:
            raise TooLargeError(f"file exceeds {max_file_bytes} bytes")
        data = await upload.read()
        fields = {k: v for k, v in form.items() if k != "file"}
        unknown = sorted(set(fields) - set(_FIELDS))
        if unknown or any(not isinstance(v, str) for v in fields.values()):
            raise ValidationError("form", f"unexpected parts: {unknown or 'file parts'}")
        return UploadRequest(
            filename=upload.filename or "",
            data=data,
            external_id=_text(fields.get("external_id")),
            title=_text(fields.get("title")),
            metadata=_metadata(_text(fields.get("metadata"))),
        )
    finally:
        await form.close()


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _metadata(raw: str | None) -> dict[str, Any] | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValidationError("metadata", "must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValidationError("metadata", "must be a JSON object")
    return value


@router.post(
    "",
    status_code=202,
    response_model=UploadOut,
    responses={
        200: {"model": UploadOut, "description": "Unchanged: no new version"},
        **error_responses(401, 404, 413, 415, 422, 429),
    },
    openapi_extra=_UPLOAD_BODY,
)
async def upload_document(
    collection_id: UUID, request: Request, response: Response, principal: Tenant, deps: Deps
) -> UploadOut:
    """Upload a file (multipart `file`, optional `external_id`, `title`, `metadata`).

    202 with document id, version and job id; 200 `unchanged: true` when the file, title
    and chunking settings equal the current version's (metadata is still updated).
    """
    upload = await _read_upload(request, deps.documents.max_file_bytes)
    result = await deps.documents.upload(principal.tenant, CollectionId(collection_id), upload)
    if result.unchanged:
        response.status_code = 200
    return UploadOut.of(result)


@router.get("", response_model=Page[DocumentOut], responses=error_responses(401, 404, 422))
async def list_documents(
    collection_id: UUID,
    principal: Tenant,
    deps: Deps,
    limit: Limit = DEFAULT_LIMIT,
    cursor: Cursor = None,
) -> Page[DocumentOut]:
    """Documents of a collection ordered by id."""
    after = decode_cursor(cursor)
    records = await deps.documents.list_page(
        principal.tenant,
        CollectionId(collection_id),
        None if after is None else DocumentId(after),
        limit + 1,
    )
    return Page[DocumentOut](
        data=[DocumentOut.of(r) for r in records[:limit]],
        next_cursor=next_cursor([r.id for r in records], limit),
    )


@router.get("/{document_id}", response_model=DocumentDetailOut, responses=error_responses(401, 404))
async def get_document(
    collection_id: UUID, document_id: UUID, principal: Tenant, deps: Deps
) -> DocumentDetailOut:
    """Status (with error), current version, chunk count and versions."""
    info = await deps.documents.get(
        principal.tenant, CollectionId(collection_id), DocumentId(document_id)
    )
    return DocumentDetailOut.of_info(info)


@router.patch(
    "/{document_id}", response_model=DocumentDetailOut, responses=error_responses(401, 404, 422)
)
async def update_document(
    collection_id: UUID, document_id: UUID, body: DocumentPatch, principal: Tenant, deps: Deps
) -> DocumentDetailOut:
    """Change metadata and/or the display title without re-upload."""
    info = await deps.documents.update(
        principal.tenant,
        CollectionId(collection_id),
        DocumentId(document_id),
        metadata=body.metadata,
        title=body.title,
    )
    return DocumentDetailOut.of_info(info)


@router.delete("/{document_id}", status_code=204, responses=error_responses(401, 404))
async def delete_document(
    collection_id: UUID, document_id: UUID, principal: Tenant, deps: Deps
) -> Response:
    """Delete every version, membership, chunk, vector and blob in one transaction."""
    await deps.documents.delete(
        principal.tenant, CollectionId(collection_id), DocumentId(document_id)
    )
    return Response(status_code=204)


@router.get(
    "/{document_id}/text",
    response_model=None,
    responses={
        200: {
            "content": {
                "text/plain": {"schema": {"type": "string"}},
                "application/json": {"schema": ElementsOut.model_json_schema()},
            },
            "description": "Text (default) or `?format=elements`",
        },
        **error_responses(401, 404, 409),
    },
)
async def document_text(
    collection_id: UUID,
    document_id: UUID,
    principal: Tenant,
    deps: Deps,
    format: Annotated[Literal["text", "elements"], Query()] = "text",  # noqa: A002 - wire name
) -> Response:
    """Current version text (elements joined by blank lines) or its element list."""
    cid, did = CollectionId(collection_id), DocumentId(document_id)
    if format == "elements":
        elements = await deps.documents.elements(principal.tenant, cid, did)
        body = ElementsOut(data=[ElementOut.of(e) for e in elements])
        return Response(body.model_dump_json(), media_type="application/json")
    text = await deps.documents.text(principal.tenant, cid, did)
    return Response(text, media_type="text/plain; charset=utf-8")


_ORIGINAL = {
    200: {
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Original bytes with the sniffed media type",
    },
    **error_responses(401, 404),
}


@router.get("/{document_id}/original", response_model=None, responses=_ORIGINAL)
async def current_original(
    collection_id: UUID, document_id: UUID, principal: Tenant, deps: Deps
) -> Response:
    """Original of the current version (404 `original_not_retained` when removed)."""
    media_type, data = await deps.documents.original(
        principal.tenant, CollectionId(collection_id), DocumentId(document_id)
    )
    return Response(data, media_type=media_type)


@router.get("/{document_id}/versions/{version}/original", response_model=None, responses=_ORIGINAL)
async def version_original(
    collection_id: UUID,
    document_id: UUID,
    version: Annotated[int, Path(ge=1)],
    principal: Tenant,
    deps: Deps,
) -> Response:
    """Original of any version, superseded ones included."""
    media_type, data = await deps.documents.original(
        principal.tenant, CollectionId(collection_id), DocumentId(document_id), version
    )
    return Response(data, media_type=media_type)
