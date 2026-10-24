# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from footnote.domain.documents import (
    ChunkBatch,
    ChunkDraft,
    ChunkId,
    DocumentId,
    Element,
    ElementKind,
    Membership,
    VersionId,
)
from footnote.domain.settings import Language
from tests.unit.api.conftest import SPACE_A, ApiRig, error_of
from tests.unit.app.fakes import DIMS

CONTENT = b"one\n\ntwo"


async def upload(
    rig: ApiRig, data: bytes = CONTENT, name: str = "a.txt", external_id: str | None = None
) -> tuple[DocumentId, VersionId]:
    r = await rig.client.post(
        rig.col_url(suffix="/documents"),
        headers=rig.auth(),
        files={"file": (name, data)},
        data={"external_id": external_id} if external_id else {},
    )
    assert r.status_code == 202, r.text
    doc_id = DocumentId(UUID(r.json()["document_id"]))
    version_id = next(
        v.id
        for v in rig.db.versions.values()
        if v.document_id == doc_id and v.version == r.json()["version"]
    )
    return doc_id, version_id


async def index(rig: ApiRig, doc_id: DocumentId, version_id: VersionId) -> None:
    elements = [Element(ElementKind.PARAGRAPH, "one", 0), Element(ElementKind.PARAGRAPH, "two", 1)]
    await rig.documents.store_elements(rig.tenant_a, version_id, elements)
    await rig.index.swap(rig.tenant_a, doc_id, version_id, [])


def doc_url(rig: ApiRig, doc_id: DocumentId, suffix: str) -> str:
    return rig.col_url(suffix=f"/documents/{doc_id}{suffix}")


async def test_text_before_indexing_is_not_ready(rig: ApiRig) -> None:
    doc_id, _ = await upload(rig)
    for suffix in ("/text", "/text?format=elements"):
        r = await rig.client.get(doc_url(rig, doc_id, suffix), headers=rig.auth())
        assert r.status_code == 409
        assert error_of(r)["code"] == "document_not_ready"


async def test_text_returns_elements_joined_by_blank_lines(rig: ApiRig) -> None:
    doc_id, version_id = await upload(rig)
    await index(rig, doc_id, version_id)
    r = await rig.client.get(doc_url(rig, doc_id, "/text"), headers=rig.auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "one\n\ntwo"


async def test_text_elements_format_returns_json(rig: ApiRig) -> None:
    doc_id, version_id = await upload(rig)
    await index(rig, doc_id, version_id)
    r = await rig.client.get(
        doc_url(rig, doc_id, "/text"), headers=rig.auth(), params={"format": "elements"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    data = r.json()["data"]
    assert [(e["kind"], e["text"], e["order"]) for e in data] == [
        ("paragraph", "one", 0),
        ("paragraph", "two", 1),
    ]
    assert data[0]["heading_path"] == []
    assert data[0]["page"] is None


async def test_text_rejects_an_unknown_format(rig: ApiRig) -> None:
    doc_id, version_id = await upload(rig)
    await index(rig, doc_id, version_id)
    r = await rig.client.get(
        doc_url(rig, doc_id, "/text"), headers=rig.auth(), params={"format": "bogus"}
    )
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_text_of_unknown_document_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.get(doc_url(rig, DocumentId(uuid4()), "/text"), headers=rig.auth())
    assert r.status_code == 404


@pytest.mark.parametrize("suffix", ["/original", "/versions/1/original"])
async def test_original_returns_the_uploaded_bytes(rig: ApiRig, suffix: str) -> None:
    doc_id, version_id = await upload(rig, b"# Title\n\nbody", "a.md")
    await index(rig, doc_id, version_id)
    r = await rig.client.get(doc_url(rig, doc_id, suffix), headers=rig.auth())
    assert r.status_code == 200
    assert r.content == b"# Title\n\nbody"
    assert r.headers["content-type"].startswith("text/markdown")


async def test_current_original_before_indexing_is_not_found(rig: ApiRig) -> None:
    doc_id, _ = await upload(rig)
    r = await rig.client.get(doc_url(rig, doc_id, "/original"), headers=rig.auth())
    assert r.status_code == 404


async def test_original_of_a_superseded_version_is_served(rig: ApiRig) -> None:
    doc_id, v1 = await upload(rig, external_id="x")
    await index(rig, doc_id, v1)
    same_doc, v2 = await upload(rig, b"second", external_id="x")
    assert same_doc == doc_id
    await index(rig, doc_id, v2)
    old = await rig.client.get(doc_url(rig, doc_id, "/versions/1/original"), headers=rig.auth())
    new = await rig.client.get(doc_url(rig, doc_id, "/original"), headers=rig.auth())
    assert (old.status_code, old.content) == (200, CONTENT)
    assert (new.status_code, new.content) == (200, b"second")


@pytest.mark.parametrize(
    ("version", "status", "code"),
    [
        pytest.param(9, 404, "not_found", id="unknown-version"),
        pytest.param(0, 422, "validation_failed", id="version-zero"),
        pytest.param(-1, 422, "validation_failed", id="negative-version"),
    ],
)
async def test_original_rejects_bad_versions(
    rig: ApiRig, version: int, status: int, code: str
) -> None:
    doc_id, version_id = await upload(rig)
    await index(rig, doc_id, version_id)
    r = await rig.client.get(
        doc_url(rig, doc_id, f"/versions/{version}/original"), headers=rig.auth()
    )
    assert r.status_code == status
    assert error_of(r)["code"] == code


@pytest.mark.parametrize("suffix", ["/original", "/versions/1/original"])
async def test_original_not_retained_after_blob_removal(rig: ApiRig, suffix: str) -> None:
    doc_id, version_id = await upload(rig)
    await index(rig, doc_id, version_id)
    rig.db.blobs.pop((rig.tenant_a, version_id))
    r = await rig.client.get(doc_url(rig, doc_id, suffix), headers=rig.auth())
    assert r.status_code == 404
    assert error_of(r)["code"] == "original_not_retained"


async def make_chunk(rig: ApiRig, doc_id: DocumentId, version_id: VersionId) -> ChunkId:
    draft = ChunkDraft(0, "alpha beta gamma", ("Intro",), "alpha beta gamma", "h1", 3, 1, 2)
    batch = ChunkBatch(rig.collection_a.id, doc_id, "A", Language.SIMPLE, [draft])
    chunk_id = (await rig.index.insert_chunks(rig.tenant_a, batch))["h1"]
    await rig.index.insert_vectors(
        rig.tenant_a, rig.collection_a.id, SPACE_A, [(chunk_id, [1.0] * DIMS)]
    )
    await rig.index.swap(
        rig.tenant_a, doc_id, version_id, [Membership(version_id, 0, chunk_id, 1, 2)]
    )
    return chunk_id


async def test_chunk_returns_text_and_memberships(rig: ApiRig) -> None:
    doc_id, version_id = await upload(rig)
    chunk_id = await make_chunk(rig, doc_id, version_id)
    r = await rig.client.get(rig.col_url(suffix=f"/chunks/{chunk_id}"), headers=rig.auth())
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == str(chunk_id)
    assert body["collection_id"] == str(rig.collection_a.id)
    assert body["document_id"] == str(doc_id)
    assert body["text"] == "alpha beta gamma"
    assert body["token_count"] == 3
    assert body["memberships"] == [{"version": 1, "ordinal": 0, "page_from": 1, "page_to": 2}]
    doc = await rig.client.get(doc_url(rig, doc_id, ""), headers=rig.auth())
    assert doc.json()["chunk_count"] == 1


async def test_chunk_lookup_of_unknown_id_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.get(rig.col_url(suffix=f"/chunks/{uuid4()}"), headers=rig.auth())
    assert r.status_code == 404
    assert error_of(r)["code"] == "chunk_not_found"


async def test_chunk_lookup_rejects_a_malformed_id(rig: ApiRig) -> None:
    r = await rig.client.get(rig.col_url(suffix="/chunks/not-a-uuid"), headers=rig.auth())
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"
