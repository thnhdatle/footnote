# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from uuid import UUID

import pytest

from footnote.domain.documents import ChunkBatch, ChunkDraft, Membership
from footnote.domain.settings import Language
from tests.unit.api.conftest import SPACE_A, ApiRig, error_of
from tests.unit.app.fakes import hash_vector


async def _seed(rig: ApiRig) -> dict[str, str]:
    up = await rig.client.post(
        rig.col_url(suffix="/documents"),
        headers=rig.auth(),
        files={"file": ("a.txt", b"alpha beta")},
        data={"external_id": "doc-1"},
    )
    assert up.status_code == 202, up.text
    body = up.json()
    doc_id = body["document_id"]
    version_id = next(v for (t, v) in rig.db.versions if t == rig.tenant_a)
    draft = ChunkDraft(0, "alpha beta", (), "alpha beta", "h" * 64, 2, 1, 1)
    ids = await rig.index.insert_chunks(
        rig.tenant_a,
        ChunkBatch(rig.collection_a.id, UUID(doc_id), "a.txt", Language.SIMPLE, [draft]),  # type: ignore[arg-type]
    )
    chunk_id = ids["h" * 64]
    await rig.index.insert_vectors(
        rig.tenant_a, rig.collection_a.id, SPACE_A, [(chunk_id, hash_vector("alpha"))]
    )
    await rig.index.swap(
        rig.tenant_a,
        UUID(doc_id),
        version_id,
        [Membership(version_id, 0, chunk_id, 1, 1)],  # type: ignore[arg-type]
    )
    return {"doc": doc_id, "job": body["job_id"], "chunk": str(chunk_id)}


def _cross_tenant_requests(rig: ApiRig, ids: dict[str, str]) -> list[tuple[str, str, dict]]:
    col = rig.col_url()
    doc = f"{col}/documents/{ids['doc']}"
    return [
        ("GET", col, {}),
        ("PATCH", col, {"json": {"retrieval": {"k": 5}}}),
        ("DELETE", col, {}),
        ("GET", f"{col}/webhook-deliveries", {}),
        ("GET", f"{col}/documents", {}),
        ("POST", f"{col}/documents", {"files": {"file": ("b.txt", b"intruder")}}),
        ("GET", doc, {}),
        ("PATCH", doc, {"json": {"title": "pwned"}}),
        ("DELETE", doc, {}),
        ("GET", f"{doc}/text", {}),
        ("GET", f"{doc}/text?format=elements", {}),
        ("GET", f"{doc}/original", {}),
        ("GET", f"{doc}/versions/1/original", {}),
        ("GET", f"{col}/chunks/{ids['chunk']}", {}),
        ("POST", f"{col}/search", {"json": {"query": "alpha"}}),
        ("POST", f"{col}/answers", {"json": {"question": "alpha beta"}}),
        ("POST", f"{col}/answers", {"json": {"question": "alpha beta", "stream": True}}),
        ("GET", f"/v1/jobs/{ids['job']}", {}),
    ]


async def test_other_tenants_ids_are_404_on_every_route(rig: ApiRig) -> None:
    ids = await _seed(rig)
    before_docs = dict(rig.db.docs)
    before_cols = {k: (c.status, c.settings) for k, c in rig.db.cols.items()}
    for method, url, kwargs in _cross_tenant_requests(rig, ids):
        response = await rig.client.request(method, url, headers=rig.auth(rig.key_b), **kwargs)
        assert response.status_code == 404, (method, url, response.text)
        assert error_of(response)["code"] in {"not_found", "chunk_not_found"}, (method, url)
    assert dict(rig.db.docs) == before_docs
    assert {k: (c.status, c.settings) for k, c in rig.db.cols.items()} == before_cols
    assert not rig.db.jobs_of(rig.tenant_b)


async def test_owner_still_sees_everything_after_the_cross_tenant_attempts(rig: ApiRig) -> None:
    ids = await _seed(rig)
    for method, url, kwargs in _cross_tenant_requests(rig, ids):
        await rig.client.request(method, url, headers=rig.auth(rig.key_b), **kwargs)
    doc = await rig.client.get(rig.col_url(suffix=f"/documents/{ids['doc']}"), headers=rig.auth())
    assert doc.status_code == 200
    assert doc.json()["title"] == "a.txt"
    chunk = await rig.client.get(rig.col_url(suffix=f"/chunks/{ids['chunk']}"), headers=rig.auth())
    assert chunk.status_code == 200
    job = await rig.client.get(f"/v1/jobs/{ids['job']}", headers=rig.auth())
    assert job.status_code == 200


@pytest.mark.parametrize("path", ["/v1/collections"])
async def test_lists_show_only_own_resources(rig: ApiRig, path: str) -> None:
    a = await rig.client.get(path, headers=rig.auth())
    b = await rig.client.get(path, headers=rig.auth(rig.key_b))
    assert [c["id"] for c in a.json()["data"]] == [str(rig.collection_a.id)]
    assert [c["id"] for c in b.json()["data"]] == [str(rig.collection_b.id)]


async def test_same_collection_name_in_two_tenants_is_allowed(rig: ApiRig) -> None:
    for key in (rig.key_a, rig.key_b):
        response = await rig.client.post(
            "/v1/collections", headers=rig.auth(key), json={"name": "shared-name"}
        )
        assert response.status_code == 201, response.text


async def test_tenant_b_documents_listing_of_own_collection_excludes_a(rig: ApiRig) -> None:
    await _seed(rig)
    response = await rig.client.get(
        rig.col_url(rig.collection_b, "/documents"), headers=rig.auth(rig.key_b)
    )
    assert response.status_code == 200
    assert response.json()["data"] == []
