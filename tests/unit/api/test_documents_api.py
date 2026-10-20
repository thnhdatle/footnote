# SPDX-License-Identifier: Apache-2.0
"""Documents API: upload validation and limits, quotas, list, get, patch and delete."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from footnote.api.app import create_app
from footnote.domain.documents import DocumentId, JobId
from footnote.domain.tenancy import Quotas
from tests.unit.api.conftest import MAX_FILE_BYTES, ApiRig, build_rig, error_of

BODY = b"hello world\n\nsecond paragraph"


@asynccontextmanager
async def custom_rig(**knobs: Any) -> AsyncIterator[ApiRig]:
    rig = build_rig(**knobs)
    async with rig.client, rig.admin_client:
        yield rig


def docs_url(rig: ApiRig, suffix: str = "") -> str:
    return rig.col_url(suffix=f"/documents{suffix}")


async def upload(
    rig: ApiRig,
    data: bytes = BODY,
    *,
    name: str = "a.txt",
    form: dict[str, str] | None = None,
) -> httpx.Response:
    return await rig.client.post(
        docs_url(rig), headers=rig.auth(), files={"file": (name, data)}, data=form or {}
    )


def ingest_jobs(rig: ApiRig) -> int:
    return len(rig.db.jobs_of(rig.tenant_a, "ingest"))


async def test_upload_accepts_and_queues_ingest(rig: ApiRig) -> None:
    r = await upload(rig, form={"external_id": "x", "title": "T", "metadata": '{"a": 1}'})
    assert r.status_code == 202
    body = r.json()
    assert body["unchanged"] is False
    assert body["version"] == 1
    assert UUID(body["document_id"])
    job = rig.db.jobs[(rig.tenant_a, JobId(UUID(body["job_id"])))]
    assert job.kind == "ingest"
    doc = (
        await rig.client.get(docs_url(rig, f"/{body['document_id']}"), headers=rig.auth())
    ).json()
    assert (doc["external_id"], doc["title"], doc["metadata"]) == ("x", "T", {"a": 1})
    assert doc["status"] == "queued"


async def test_upload_without_title_uses_the_filename(rig: ApiRig) -> None:
    r = await upload(rig, name="notes.txt")
    doc = (
        await rig.client.get(docs_url(rig, f"/{r.json()['document_id']}"), headers=rig.auth())
    ).json()
    assert doc["title"] == "notes.txt"


async def test_upload_same_file_again_is_unchanged(rig: ApiRig) -> None:
    first = await upload(rig, form={"external_id": "x"})
    doc_id = DocumentId(UUID(first.json()["document_id"]))
    version_id = next(v.id for v in rig.db.versions.values() if v.document_id == doc_id)
    await rig.documents.store_elements(rig.tenant_a, version_id, [])
    await rig.index.swap(rig.tenant_a, doc_id, version_id, [])
    jobs_before = ingest_jobs(rig)
    again = await upload(rig, form={"external_id": "x", "metadata": '{"k": "v"}'})
    assert again.status_code == 200
    body = again.json()
    assert body["unchanged"] is True
    assert body["document_id"] == str(doc_id)
    assert body["job_id"] is None
    assert ingest_jobs(rig) == jobs_before


async def test_upload_changed_file_adds_a_version(rig: ApiRig) -> None:
    first = await upload(rig, form={"external_id": "x"})
    second = await upload(rig, b"different content", form={"external_id": "x"})
    assert second.status_code == 202
    assert second.json()["document_id"] == first.json()["document_id"]
    assert second.json()["version"] == 2


@pytest.mark.parametrize(
    ("form", "field"),
    [
        pytest.param({"metadata": "[1]"}, "metadata", id="metadata-array"),
        pytest.param({"metadata": "nope"}, "metadata", id="metadata-not-json"),
        pytest.param({"title": "t" * 513}, "title", id="title-too-long"),
        pytest.param({"external_id": "e" * 513}, "external_id", id="external-id-too-long"),
        pytest.param({"external_id": ""}, "external_id", id="external-id-empty"),
    ],
)
async def test_upload_rejects_invalid_fields(rig: ApiRig, form: dict[str, str], field: str) -> None:
    r = await upload(rig, form=form)
    assert r.status_code == 422
    err = error_of(r)
    assert err["code"] == "validation_failed"
    assert err["details"][0]["field"] == field
    assert ingest_jobs(rig) == 0


async def test_upload_rejects_oversize_metadata(rig: ApiRig) -> None:
    deps = replace(
        rig.deps, settings=replace(rig.deps.settings, multipart_overhead_bytes=64 * 1024)
    )
    big = '{"a": "' + "x" * 17000 + '"}'
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(deps)), base_url="http://test"
    ) as client:
        r = await client.post(
            docs_url(rig),
            headers=rig.auth(),
            files={"file": ("a.txt", BODY)},
            data={"metadata": big},
        )
    assert r.status_code == 422
    assert error_of(r)["details"][0]["field"] == "metadata"


async def test_upload_rejects_an_empty_file(rig: ApiRig) -> None:
    r = await upload(rig, b"")
    assert r.status_code == 422
    assert error_of(r)["details"][0]["field"] == "file"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"files": {"other": ("a.txt", b"hi")}}, id="no-file-part"),
        pytest.param({"data": {"title": "x"}}, id="urlencoded-form"),
        pytest.param({"json": {"file": "hi"}}, id="json-body"),
    ],
)
async def test_upload_requires_a_multipart_file(rig: ApiRig, kwargs: dict[str, Any]) -> None:
    r = await rig.client.post(docs_url(rig), headers=rig.auth(), **kwargs)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_upload_rejects_a_file_over_the_limit(rig: ApiRig) -> None:
    r = await upload(rig, b"a" * (MAX_FILE_BYTES + 1))
    assert r.status_code == 413
    assert error_of(r)["code"] == "file_too_large"


async def test_upload_accepts_a_file_at_the_limit(rig: ApiRig) -> None:
    r = await upload(rig, b"a" * MAX_FILE_BYTES)
    assert r.status_code == 202


async def test_upload_rejects_a_body_far_over_the_limit(rig: ApiRig) -> None:
    r = await upload(rig, b"a" * (64 * 1024))
    assert r.status_code == 413
    assert error_of(r)["code"] in {"file_too_large", "request_too_large"}
    assert ingest_jobs(rig) == 0


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        pytest.param(b"binary\x00data", 415, "unsupported_media_type", id="nul-byte"),
        pytest.param(b"%PDF-ENCRYPTED", 422, "pdf_encrypted", id="encrypted-pdf"),
    ],
)
async def test_upload_rejects_content_by_sniffing(
    rig: ApiRig, data: bytes, status: int, code: str
) -> None:
    r = await upload(rig, data, name="a.pdf")
    assert r.status_code == status
    assert error_of(r)["code"] == code


async def test_upload_to_unknown_collection_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.post(
        f"/v1/collections/{uuid4()}/documents",
        headers=rig.auth(),
        files={"file": ("a.txt", BODY)},
    )
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_upload_enforces_max_documents() -> None:
    async with custom_rig(quotas=Quotas(max_documents=1)) as rig:
        assert (await upload(rig, form={"external_id": "a"})).status_code == 202
        blocked = await upload(rig, form={"external_id": "b"})
        assert blocked.status_code == 429
        assert error_of(blocked)["code"] == "quota_exceeded"
        again = await upload(rig, b"new content", form={"external_id": "a"})
        assert again.status_code == 202


@pytest.mark.parametrize(
    ("size", "status"),
    [pytest.param(10, 202, id="at-quota"), pytest.param(11, 429, id="over-quota")],
)
async def test_upload_enforces_daily_byte_quota(size: int, status: int) -> None:
    async with custom_rig(quotas=Quotas(max_upload_bytes_per_day=10)) as rig:
        r = await upload(rig, b"a" * size)
        assert r.status_code == status
        if status == 429:
            assert error_of(r)["code"] == "quota_exceeded"


async def test_list_documents_paginates_without_repeats(rig: ApiRig) -> None:
    ids = [(await upload(rig, f"doc {n}".encode())).json()["document_id"] for n in range(5)]
    seen: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        r = await rig.client.get(docs_url(rig), headers=rig.auth(), params=params)
        assert r.status_code == 200
        seen += [d["id"] for d in r.json()["data"]]
        cursor = r.json()["next_cursor"]
        if cursor is None:
            break
    assert sorted(seen) == sorted(ids)
    assert len(set(seen)) == 5


@pytest.mark.parametrize(
    "params",
    [{"cursor": "garbage"}, {"limit": 0}, {"limit": 501}],
    ids=["malformed-cursor", "limit-zero", "limit-too-large"],
)
async def test_list_documents_rejects_bad_paging(rig: ApiRig, params: dict[str, Any]) -> None:
    r = await rig.client.get(docs_url(rig), headers=rig.auth(), params=params)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_list_documents_of_unknown_collection_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.get(f"/v1/collections/{uuid4()}/documents", headers=rig.auth())
    assert r.status_code == 404


async def test_get_document_shows_versions_and_chunk_count(rig: ApiRig) -> None:
    doc_id = (await upload(rig, form={"external_id": "x"})).json()["document_id"]
    r = await rig.client.get(docs_url(rig, f"/{doc_id}"), headers=rig.auth())
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == doc_id
    assert body["collection_id"] == str(rig.collection_a.id)
    assert body["chunk_count"] == 0
    assert body["current_version"] is None
    [version] = body["versions"]
    assert version["version"] == 1
    assert version["media_type"] == "text/plain"
    assert version["bytes"] == len(BODY)
    assert version["has_original"] is True
    assert len(version["sha256"]) == 64


async def test_get_unknown_document_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.get(docs_url(rig, f"/{uuid4()}"), headers=rig.auth())
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_patch_changes_metadata_and_title(rig: ApiRig) -> None:
    doc_id = (await upload(rig)).json()["document_id"]
    r = await rig.client.patch(
        docs_url(rig, f"/{doc_id}"),
        headers=rig.auth(),
        json={"metadata": {"lang": "en"}, "title": "Renamed"},
    )
    assert r.status_code == 200
    assert (r.json()["metadata"], r.json()["title"]) == ({"lang": "en"}, "Renamed")
    only_title = await rig.client.patch(
        docs_url(rig, f"/{doc_id}"), headers=rig.auth(), json={"title": "Again"}
    )
    assert only_title.json()["metadata"] == {"lang": "en"}
    assert only_title.json()["title"] == "Again"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"metadata": {"a": "x" * 17000}}, id="oversize-metadata"),
        pytest.param({"metadata": [1]}, id="metadata-not-object"),
        pytest.param({"title": "t" * 513}, id="title-too-long"),
        pytest.param({"surprise": 1}, id="unknown-field"),
    ],
)
async def test_patch_rejects_invalid_bodies(rig: ApiRig, body: dict[str, Any]) -> None:
    doc_id = (await upload(rig)).json()["document_id"]
    r = await rig.client.patch(docs_url(rig, f"/{doc_id}"), headers=rig.auth(), json=body)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_patch_unknown_document_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.patch(
        docs_url(rig, f"/{uuid4()}"), headers=rig.auth(), json={"title": "x"}
    )
    assert r.status_code == 404


async def test_delete_removes_the_document(rig: ApiRig) -> None:
    doc_id = (await upload(rig)).json()["document_id"]
    r = await rig.client.delete(docs_url(rig, f"/{doc_id}"), headers=rig.auth())
    assert r.status_code == 204
    assert r.content == b""
    assert (
        await rig.client.get(docs_url(rig, f"/{doc_id}"), headers=rig.auth())
    ).status_code == 404
    assert (
        await rig.client.delete(docs_url(rig, f"/{doc_id}"), headers=rig.auth())
    ).status_code == 404
