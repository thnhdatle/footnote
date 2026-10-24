# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from footnote.domain.deliveries import DeliveryRecord
from footnote.domain.documents import CollectionId, JobId
from footnote.domain.tenancy import Quotas
from tests.unit.api.conftest import ApiRig, build_rig, error_of

WEBHOOK_URL = "https://hooks.example.com/in"


@asynccontextmanager
async def custom_rig(**knobs: Any) -> AsyncIterator[ApiRig]:
    rig = build_rig(**knobs)
    async with rig.client, rig.admin_client:
        yield rig


async def create(rig: ApiRig, **body: Any) -> Any:
    body.setdefault("name", "fresh")
    return await rig.client.post("/v1/collections", headers=rig.auth(), json=body)


def job_kind(rig: ApiRig, job_id: str) -> str:
    return rig.db.jobs[(rig.tenant_a, JobId(UUID(job_id)))].kind


async def test_create_applies_defaults(rig: ApiRig) -> None:
    r = await create(rig)
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "fresh"
    assert body["status"] == "active"
    assert body["active_space"] == "space-a"
    assert body["embedding_space"] == "space-a"
    assert body["reembed"] is None
    assert body["chunking"]["max_tokens"] == 400
    assert body["retrieval"]["k"] == 8
    assert body["retrieval"]["rrf_weights"] == {"lexical": 0.2, "vector": 1.0}
    assert body["answer"]["prompt_template"] == "answer_v1"
    assert body["webhook"] == {
        "url": None,
        "events": body["webhook"]["events"],
        "has_secret": False,
    }
    assert body["job_id"] is None
    assert body["job_ids"] == []
    assert body["webhook_secret"] is None


async def test_create_applies_settings_patch(rig: ApiRig) -> None:
    r = await create(
        rig,
        chunking={"max_tokens": 200},
        retrieval={"k": 3, "rrf_weights": {"lexical": 2}, "rerank": False},
        answer={"unsupported_as_abstain": True},
        language="english",
    )
    assert r.status_code == 201
    body = r.json()
    assert body["chunking"]["max_tokens"] == 200
    assert body["chunking"]["overlap_tokens"] == 40
    assert body["retrieval"]["k"] == 3
    assert body["retrieval"]["rerank"] is False
    assert body["retrieval"]["rrf_weights"] == {"lexical": 2.0, "vector": 1.0}
    assert body["answer"]["unsupported_as_abstain"] is True
    assert body["language"] == "english"


async def test_create_with_webhook_returns_secret_once(rig: ApiRig) -> None:
    r = await create(rig, webhook={"url": WEBHOOK_URL})
    assert r.status_code == 201
    body = r.json()
    assert body["webhook_secret"].startswith("whsec_")
    assert body["webhook"]["url"] == WEBHOOK_URL
    assert body["webhook"]["has_secret"] is True
    again = await rig.client.get(f"/v1/collections/{body['id']}", headers=rig.auth())
    assert again.status_code == 200
    assert "webhook_secret" not in again.json()
    assert again.json()["webhook"]["has_secret"] is True


@pytest.mark.parametrize(
    ("url", "table"),
    [
        pytest.param("https://nowhere.example.com/x", {}, id="unresolvable"),
        pytest.param(
            "https://internal.example.com/x", {"internal.example.com": ["10.0.0.5"]}, id="private"
        ),
        pytest.param(
            "https://loop.example.com/x", {"loop.example.com": ["127.0.0.1"]}, id="loopback"
        ),
        pytest.param("http://hooks.example.com/x", {}, id="http-scheme"),
    ],
)
async def test_create_rejects_unsafe_webhook_url(
    rig: ApiRig, url: str, table: dict[str, list[str]]
) -> None:
    rig.resolver.table.update(table)
    r = await create(rig, webhook={"url": url})
    assert r.status_code == 422
    assert error_of(r)["code"] == "webhook_url_rejected"


async def test_create_duplicate_name_conflicts(rig: ApiRig) -> None:
    r = await create(rig, name="docs")
    assert r.status_code == 409
    assert error_of(r)["code"] == "collection_exists"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"embedding_space": "no-such-space"}, id="unknown-space"),
        pytest.param({"surprise": 1}, id="extra-field"),
        pytest.param({"retrieval": {"surprise": 1}}, id="extra-nested-field"),
        pytest.param({"name": ""}, id="empty-name"),
        pytest.param({"name": "x" * 257}, id="long-name"),
        pytest.param({"language": "klingon"}, id="unknown-language"),
    ],
)
async def test_create_rejects_invalid_body(rig: ApiRig, body: dict[str, Any]) -> None:
    r = await create(rig, **body)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_create_rejects_beyond_max_collections() -> None:
    async with custom_rig(quotas=Quotas(max_collections=1)) as rig:
        r = await create(rig)
        assert r.status_code == 429
        assert error_of(r)["code"] == "quota_exceeded"


async def test_list_paginates_without_repeats(rig: ApiRig) -> None:
    for n in range(4):
        assert (await create(rig, name=f"c{n}")).status_code == 201
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, Any] = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        r = await rig.client.get("/v1/collections", headers=rig.auth(), params=params)
        assert r.status_code == 200
        body = r.json()
        assert len(body["data"]) <= 2
        seen += [c["id"] for c in body["data"]]
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == 5
    assert len(set(seen)) == 5
    assert pages == 3
    assert seen == sorted(seen)


async def test_list_hides_other_tenants_collections(rig: ApiRig) -> None:
    r = await rig.client.get("/v1/collections", headers=rig.auth())
    assert [c["id"] for c in r.json()["data"]] == [str(rig.collection_a.id)]
    assert r.json()["next_cursor"] is None


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"cursor": "not-a-cursor"}, id="malformed-cursor"),
        pytest.param({"cursor": "Y3gAAAAAAA"}, id="wrong-prefix"),
        pytest.param({"limit": 0}, id="limit-zero"),
        pytest.param({"limit": 501}, id="limit-too-large"),
    ],
)
async def test_list_rejects_bad_paging(rig: ApiRig, params: dict[str, Any]) -> None:
    r = await rig.client.get("/v1/collections", headers=rig.auth(), params=params)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_get_returns_collection_with_jobs(rig: ApiRig) -> None:
    r = await rig.client.get(rig.col_url(), headers=rig.auth())
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == str(rig.collection_a.id)
    assert body["jobs"] == []
    assert body["chunk_count"] == 0
    patched = await rig.client.patch(
        rig.col_url(), headers=rig.auth(), json={"chunking": {"max_tokens": 300}}
    )
    job_id = patched.json()["job_id"]
    again = await rig.client.get(rig.col_url(), headers=rig.auth())
    jobs = again.json()["jobs"]
    assert [(j["id"], j["kind"], j["status"]) for j in jobs] == [(job_id, "reindex", "queued")]


@pytest.mark.parametrize("method", ["get", "patch", "delete"])
async def test_unknown_collection_is_not_found(rig: ApiRig, method: str) -> None:
    kwargs: dict[str, Any] = {"json": {"retrieval": {"k": 2}}} if method == "patch" else {}
    r = await rig.client.request(method, f"/v1/collections/{uuid4()}", headers=rig.auth(), **kwargs)
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_patch_retrieval_only_starts_no_job(rig: ApiRig) -> None:
    r = await rig.client.patch(rig.col_url(), headers=rig.auth(), json={"retrieval": {"k": 5}})
    assert r.status_code == 200
    assert r.json()["retrieval"]["k"] == 5
    assert r.json()["job_id"] is None
    assert r.json()["job_ids"] == []
    assert not rig.db.jobs_of(rig.tenant_a)


async def test_patch_chunking_starts_reindex(rig: ApiRig) -> None:
    r = await rig.client.patch(
        rig.col_url(), headers=rig.auth(), json={"chunking": {"max_tokens": 500}}
    )
    assert r.status_code == 202
    body = r.json()
    assert body["chunking"]["max_tokens"] == 500
    assert job_kind(rig, body["job_id"]) == "reindex"
    assert body["job_ids"] == [body["job_id"]]


async def test_patch_language_starts_fts_reconfig(rig: ApiRig) -> None:
    r = await rig.client.patch(rig.col_url(), headers=rig.auth(), json={"language": "english"})
    assert r.status_code == 202
    job = rig.db.jobs[(rig.tenant_a, JobId(UUID(r.json()["job_id"])))]
    assert job.kind == "fts_reconfig"
    assert job.args["language"] == "english"


async def test_patch_chunking_and_language_start_two_jobs(rig: ApiRig) -> None:
    r = await rig.client.patch(
        rig.col_url(),
        headers=rig.auth(),
        json={"chunking": {"max_tokens": 500}, "language": "english"},
    )
    assert r.status_code == 202
    body = r.json()
    assert len(body["job_ids"]) == 2
    assert body["job_id"] == body["job_ids"][0]
    assert sorted(job_kind(rig, j) for j in body["job_ids"]) == ["fts_reconfig", "reindex"]


async def test_patch_embedding_space_starts_reembed_and_can_be_cancelled(rig: ApiRig) -> None:
    r = await rig.client.patch(
        rig.col_url(), headers=rig.auth(), json={"embedding_space": "space-b"}
    )
    assert r.status_code == 202
    body = r.json()
    assert job_kind(rig, body["job_id"]) == "reembed"
    assert body["reembed"]["target_space"] == "space-b"
    assert body["embedding_space"] == "space-b"
    assert body["active_space"] == "space-a"
    back = await rig.client.patch(
        rig.col_url(), headers=rig.auth(), json={"embedding_space": "space-a"}
    )
    assert back.status_code == 200
    assert back.json()["reembed"] is None
    assert back.json()["embedding_space"] == "space-a"


async def test_patch_unknown_space_is_rejected(rig: ApiRig) -> None:
    r = await rig.client.patch(rig.col_url(), headers=rig.auth(), json={"embedding_space": "nope"})
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_patch_rotate_secret_issues_a_new_secret(rig: ApiRig) -> None:
    first = (await create(rig, webhook={"url": WEBHOOK_URL})).json()
    url = f"/v1/collections/{first['id']}"
    r = await rig.client.patch(
        url, headers=rig.auth(), json={"webhook": {"url": WEBHOOK_URL, "rotate_secret": True}}
    )
    assert r.status_code == 200
    assert r.json()["webhook_secret"].startswith("whsec_")
    assert r.json()["webhook_secret"] != first["webhook_secret"]


async def test_patch_rotate_secret_without_a_url_is_rejected(rig: ApiRig) -> None:
    r = await rig.client.patch(
        rig.col_url(), headers=rig.auth(), json={"webhook": {"rotate_secret": True}}
    )
    assert r.status_code == 422
    err = error_of(r)
    assert err["code"] == "validation_failed"
    assert err["details"][0]["field"] == "webhook.rotate_secret"


async def test_patch_null_webhook_url_removes_the_webhook(rig: ApiRig) -> None:
    first = (await create(rig, webhook={"url": WEBHOOK_URL})).json()
    r = await rig.client.patch(
        f"/v1/collections/{first['id']}", headers=rig.auth(), json={"webhook": {"url": None}}
    )
    assert r.status_code == 200
    assert r.json()["webhook"]["url"] is None
    assert r.json()["webhook"]["has_secret"] is False


async def test_patch_unknown_field_is_rejected(rig: ApiRig) -> None:
    r = await rig.client.patch(rig.col_url(), headers=rig.auth(), json={"name": "renamed"})
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_delete_accepts_a_job_and_hides_the_collection(rig: ApiRig) -> None:
    r = await rig.client.delete(rig.col_url(), headers=rig.auth())
    assert r.status_code == 202
    assert job_kind(rig, r.json()["job_id"]) == "collection_delete"
    gone = await rig.client.get(rig.col_url(), headers=rig.auth())
    assert gone.status_code == 404
    listed = await rig.client.get("/v1/collections", headers=rig.auth())
    assert listed.json()["data"] == []


async def test_webhook_deliveries_lists_records(rig: ApiRig) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    records = [
        DeliveryRecord(
            uuid4(),
            CollectionId(rig.collection_a.id),
            "document.ready",
            WEBHOOK_URL,
            "delivered" if n == 0 else "failed",
            n + 1,
            200 if n == 0 else 500,
            None if n == 0 else "boom",
            now,
            now,
            now if n == 0 else None,
        )
        for n in range(3)
    ]
    rig.deliveries.records[(rig.tenant_a, rig.collection_a.id)] = records
    r = await rig.client.get(rig.col_url(suffix="/webhook-deliveries"), headers=rig.auth())
    assert r.status_code == 200
    body = r.json()
    assert body["next_cursor"] is None
    assert [d["id"] for d in body["data"]] == [str(d.id) for d in records]
    assert body["data"][0]["status"] == "delivered"
    assert body["data"][1]["error"] == "boom"
    limited = await rig.client.get(
        rig.col_url(suffix="/webhook-deliveries"), headers=rig.auth(), params={"limit": 2}
    )
    assert len(limited.json()["data"]) == 2


async def test_webhook_deliveries_of_unknown_collection_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.get(f"/v1/collections/{uuid4()}/webhook-deliveries", headers=rig.auth())
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"
