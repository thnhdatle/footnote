# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from footnote.domain.documents import JobId
from tests.unit.api.conftest import ADMIN_KEY, ApiRig, build_rig, error_of

TENANTS = "/v1/admin/tenants"
SPACES = "/v1/admin/embedding-spaces"


def space_body(name: str = "space-c", **extra: Any) -> dict[str, Any]:
    return {"name": name, "provider": "onnx_local", "model": "m", "dimensions": 16, **extra}


async def create_tenant(rig: ApiRig, **body: Any) -> dict[str, Any]:
    r = await rig.client.post(TENANTS, headers=rig.admin, json={"name": "acme", **body})
    assert r.status_code == 201, r.text
    created: dict[str, Any] = r.json()
    return created


async def test_create_tenant_applies_default_quotas(rig: ApiRig) -> None:
    body = await create_tenant(rig)
    assert body["name"] == "acme"
    assert body["status"] == "active"
    assert body["quotas"] == {
        "max_collections": 10,
        "max_documents": 0,
        "max_chunks": 0,
        "max_upload_bytes_per_day": 0,
    }
    assert UUID(body["id"])


async def test_create_tenant_merges_partial_quotas_onto_defaults(rig: ApiRig) -> None:
    body = await create_tenant(rig, quotas={"max_documents": 5})
    assert body["quotas"]["max_collections"] == 10
    assert body["quotas"]["max_documents"] == 5


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"name": ""}, id="empty-name"),
        pytest.param({"name": "x", "quotas": {"max_collections": -1}}, id="negative-quota"),
        pytest.param({"name": "x", "quotas": {"surprise": 1}}, id="unknown-quota"),
        pytest.param({"name": "x", "surprise": 1}, id="unknown-field"),
        pytest.param({}, id="no-name"),
    ],
)
async def test_create_tenant_rejects_invalid_bodies(rig: ApiRig, body: dict[str, Any]) -> None:
    r = await rig.client.post(TENANTS, headers=rig.admin, json=body)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_list_tenants_paginates_without_repeats(rig: ApiRig) -> None:
    for n in range(3):
        await create_tenant(rig, name=f"t{n}")
    seen: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        r = await rig.client.get(TENANTS, headers=rig.admin, params=params)
        assert r.status_code == 200
        seen += [t["id"] for t in r.json()["data"]]
        cursor = r.json()["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 5
    assert str(rig.tenant_a) in seen


async def _walk(rig: ApiRig, url: str, key: str, limit: int) -> list[Any]:
    seen: list[Any] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": limit, **({"cursor": cursor} if cursor else {})}
        r = await rig.client.get(url, headers=rig.admin, params=params)
        assert r.status_code == 200
        seen += [row[key] for row in r.json()["data"]]
        cursor = r.json()["next_cursor"]
        if cursor is None:
            return seen


async def test_list_keys_paginates_without_repeats(rig: ApiRig) -> None:
    for n in range(3):
        await issue_key(rig, name=f"k{n}")
    seen = await _walk(rig, f"{TENANTS}/{rig.tenant_a}/keys", "id", 2)
    assert len(seen) == len(set(seen)) == 4
    assert seen == sorted(seen)


async def test_list_spaces_paginates_without_repeats(rig: ApiRig) -> None:
    await rig.client.post(SPACES, headers=rig.admin, json=space_body())
    seen = await _walk(rig, SPACES, "name", 2)
    assert sorted(seen) == ["space-a", "space-b", "space-c"]


@pytest.mark.parametrize("path", [f"{TENANTS}/{{tenant}}/keys", SPACES], ids=["keys", "spaces"])
@pytest.mark.parametrize(
    "params",
    [{"cursor": "garbage"}, {"limit": 0}, {"limit": 501}],
    ids=["malformed-cursor", "limit-zero", "limit-too-large"],
)
async def test_admin_lists_reject_bad_paging(
    rig: ApiRig, path: str, params: dict[str, Any]
) -> None:
    r = await rig.client.get(path.format(tenant=rig.tenant_a), headers=rig.admin, params=params)
    assert r.status_code == 422


@pytest.mark.parametrize(
    "params",
    [{"cursor": "garbage"}, {"limit": 0}, {"limit": 501}],
    ids=["malformed-cursor", "limit-zero", "limit-too-large"],
)
async def test_list_tenants_rejects_bad_paging(rig: ApiRig, params: dict[str, Any]) -> None:
    r = await rig.client.get(TENANTS, headers=rig.admin, params=params)
    assert r.status_code == 422


async def test_get_tenant(rig: ApiRig) -> None:
    r = await rig.client.get(f"{TENANTS}/{rig.tenant_a}", headers=rig.admin)
    assert r.status_code == 200
    assert r.json()["id"] == str(rig.tenant_a)


@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("GET", ""),
        ("PATCH", ""),
        ("DELETE", ""),
        ("GET", "/usage"),
        ("GET", "/keys"),
        ("POST", "/keys"),
    ],
)
async def test_unknown_tenant_is_not_found(rig: ApiRig, method: str, suffix: str) -> None:
    body = {"name": "x"} if method in {"PATCH", "POST"} else None
    r = await rig.client.request(
        method, f"{TENANTS}/{uuid4()}{suffix}", headers=rig.admin, json=body
    )
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_patch_tenant_changes_name_and_partial_quotas(rig: ApiRig) -> None:
    tenant = await create_tenant(rig, quotas={"max_documents": 5, "max_chunks": 7})
    r = await rig.client.patch(
        f"{TENANTS}/{tenant['id']}",
        headers=rig.admin,
        json={"name": "renamed", "quotas": {"max_documents": 6}},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "renamed"
    assert r.json()["quotas"] == {
        "max_collections": 10,
        "max_documents": 6,
        "max_chunks": 7,
        "max_upload_bytes_per_day": 0,
    }
    name_only = await rig.client.patch(
        f"{TENANTS}/{tenant['id']}", headers=rig.admin, json={"name": "again"}
    )
    assert name_only.json()["quotas"]["max_documents"] == 6


@pytest.mark.parametrize(
    "body",
    [{"quotas": {"max_collections": -1}}, {"name": ""}, {"surprise": 1}],
    ids=["negative-quota", "empty-name", "unknown-field"],
)
async def test_patch_tenant_rejects_invalid_bodies(rig: ApiRig, body: dict[str, Any]) -> None:
    r = await rig.client.patch(f"{TENANTS}/{rig.tenant_a}", headers=rig.admin, json=body)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_delete_tenant_enqueues_a_job_and_revokes_its_keys(rig: ApiRig) -> None:
    before = await rig.client.get("/v1/collections", headers=rig.auth())
    assert before.status_code == 200
    r = await rig.client.delete(f"{TENANTS}/{rig.tenant_a}", headers=rig.admin)
    assert r.status_code == 202
    job = rig.db.jobs[(rig.tenant_a, JobId(UUID(r.json()["job_id"])))]
    assert job.kind == "tenant_delete"
    assert job.args == {"tenant_id": str(rig.tenant_a)}
    status = await rig.client.get(f"{TENANTS}/{rig.tenant_a}", headers=rig.admin)
    assert status.json()["status"] == "deleting"
    after = await rig.client.get("/v1/collections", headers=rig.auth())
    assert after.status_code == 401
    other = await rig.client.get("/v1/collections", headers=rig.auth(rig.key_b))
    assert other.status_code == 200


async def test_usage_counts_the_tenants_rows(rig: ApiRig) -> None:
    r = await rig.client.get(f"{TENANTS}/{rig.tenant_a}/usage", headers=rig.admin)
    assert r.status_code == 200
    assert r.json() == {
        "collections": 1,
        "documents": 0,
        "chunks": 0,
        "vectors": 0,
        "stored_bytes": 0,
    }


def ndjson(response: httpx.Response) -> list[dict[str, Any]]:
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


@pytest.mark.parametrize(
    ("params", "types"),
    [
        pytest.param({}, ["tenant", "collection"], id="plain"),
        pytest.param({"include_vectors": "true"}, ["tenant", "collection", "vector"], id="vectors"),
        pytest.param(
            {"include_originals": "true"}, ["tenant", "collection", "version"], id="originals"
        ),
        pytest.param(
            {"include_vectors": "true", "include_originals": "true"},
            ["tenant", "collection", "vector", "version"],
            id="both",
        ),
    ],
)
async def test_export_streams_ndjson_records(
    rig: ApiRig, params: dict[str, str], types: list[str]
) -> None:
    r = await rig.client.get(f"{TENANTS}/{rig.tenant_a}/export", headers=rig.admin, params=params)
    assert r.status_code == 200
    lines = ndjson(r)
    assert [line["type"] for line in lines] == types
    assert all(set(line) == {"type", "data"} for line in lines)
    assert lines[0]["data"]["id"] == str(rig.tenant_a)
    assert rig.exporter.closed == 1


async def test_export_of_unknown_tenant_is_a_json_error(rig: ApiRig) -> None:
    r = await rig.client.get(f"{TENANTS}/{uuid4()}/export", headers=rig.admin)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert error_of(r)["code"] == "not_found"


async def issue_key(rig: ApiRig, **body: Any) -> httpx.Response:
    return await rig.client.post(f"{TENANTS}/{rig.tenant_a}/keys", headers=rig.admin, json=body)


async def test_create_key_returns_the_secret_once(rig: ApiRig) -> None:
    r = await issue_key(rig, name="ci")
    assert r.status_code == 201
    created = r.json()
    assert created["key"].startswith("fn_")
    assert created["name"] == "ci"
    assert created["revoked"] is False
    assert created["tenant_id"] == str(rig.tenant_a)
    works = await rig.client.get("/v1/collections", headers=rig.auth(created["key"]))
    assert works.status_code == 200
    listed = await rig.client.get(f"{TENANTS}/{rig.tenant_a}/keys", headers=rig.admin)
    assert listed.status_code == 200
    assert listed.json()["next_cursor"] is None
    entry = next(k for k in listed.json()["data"] if k["id"] == created["id"])
    assert "key" not in entry
    assert entry["revoked"] is False
    assert len(listed.json()["data"]) == 2


async def test_create_key_accepts_a_future_expiry(rig: ApiRig) -> None:
    r = await issue_key(rig, expires_at="2999-01-01T00:00:00Z")
    assert r.status_code == 201
    assert r.json()["expires_at"].startswith("2999-01-01T00:00:00")


@pytest.mark.parametrize(
    "expires_at",
    [
        pytest.param("2020-01-01T00:00:00Z", id="past"),
        pytest.param("2999-01-01T00:00:00", id="naive"),
        pytest.param("not a date", id="malformed"),
    ],
)
async def test_create_key_rejects_bad_expiry(rig: ApiRig, expires_at: str) -> None:
    r = await issue_key(rig, expires_at=expires_at)
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_revoke_key_blocks_it(rig: ApiRig) -> None:
    created = (await issue_key(rig)).json()
    url = f"{TENANTS}/{rig.tenant_a}/keys/{created['id']}"
    r = await rig.client.delete(url, headers=rig.admin)
    assert r.status_code == 204
    blocked = await rig.client.get("/v1/collections", headers=rig.auth(created["key"]))
    assert blocked.status_code == 401
    listed = await rig.client.get(f"{TENANTS}/{rig.tenant_a}/keys", headers=rig.admin)
    assert next(k for k in listed.json()["data"] if k["id"] == created["id"])["revoked"] is True


@pytest.mark.parametrize("action", ["revoke", "rotate"])
async def test_unknown_key_is_not_found(rig: ApiRig, action: str) -> None:
    url = f"{TENANTS}/{rig.tenant_a}/keys/{uuid4()}"
    r = (
        await rig.client.delete(url, headers=rig.admin)
        if action == "revoke"
        else await rig.client.post(f"{url}/rotate", headers=rig.admin)
    )
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_rotate_key_issues_a_working_replacement(rig: ApiRig) -> None:
    created = (await issue_key(rig)).json()
    r = await rig.client.post(
        f"{TENANTS}/{rig.tenant_a}/keys/{created['id']}/rotate", headers=rig.admin
    )
    assert r.status_code == 201
    new = r.json()
    assert new["key"] != created["key"]
    assert new["id"] != created["id"]
    fresh = await rig.client.get("/v1/collections", headers=rig.auth(new["key"]))
    assert fresh.status_code == 200
    old = await rig.client.get("/v1/collections", headers=rig.auth(created["key"]))
    assert old.status_code == 200


@pytest.mark.parametrize(
    ("dimensions", "index_type"),
    [(16, "vector"), (2000, "vector"), (3000, "halfvec")],
)
async def test_register_space_picks_the_index_type(
    rig: ApiRig, dimensions: int, index_type: str
) -> None:
    r = await rig.client.post(SPACES, headers=rig.admin, json=space_body(dimensions=dimensions))
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "space-c"
    assert body["dimensions"] == dimensions
    assert body["index_type"] == index_type
    assert body["normalise"] is True
    assert body["send_dimensions"] is False


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(space_body(dimensions=4001), id="dimensions-too-large"),
        pytest.param(
            {"name": "space-c", "provider": "onnx_local", "model": "m"}, id="no-dimensions"
        ),
        pytest.param(space_body("Bad Name"), id="bad-name"),
        pytest.param(space_body(provider="other"), id="unknown-provider"),
        pytest.param(space_body(dimensions=0), id="zero-dimensions"),
        pytest.param(space_body(calibrated_threshold=1.5), id="bad-threshold"),
    ],
)
async def test_register_space_rejects_invalid_bodies(rig: ApiRig, body: dict[str, Any]) -> None:
    r = await rig.client.post(SPACES, headers=rig.admin, json=body)
    assert r.status_code == 422


async def test_register_space_rejects_unsupported_dimensions_with_its_code(rig: ApiRig) -> None:
    r = await rig.client.post(SPACES, headers=rig.admin, json=space_body(dimensions=4001))
    assert error_of(r)["code"] == "embedding_dimension_unsupported"


async def test_list_spaces_includes_registered_ones(rig: ApiRig) -> None:
    await rig.client.post(SPACES, headers=rig.admin, json=space_body())
    r = await rig.client.get(SPACES, headers=rig.admin)
    assert r.status_code == 200
    assert [s["name"] for s in r.json()["data"]] == ["space-a", "space-b", "space-c"]
    assert r.json()["next_cursor"] is None


async def test_delete_unused_space(rig: ApiRig) -> None:
    await rig.client.post(SPACES, headers=rig.admin, json=space_body())
    r = await rig.client.delete(f"{SPACES}/space-c", headers=rig.admin)
    assert r.status_code == 204
    names = [s["name"] for s in (await rig.client.get(SPACES, headers=rig.admin)).json()["data"]]
    assert "space-c" not in names


async def test_delete_space_in_use_conflicts(rig: ApiRig) -> None:
    r = await rig.client.delete(f"{SPACES}/space-a", headers=rig.admin)
    assert r.status_code == 409
    assert error_of(r)["code"] == "space_in_use"


async def test_delete_unknown_space_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.delete(f"{SPACES}/nope", headers=rig.admin)
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_ninth_space_hits_the_limit(rig: ApiRig) -> None:
    for n in range(6):
        r = await rig.client.post(SPACES, headers=rig.admin, json=space_body(f"extra-{n}"))
        assert r.status_code == 201
    r = await rig.client.post(SPACES, headers=rig.admin, json=space_body("one-too-many"))
    assert r.status_code == 409
    assert error_of(r)["code"] == "space_limit_reached"


ROUTES: list[Any] = [
    pytest.param("POST", TENANTS, {"name": "x"}, id="create-tenant"),
    pytest.param("GET", TENANTS, None, id="list-tenants"),
    pytest.param("GET", f"{TENANTS}/{{tenant}}", None, id="get-tenant"),
    pytest.param("PATCH", f"{TENANTS}/{{tenant}}", {"name": "y"}, id="patch-tenant"),
    pytest.param("DELETE", f"{TENANTS}/{{tenant}}", None, id="delete-tenant"),
    pytest.param("GET", f"{TENANTS}/{{tenant}}/usage", None, id="usage"),
    pytest.param("GET", f"{TENANTS}/{{tenant}}/export", None, id="export"),
    pytest.param("POST", f"{TENANTS}/{{tenant}}/keys", {"name": "k"}, id="create-key"),
    pytest.param("GET", f"{TENANTS}/{{tenant}}/keys", None, id="list-keys"),
    pytest.param("DELETE", f"{TENANTS}/{{tenant}}/keys/{{key}}", None, id="revoke-key"),
    pytest.param("POST", f"{TENANTS}/{{tenant}}/keys/{{key}}/rotate", None, id="rotate-key"),
    pytest.param("POST", SPACES, space_body(), id="register-space"),
    pytest.param("GET", SPACES, None, id="list-spaces"),
    pytest.param("DELETE", f"{SPACES}/space-b", None, id="delete-space"),
]


def headers_for(rig: ApiRig, kind: str) -> dict[str, str]:
    return {
        "none": {},
        "tenant-key": rig.auth(),
        "wrong-admin-key": {"Authorization": "Bearer not-the-admin-key"},
        "basic-scheme": {"Authorization": f"Basic {ADMIN_KEY}"},
    }[kind]


@pytest.mark.parametrize("kind", ["none", "tenant-key", "wrong-admin-key", "basic-scheme"])
@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
async def test_admin_routes_reject_other_credentials(
    rig: ApiRig, method: str, path: str, body: Any, kind: str
) -> None:
    key_id = next(iter(rig.keys.keys.values())).id
    url = path.format(tenant=rig.tenant_a, key=key_id)
    r = await rig.client.request(method, url, headers=headers_for(rig, kind), json=body)
    assert r.status_code == 401
    assert error_of(r)["code"] == "unauthenticated"
    assert r.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
async def test_admin_routes_are_disabled_without_an_admin_key(
    method: str, path: str, body: Any
) -> None:
    rig = build_rig(admin_key="")
    async with rig.client, rig.admin_client:
        key_id = next(iter(rig.keys.keys.values())).id
        url = path.format(tenant=rig.tenant_a, key=key_id)
        for token in (ADMIN_KEY, "x"):
            r = await rig.client.request(
                method, url, headers={"Authorization": f"Bearer {token}"}, json=body
            )
            assert r.status_code == 401
            assert error_of(r)["code"] == "unauthenticated"
            assert r.headers["www-authenticate"].startswith("Bearer")
