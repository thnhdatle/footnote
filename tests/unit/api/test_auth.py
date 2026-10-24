# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from footnote.api.app import build_app
from footnote.api.auth import MAX_TOKEN_CHARS
from tests.unit.api.conftest import ADMIN_KEY, ApiRig, build_rig, error_of

ANY_ID = "00000000-0000-0000-0000-00000000abcd"


def _tenant_routes() -> list[tuple[str, str]]:
    out = []
    for path, item in build_app(None).openapi()["paths"].items():
        if not path.startswith(("/v1/collections", "/v1/jobs")):
            continue
        concrete = re.sub(r"\{version\}", "1", path)
        concrete = re.sub(r"\{[a-z_]+\}", ANY_ID, concrete)
        out.extend((method.upper(), concrete) for method in item)
    return sorted(out)


def _admin_routes() -> list[tuple[str, str]]:
    out = []
    for path, item in build_app(None).openapi()["paths"].items():
        if path.startswith("/v1/admin"):
            concrete = re.sub(r"\{name\}", "space-x", path)
            concrete = re.sub(r"\{[a-z_]+\}", ANY_ID, concrete)
            out.extend((method.upper(), concrete) for method in item)
    return sorted(out)


TENANT_ROUTES = _tenant_routes()
ADMIN_ROUTES = _admin_routes()


def test_route_tables_cover_every_router() -> None:
    assert len(TENANT_ROUTES) == 18
    assert len(ADMIN_ROUTES) == 14


def _assert_unauthenticated(response: object) -> None:
    import httpx  # noqa: PLC0415 - typing helper only

    assert isinstance(response, httpx.Response)
    assert response.status_code == 401, response.text
    assert error_of(response)["code"] == "unauthenticated"
    assert response.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(("method", "path"), TENANT_ROUTES)
async def test_tenant_route_without_key_is_401(rig: ApiRig, method: str, path: str) -> None:
    response = await rig.client.request(method, path, content=b"{not json")
    _assert_unauthenticated(response)
    assert rig.keys.calls == []


@pytest.mark.parametrize(
    ("name", "header"),
    [
        ("basic_scheme", "Basic Zm46YmFy"),
        ("empty_bearer", "Bearer "),
        ("unknown_key", "Bearer fn_unknownkey"),
        ("admin_key_on_tenant_route", f"Bearer {ADMIN_KEY}"),
        ("no_scheme", "fn_test0000"),
    ],
)
async def test_tenant_auth_rejects_bad_credentials(rig: ApiRig, name: str, header: str) -> None:
    response = await rig.client.get("/v1/collections", headers={"Authorization": header})
    _assert_unauthenticated(response)


async def test_tenant_auth_rejects_overlong_token_without_lookup(rig: ApiRig) -> None:
    token = "fn_" + "a" * MAX_TOKEN_CHARS
    response = await rig.client.get("/v1/collections", headers=rig.auth(token))
    _assert_unauthenticated(response)
    assert rig.keys.calls == []


async def test_revoked_key_is_401_and_other_keys_still_work(rig: ApiRig) -> None:
    second = rig.keys.issue(rig.tenant_a, "second").key
    rig.keys.revoke(rig.key_a)
    _assert_unauthenticated(await rig.client.get("/v1/collections", headers=rig.auth()))
    ok = await rig.client.get("/v1/collections", headers=rig.auth(second))
    assert ok.status_code == 200


async def test_expired_key_is_401(rig: ApiRig) -> None:
    from dataclasses import replace  # noqa: PLC0415

    record = rig.keys.keys[rig.key_a]
    rig.keys.keys[rig.key_a] = replace(record, expires_at=datetime.now(UTC) - timedelta(1))
    _assert_unauthenticated(await rig.client.get("/v1/collections", headers=rig.auth()))


async def test_valid_key_resolves_tenant_and_logs_it() -> None:
    rig = build_rig(access_log=True)
    async with rig.client, rig.admin_client:
        response = await rig.client.get("/v1/collections", headers=rig.auth())
    assert response.status_code == 200
    ids = [UUID(c["id"]) for c in response.json()["data"]]
    assert ids == [rig.collection_a.id]
    lines = [json.loads(line) for line in rig.log.getvalue().splitlines()]
    access = [line for line in lines if line["msg"] == "request"]
    assert access[-1]["tenant"] == str(rig.tenant_a)
    assert access[-1]["request_id"] == response.headers["x-request-id"]
    assert rig.key_a not in rig.log.getvalue()


async def test_tenant_context_does_not_leak_into_the_next_request() -> None:
    rig = build_rig(access_log=True)
    async with rig.client, rig.admin_client:
        await rig.client.get("/v1/collections", headers=rig.auth())
        await rig.client.get("/healthz")
    lines = [json.loads(line) for line in rig.log.getvalue().splitlines()]
    assert "tenant" not in lines[-1]


@pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
@pytest.mark.parametrize("key", ["none", "tenant", "wrong"])
async def test_admin_route_rejects_non_admin_credentials(
    rig: ApiRig, method: str, path: str, key: str
) -> None:
    headers = {
        "none": {},
        "tenant": rig.auth(),
        "wrong": {"Authorization": f"Bearer {ADMIN_KEY}x"},
    }[key]
    response = await rig.client.request(method, path, headers=headers, json={})
    _assert_unauthenticated(response)


async def test_admin_api_disabled_without_admin_key() -> None:
    rig = build_rig(admin_key="")
    async with rig.client, rig.admin_client:
        for token in ("", ADMIN_KEY):
            response = await rig.client.get(
                "/v1/admin/tenants", headers={"Authorization": f"Bearer {token}"}
            )
            _assert_unauthenticated(response)


async def test_admin_key_grants_admin_routes(rig: ApiRig) -> None:
    response = await rig.client.get("/v1/admin/tenants", headers=rig.admin)
    assert response.status_code == 200
    ids = {UUID(t["id"]) for t in response.json()["data"]}
    assert {rig.tenant_a, rig.tenant_b} <= ids
