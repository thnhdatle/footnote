# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from footnote.api import ReadinessCheck
from footnote.api.app import create_app
from footnote.api.deps import ApiDeps
from tests.unit.api.conftest import ApiRig, build_rig, error_of

ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@asynccontextmanager
async def custom_rig(**knobs: Any) -> AsyncIterator[ApiRig]:
    rig = build_rig(**knobs)
    async with rig.client, rig.admin_client:
        yield rig


@asynccontextmanager
async def serve(deps: ApiDeps) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    app = create_app(deps)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield app, client


def log_lines(rig: ApiRig) -> list[dict[str, Any]]:
    return [json.loads(line) for line in rig.log.getvalue().splitlines() if line.strip()]


async def test_healthz_is_ok(rig: ApiRig) -> None:
    r = await rig.client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json().get("checks") is None


async def test_readyz_is_ready(rig: ApiRig) -> None:
    r = await rig.client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"fake": []}}


async def test_readyz_reports_problems(rig: ApiRig) -> None:
    rig.readiness_problems.append("db down")
    r = await rig.client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "not_ready", "checks": {"fake": ["db down"]}}
    rig.readiness_problems.clear()
    assert (await rig.client.get("/readyz")).status_code == 200


async def test_readyz_treats_a_raising_probe_as_a_problem(rig: ApiRig) -> None:
    async def failing_probe() -> list[str]:
        msg = "connection refused"
        raise ConnectionError(msg)

    deps = replace(rig.deps, readiness=(ReadinessCheck("boom", failing_probe),))
    async with serve(deps) as (_, client):
        r = await client.get("/readyz")
    assert r.status_code == 503
    problems = r.json()["checks"]["boom"]
    assert len(problems) == 1
    assert "connection refused" in problems[0]


async def test_readyz_times_out_a_hanging_probe(rig: ApiRig) -> None:
    async def hanging_probe() -> list[str]:
        await asyncio.Event().wait()
        return []

    deps = replace(
        rig.deps,
        readiness=(
            ReadinessCheck("slow", hanging_probe),
            ReadinessCheck("fake", rig.deps.readiness[0].probe),
        ),
        settings=replace(rig.deps.settings, readiness_timeout=0.05),
    )
    async with serve(deps) as (_, client):
        r = await client.get("/readyz")
    assert r.status_code == 503
    checks = r.json()["checks"]
    assert "timed out" in checks["slow"][0]
    assert checks["fake"] == []


async def test_healthz_is_unavailable_while_draining(rig: ApiRig) -> None:
    async with serve(rig.deps) as (app, client):
        app.state.draining = True
        r = await client.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "draining"


async def test_version_reports_build_information(rig: ApiRig) -> None:
    r = await rig.client.get("/v1/version")
    assert r.status_code == 200
    body = r.json()
    assert (body["version"], body["commit"], body["build_date"]) == (
        "1.2.3",
        "abc123",
        "2026-10-20",
    )
    assert re.fullmatch(r"3\.\d+\.\d+", body["python_version"])


@pytest.mark.parametrize("path", ["/healthz", "/readyz"])
async def test_admin_listener_serves_health(rig: ApiRig, path: str) -> None:
    r = await rig.admin_client.get(path)
    assert r.status_code == 200


async def test_admin_listener_readyz_reports_problems(rig: ApiRig) -> None:
    rig.readiness_problems.append("db down")
    r = await rig.admin_client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"] == {"fake": ["db down"]}


async def test_admin_listener_serves_prometheus_metrics(rig: ApiRig) -> None:
    r = await rig.admin_client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "footnote_build_info" in r.text


@pytest.mark.parametrize("path", ["/v1/version", "/v1/collections", "/openapi.json"])
async def test_admin_listener_has_no_data_plane_routes(rig: ApiRig, path: str) -> None:
    r = await rig.admin_client.get(path, headers=rig.auth())
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_data_plane_does_not_serve_metrics(rig: ApiRig) -> None:
    assert (await rig.client.get("/metrics")).status_code == 404


@pytest.mark.parametrize("request_id", ["abc-123_X.y", "a" * 128, "01HZXK9Q3M5N7P8R2S4T6V8W0X"])
async def test_valid_request_id_is_echoed(rig: ApiRig, request_id: str) -> None:
    r = await rig.client.get("/healthz", headers={"X-Request-Id": request_id})
    assert r.headers["x-request-id"] == request_id
    err = await rig.client.get("/v1/collections", headers={"X-Request-Id": request_id})
    assert error_of(err)["request_id"] == request_id


@pytest.mark.parametrize("request_id", ["bad id!", "a" * 129, "a" * 200, "café", ""])
async def test_invalid_request_id_is_replaced_by_a_ulid(rig: ApiRig, request_id: str) -> None:
    headers = httpx.Headers([(b"x-request-id", request_id.encode("latin-1"))])
    r = await rig.client.get("/v1/collections", headers=headers)
    assert ULID.match(r.headers["x-request-id"])
    assert error_of(r)["request_id"] == r.headers["x-request-id"]


async def test_missing_request_id_is_generated_per_request(rig: ApiRig) -> None:
    first = await rig.client.get("/healthz")
    second = await rig.client.get("/healthz")
    assert ULID.match(first.headers["x-request-id"])
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


async def test_unknown_route_is_a_not_found_envelope(rig: ApiRig) -> None:
    r = await rig.client.get("/v1/nothing/here")
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"
    text = rig.deps.metrics.render()[0].decode()
    assert any(
        line.startswith("footnote_http_requests_total{")
        and 'route="other"' in line
        and 'status="404"' in line
        for line in text.splitlines()
    )
    assert "/v1/nothing/here" not in text


async def test_wrong_method_is_a_method_not_allowed_envelope(rig: ApiRig) -> None:
    r = await rig.client.post("/healthz")
    assert r.status_code == 405
    assert error_of(r)["code"] == "method_not_allowed"


async def test_malformed_json_is_a_validation_error(rig: ApiRig) -> None:
    r = await rig.client.post(
        "/v1/collections",
        headers={**rig.auth(), "Content-Type": "application/json"},
        content=b"{not json",
    )
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


async def test_validation_errors_name_the_invalid_fields(rig: ApiRig) -> None:
    r = await rig.client.post("/v1/collections", headers=rig.auth(), json={"name": 5})
    err = error_of(r)
    assert err["code"] == "validation_failed"
    assert err["details"][0]["field"] == "name"


async def test_declared_json_body_over_the_limit_is_rejected() -> None:
    async with custom_rig(max_json_bytes=100) as rig:
        r = await rig.client.post("/v1/collections", headers=rig.auth(), json={"name": "x" * 200})
        assert r.status_code == 413
        assert error_of(r)["code"] == "request_too_large"


async def test_streamed_json_body_over_the_limit_is_rejected() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"name": "'
        yield b"x" * 200
        yield b'"}'

    async with custom_rig(max_json_bytes=100) as rig:
        r = await rig.client.post(
            "/v1/collections",
            headers={**rig.auth(), "Content-Type": "application/json"},
            content=chunks(),
        )
        assert r.status_code == 413
        assert error_of(r)["code"] == "request_too_large"


async def test_json_body_at_the_limit_is_accepted() -> None:
    body = json.dumps({"name": "x" * 88}).encode()
    assert len(body) == 100
    async with custom_rig(max_json_bytes=100) as rig:
        r = await rig.client.post(
            "/v1/collections",
            headers={**rig.auth(), "Content-Type": "application/json"},
            content=body,
        )
        assert r.status_code == 201


async def test_handler_exception_is_a_generic_500(
    rig: ApiRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*_: Any, **__: Any) -> Any:
        msg = "secret detail"
        raise RuntimeError(msg)

    monkeypatch.setattr(rig.deps.collections, "get", boom)
    r = await rig.client.get(rig.col_url(), headers=rig.auth())
    assert r.status_code == 500
    err = error_of(r)
    assert err["code"] == "internal"
    assert err["message"] == "internal error"
    assert "secret detail" not in r.text
    assert "request failed" in rig.log.getvalue()
    assert "secret detail" in rig.log.getvalue()


async def test_authentication_exception_is_a_generic_500(
    rig: ApiRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(_key: str) -> Any:
        msg = "key store down"
        raise RuntimeError(msg)

    monkeypatch.setattr(rig.keys, "resolve", boom)
    r = await rig.client.get("/v1/collections", headers=rig.auth())
    assert r.status_code == 500
    assert error_of(r)["code"] == "internal"
    assert "key store down" not in r.text
    assert "request failed" in rig.log.getvalue()


async def test_failed_requests_are_counted_with_status_500(
    rig: ApiRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError

    monkeypatch.setattr(rig.deps.collections, "get", boom)
    await rig.client.get(rig.col_url(), headers=rig.auth())
    text = rig.deps.metrics.render()[0].decode()
    assert any(
        line.startswith("footnote_http_requests_total{")
        and 'route="/v1/collections/{collection_id}"' in line
        and 'status="500"' in line
        for line in text.splitlines()
    )


async def test_access_log_writes_one_line_per_request() -> None:
    async with custom_rig(access_log=True) as rig:
        r = await rig.client.get("/v1/collections", headers=rig.auth())
        assert r.status_code == 200
        lines = log_lines(rig)
        assert len(lines) == 1
        line = lines[0]
        assert line["msg"] == "request"
        assert line["operation"] == "GET /v1/collections"
        assert line["status"] == 200
        assert line["request_id"] == r.headers["x-request-id"]
        assert "duration_ms" in line
        await rig.client.get("/healthz")
        assert len(log_lines(rig)) == 2


async def test_access_log_is_off_by_default(rig: ApiRig) -> None:
    r = await rig.client.get("/v1/collections", headers=rig.auth())
    assert r.status_code == 200
    assert rig.log.getvalue() == ""
