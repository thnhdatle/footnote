# SPDX-License-Identifier: Apache-2.0
"""Request middleware."""

from __future__ import annotations

import re
import time
import traceback
from collections.abc import Callable, MutableMapping
from typing import Any

import orjson
import structlog
from opentelemetry import propagate
from opentelemetry.trace import SpanKind
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from ulid import ULID

from footnote.api.deps import ApiDeps
from footnote.api.errors import INTERNAL, envelope
from footnote.domain.errors import TooLargeError
from footnote.observability import tracer

REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
UNMATCHED_ROUTE = "other"

type Limit = Callable[[Scope], int]


class BodyTooLargeError(TooLargeError):
    code = "request_too_large"


def request_id_from(headers: list[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1")
            if _REQUEST_ID.fullmatch(candidate):
                return candidate
    return str(ULID())


def route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    app = scope.get("app")
    for candidate in getattr(getattr(app, "router", None), "routes", ()):
        candidate_path = getattr(candidate, "path", None)
        if not isinstance(candidate_path, str):
            continue
        match, _ = candidate.matches(scope)
        if match is Match.FULL:
            return candidate_path
    return UNMATCHED_ROUTE


def _limited(receive: Receive, limit: int) -> Receive:
    seen = 0

    async def wrapped() -> Message:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > limit:
                raise BodyTooLargeError(f"request body exceeds {limit} bytes")
        return message

    return wrapped


class RequestContextMiddleware:
    """Adds a request id, tracing, metrics, access log and body limit to each request."""

    def __init__(self, app: ASGIApp, deps: ApiDeps, body_limit: Limit) -> None:
        self.app = app
        self.deps = deps
        self.body_limit = body_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = request_id_from(scope.get("headers", []))
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        state: dict[str, Any] = {"status": 500, "started": False}
        start = time.perf_counter()
        carrier = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        method = scope["method"]
        with tracer().start_as_current_span(
            f"{method}", context=propagate.extract(carrier), kind=SpanKind.SERVER
        ) as span:
            try:
                await self._serve(scope, receive, send, request_id, state)
            finally:
                route = route_template(scope)
                span.update_name(f"{method} {route}")
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", state["status"])
                self._observe(method, route, state["status"], time.perf_counter() - start)
                structlog.contextvars.clear_contextvars()

    async def _serve(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        request_id: str,
        state: MutableMapping[str, Any],
    ) -> None:
        async def tracked_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["started"] = True
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        limit = self.body_limit(scope)
        too_large = _declared_length(scope) > limit
        try:
            if too_large:
                raise BodyTooLargeError(f"request body exceeds {limit} bytes")
            await self.app(scope, _limited(receive, limit), tracked_send)
        except BodyTooLargeError as exc:
            if state["started"]:
                raise
            await _send_error(tracked_send, 413, exc.code, exc.message, request_id)
        except Exception:
            self.deps.logger.error(
                "request failed", operation=route_template(scope), error=traceback.format_exc()
            )
            if state["started"]:
                raise
            await _send_error(tracked_send, 500, INTERNAL, "internal error", request_id)

    def _observe(self, method: str, route: str, status: int, seconds: float) -> None:
        metrics = self.deps.metrics
        metrics.http_requests.labels(route, method, str(status)).inc()
        metrics.http_duration.labels(route).observe(seconds)
        if self.deps.settings.access_log:
            self.deps.logger.info(
                "request",
                operation=f"{method} {route}",
                status=status,
                duration_ms=round(seconds * 1000, 3),
            )


def _declared_length(scope: Scope) -> int:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


async def _send_error(send: Send, status: int, code: str, message: str, request_id: str) -> None:
    body = orjson.dumps(envelope(code, message, request_id))
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
