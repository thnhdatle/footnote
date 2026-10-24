# SPDX-License-Identifier: Apache-2.0
"""Search and answer routes."""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from footnote.api.auth import Tenant
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.errors import INTERNAL
from footnote.api.schemas.common import error_responses
from footnote.api.schemas.search import (
    AnswerIn,
    AnswerOut,
    CitationOut,
    HitOut,
    SearchIn,
    SearchOut,
    to_filters,
)
from footnote.app.answer import AnswerEvent, AnswerRequest
from footnote.app.search import SearchRequest, SearchResult
from footnote.domain.answer import Answer
from footnote.domain.documents import CollectionId
from footnote.domain.errors import DomainError
from footnote.domain.tenancy import TenantId
from footnote.observability import Metrics

router = APIRouter(prefix="/v1/collections/{collection_id}", tags=["retrieval"])
Deps = Annotated[ApiDeps, Depends(get_deps)]

SSE_MEDIA_TYPE = "text/event-stream"


def _trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


def observe_search(metrics: Metrics, result: SearchResult) -> None:
    for stage, seconds in result.stage_seconds.items():
        metrics.search_duration.labels(stage).observe(seconds)
    if result.exact_scan_reason in ("small_collection", "fallback"):
        metrics.observe_exact_scan(result.exact_scan_reason)  # type: ignore[arg-type]


@router.post("/search", response_model=SearchOut, responses=error_responses(401, 404, 422))
async def search(collection_id: UUID, body: SearchIn, principal: Tenant, deps: Deps) -> SearchOut:
    """Hybrid (default), vector or lexical search with optional rerank."""
    result = await deps.search.search(
        principal.tenant,
        CollectionId(collection_id),
        SearchRequest(
            query=body.query,
            mode=body.mode,
            rerank=body.rerank,
            k=body.k,
            filters=to_filters(body.filters),
        ),
    )
    observe_search(deps.metrics, result)
    return SearchOut.of(result, debug=body.debug)


def _outcome(answer: Answer) -> str:
    if answer.abstained:
        return "abstained"
    return "unsupported" if answer.unsupported else "answered"


@router.post(
    "/answers",
    response_model=AnswerOut,
    responses={
        200: {
            "content": {
                SSE_MEDIA_TYPE: {
                    "schema": {
                        "type": "string",
                        "description": "retrieval, delta, citations, done | error events",
                    }
                }
            },
            "description": "JSON answer, or SSE events with `stream: true`",
        },
        **error_responses(401, 404, 409, 422, 502, 503),
    },
)
async def answer(
    collection_id: UUID, body: AnswerIn, principal: Tenant, deps: Deps
) -> AnswerOut | Response:
    """Answer from the collection with verified `[n]` citations, or abstain."""
    request = AnswerRequest(
        question=body.question, filters=to_filters(body.filters), mode=body.mode, rerank=body.rerank
    )
    cid = CollectionId(collection_id)
    if body.stream:
        return await _stream(deps, principal.tenant, cid, request)
    try:
        outcome = await deps.answers.answer(principal.tenant, cid, request)
    except Exception:
        deps.metrics.observe_answer("error")
        raise
    observe_search(deps.metrics, outcome.search)
    deps.metrics.observe_answer(_outcome(outcome.answer))  # type: ignore[arg-type]
    return AnswerOut.of(outcome.answer, _trace_id())


async def _close(events: AsyncIterator[AnswerEvent]) -> None:
    close = getattr(events, "aclose", None)
    if close is not None:
        await close()


def sse(event: str, data: Any) -> bytes:
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


def event_payload(event: AnswerEvent, trace_id: str | None) -> dict[str, Any]:
    data = event.data
    if event.event == "retrieval":
        return {"hits": [HitOut.of(h).model_dump(mode="json") for h in data["hits"]]}
    if event.event == "citations":
        return {
            **data,
            "citations": [CitationOut.of(c).model_dump(mode="json") for c in data["citations"]],
        }
    if event.event == "done":
        return {**data, "trace_id": trace_id}
    return dict(data)


async def _stream(
    deps: ApiDeps, tenant: TenantId, collection_id: CollectionId, request: AnswerRequest
) -> Response:
    events = deps.answers.stream(tenant, collection_id, request)
    try:
        first = await anext(events)
    except Exception:
        deps.metrics.observe_answer("error")
        await _close(events)
        raise
    trace_id = _trace_id()

    async def body() -> AsyncIterator[bytes]:
        outcome = "error"
        try:
            event: AnswerEvent | None = first
            while event is not None:
                if event.event == "done":
                    done = event.data
                    outcome = (
                        "abstained"
                        if done.get("abstained")
                        else "unsupported"
                        if done.get("unsupported")
                        else "answered"
                    )
                yield sse(event.event, event_payload(event, trace_id))
                event = await anext(events, None)
        except DomainError as exc:
            yield sse("error", {"code": exc.code, "message": exc.message})
        except Exception:
            deps.logger.error(
                "answer stream failed",
                collection_id=str(collection_id),
                error=traceback.format_exc(),
            )
            yield sse("error", {"code": INTERNAL, "message": "internal error"})
        finally:
            await _close(events)
            deps.metrics.observe_answer(outcome)  # type: ignore[arg-type]

    return StreamingResponse(
        body(),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
