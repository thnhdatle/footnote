# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest

from footnote.domain.settings import AnswerSettings, CollectionSettings
from tests.unit.api.conftest import ANSWER_TEXT, ApiRig, build_rig, error_of
from tests.unit.app.fakes import FakeLLM


@asynccontextmanager
async def custom_rig(**knobs: Any) -> AsyncIterator[ApiRig]:
    rig = build_rig(**knobs)
    async with rig.client, rig.admin_client:
        yield rig


def search_url(rig: ApiRig) -> str:
    return rig.col_url(suffix="/search")


def answers_url(rig: ApiRig) -> str:
    return rig.col_url(suffix="/answers")


def metrics_text(rig: ApiRig) -> str:
    body, _ = rig.deps.metrics.render()
    return body.decode()


def counter(rig: ApiRig, outcome: str) -> float:
    prefix = f'footnote_answers_total{{outcome="{outcome}"}} '
    for line in metrics_text(rig).splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    return 0.0


def sse_events(response: httpx.Response) -> list[tuple[str, Any]]:
    events = []
    for frame in response.text.split("\n\n"):
        if not frame:
            continue
        event_line, data_line = frame.split("\n")
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")
        events.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


@pytest.mark.parametrize("mode", [None, "hybrid", "lexical", "vector"])
async def test_search_returns_hits(rig: ApiRig, mode: str | None) -> None:
    body: dict[str, Any] = {"query": "alpha beta"}
    if mode:
        body["mode"] = mode
    r = await rig.client.post(search_url(rig), headers=rig.auth(), json=body)
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits
    assert hits[0]["text"] == "alpha beta gamma"
    assert hits[0]["title"] == "Greek"
    assert hits[0]["debug"] is None
    assert r.json()["debug"] is None


async def test_search_debug_adds_stage_details(rig: ApiRig) -> None:
    r = await rig.client.post(
        search_url(rig), headers=rig.auth(), json={"query": "alpha beta", "debug": True}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["debug"]["space"] == "space-a"
    assert set(body["debug"]) >= {"reranked", "exact_scan_reason", "stage_seconds"}
    assert all(h["debug"] is not None for h in body["hits"])
    assert set(body["hits"][0]["debug"]) == {"rrf_score", "rerank_score", "ranks", "scores"}


async def test_search_k_limits_the_hits(rig: ApiRig) -> None:
    r = await rig.client.post(search_url(rig), headers=rig.auth(), json={"query": "alpha", "k": 1})
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 1


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"query": "x", "k": 0}, id="k-0"),
        pytest.param({"query": "x", "k": 101}, id="k-101"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": "q" * 4001}, id="long-query"),
        pytest.param({}, id="no-query"),
        pytest.param({"query": "x", "mode": "fuzzy"}, id="unknown-mode"),
        pytest.param({"query": "x", "extra": 1}, id="extra-field"),
        pytest.param({"query": "x", "filters": {"metadata": {"a": {"in": []}}}}, id="empty-in"),
        pytest.param({"query": "x", "filters": {"document_ids": ["nope"]}}, id="bad-document-id"),
        pytest.param({"query": "x", "filters": {"created_after": "yesterday"}}, id="bad-date"),
    ],
)
async def test_search_rejects_invalid_requests(rig: ApiRig, body: dict[str, Any]) -> None:
    r = await rig.client.post(search_url(rig), headers=rig.auth(), json=body)
    assert r.status_code == 422
    err = error_of(r)
    assert err["code"] == "validation_failed"
    assert err["details"]


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param({"metadata": {"lang": "en"}}, id="scalar"),
        pytest.param({"metadata": {"year": 2026, "draft": False, "n": None}}, id="scalars"),
        pytest.param({"metadata": {"lang": {"in": ["en", "fr"]}}}, id="in"),
        pytest.param({"document_ids": [str(uuid4())]}, id="document-ids"),
        pytest.param({"created_after": "2026-01-01T00:00:00Z"}, id="created-after"),
        pytest.param(
            {"created_after": "2026-01-01T00:00:00Z", "created_before": "2027-01-01T00:00:00Z"},
            id="created-between",
        ),
    ],
)
async def test_search_accepts_filters(rig: ApiRig, filters: dict[str, Any]) -> None:
    r = await rig.client.post(
        search_url(rig), headers=rig.auth(), json={"query": "alpha", "filters": filters}
    )
    assert r.status_code == 200


async def test_search_unknown_collection_is_not_found(rig: ApiRig) -> None:
    r = await rig.client.post(
        f"/v1/collections/{uuid4()}/search", headers=rig.auth(), json={"query": "alpha"}
    )
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_search_records_metrics_with_the_route_template(rig: ApiRig) -> None:
    await rig.client.post(search_url(rig), headers=rig.auth(), json={"query": "alpha"})
    text = metrics_text(rig)
    assert "footnote_search_duration_seconds" in text
    route = 'route="/v1/collections/{collection_id}/search"'
    assert any(
        line.startswith("footnote_http_requests_total{")
        and route in line
        and 'status="200"' in line
        for line in text.splitlines()
    )
    assert str(rig.collection_a.id) not in text


async def test_answer_returns_text_citations_and_usage(rig: ApiRig) -> None:
    r = await rig.client.post(answers_url(rig), headers=rig.auth(), json={"question": "beta gamma"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == ANSWER_TEXT
    assert body["abstained"] is False
    assert body["abstention_reason"] is None
    assert body["unsupported"] is False
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["title"] == "Greek"
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert counter(rig, "answered") == 1
    assert "footnote_search_duration_seconds" in metrics_text(rig)


async def test_answer_abstains_without_enough_evidence() -> None:
    settings = CollectionSettings(
        embedding_space="space-a", answer=AnswerSettings(evidence_threshold_rerank=0.99)
    )
    async with custom_rig(settings=settings) as rig:
        r = await rig.client.post(
            answers_url(rig), headers=rig.auth(), json={"question": "zzz qqq"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] is None
        assert body["abstained"] is True
        assert body["abstention_reason"] == "insufficient_evidence"
        assert body["citations"] == []
        assert rig.llm.chat_calls == []
        assert counter(rig, "abstained") == 1


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
async def test_answer_without_llm_is_unavailable(stream: bool) -> None:
    async with custom_rig(llm=FakeLLM("x", configured=False)) as rig:
        r = await rig.client.post(
            answers_url(rig), headers=rig.auth(), json={"question": "beta", "stream": stream}
        )
        assert r.status_code == 503
        assert r.headers["content-type"].startswith("application/json")
        assert error_of(r)["code"] == "llm_not_configured"
        assert counter(rig, "error") == 1


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": "q" * 4001}, id="long-question"),
        pytest.param({"question": "x", "extra": 1}, id="extra-field"),
        pytest.param({"question": "x", "filters": {"metadata": {"a": {"in": []}}}}, id="empty-in"),
    ],
)
async def test_answer_rejects_invalid_requests(
    rig: ApiRig, body: dict[str, Any], stream: bool
) -> None:
    r = await rig.client.post(answers_url(rig), headers=rig.auth(), json={**body, "stream": stream})
    assert r.status_code == 422
    assert error_of(r)["code"] == "validation_failed"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
async def test_answer_unknown_collection_is_not_found(rig: ApiRig, stream: bool) -> None:
    r = await rig.client.post(
        f"/v1/collections/{uuid4()}/answers",
        headers=rig.auth(),
        json={"question": "beta", "stream": stream},
    )
    assert r.status_code == 404
    assert error_of(r)["code"] == "not_found"


async def test_stream_emits_events_in_order(rig: ApiRig) -> None:
    r = await rig.client.post(
        answers_url(rig), headers=rig.auth(), json={"question": "beta gamma", "stream": True}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = sse_events(r)
    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert names[-2:] == ["citations", "done"]
    assert set(names[1:-2]) == {"delta"}
    assert len(names) > 3
    assert events[0][1]["hits"][0]["text"] == "alpha beta gamma"
    assert "".join(data["text"] for name, data in events if name == "delta") == ANSWER_TEXT
    citations = events[-2][1]
    assert citations["answer"] == ANSWER_TEXT
    assert citations["citations"][0]["n"] == 1
    done = events[-1][1]
    assert done["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert done["abstained"] is False
    assert counter(rig, "answered") == 1


async def test_stream_abstention_has_no_deltas() -> None:
    settings = CollectionSettings(
        embedding_space="space-a", answer=AnswerSettings(evidence_threshold_rerank=0.99)
    )
    async with custom_rig(settings=settings) as rig:
        r = await rig.client.post(
            answers_url(rig), headers=rig.auth(), json={"question": "zzz qqq", "stream": True}
        )
        assert r.status_code == 200
        events = sse_events(r)
        assert [name for name, _ in events] == ["retrieval", "citations", "done"]
        assert events[1][1]["citations"] == []
        assert events[1][1]["answer"] is None
        assert events[2][1]["abstained"] is True
        assert events[2][1]["abstention_reason"] == "insufficient_evidence"
        assert counter(rig, "abstained") == 1


async def test_stream_reports_a_failing_llm_as_an_error_event() -> None:
    async with custom_rig(llm=FakeLLM(ANSWER_TEXT, fail_after=1)) as rig:
        r = await rig.client.post(
            answers_url(rig), headers=rig.auth(), json={"question": "beta gamma", "stream": True}
        )
        assert r.status_code == 200
        events = sse_events(r)
        assert [name for name, _ in events] == ["retrieval", "delta", "error"]
        assert events[-1][1]["code"] == "upstream_failed"
        assert counter(rig, "error") == 1
        assert counter(rig, "answered") == 0
