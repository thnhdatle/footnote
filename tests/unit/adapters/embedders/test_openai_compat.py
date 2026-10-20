# SPDX-License-Identifier: Apache-2.0
"""OpenAICompatEmbedder against an httpx mock transport (the fakellm wire shape)."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import replace

import httpx
import pytest

from footnote.adapters.embedders import OpenAICompatEmbedder, OpenAICompatSettings
from footnote.domain.errors import UnavailableError, UpstreamError, ValidationError

SETTINGS = OpenAICompatSettings(
    base_url="http://provider/v1/", model="m", dimensions=3, api_key="secret"
)
Handler = Callable[[httpx.Request], httpx.Response]


def _vec(text: str) -> list[float]:
    return [float(len(text)), 1.0, 2.0]


def _ok(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    data = [{"index": i, "embedding": _vec(t)} for i, t in enumerate(body["input"])]
    return httpx.Response(200, json={"data": list(reversed(data))})


class Rig:
    def __init__(self, handler: Handler, settings: OpenAICompatSettings = SETTINGS) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        async def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
        self.embedder = OpenAICompatEmbedder(
            settings, client=self.client, sleep=sleep, rand=lambda: 0.5
        )


async def test_posts_to_embeddings_with_headers_and_order() -> None:
    rig = Rig(_ok, replace(SETTINGS, extra_headers={"X-Fake-Scenario": "ok"}))
    out = await rig.embedder.embed_documents(["a", "bbb"])
    request = rig.requests[0]
    assert str(request.url) == "http://provider/v1/embeddings"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["x-fake-scenario"] == "ok"
    body = json.loads(request.content)
    assert body == {"model": "m", "input": ["a", "bbb"]}
    assert [round(math.hypot(*v), 6) for v in out] == [1.0, 1.0]
    assert out[1][0] > out[0][0]  # response was reversed; order restored by index


async def test_sends_dimensions_when_asked() -> None:
    rig = Rig(_ok, replace(SETTINGS, send_dimensions=True))
    await rig.embedder.embed_query("a")
    assert json.loads(rig.requests[0].content)["dimensions"] == 3


async def test_no_authorization_without_key() -> None:
    rig = Rig(_ok, replace(SETTINGS, api_key=""))
    await rig.embedder.embed_query("a")
    assert "authorization" not in rig.requests[0].headers


async def test_batches_by_batch_size() -> None:
    rig = Rig(_ok, replace(SETTINGS, batch_size=2))
    out = await rig.embedder.embed_documents(["a", "bb", "ccc", "dddd", "e"])
    assert [len(json.loads(r.content)["input"]) for r in rig.requests] == [2, 2, 1]
    assert len(out) == 5
    assert out[3][0] > out[2][0] > out[1][0]


async def test_empty_input_makes_no_request() -> None:
    rig = Rig(_ok)
    assert await rig.embedder.embed_documents([]) == []
    assert rig.requests == []


async def test_retries_429_honouring_retry_after_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return _ok(request)

    rig = Rig(handler)
    assert len(await rig.embedder.embed_documents(["a"])) == 1
    assert calls == 3
    assert rig.sleeps == [1.0, 1.0]  # backoff 0.25 and 0.5 are below Retry-After


async def test_retry_after_is_capped_and_garbage_ignored() -> None:
    answers = iter([("9999", 500), ("soon", 500), ("", 200)])

    def handler(request: httpx.Request) -> httpx.Response:
        value, status = next(answers)
        if status == 200:
            return _ok(request)
        return httpx.Response(status, headers={"Retry-After": value})

    rig = Rig(handler)
    await rig.embedder.embed_documents(["a"])
    assert rig.sleeps == [30.0, 0.5]


async def test_exhausted_retries_raise_upstream_error() -> None:
    rig = Rig(lambda _r: httpx.Response(500))
    with pytest.raises(UpstreamError) as info:
        await rig.embedder.embed_documents(["a"])
    assert info.value.code == "embedding_upstream_failed"
    assert len(rig.requests) == SETTINGS.max_retries + 1
    assert len(rig.sleeps) == SETTINGS.max_retries


@pytest.mark.parametrize(
    ("name", "response"),
    [
        ("truncated json", httpx.Response(200, content=b'{"data": [{"index": 0, "emb')),
        ("no data", httpx.Response(200, json={"object": "list"})),
        ("wrong count", httpx.Response(200, json={"data": []})),
        ("bad element", httpx.Response(200, json={"data": [{"index": 0, "embedding": "x"}]})),
    ],
)
async def test_malformed_responses_are_retried(name: str, response: httpx.Response) -> None:
    rig = Rig(lambda _r: response, replace(SETTINGS, max_retries=1))
    with pytest.raises(UpstreamError):
        await rig.embedder.embed_documents(["a"])
    assert len(rig.requests) == 2, name


async def test_transport_errors_are_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
        return _ok(request)

    rig = Rig(handler)
    assert len(await rig.embedder.embed_documents(["a"])) == 1
    assert calls == 2


async def test_client_errors_are_not_retried() -> None:
    rig = Rig(lambda _r: httpx.Response(401, json={"error": "nope"}))
    with pytest.raises(UpstreamError) as info:
        await rig.embedder.embed_documents(["a"])
    assert info.value.code == "embedding_request_rejected"
    assert len(rig.requests) == 1


async def test_dimension_mismatch_is_final() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    rig = Rig(handler)
    with pytest.raises(UpstreamError) as info:
        await rig.embedder.embed_documents(["a"])
    assert info.value.code == "embedding_dimension_mismatch"
    assert len(rig.requests) == 1


async def test_concurrency_is_bounded() -> None:
    running = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return _ok(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = replace(SETTINGS, batch_size=1, max_concurrency=2)
    embedder = OpenAICompatEmbedder(settings, client=client)
    out = await embedder.embed_documents([str(i) for i in range(8)])
    assert len(out) == 8
    assert peak == 2


async def test_owned_client_is_closed() -> None:
    embedder = OpenAICompatEmbedder(SETTINGS)
    await embedder.aclose()
    rig = Rig(_ok)
    await rig.embedder.aclose()
    assert not rig.client.is_closed


@pytest.mark.parametrize(
    ("name", "settings", "error"),
    [
        ("no url", replace(SETTINGS, base_url=""), UnavailableError),
        ("batch", replace(SETTINGS, batch_size=0), ValidationError),
        ("concurrency", replace(SETTINGS, max_concurrency=0), ValidationError),
    ],
)
def test_invalid_settings(
    name: str, settings: OpenAICompatSettings, error: type[Exception]
) -> None:
    with pytest.raises(error):
        OpenAICompatEmbedder(settings)
