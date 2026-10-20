# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OpenAI-compatible chat client over an httpx mock transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from footnote.adapters.llm import LLMConfig, OpenAICompatLLM
from footnote.domain.answer import Usage
from footnote.domain.errors import UnavailableError, UpstreamError
from footnote.domain.ports import ChatMessage

MESSAGES = [ChatMessage("user", "hello")]
Handler = Callable[[httpx.Request], httpx.Response]


def _completion(content: str, **message: Any) -> dict[str, Any]:
    return {
        "model": "served-model",
        "choices": [{"message": {"role": "assistant", "content": content, **message}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


class Harness:
    """Builds a client over a scripted handler and records sleeps."""

    def __init__(self, handler: Handler, **config: Any) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        cfg = LLMConfig(base_url="http://llm.test/v1/", model="m", **config)
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
        self.llm = OpenAICompatLLM(cfg, self.client, sleep=self._sleep, rand=lambda: 0.5)

    async def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)  # type: ignore[no-any-return]


def _sse(*events: Any, done: bool = True) -> httpx.Response:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return httpx.Response(
        200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
    )


def _chunk(text: str | None = None, **extra: Any) -> dict[str, Any]:
    delta: dict[str, Any] = {} if text is None else {"content": text}
    return {"model": "served-model", "choices": [{"delta": delta}], **extra}


async def test_chat_returns_clean_text_and_usage() -> None:
    h = Harness(
        lambda _r: httpx.Response(
            200, json=_completion("<think>plan</think>\n\nAnswer [1]", reasoning_content="plan")
        )
    )
    result = await h.llm.chat(MESSAGES)
    assert result.text == "Answer [1]"
    assert result.model == "served-model"
    assert result.usage == Usage(7, 3)
    assert str(h.requests[0].url) == "http://llm.test/v1/chat/completions"


async def test_chat_request_body_headers_and_extra_body() -> None:
    h = Harness(
        lambda _r: httpx.Response(200, json=_completion("ok")),
        api_key="sk-test",
        extra_headers={"X-Fake-Scenario": "echo"},
        extra_body={"reasoning_effort": "none", "stream": True, "messages": []},
        temperature=0.2,
        max_tokens=64,
    )
    await h.llm.chat(MESSAGES)
    assert h.body() == {
        "model": "m",
        "temperature": 0.2,
        "max_tokens": 64,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "hello"}],
    }
    headers = h.requests[0].headers
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["x-fake-scenario"] == "echo"


async def test_not_configured() -> None:
    llm = OpenAICompatLLM(LLMConfig(), httpx.AsyncClient())
    assert not llm.configured
    with pytest.raises(UnavailableError) as info:
        await llm.chat(MESSAGES)
    assert info.value.code == "llm_not_configured"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_retries_retryable_status_with_jittered_backoff(status: int) -> None:
    calls = iter(
        [
            httpx.Response(status),
            httpx.Response(status),
            httpx.Response(200, json=_completion("fine")),
        ]
    )
    h = Harness(lambda _r: next(calls), max_retries=2, backoff_base=1.0, backoff_cap=8.0)
    result = await h.llm.chat(MESSAGES)
    assert result.text == "fine"
    assert h.sleeps == [0.5, 1.0]  # rand 0.5 * min(cap, base * 2**(attempt-1))


async def test_retry_after_header_is_honoured_and_capped() -> None:
    calls = iter(
        [
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(429, headers={"Retry-After": "999"}),
            httpx.Response(200, json=_completion("fine")),
        ]
    )
    h = Harness(lambda _r: next(calls), max_retries=2, backoff_cap=8.0)
    await h.llm.chat(MESSAGES)
    assert h.sleeps == [1.0, 8.0]


async def test_exhausted_retries_raise_upstream_error() -> None:
    h = Harness(lambda _r: httpx.Response(500), max_retries=1)
    with pytest.raises(UpstreamError, match="after 2 attempts"):
        await h.llm.chat(MESSAGES)
    assert len(h.requests) == 2


async def test_client_error_is_not_retried() -> None:
    h = Harness(lambda _r: httpx.Response(400, json={"error": "bad"}), max_retries=3)
    with pytest.raises(UpstreamError, match="status 400"):
        await h.llm.chat(MESSAGES)
    assert len(h.requests) == 1


async def test_transport_errors_and_timeouts_are_retried() -> None:
    state = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ReadTimeout("slow")
        if state["n"] == 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=_completion("back"))

    h = Harness(handler, max_retries=2)
    assert (await h.llm.chat(MESSAGES)).text == "back"
    assert len(h.sleeps) == 2


async def test_zero_retries_means_one_attempt() -> None:
    h = Harness(lambda _r: httpx.Response(503), max_retries=0)
    with pytest.raises(UpstreamError):
        await h.llm.chat(MESSAGES)
    assert h.sleeps == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b'{"choices": [{"mess'),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"nothing": 1}),
    ],
    ids=["truncated", "no-choice", "no-choices-key"],
)
async def test_invalid_body_raises_upstream_error(response: httpx.Response) -> None:
    h = Harness(lambda _r: response)
    with pytest.raises(UpstreamError, match="invalid response body"):
        await h.llm.chat(MESSAGES)


async def test_missing_usage_defaults_to_zero() -> None:
    body = _completion("x")
    del body["usage"]
    h = Harness(lambda _r: httpx.Response(200, json=body))
    assert (await h.llm.chat(MESSAGES)).usage == Usage()


async def test_stream_yields_content_deltas_then_usage() -> None:
    h = Harness(
        lambda _r: _sse(
            _chunk(None, choices=[{"delta": {"role": "assistant"}}]),
            _chunk(None, choices=[{"delta": {"reasoning_content": "thinking"}}]),
            _chunk("Hel"),
            _chunk("lo [1]"),
            {
                "model": "served-model",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    deltas = [d async for d in h.llm.stream(MESSAGES)]
    assert [d.text for d in deltas] == ["Hel", "lo [1]", ""]
    assert deltas[-1].usage == Usage(5, 2)
    assert deltas[-1].model == "served-model"
    assert h.body()["stream"] is True
    assert h.body()["stream_options"] == {"include_usage": True}


async def test_stream_retries_before_first_byte() -> None:
    calls = iter([httpx.Response(502), _sse(_chunk("ok"))])
    h = Harness(lambda _r: next(calls), max_retries=1)
    assert [d.text async for d in h.llm.stream(MESSAGES)] == ["ok"]
    assert len(h.sleeps) == 1


async def test_stream_cut_without_done_raises() -> None:
    h = Harness(lambda _r: _sse(_chunk("partial"), done=False))
    seen: list[str] = []

    async def consume() -> None:
        async for delta in h.llm.stream(MESSAGES):
            seen.append(delta.text)

    with pytest.raises(UpstreamError, match="before"):
        await consume()
    assert seen == ["partial"]


@pytest.mark.parametrize("event", ["{not json", "[1, 2]"], ids=["malformed", "not-object"])
async def test_stream_malformed_event_raises(event: str) -> None:
    h = Harness(lambda _r: _sse(event))
    with pytest.raises(UpstreamError, match="malformed"):
        _ = [d async for d in h.llm.stream(MESSAGES)]


async def test_stream_not_configured() -> None:
    llm = OpenAICompatLLM(LLMConfig(), httpx.AsyncClient())
    with pytest.raises(UnavailableError):
        _ = [d async for d in llm.stream(MESSAGES)]
