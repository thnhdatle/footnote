# SPDX-License-Identifier: Apache-2.0
"""Chat and streaming client for OpenAI-compatible servers."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from footnote.domain.answer import Usage
from footnote.domain.citations import strip_reasoning
from footnote.domain.errors import UnavailableError, UpstreamError
from footnote.domain.ports import ChatDelta, ChatMessage, ChatResult
from footnote.domain.webhooks import backoff_seconds
from footnote.observability import inject_traceparent, set_chat_attributes, tracer

_PROVIDER = "openai"
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_PROTECTED_KEYS = frozenset({"messages", "stream", "stream_options"})
_DATA = "data:"
_DONE = "[DONE]"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_cap: float = 8.0


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    return Usage(int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0))


def _retry_after(response: httpx.Response, cap: float) -> float | None:
    try:
        return min(max(float(response.headers["retry-after"]), 0.0), cap)
    except (KeyError, ValueError):
        return None


class OpenAICompatLLM:
    """Chat client for an OpenAI-compatible server."""

    def __init__(
        self,
        config: LLMConfig,
        client: httpx.AsyncClient,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._cfg = config
        self._client = client
        self._sleep = sleep
        self._rand = rand

    @property
    def configured(self) -> bool:
        return bool(self._cfg.base_url.strip())

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """Completes a chat and returns the text."""
        return await self.complete(_raw(messages))

    async def complete(self, messages: Sequence[Mapping[str, Any]]) -> ChatResult:
        """Completes a chat given as OpenAI message objects, such as ones with image parts."""
        span = self._start_span()
        with trace.use_span(span, end_on_exit=True):
            response = await self._open(self._body(messages, stream=False), span)
            try:
                payload = orjson.loads(await response.aread())
                choice = payload["choices"][0]["message"]
                content = choice.get("content") or ""
                model = str(payload.get("model") or self._cfg.model)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"llm response failed: {exc.__class__.__name__}") from exc
            except (orjson.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as exc:
                raise UpstreamError("llm returned an invalid response body") from exc
            finally:
                await response.aclose()
            usage = _usage(payload.get("usage")) or Usage()
            self._set_attributes(span, model, usage)
        return ChatResult(text=strip_reasoning(str(content)), model=model, usage=usage)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]:
        """Streams the content deltas of a chat."""
        span = self._start_span()
        model: str | None = None
        usage: Usage | None = None
        try:
            response = await self._open(self._body(_raw(messages), stream=True), span)
            try:
                done = False
                async for line in response.aiter_lines():
                    parsed = _parse_sse(line)
                    if parsed is None:
                        continue
                    if parsed == _DONE:
                        done = True
                        break
                    delta = _delta(parsed)
                    if delta is not None:
                        model = delta.model or model
                        usage = delta.usage or usage
                        yield delta
                if not done:
                    raise UpstreamError("llm stream ended before [DONE]")
            except httpx.HTTPError as exc:
                raise UpstreamError(f"llm stream failed: {exc.__class__.__name__}") from exc
            finally:
                await response.aclose()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            self._set_attributes(span, model, usage)
            span.end()

    def _start_span(self) -> Span:
        span = tracer().start_span(f"chat {self._cfg.model}", kind=SpanKind.CLIENT)
        self._set_attributes(span, None, None)
        return span

    def _set_attributes(self, span: Span, model: str | None, usage: Usage | None) -> None:
        set_chat_attributes(
            span,
            provider=_PROVIDER,
            request_model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            response_model=model,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
        )

    def _body(self, messages: Sequence[Mapping[str, Any]], *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        body.update({k: v for k, v in self._cfg.extra_body.items() if k not in _PROTECTED_KEYS})
        body["messages"] = list(messages)
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _headers(self, span: Span) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        headers.update(self._cfg.extra_headers)
        inject_traceparent(headers, span)
        return headers

    async def _open(self, body: dict[str, Any], span: Span) -> httpx.Response:
        if not self.configured:
            raise UnavailableError("llm.base_url is empty", code="llm_not_configured")
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        content = orjson.dumps(body)
        last = "no attempt made"
        for attempt in range(1, self._cfg.max_retries + 2):
            delay: float | None = None
            try:
                request = self._client.build_request(
                    "POST",
                    url,
                    content=content,
                    headers=self._headers(span),
                    timeout=self._cfg.timeout,
                )
                response = await self._client.send(request, stream=True)
            except httpx.TransportError as exc:
                last = f"transport error {exc.__class__.__name__}"
            else:
                if response.status_code < httpx.codes.BAD_REQUEST:
                    return response
                await response.aclose()
                last = f"status {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise UpstreamError(f"llm request failed: {last}")
                delay = _retry_after(response, self._cfg.backoff_cap)
            if attempt <= self._cfg.max_retries:
                await self._sleep(
                    delay
                    if delay is not None
                    else backoff_seconds(
                        attempt, self._cfg.backoff_base, self._cfg.backoff_cap, self._rand()
                    )
                )
        raise UpstreamError(
            f"llm request failed after {self._cfg.max_retries + 1} attempts: {last}"
        )


def _raw(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _parse_sse(line: str) -> Any | None:
    if not line.startswith(_DATA):
        return None
    data = line[len(_DATA) :].strip()
    if data == _DONE:
        return _DONE
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise UpstreamError("llm stream sent a malformed event") from exc


def _delta(event: Any) -> ChatDelta | None:
    if not isinstance(event, dict):
        raise UpstreamError("llm stream sent a malformed event")
    usage = _usage(event.get("usage"))
    text = ""
    choices = event.get("choices") or []
    if choices and isinstance(choices[0], dict):
        content = (choices[0].get("delta") or {}).get("content")
        text = content if isinstance(content, str) else ""
    if not text and usage is None:
        return None
    model = event.get("model")
    return ChatDelta(text=text, usage=usage, model=model if isinstance(model, str) else None)
