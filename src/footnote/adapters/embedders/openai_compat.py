# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible `/embeddings` client with batching, bounded concurrency and retries."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from footnote.domain.embedding import l2_normalise
from footnote.domain.errors import UnavailableError, UpstreamError, ValidationError
from footnote.domain.webhooks import backoff_seconds

_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER = 30.0


@dataclass(frozen=True, slots=True)
class OpenAICompatSettings:
    """`embedding.*` settings of one openai_compat space."""

    base_url: str
    model: str
    dimensions: int
    api_key: str = ""
    send_dimensions: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    batch_size: int = 64
    max_concurrency: int = 4
    timeout: float = 30.0
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 8.0


class _Retryable(Exception):  # noqa: N818 - internal control flow, not a public error
    def __init__(self, reason: str, retry_after: float = 0.0) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float:
    try:
        return min(max(float(response.headers.get("retry-after", "0")), 0.0), _MAX_RETRY_AFTER)
    except ValueError:
        return 0.0


class OpenAICompatEmbedder:
    """Implements `ports.Embedder` against an OpenAI-compatible provider."""

    def __init__(
        self,
        settings: OpenAICompatSettings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        if not settings.base_url:
            raise UnavailableError("embedding.base_url is not set", code="embedding_not_configured")
        if settings.batch_size < 1 or settings.max_concurrency < 1 or settings.max_retries < 0:
            raise ValidationError("embedding", "batch_size, max_concurrency must be >= 1")
        self._settings = settings
        self._url = settings.base_url.rstrip("/") + "/embeddings"
        self._client = client or httpx.AsyncClient(timeout=settings.timeout)
        self._owns_client = client is None
        self._sleep = sleep
        self._rand = rand
        self._gate = asyncio.Semaphore(settings.max_concurrency)

    async def aclose(self) -> None:
        """Close the HTTP client when this adapter created it."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        headers.update(self._settings.extra_headers)
        return headers

    def _body(self, texts: Sequence[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._settings.model, "input": list(texts)}
        if self._settings.send_dimensions:
            body["dimensions"] = self._settings.dimensions
        return body

    def _parse(self, response: httpx.Response, count: int) -> list[list[float]]:
        if response.status_code in _RETRY_STATUSES:
            raise _Retryable(f"status {response.status_code}", _retry_after(response))
        if response.status_code >= 400:  # noqa: PLR2004
            raise UpstreamError(
                f"embedding provider rejected the request (status {response.status_code})",
                code="embedding_request_rejected",
            )
        try:
            rows = sorted(response.json()["data"], key=lambda r: r["index"])
            vectors = [[float(x) for x in r["embedding"]] for r in rows]
        except (ValueError, KeyError, TypeError) as exc:
            raise _Retryable("malformed response body") from exc
        if len(vectors) != count:
            raise _Retryable(f"provider returned {len(vectors)} vectors for {count} inputs")
        for vector in vectors:
            if len(vector) != self._settings.dimensions:
                raise UpstreamError(
                    f"provider returned {len(vector)} dimensions, "
                    f"space expects {self._settings.dimensions}",
                    code="embedding_dimension_mismatch",
                )
        return vectors

    async def _attempt(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                self._url,
                json=self._body(texts),
                headers=self._headers(),
                timeout=self._settings.timeout,
            )
        except httpx.HTTPError as exc:
            raise _Retryable(type(exc).__name__) from exc
        return self._parse(response, len(texts))

    async def _post_batch(self, texts: Sequence[str]) -> list[list[float]]:
        attempts = self._settings.max_retries + 1
        last = ""
        async with self._gate:
            for attempt in range(1, attempts + 1):
                try:
                    return await self._attempt(texts)
                except _Retryable as exc:
                    last = str(exc)
                    if attempt == attempts:
                        break
                    delay = backoff_seconds(
                        attempt,
                        self._settings.backoff_base,
                        self._settings.backoff_cap,
                        self._rand(),
                    )
                    await self._sleep(max(delay, exc.retry_after))
        raise UpstreamError(
            f"embedding provider failed after {attempts} attempts: {last}",
            code="embedding_upstream_failed",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches; every vector is L2-normalised."""
        size = self._settings.batch_size
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        tasks = [asyncio.ensure_future(self._post_batch(b)) for b in batches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [l2_normalise(v) for batch in results for v in batch]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query (no prefix for provider models)."""
        return (await self.embed_documents([text]))[0]
