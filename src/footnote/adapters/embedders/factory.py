# SPDX-License-Identifier: Apache-2.0
"""`EmbedderFactory`: serves a registered space with the adapter of its provider.

`on_embed(provider, texts)` is called once per embedding call (documents or query) with
the number of texts, feeding `footnote_embedding_requests_total` and
`footnote_embedding_texts_total`; `aclose()` releases the HTTP clients of remote spaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from footnote.adapters.embedders.onnx_local import OnnxEmbedder
from footnote.adapters.embedders.openai_compat import OpenAICompatEmbedder, OpenAICompatSettings
from footnote.adapters.models.manifest import Manifest
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import UnavailableError
from footnote.domain.ports import Embedder

PROVIDER_ONNX = "onnx_local"
PROVIDER_OPENAI = "openai_compat"

type EmbedObserver = Callable[[str, int], None]


class _Metered:
    """Embedder decorator reporting each call to the observer."""

    def __init__(self, inner: Embedder, provider: str, observe: EmbedObserver) -> None:
        self.inner = inner
        self._provider = provider
        self._observe = observe

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = await self.inner.embed_documents(texts)
        if texts:
            self._observe(self._provider, len(texts))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vector = await self.inner.embed_query(text)
        self._observe(self._provider, 1)
        return vector


@dataclass(frozen=True, slots=True)
class OpenAICompatDefaults:
    """Connection settings shared by every openai_compat space (`embedding.*`)."""

    base_url: str = ""
    api_key: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    batch_size: int = 64
    max_concurrency: int = 4
    timeout: float = 30.0
    max_retries: int = 3


class DefaultEmbedderFactory:
    """Builds and caches one embedder per space id."""

    def __init__(  # noqa: PLR0913 - two optional metric observers
        self,
        manifest: Manifest,
        models_dir: Path,
        remote: OpenAICompatDefaults,
        *,
        intra_op_threads: int = 0,
        on_truncated: Callable[[int], None] | None = None,
        on_embed: EmbedObserver | None = None,
    ) -> None:
        self._manifest = manifest
        self._models_dir = models_dir
        self._remote = remote
        self._threads = intra_op_threads
        self._on_truncated = on_truncated
        self._on_embed = on_embed
        self._cache: dict[int, Embedder] = {}
        self._remote_clients: list[OpenAICompatEmbedder] = []

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        """Embedder for `space`; UnavailableError for an unknown provider or model."""
        cached = self._cache.get(space.id)
        if cached is None:
            cached = self._build(space)
            if self._on_embed is not None:
                cached = _Metered(cached, space.provider, self._on_embed)
            self._cache[space.id] = cached
        return cached

    async def aclose(self) -> None:
        """Close the HTTP clients of the openai_compat embedders built so far."""
        clients, self._remote_clients = self._remote_clients, []
        for embedder in clients:
            await embedder.aclose()

    def _build(self, space: EmbeddingSpace) -> Embedder:
        if space.provider == PROVIDER_ONNX:
            entry = self._manifest.models.get(space.model)
            if entry is None or entry.kind != "embedder":
                raise UnavailableError(f"model {space.model!r} is not an embedder in the manifest")
            return OnnxEmbedder.from_manifest(
                self._models_dir,
                entry,
                intra_op_threads=self._threads,
                on_truncated=self._on_truncated,
            )
        if space.provider == PROVIDER_OPENAI:
            r = self._remote
            remote = OpenAICompatEmbedder(
                OpenAICompatSettings(
                    base_url=r.base_url,
                    model=space.model,
                    dimensions=space.dimensions,
                    api_key=r.api_key,
                    send_dimensions=space.send_dimensions,
                    extra_headers=r.extra_headers,
                    batch_size=r.batch_size,
                    max_concurrency=r.max_concurrency,
                    timeout=r.timeout,
                    max_retries=r.max_retries,
                )
            )
            self._remote_clients.append(remote)
            return remote
        raise UnavailableError(f"unknown embedding provider {space.provider!r}")
