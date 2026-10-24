# SPDX-License-Identifier: Apache-2.0
"""In-process runtime for evaluation runs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from footnote import bootstrap
from footnote.app.evaluation.cache import CachingEmbedder, cache_key_for_space, mark_warm
from footnote.app.evaluation.retrieval import EvalServices, Shipped
from footnote.config import Settings
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.ports import Embedder, Reranker
from footnote.domain.tenancy import TenantId

RERANKER_VARIANTS = ("int8", "fp32")
DATASETS_MANIFEST = Path("eval/datasets.yaml")


@dataclass
class EvalRuntime:
    settings: Settings
    runtime: bootstrap.Runtime
    services: EvalServices
    shipped: Shipped
    caches: dict[str, CachingEmbedder] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)

    def mark_warm(self, dataset: str, texts: int) -> bool:
        cache = self.caches.get(self.settings.embedding.default_space)
        if cache is None:
            return False
        mark_warm(self.settings.eval.cache_dir, dataset, cache.key, texts)
        return True


def dataset_sha256(name: str, manifest: Path = DATASETS_MANIFEST) -> str | None:
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    entry = (data.get("datasets") or {}).get(name) or {}
    value = entry.get("sha256")
    return str(value) if value else None


@asynccontextmanager
async def open_eval_runtime(
    settings: Settings, *, cache: bool = True
) -> AsyncIterator[EvalRuntime]:
    """Opens the in-process runtime with the embedding cache."""
    manifest = bootstrap.load_model_manifest(settings)
    caches: dict[str, CachingEmbedder] = {}

    def provenance() -> list[str]:
        lines: list[str] = []
        for space in settings.embedding.spaces:
            if space.name != settings.embedding.default_space:
                continue
            line = f"embedding space `{space.name}`: {space.provider} `{space.model}`"
            entry = manifest.models.get(space.model)
            if entry is not None:
                line += f" ({entry.repo}@{entry.revision[:12]}, model sha256 "
                line += f"{entry.files['model'].sha256[:12]})"
            lines.append(line)
        r = settings.rerank
        if r.provider == "onnx_cross_encoder":
            entry = manifest.models.get(r.model)
            if entry is not None:
                files = ", ".join(f"{k} {v.sha256[:12]}" for k, v in sorted(entry.variants.items()))
                lines.append(
                    f"reranker `{r.model}` ({entry.repo}@{entry.revision[:12]}; files {files}); "
                    f"configured variant {r.variant}"
                )
        else:
            lines.append(f"reranker provider {r.provider}")
        return lines

    def wrap(space: EmbeddingSpace, inner: Embedder) -> Embedder:
        sha = None
        if space.provider == bootstrap.PROVIDER_ONNX:
            sha = manifest.get(space.model).files["model"].sha256
        wrapped = CachingEmbedder(
            inner,
            settings.eval.cache_dir,
            cache_key_for_space(space, sha),
            cache_queries=False,
        )
        caches[space.name] = wrapped
        return wrapped

    try:
        async with bootstrap.open_runtime(settings, embedder_wrapper=wrap if cache else None) as rt:
            rerankers: dict[str, Reranker] = {}
            configured = rt.api.search.reranker
            if configured is not None:
                rerankers[settings.rerank.variant] = configured
                for variant in RERANKER_VARIANTS:
                    if variant not in rerankers:
                        extra = bootstrap.make_reranker(settings, manifest, variant)
                        if extra is not None:
                            rerankers[variant] = extra

            async def delete_tenant(tenant: TenantId) -> None:
                await bootstrap.delete_tenant_now(rt, tenant)

            services = EvalServices(
                admin=rt.admin,
                collections=rt.api.collections,
                uploader=rt.api.documents,
                ingest=rt.ingest,
                search=rt.api.search,
                rerankers=rerankers,
                delete_tenant=delete_tenant,
            )
            defaults = settings.collection_defaults()
            shipped = Shipped(
                defaults.retrieval, defaults.chunking.strategy, settings.rerank.variant
            )
            yield EvalRuntime(settings, rt, services, shipped, caches, provenance())
    finally:
        for c in caches.values():
            c.close()
