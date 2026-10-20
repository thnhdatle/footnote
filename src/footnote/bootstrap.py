# SPDX-License-Identifier: Apache-2.0
"""Composition root: config → adapters → use cases → API.

Only this module and `__main__` import both `api` and `adapters`. `open_runtime` opens the
two pools (footnote_app and footnote_admin), the HTTP clients and the ML adapters, wires
every use case, the job handlers (including the `webhook` adapter handler and
`tenant_delete` over the admin pool) and the readiness checks, registers the configured
embedding spaces, and closes everything in reverse order. `serve` runs the data plane and
the admin listener with uvicorn (optionally the worker in-process); `run_worker` runs the
queue worker next to the admin listener. Both stop on `stop` (SIGTERM/SIGINT in the CLI):
stop accepting, drain in-flight requests and jobs (bounded by `shutdown_timeout`), flush
traces, close the pools.

Usage::

    settings = load_settings()
    asyncio.run(serve(settings, with_worker=True))
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from structlog.typing import FilteringBoundLogger

from footnote.adapters.embedders import DefaultEmbedderFactory, OpenAICompatDefaults
from footnote.adapters.llm.client import LLMConfig, OpenAICompatLLM
from footnote.adapters.llm.probe import models_endpoint_problems
from footnote.adapters.models import (
    Manifest,
    ModelNeed,
    ModelReadiness,
    load_manifest,
    reranker_variant_key,
)
from footnote.adapters.parsers import ParseLimits, SandboxedParser, UploadChecks
from footnote.adapters.postgres import health as pg_health
from footnote.adapters.postgres.chunks import PgChunks
from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.export import PgTenantExport
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.migrate import find_migrations, head_revision
from footnote.adapters.postgres.roles import app_role_problems
from footnote.adapters.postgres.search import PgSearchStore, SnapshotSettings
from footnote.adapters.postgres.session import Database, PoolSettings, open_pool
from footnote.adapters.postgres.spaces import HnswParams, PgSpaceAdmin, PgSpaceReader
from footnote.adapters.postgres.tenants import PgKeyResolver, PgTenantAdmin
from footnote.adapters.postgres.webhook_deliveries import PgWebhookDeliveries
from footnote.adapters.queue.worker import Handler, JobContext, Worker, WorkerConfig
from footnote.adapters.rerankers import LazyOnnxReranker
from footnote.adapters.tokenizer import ManifestTokenizers, tokenizer_ids
from footnote.adapters.webhooks.handler import WebhookJobHandler
from footnote.adapters.webhooks.sender import SystemHostResolver, WebhookConfig, WebhookSender
from footnote.api import (
    ApiDeps,
    ApiSettings,
    ReadinessCheck,
    VersionInfo,
    create_admin_app,
    create_app,
)
from footnote.app.admin import AdminConfig, AdminDeps, AdminService
from footnote.app.answer import AnswerConfig, AnswerService
from footnote.app.collections import CollectionConfig, CollectionDeps, CollectionService
from footnote.app.documents import DocumentConfig, DocumentDeps, DocumentService
from footnote.app.gc import GcService
from footnote.app.ingest import IngestConfig, IngestDeps, IngestService
from footnote.app.jobs import JobHandlers, JobServices
from footnote.app.reembed import ReembedConfig, ReembedService
from footnote.app.reindex import ReindexService
from footnote.app.search import SearchConfig, SearchService
from footnote.app.webhooks import WebhookNotifier
from footnote.config import ConfigError, Settings
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec
from footnote.domain.errors import DomainError, ValidationError
from footnote.domain.jobs import ClaimedJob, JobKind
from footnote.domain.ports import Embedder, EmbedderFactory, Reranker
from footnote.domain.tenancy import Quotas, TenantId
from footnote.observability import Metrics, make_logger, setup_tracing

PACKAGE = "rayld-footnote"
PROVIDER_ONNX = "onnx_local"
PROVIDER_OPENAI = "openai_compat"
QUEUE_DEPTH_INTERVAL = 15.0
PROBE_TIMEOUT = 5.0
_ANY_HOST = "0.0.0.0"  # noqa: S104 - an empty listen host means every interface


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """Version and build metadata (`/v1/version`, `footnote_build_info`)."""

    version: str
    commit: str = "unknown"
    build_date: str = "unknown"


def build_info(env: Mapping[str, str] | None = None) -> BuildInfo:
    """Package version plus `FOOTNOTE_BUILD_COMMIT` / `FOOTNOTE_BUILD_DATE` (image build args)."""
    source = os.environ if env is None else env
    try:
        version = pkg_version(PACKAGE)
    except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
        version = "0.0.0"
    return BuildInfo(
        version,
        source.get("FOOTNOTE_BUILD_COMMIT") or "unknown",
        source.get("FOOTNOTE_BUILD_DATE") or "unknown",
    )


class SystemClock:
    """`ports.Clock` over the system clocks."""

    def now(self) -> float:
        """Wall-clock seconds since the epoch."""
        return time.time()

    def monotonic(self) -> float:
        """Monotonic seconds (durations)."""
        return time.monotonic()


# --- configuration checks that need the model manifest ---------------------------------


def required_dsn(settings: Settings, *, admin: bool) -> str:
    """The app or admin DSN; ConfigError (exit 2) when neither env nor `*_file` sets it."""
    section, env = (
        ("admin_database", "FOOTNOTE_ADMIN_DATABASE_URL")
        if admin
        else ("database", "FOOTNOTE_DATABASE_URL")
    )
    value = (settings.admin_database.url if admin else settings.database.url).get_secret_value()
    if not value:
        raise ConfigError([f"{section}.url: required; set {env} or {section}.url_file"])
    return value


def load_model_manifest(settings: Settings) -> Manifest:
    """The manifest named by `models.manifest`; ConfigError when unreadable."""
    try:
        return load_manifest(settings.models.manifest)
    except ValidationError as exc:
        raise ConfigError([f"models.manifest: {exc.message}"]) from exc


def manifest_problems(settings: Settings, manifest: Manifest) -> list[str]:
    """Config references the manifest cannot satisfy (every problem, not the first)."""
    problems: list[str] = []
    for i, space in enumerate(settings.embedding.spaces):
        if space.provider != PROVIDER_ONNX:
            continue
        entry = manifest.models.get(space.model)
        if entry is None or entry.kind != "embedder" or not entry.dimensions:
            problems.append(
                f"embedding.spaces.{i}.model: {space.model!r} is not a manifest embedder"
            )
        elif space.dimensions is not None and space.dimensions != entry.dimensions:
            problems.append(
                f"embedding.spaces.{i}.dimensions: model {space.model} has {entry.dimensions}"
            )
    if settings.rerank.provider == "onnx_cross_encoder":
        entry = manifest.models.get(settings.rerank.model)
        key = reranker_variant_key(settings.rerank.variant)
        if entry is None or entry.kind != "reranker":
            problems.append(f"rerank.model: {settings.rerank.model!r} is not a manifest reranker")
        elif key not in entry.variants or "tokenizer" not in entry.files:
            problems.append(f"rerank.variant: {settings.rerank.model} has no file {key!r}")
    if settings.chunking.tokenizer not in tokenizer_ids(manifest):
        problems.append(
            f"chunking.tokenizer: {settings.chunking.tokenizer!r} is not a manifest model "
            "with a tokenizer file"
        )
    return problems


def required_models(settings: Settings, manifest: Manifest) -> list[ModelNeed]:
    """The manifest files this configuration loads (ONNX spaces, reranker, tokenizer)."""
    keys: dict[str, set[str]] = {}
    for space in settings.embedding.spaces:
        if space.provider == PROVIDER_ONNX and space.model in manifest.models:
            keys.setdefault(space.model, set()).update(manifest.models[space.model].files)
    if (
        settings.rerank.provider == "onnx_cross_encoder"
        and settings.rerank.model in manifest.models
    ):
        variant = reranker_variant_key(settings.rerank.variant)
        keys.setdefault(settings.rerank.model, set()).update({"tokenizer", variant})
    if settings.chunking.tokenizer in manifest.models:
        keys.setdefault(settings.chunking.tokenizer, set()).add("tokenizer")
    return [ModelNeed(manifest.models[m], tuple(sorted(k))) for m, k in sorted(keys.items())]


def space_completer(manifest: Manifest) -> Callable[[SpaceSpec], SpaceSpec]:
    """Fills dimensions and the calibrated threshold of `onnx_local` spaces (precedence:
    request/config value, then the manifest value)."""

    def complete(spec: SpaceSpec) -> SpaceSpec:
        if spec.provider == PROVIDER_OPENAI:
            if spec.dimensions <= 0:
                raise ValidationError("dimensions", "required for provider openai_compat")
            return spec
        if spec.provider != PROVIDER_ONNX:
            raise ValidationError(
                "provider", f"{spec.provider!r} must be onnx_local or openai_compat"
            )
        entry = manifest.models.get(spec.model)
        if entry is None or entry.kind != "embedder" or not entry.dimensions:
            raise ValidationError("model", f"{spec.model!r} is not an embedder in the manifest")
        if spec.dimensions and spec.dimensions != entry.dimensions:
            raise ValidationError("dimensions", f"model {spec.model} has {entry.dimensions}")
        threshold = spec.calibrated_threshold
        return replace(
            spec,
            dimensions=entry.dimensions,
            normalise=entry.normalise,
            calibrated_threshold=entry.calibrated_threshold if threshold is None else threshold,
        )

    return complete


def space_specs(settings: Settings) -> list[SpaceSpec]:
    """`embedding.spaces` as registration specs (dimensions 0 = from the manifest)."""
    return [
        SpaceSpec(
            name=s.name,
            provider=s.provider,
            model=s.model,
            dimensions=s.dimensions or 0,
            send_dimensions=s.send_dimensions,
            normalise=s.normalise,
            calibrated_threshold=s.calibrated_threshold,
        )
        for s in settings.embedding.spaces
    ]


def _quotas(settings: Settings) -> Quotas:
    q = settings.tenancy.default_quotas
    return Quotas(q.max_collections, q.max_documents, q.max_chunks, q.max_upload_bytes_per_day)


def _admin_service(
    settings: Settings, admin_db: Database, manifest: Manifest, clock: SystemClock
) -> AdminService:
    hnsw = HnswParams(settings.embedding.hnsw.m, settings.embedding.hnsw.ef_construction)
    return AdminService(
        AdminDeps(
            PgTenantAdmin(admin_db),
            PgSpaceAdmin(admin_db, hnsw, settings.embedding.max_spaces),
            PgTenantExport(admin_db),
            clock,
            space_completer(manifest),
        ),
        AdminConfig(_quotas(settings), timedelta(seconds=settings.auth.key_rotation_grace)),
    )


# --- pools ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pools:
    """The footnote_app and footnote_admin databases."""

    app: Database
    admin: Database


def _database(settings: Settings, pool: Any, *, predicates: bool = True) -> Database:
    return Database(
        pool,
        statement_timeout_ms=int(settings.database.statement_timeout * 1000),
        app_tenant_predicates=predicates,
    )


async def _open_pools(settings: Settings, stack: AsyncExitStack, predicates: bool) -> Pools:
    app_pool = await open_pool(
        PoolSettings(
            conninfo=required_dsn(settings, admin=False),
            max_size=settings.database.pool_max,
            prepare_threshold=settings.database.prepare_threshold,
        )
    )
    stack.push_async_callback(app_pool.close)
    admin_pool = await open_pool(
        PoolSettings(
            conninfo=required_dsn(settings, admin=True),
            max_size=settings.admin_database.pool_max,
            prepare_threshold=settings.database.prepare_threshold,
        )
    )
    stack.push_async_callback(admin_pool.close)
    return Pools(
        _database(settings, app_pool, predicates=predicates), _database(settings, admin_pool)
    )


@asynccontextmanager
async def open_admin(settings: Settings) -> AsyncIterator[AdminService]:
    """AdminService over the admin pool only (CLI `tenants`, `keys`, `spaces`)."""
    manifest = load_model_manifest(settings)
    pool = await open_pool(
        PoolSettings(
            conninfo=required_dsn(settings, admin=True),
            min_size=1,
            max_size=2,
            prepare_threshold=settings.database.prepare_threshold,
        )
    )
    try:
        yield _admin_service(settings, _database(settings, pool), manifest, SystemClock())
    finally:
        await pool.close()


# --- the runtime ------------------------------------------------------------------------


@dataclass(slots=True)
class Runtime:
    """Everything a process serves from; built and closed by `open_runtime`."""

    settings: Settings
    info: BuildInfo
    metrics: Metrics
    log: FilteringBoundLogger
    pools: Pools
    api: ApiDeps
    jobs: PgJobs
    handlers: dict[str, Handler]
    admin: AdminService
    ingest: IngestService


type EmbedderWrapper = Callable[[EmbeddingSpace, Embedder], Embedder]
"""Decorates the embedder of a space (the eval harness wraps it in its cache)."""


class WrappedEmbedders:
    """`EmbedderFactory` serving `wrap(space, inner.for_space(space))`, one per space id."""

    def __init__(self, inner: EmbedderFactory, wrap: EmbedderWrapper) -> None:
        self._inner = inner
        self._wrap = wrap
        self._cache: dict[int, Embedder] = {}

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        """The wrapped embedder of `space` (built once)."""
        cached = self._cache.get(space.id)
        if cached is None:
            cached = self._wrap(space, self._inner.for_space(space))
            self._cache[space.id] = cached
        return cached


@dataclass(frozen=True, slots=True)
class _Ml:
    embedders: EmbedderFactory
    tokenizers: ManifestTokenizers
    reranker: Reranker | None
    llm: OpenAICompatLLM


def make_reranker(
    settings: Settings, manifest: Manifest, variant: str | None = None, arch: str | None = None
) -> Reranker | None:
    """The configured reranker (`rerank.*`), optionally with another file `variant`
    (`int8` | `fp32`) and, for `int8`, another `arch` (`arm64` | `amd64`; default: this
    host; ONNX Runtime runs either INT8 file on any CPU, used by `eval calibrate`); None
    with the `noop` provider. Files load on the first score."""
    r = settings.rerank
    if r.provider != "onnx_cross_encoder":
        return None
    return LazyOnnxReranker(
        settings.models.dir,
        manifest.get(r.model),
        variant=variant or r.variant,
        arch=arch,
        max_tokens=r.max_tokens,
        batch_size=r.batch_size,
        intra_op_threads=settings.models.intra_op_threads,
    )


async def _ml(
    settings: Settings,
    manifest: Manifest,
    metrics: Metrics,
    stack: AsyncExitStack,
    embedder_wrapper: EmbedderWrapper | None = None,
) -> _Ml:
    e = settings.embedding
    embedders = DefaultEmbedderFactory(
        manifest,
        settings.models.dir,
        OpenAICompatDefaults(
            base_url=e.base_url,
            api_key=e.api_key.get_secret_value(),
            extra_headers=dict(e.extra_headers),
            batch_size=e.batch_size,
            max_concurrency=e.max_concurrency,
            timeout=e.timeout,
            max_retries=e.max_retries,
        ),
        intra_op_threads=settings.models.intra_op_threads,
        on_truncated=metrics.embedding_truncated.inc,
        on_embed=metrics.observe_embedding,
    )
    stack.push_async_callback(embedders.aclose)
    served: EmbedderFactory = embedders
    if embedder_wrapper is not None:
        served = WrappedEmbedders(embedders, embedder_wrapper)
    reranker = make_reranker(settings, manifest)
    llm_client = httpx.AsyncClient(timeout=settings.llm.timeout)
    stack.push_async_callback(llm_client.aclose)
    llm = OpenAICompatLLM(_llm_config(settings), llm_client)
    return _Ml(served, ManifestTokenizers(manifest, settings.models.dir), reranker, llm)


def _llm_config(settings: Settings) -> LLMConfig:
    c = settings.llm
    return LLMConfig(
        base_url=c.base_url,
        model=c.model,
        api_key=c.api_key.get_secret_value(),
        extra_headers=dict(c.extra_headers),
        extra_body=dict(c.extra_body),
        temperature=c.temperature,
        max_tokens=c.max_tokens,
        timeout=c.timeout,
        max_retries=c.max_retries,
    )


@dataclass(frozen=True, slots=True)
class _Repos:
    collections: PgCollections
    documents: PgDocuments
    index: PgIndex
    jobs: PgJobs
    spaces: PgSpaceReader
    tenants: PgTenantAdmin


def _repos(pools: Pools) -> _Repos:
    app = pools.app
    return _Repos(
        PgCollections(app),
        PgDocuments(app),
        PgIndex(app),
        PgJobs(app),
        PgSpaceReader(app),
        PgTenantAdmin(pools.admin),
    )


def _parse_limits(settings: Settings) -> ParseLimits:
    i = settings.ingest
    return ParseLimits(
        max_pages=i.max_pages,
        max_uncompressed_bytes=i.max_uncompressed_bytes,
        max_archive_members=i.max_archive_members,
        parse_timeout=i.parse_timeout,
    )


def _ingest(settings: Settings, repos: _Repos, ml: _Ml, metrics: Metrics) -> IngestService:
    clock = SystemClock()
    deps = IngestDeps(
        repos.collections,
        repos.documents,
        repos.index,
        repos.spaces,
        SandboxedParser(_parse_limits(settings)),
        ml.embedders,
        ml.tokenizers,
        WebhookNotifier(repos.jobs, clock),
        clock,
        repos.tenants,
    )
    config = IngestConfig(
        gc_grace=timedelta(seconds=settings.ingest.gc_grace),
        retain_originals=settings.ingest.retain_originals,
        batch_size=settings.embedding.batch_size,
    )

    def on_stage(stage: str, seconds: float) -> None:
        metrics.ingest_duration.labels(stage).observe(seconds)

    return IngestService(deps, config, on_stage=on_stage)


def _reembed(settings: Settings, repos: _Repos, ml: _Ml) -> ReembedService:
    return ReembedService(
        repos.collections,
        repos.index,
        repos.spaces,
        ml.embedders,
        repos.jobs,
        config=ReembedConfig(
            batch_size=settings.embedding.batch_size,
            gc_grace=timedelta(seconds=settings.ingest.gc_grace),
        ),
    )


def _search(settings: Settings, pools: Pools, ml: _Ml) -> SearchService:
    r = settings.retrieval
    store = PgSearchStore(
        pools.app,
        SnapshotSettings(
            ef_search=r.ef_search,
            iterative_scan=r.iterative_scan,
            max_scan_tuples=r.max_scan_tuples,
        ),
    )
    return SearchService(
        store,
        ml.embedders,
        ml.reranker,
        SystemClock(),
        SearchConfig(exact_scan_max=r.exact_scan_max, default_mode=r.default_mode),
    )


def _collections(
    settings: Settings, pools: Pools, repos: _Repos, reembed: ReembedService, manifest: Manifest
) -> CollectionService:
    return CollectionService(
        CollectionDeps(
            repos.collections,
            repos.jobs,
            repos.spaces,
            reembed,
            SystemHostResolver(),
            PgWebhookDeliveries(pools.app),
            repos.tenants,
        ),
        CollectionConfig(
            defaults=settings.collection_defaults(),
            tokenizer_ids=tokenizer_ids(manifest),
            allow_http_webhooks=settings.webhooks.allow_http,
            allow_private_webhooks=settings.webhooks.allow_private,
        ),
    )


def _documents(settings: Settings, pools: Pools, repos: _Repos) -> DocumentService:
    return DocumentService(
        DocumentDeps(
            repos.collections,
            repos.documents,
            repos.index,
            PgChunks(pools.app),
            UploadChecks(_parse_limits(settings)),
            repos.tenants,
        ),
        DocumentConfig(max_file_bytes=settings.ingest.max_file_bytes),
    )


def _webhook_handler(
    settings: Settings, pools: Pools, repos: _Repos, metrics: Metrics, stack: AsyncExitStack
) -> WebhookJobHandler:
    w = settings.webhooks
    client = httpx.AsyncClient(follow_redirects=False, timeout=w.timeout)
    stack.push_async_callback(client.aclose)
    sender = WebhookSender(
        client,
        WebhookConfig(
            allow_http=w.allow_http,
            allow_private=w.allow_private,
            timeout=w.timeout,
            max_attempts=w.max_attempts,
        ),
    )

    def on_attempt(status: str) -> None:
        metrics.webhook_deliveries.labels(status).inc()

    return WebhookJobHandler(
        repos.jobs, repos.collections, PgWebhookDeliveries(pools.app), sender, on_attempt=on_attempt
    )


def job_handlers(services: JobServices, webhook: Handler) -> dict[str, Handler]:
    """Every job kind's handler: the app-layer handlers plus the `webhook` adapter handler."""
    handlers: dict[str, Handler] = dict(JobHandlers(services).handlers())
    handlers[JobKind.WEBHOOK.value] = webhook
    missing = {k.value for k in JobKind} - handlers.keys()
    if missing:  # pragma: no cover - guarded by a unit test
        raise RuntimeError(f"no handler for job kinds {sorted(missing)}")
    return handlers


# --- readiness --------------------------------------------------------------------------


def _readiness(
    settings: Settings, pools: Pools, models: ModelReadiness, client: httpx.AsyncClient
) -> tuple[ReadinessCheck, ...]:
    head = head_revision(find_migrations())

    async def migrations() -> list[str]:
        return await pg_health.migration_problems(pools.app, head)

    async def versions() -> list[str]:
        return await pg_health.version_problems(pools.app)

    async def admin_pool() -> list[str]:
        return await pg_health.ping(pools.admin)

    async def app_pool() -> list[str]:
        return await pg_health.ping(pools.app)

    async def roles() -> list[str]:
        return await app_role_problems(pools.app)

    checks = [
        ReadinessCheck("database", app_pool),
        ReadinessCheck("admin_database", admin_pool),
        ReadinessCheck("migrations", migrations),
        ReadinessCheck("postgres_versions", versions),
        ReadinessCheck("app_role", roles),
        ReadinessCheck("models", models.problems),
    ]
    checks.extend(_endpoint_checks(settings, client))
    return tuple(checks)


def _endpoint_checks(settings: Settings, client: httpx.AsyncClient) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    llm = settings.llm
    if llm.base_url.strip():

        async def llm_models() -> list[str]:
            return await models_endpoint_problems(
                client,
                llm.base_url,
                api_key=llm.api_key.get_secret_value(),
                headers=llm.extra_headers,
            )

        checks.append(ReadinessCheck("llm", llm_models))
    e = settings.embedding
    if e.base_url.strip() and any(s.provider == PROVIDER_OPENAI for s in e.spaces):

        async def embedding_models() -> list[str]:
            return await models_endpoint_problems(
                client, e.base_url, api_key=e.api_key.get_secret_value(), headers=e.extra_headers
            )

        checks.append(ReadinessCheck("embedding", embedding_models))
    return checks


# --- open_runtime -----------------------------------------------------------------------


async def register_spaces(admin: AdminService, settings: Settings) -> None:
    """Register `embedding.spaces` over the admin pool (idempotent by name)."""
    for spec in space_specs(settings):
        await admin.register_space(spec)


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    info: BuildInfo | None = None,
    env: Mapping[str, str] | None = None,
    embedder_wrapper: EmbedderWrapper | None = None,
) -> AsyncIterator[Runtime]:
    """Open pools, clients and adapters, wire the use cases; close all on exit.

    `embedder_wrapper` decorates every space's embedder (eval cache only).
    """
    info = info or build_info(env)
    log = make_logger(settings.log, component="bootstrap")
    manifest = load_model_manifest(settings)
    problems = manifest_problems(settings, manifest)
    if problems:
        raise ConfigError(problems)
    predicates = not settings.tenant_predicates_disabled(env)
    if not predicates:
        log.warning(
            "testing.disable_app_tenant_predicates is active (tenant-isolation fault test): "
            "repositories omit tenant predicates and rely on RLS alone"
        )
    metrics = Metrics(version=info.version, commit=info.commit)
    async with AsyncExitStack() as stack:
        pools = await _open_pools(settings, stack, predicates)
        runtime = await _wire(
            settings,
            info,
            metrics,
            pools,
            manifest=manifest,
            stack=stack,
            embedder_wrapper=embedder_wrapper,
        )
        await register_spaces(runtime.admin, settings)
        yield runtime


async def _wire(  # noqa: PLR0913 - the composition root's inputs
    settings: Settings,
    info: BuildInfo,
    metrics: Metrics,
    pools: Pools,
    *,
    manifest: Manifest,
    stack: AsyncExitStack,
    embedder_wrapper: EmbedderWrapper | None = None,
) -> Runtime:
    ml = await _ml(settings, manifest, metrics, stack, embedder_wrapper)
    repos = _repos(pools)
    reembed = _reembed(settings, repos, ml)
    search = _search(settings, pools, ml)
    admin = _admin_service(settings, pools.admin, manifest, SystemClock())
    probe_client = httpx.AsyncClient(timeout=PROBE_TIMEOUT)
    stack.push_async_callback(probe_client.aclose)
    models = ModelReadiness(settings.models.dir, required_models(settings, manifest))
    api = ApiDeps(
        collections=_collections(settings, pools, repos, reembed, manifest),
        documents=_documents(settings, pools, repos),
        admin=admin,
        search=search,
        answers=AnswerService(
            search,
            ml.llm,
            AnswerConfig(
                settings.answer.evidence_threshold_rerank, settings.answer.evidence_threshold_vector
            ),
        ),
        jobs=repos.jobs,
        keys=PgKeyResolver(pools.app),
        metrics=metrics,
        logger=make_logger(settings.log, component="api"),
        settings=ApiSettings(
            admin_key=settings.auth.admin_key.get_secret_value(),
            version=VersionInfo(info.version, info.commit, info.build_date),
            access_log=settings.log.access_log,
        ),
        readiness=_readiness(settings, pools, models, probe_client),
    )
    ingest = _ingest(settings, repos, ml, metrics)
    services = JobServices(
        jobs=repos.jobs,
        collections=repos.collections,
        ingest=ingest,
        reindex=ReindexService(repos.collections, repos.documents),
        reembed=reembed,
        gc=GcService(repos.index, retain_originals=settings.ingest.retain_originals),
        tenants=repos.tenants,
        max_attempts=settings.ingest.job_max_attempts,
    )
    webhook = _webhook_handler(settings, pools, repos, metrics, stack)
    return Runtime(
        settings=settings,
        info=info,
        metrics=metrics,
        log=make_logger(settings.log, component="bootstrap"),
        pools=pools,
        api=api,
        jobs=repos.jobs,
        handlers=job_handlers(services, webhook),
        admin=admin,
        ingest=ingest,
    )


async def delete_tenant_now(rt: Runtime, tenant: TenantId) -> None:
    """Delete a tenant and all its data in this process (eval temporary tenants):
    `AdminService.delete_tenant` revokes its keys and queues `tenant_delete`, whose handler
    runs here directly instead of waiting for a worker (it removes the job row too)."""
    job_id = await rt.admin.delete_tenant(tenant)
    handler = rt.handlers[JobKind.TENANT_DELETE.value]
    await handler(
        JobContext(ClaimedJob(job_id, tenant, JobKind.TENANT_DELETE.value), rt.jobs, worker_id())
    )


# --- processes --------------------------------------------------------------------------


def split_listen(address: str) -> tuple[str, int]:
    """`host:port` → (host, port); `[v6]:port` unwrapped; an empty host binds all."""
    host, _, port = address.rpartition(":")
    host = host.strip("[]") or _ANY_HOST
    return host, int(port)


def bind(address: str) -> socket.socket:
    """A listening TCP socket for `address` (port 0 picks a free port)."""
    host, port = split_listen(address)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(2048)
    except OSError:
        sock.close()
        raise
    return sock


def bound_address(sock: socket.socket) -> str:
    """`host:port` actually bound (resolves port 0)."""
    host, port = sock.getsockname()[:2]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


class _Server(uvicorn.Server):
    """uvicorn server whose signals are handled by the composition root."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """No-op: `serve`/`run_worker` own SIGTERM/SIGINT."""
        yield


def _server(app: FastAPI, settings: Settings) -> _Server:
    return _Server(
        uvicorn.Config(
            app,
            log_config=None,
            log_level="warning",
            access_log=False,
            lifespan="off",
            server_header=False,
            timeout_graceful_shutdown=math.ceil(settings.shutdown_timeout),
        )
    )


def worker_id() -> str:
    """Unique id of this process's worker (`hostname-pid-random`)."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def make_worker(rt: Runtime) -> Worker:
    """The queue worker over every job handler (`ingest.*` settings)."""
    i = rt.settings.ingest
    return Worker(
        rt.jobs,
        rt.handlers,
        WorkerConfig(
            worker_id=worker_id(),
            concurrency=i.workers,
            lease=timedelta(seconds=i.job_lease),
            max_attempts=i.job_max_attempts,
            poll_interval=i.poll_interval,
            shutdown_timeout=rt.settings.shutdown_timeout,
        ),
        log=make_logger(rt.settings.log, component="queue"),
        on_outcome=rt.metrics.observe_job,
    )


async def report_queue_depth(rt: Runtime, stop: asyncio.Event, interval: float) -> None:
    """Refresh `footnote_jobs_queue_depth{kind}` from the admin pool until `stop`."""
    while not stop.is_set():
        try:
            depth = await pg_health.queue_depth(rt.pools.admin)
        except Exception as exc:  # a DB blip must not stop the worker
            rt.log.warning("queue depth query failed", error=str(exc))
        else:
            for kind in JobKind:
                rt.metrics.jobs_queue_depth.labels(kind.value).set(depth.get(kind.value, 0))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)


async def _run_worker_tasks(rt: Runtime, stop: asyncio.Event) -> None:
    worker = make_worker(rt)
    depth = asyncio.create_task(report_queue_depth(rt, stop, QUEUE_DEPTH_INTERVAL))
    try:
        await worker.run(stop)
    finally:
        stop.set()
        await depth


type ReadyCallback = Callable[[Mapping[str, str]], None]


@dataclass(frozen=True, slots=True)
class ServeOptions:
    """What `run_process` serves: the data plane (serve) and/or the worker."""

    data_plane: bool = True
    worker: bool = False
    on_ready: ReadyCallback | None = None


async def run_process(
    settings: Settings,
    options: ServeOptions,
    stop: asyncio.Event,
    env: Mapping[str, str] | None = None,
) -> None:
    """Serve until `stop` is set, then shut down gracefully."""
    tracing = setup_tracing(version=build_info(env).version, env=env)
    try:
        async with open_runtime(settings, env=env) as rt:
            await _serve_runtime(rt, options, stop)
    finally:
        tracing.shutdown()


async def _serve_runtime(rt: Runtime, options: ServeOptions, stop: asyncio.Event) -> None:
    settings = rt.settings
    admin_sock = bind(settings.admin.listen)
    data_sock = bind(settings.server.listen) if options.data_plane else None
    admin_app = create_admin_app(rt.api)
    admin_server = _server(admin_app, settings)
    apps = [admin_app]
    servers: list[tuple[_Server, asyncio.Task[None]]] = []
    if data_sock is not None:
        data_app = create_app(rt.api)
        apps.append(data_app)
        data_server = _server(data_app, settings)
        servers.append((data_server, asyncio.create_task(data_server.serve([data_sock]))))
    worker_stop = asyncio.Event()
    worker = asyncio.create_task(_run_worker_tasks(rt, worker_stop)) if options.worker else None
    admin_task = asyncio.create_task(admin_server.serve([admin_sock]))
    addresses = {"admin": bound_address(admin_sock)}
    if data_sock is not None:
        addresses["data"] = bound_address(data_sock)
    for name, address in addresses.items():
        rt.log.info(f"{name} listener on http://{address}")
    if options.on_ready is not None:
        options.on_ready(addresses)
    watched: set[asyncio.Future[Any]] = {admin_task, *(t for _, t in servers)}
    if worker is not None:
        watched.add(worker)
    stopper = asyncio.create_task(stop.wait())
    await asyncio.wait({stopper, *watched}, return_when=asyncio.FIRST_COMPLETED)
    stopper.cancel()
    rt.log.info("shutting down")
    await _shutdown(apps, servers, worker, worker_stop, (admin_server, admin_task))


async def _shutdown(
    apps: Sequence[FastAPI],
    servers: Sequence[tuple[_Server, asyncio.Task[None]]],
    worker: asyncio.Task[None] | None,
    worker_stop: asyncio.Event,
    admin: tuple[_Server, asyncio.Task[None]],
) -> None:
    """Drain order: data plane and worker first (bounded), the admin listener last."""
    for app in apps:
        app.state.draining = True
    for server, _ in servers:
        server.should_exit = True
    worker_stop.set()
    pending: list[asyncio.Task[None]] = [t for _, t in servers]
    if worker is not None:
        pending.append(worker)
    results = await asyncio.gather(*pending, return_exceptions=True)
    admin[0].should_exit = True
    results.append((await asyncio.gather(admin[1], return_exceptions=True))[0])
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


def is_domain_error(exc: BaseException) -> bool:
    """Startup failures shown as one line (no traceback) by the CLI."""
    return isinstance(exc, DomainError | OSError)
