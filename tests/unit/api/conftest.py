# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest_asyncio

from footnote.api import ApiDeps, ApiSettings, ReadinessCheck, VersionInfo
from footnote.api.admin_listener import create_admin_app
from footnote.api.app import create_app
from footnote.app.admin import AdminConfig, AdminDeps, AdminService
from footnote.app.answer import AnswerService
from footnote.app.collections import CollectionConfig, CollectionDeps, CollectionService
from footnote.app.documents import DocumentConfig, DocumentDeps, DocumentService
from footnote.app.reembed import ReembedService
from footnote.app.search import SearchService
from footnote.config import LogConfig
from footnote.domain.documents import CollectionRecord, NewCollection
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.settings import CollectionSettings
from footnote.domain.tenancy import Quotas, TenantId
from footnote.observability import Metrics, make_logger
from tests.unit.app.fakes import (
    DIMS,
    FakeClock,
    FakeCollection,
    FakeCollections,
    FakeDb,
    FakeDocuments,
    FakeEmbedderFactory,
    FakeIndex,
    FakeJobs,
    FakeLLM,
    FakeReranker,
)

from .fakes import (
    FakeChunkReader,
    FakeDeliveries,
    FakeExporter,
    FakeInspector,
    FakeKeys,
    FakeResolver,
    FakeSpaceRegistry,
    FakeTenantAdmin,
    TenantSearchStore,
)

ADMIN_KEY = "admin-secret-key"
SPACE_A = EmbeddingSpace(1, "space-a", "fake", "model-a", DIMS, calibrated_threshold=0.1)
SPACE_B = EmbeddingSpace(2, "space-b", "fake", "model-b", DIMS)
MAX_FILE_BYTES = 4096
ANSWER_TEXT = "Beta is gamma [1]."


@dataclass
class ApiRig:
    db: FakeDb
    keys: FakeKeys
    tenants: FakeTenantAdmin
    collections: FakeCollections
    documents: FakeDocuments
    index: FakeIndex
    jobs: FakeJobs
    spaces: FakeSpaceRegistry
    exporter: FakeExporter
    resolver: FakeResolver
    deliveries: FakeDeliveries
    llm: FakeLLM
    search_collection: FakeCollection
    deps: ApiDeps
    log: io.StringIO
    tenant_a: TenantId
    tenant_b: TenantId
    key_a: str
    key_b: str
    collection_a: CollectionRecord
    collection_b: CollectionRecord
    readiness_problems: list[str] = field(default_factory=list)
    client: httpx.AsyncClient = field(init=False)
    admin_client: httpx.AsyncClient = field(init=False)

    def auth(self, key: str | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {key or self.key_a}"}

    @property
    def admin(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {ADMIN_KEY}"}

    def col_url(self, record: CollectionRecord | None = None, suffix: str = "") -> str:
        return f"/v1/collections/{(record or self.collection_a).id}{suffix}"


def build_rig(  # noqa: PLR0913 - test knobs
    *,
    quotas: Quotas | None = None,
    llm: FakeLLM | None = None,
    admin_key: str = ADMIN_KEY,
    access_log: bool = False,
    max_json_bytes: int = 64 * 1024,
    settings: CollectionSettings | None = None,
) -> ApiRig:
    db = FakeDb()
    db.add_space(SPACE_A)
    db.add_space(SPACE_B)
    keys = FakeKeys(db)
    tenants = FakeTenantAdmin(db, keys)
    tenant_a = db.add_tenant(quotas)
    tenant_b = db.add_tenant(quotas)
    collections = FakeCollections(db)
    documents = FakeDocuments(db)
    index = FakeIndex(db)
    jobs = FakeJobs(db)
    spaces = FakeSpaceRegistry(db)
    exporter = FakeExporter(db)
    resolver = FakeResolver()
    deliveries = FakeDeliveries()
    clock = FakeClock()
    embedders = FakeEmbedderFactory()
    base = settings or CollectionSettings(embedding_space=SPACE_A.name)
    collection_a = collections.create_now(tenant_a, NewCollection("docs", base, SPACE_A.id))
    collection_b = collections.create_now(tenant_b, NewCollection("docs", base, SPACE_A.id))
    reembed = ReembedService(collections, index, spaces, embedders, jobs)
    search_collection = FakeCollection(SPACE_A, base)
    search_collection.add("alpha beta gamma", title="Greek")
    search_collection.add("delta epsilon", title="More")
    store = TenantSearchStore(db, search_collection)
    search = SearchService(store, embedders, FakeReranker(calibrated_threshold=0.5), clock)
    llm = llm or FakeLLM(ANSWER_TEXT)
    log = io.StringIO()
    problems: list[str] = []

    async def probe() -> list[str]:
        return list(problems)

    deps = ApiDeps(
        collections=CollectionService(
            CollectionDeps(collections, jobs, spaces, reembed, resolver, deliveries, tenants),
            CollectionConfig(defaults=base, tokenizer_ids=frozenset({"bge-small-en-v1.5"})),
        ),
        documents=DocumentService(
            DocumentDeps(
                collections, documents, index, FakeChunkReader(db), FakeInspector(), tenants
            ),
            DocumentConfig(max_file_bytes=MAX_FILE_BYTES),
        ),
        admin=AdminService(
            AdminDeps(tenants, spaces, exporter, clock),
            AdminConfig(default_quotas=Quotas(max_collections=10)),
        ),
        search=search,
        answers=AnswerService(search, llm),
        jobs=jobs,
        keys=keys,
        metrics=Metrics(version="test"),
        logger=make_logger(LogConfig(), component="api", stream=log, strict=True),
        settings=ApiSettings(
            admin_key=admin_key,
            version=VersionInfo("1.2.3", "abc123", "2026-10-20"),
            access_log=access_log,
            max_json_bytes=max_json_bytes,
            multipart_overhead_bytes=8 * 1024,
        ),
        readiness=(ReadinessCheck("fake", probe),),
    )
    rig = ApiRig(
        db=db,
        keys=keys,
        tenants=tenants,
        collections=collections,
        documents=documents,
        index=index,
        jobs=jobs,
        spaces=spaces,
        exporter=exporter,
        resolver=resolver,
        deliveries=deliveries,
        llm=llm,
        search_collection=search_collection,
        deps=deps,
        log=log,
        tenant_a=tenant_a,
        tenant_b=tenant_b,
        key_a=keys.issue(tenant_a).key,
        key_b=keys.issue(tenant_b).key,
        collection_a=collection_a,
        collection_b=collection_b,
        readiness_problems=problems,
    )
    rig.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(deps)), base_url="http://test"
    )
    rig.admin_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_admin_app(deps)), base_url="http://admin"
    )
    return rig


@pytest_asyncio.fixture
async def rig() -> AsyncIterator[ApiRig]:
    r = build_rig()
    async with r.client, r.admin_client:
        yield r


def error_of(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert set(body) == {"error"}, body
    error: dict[str, Any] = body["error"]
    assert {"code", "message", "request_id"} <= set(error), error
    assert error["request_id"] == response.headers["x-request-id"]
    return error
