# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
import pytest
import pytest_asyncio

from footnote.adapters.parsers import ParseLimits, SandboxedParser, UploadChecks
from footnote.adapters.postgres.chunks import PgChunks
from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.export import PgTenantExport
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.search import PgSearchStore
from footnote.adapters.postgres.session import Database, Pool
from footnote.adapters.postgres.spaces import HnswParams, PgSpaceAdmin, PgSpaceReader
from footnote.adapters.postgres.tenants import PgKeyResolver, PgTenantAdmin
from footnote.adapters.postgres.webhook_deliveries import PgWebhookDeliveries
from footnote.api import ApiDeps, ApiSettings, create_app
from footnote.app.admin import AdminDeps, AdminService
from footnote.app.answer import AnswerService
from footnote.app.collections import CollectionConfig, CollectionDeps, CollectionService
from footnote.app.documents import DocumentDeps, DocumentService
from footnote.app.ingest import IngestDeps, IngestService, IngestTarget
from footnote.app.reembed import ReembedService
from footnote.app.search import SearchService
from footnote.app.webhooks import WebhookNotifier
from footnote.config import LogConfig
from footnote.domain.documents import CollectionId, DocumentId, JobId, VersionId
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.ports import Embedder, TokenCounter
from footnote.domain.settings import ChunkingSettings, ChunkingStrategyName, CollectionSettings
from footnote.domain.tenancy import TenantId
from footnote.observability import Metrics, make_logger
from tests.integration.world import SPACE_NAME, hash_vector
from tests.unit.app.fakes import FakeLLM

pytestmark = pytest.mark.integration

ADMIN_KEY = "it-admin-key"
DOCUMENT = b"# Orchards\n\nApple trees bloom in spring.\n\n## Harvest\n\nPears ripen in autumn.\n"


class _Clock:
    def now(self) -> float:
        return 1_760_000_000.0

    def monotonic(self) -> float:
        return asyncio.get_running_loop().time()


class _Words:
    def count(self, text: str) -> int:
        return len(text.split())


class _Tokenizers:
    def for_tokenizer(self, tokenizer_id: str) -> TokenCounter:
        return _Words()


class _Embedder:
    def __init__(self, space: EmbeddingSpace) -> None:
        self._space = space

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [hash_vector(f"{self._space.name}:{t}", self._space.dimensions) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return hash_vector(f"{self._space.name}:{text}", self._space.dimensions)


class _Embedders:
    def for_space(self, space: EmbeddingSpace) -> Embedder:
        return _Embedder(space)


class _Resolver:
    async def resolve(self, host: str, port: int) -> Sequence[str]:
        return ["93.184.216.34"]


@dataclass
class Stack:
    client: httpx.AsyncClient
    ingest: IngestService
    jobs: PgJobs

    async def run_ingest(self, tenant: TenantId, job_id: str) -> None:
        job = await self.jobs.get(tenant, JobId(uuid.UUID(job_id)))
        args = job.args
        await self.ingest.ingest(
            tenant,
            IngestTarget(
                CollectionId(uuid.UUID(args["collection_id"])),
                DocumentId(uuid.UUID(args["document_id"])),
                VersionId(uuid.UUID(args["version_id"])),
            ),
        )


def wire(app_db: Database, admin_db: Database) -> tuple[ApiDeps, IngestService, PgJobs]:
    collections = PgCollections(app_db)
    documents = PgDocuments(app_db)
    index = PgIndex(app_db)
    jobs = PgJobs(app_db)
    spaces = PgSpaceReader(app_db)
    tenants = PgTenantAdmin(admin_db)
    embedders = _Embedders()
    clock = _Clock()
    reembed = ReembedService(collections, index, spaces, embedders, jobs)
    defaults = CollectionSettings(
        embedding_space=SPACE_NAME,
        chunking=ChunkingSettings(
            strategy=ChunkingStrategyName.HEADING, max_tokens=64, overlap_tokens=0, min_tokens=0
        ),
    )
    search = SearchService(PgSearchStore(app_db), embedders, None, clock)
    deps = ApiDeps(
        collections=CollectionService(
            CollectionDeps(
                collections,
                jobs,
                spaces,
                reembed,
                _Resolver(),
                PgWebhookDeliveries(app_db),
                tenants,
            ),
            CollectionConfig(defaults=defaults),
        ),
        documents=DocumentService(
            DocumentDeps(
                collections,
                documents,
                index,
                PgChunks(app_db),
                UploadChecks(ParseLimits()),
                tenants,
            )
        ),
        admin=AdminService(
            AdminDeps(
                tenants, PgSpaceAdmin(admin_db, HnswParams()), PgTenantExport(admin_db), clock
            )
        ),
        search=search,
        answers=AnswerService(search, FakeLLM("Apple trees bloom [1].")),
        jobs=jobs,
        keys=PgKeyResolver(app_db),
        metrics=Metrics(version="it"),
        logger=make_logger(LogConfig(), component="api", stream=io.StringIO(), strict=True),
        settings=ApiSettings(admin_key=ADMIN_KEY),
    )
    ingest = IngestService(
        IngestDeps(
            collections,
            documents,
            index,
            spaces,
            SandboxedParser(ParseLimits(parse_timeout=60)),
            embedders,
            _Tokenizers(),
            WebhookNotifier(jobs, clock),
            clock,
            tenants,
        )
    )
    return deps, ingest, jobs


@pytest.fixture(params=[True, False], ids=["predicates", "rls-only"])
def app_db_mode(app_pool: Pool, request: pytest.FixtureRequest) -> Database:
    return Database(app_pool, app_tenant_predicates=bool(request.param))


@pytest_asyncio.fixture
async def stack(app_db_mode: Database, admin_db: Database) -> AsyncIterator[Stack]:
    deps, ingest, jobs = wire(app_db_mode, admin_db)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://it") as client:
        yield Stack(client, ingest, jobs)


ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def _tenant(stack: Stack, name: str) -> tuple[str, str, str]:
    created = await stack.client.post(
        "/v1/admin/tenants", headers=ADMIN, json={"name": f"{name}-{uuid.uuid4().hex[:6]}"}
    )
    assert created.status_code == 201, created.text
    tenant_id = created.json()["id"]
    key = await stack.client.post(
        f"/v1/admin/tenants/{tenant_id}/keys", headers=ADMIN, json={"name": "ci"}
    )
    assert key.status_code == 201, key.text
    return tenant_id, key.json()["key"], key.json()["id"]


@dataclass
class Seeded:
    tenant: str
    key: str
    key_id: str
    collection: str
    document: str
    job: str
    chunk: str


async def _seed(stack: Stack, name: str) -> Seeded:
    tenant, key, key_id = await _tenant(stack, name)
    col = await stack.client.post("/v1/collections", headers=bearer(key), json={"name": "orchard"})
    assert col.status_code == 201, col.text
    cid = col.json()["id"]
    up = await stack.client.post(
        f"/v1/collections/{cid}/documents",
        headers=bearer(key),
        files={"file": ("orchard.md", DOCUMENT, "application/pdf")},
        data={"external_id": "orchard", "metadata": orjson.dumps({"team": name}).decode()},
    )
    assert up.status_code == 202, up.text
    await stack.run_ingest(TenantId(uuid.UUID(tenant)), up.json()["job_id"])
    hits = await stack.client.post(
        f"/v1/collections/{cid}/search", headers=bearer(key), json={"query": "apple trees"}
    )
    assert hits.status_code == 200, hits.text
    assert hits.json()["hits"], hits.text
    return Seeded(
        tenant,
        key,
        key_id,
        cid,
        up.json()["document_id"],
        up.json()["job_id"],
        hits.json()["hits"][0]["chunk_id"],
    )


def _cross_requests(s: Seeded) -> list[tuple[str, str, dict[str, Any]]]:
    col = f"/v1/collections/{s.collection}"
    doc = f"{col}/documents/{s.document}"
    return [
        ("GET", col, {}),
        ("PATCH", col, {"json": {"chunking": {"max_tokens": 128}}}),
        ("DELETE", col, {}),
        ("GET", f"{col}/webhook-deliveries", {}),
        ("GET", f"{col}/documents", {}),
        ("POST", f"{col}/documents", {"files": {"file": ("x.txt", b"intruder text")}}),
        ("GET", doc, {}),
        ("PATCH", doc, {"json": {"metadata": {"owned": True}}}),
        ("DELETE", doc, {}),
        ("GET", f"{doc}/text", {}),
        ("GET", f"{doc}/original", {}),
        ("GET", f"{doc}/versions/1/original", {}),
        ("GET", f"{col}/chunks/{s.chunk}", {}),
        ("POST", f"{col}/search", {"json": {"query": "apple"}}),
        ("POST", f"{col}/answers", {"json": {"question": "apple trees"}}),
        ("POST", f"{col}/answers", {"json": {"question": "apple trees", "stream": True}}),
        ("GET", f"/v1/jobs/{s.job}", {}),
    ]


async def test_owner_reads_everything_it_uploaded(stack: Stack) -> None:
    a = await _seed(stack, "owner")
    headers = bearer(a.key)
    col = f"/v1/collections/{a.collection}"
    doc = await stack.client.get(f"{col}/documents/{a.document}", headers=headers)
    assert doc.status_code == 200
    body = doc.json()
    assert (body["status"], body["current_version"], body["chunk_count"]) == ("ready", 1, 2)
    assert body["versions"][0]["media_type"] == "text/markdown"
    assert body["versions"][0]["sha256"] == hashlib.sha256(DOCUMENT).hexdigest()
    text = await stack.client.get(f"{col}/documents/{a.document}/text", headers=headers)
    assert "Apple trees bloom in spring." in text.text
    original = await stack.client.get(f"{col}/documents/{a.document}/original", headers=headers)
    assert original.content == DOCUMENT
    assert original.headers["content-type"].startswith("text/markdown")
    chunk = await stack.client.get(f"{col}/chunks/{a.chunk}", headers=headers)
    assert chunk.status_code == 200
    assert chunk.json()["memberships"][0]["version"] == 1
    job = await stack.client.get(f"/v1/jobs/{a.job}", headers=headers)
    assert (job.status_code, job.json()["kind"]) == (200, "ingest")
    same = await stack.client.post(
        f"{col}/documents",
        headers=headers,
        files={"file": ("orchard.md", DOCUMENT)},
        data={"external_id": "orchard"},
    )
    assert (same.status_code, same.json()["unchanged"]) == (200, True)


async def test_other_tenant_gets_404_everywhere_and_changes_nothing(stack: Stack) -> None:
    a = await _seed(stack, "victim")
    b_tenant, b_key, _ = await _tenant(stack, "intruder")
    before = await stack.client.get(
        f"/v1/collections/{a.collection}/documents/{a.document}", headers=bearer(a.key)
    )
    for method, url, kwargs in _cross_requests(a):
        response = await stack.client.request(method, url, headers=bearer(b_key), **kwargs)
        assert response.status_code == 404, (method, url, response.text)
        assert response.json()["error"]["code"] in {"not_found", "chunk_not_found"}
    after = await stack.client.get(
        f"/v1/collections/{a.collection}/documents/{a.document}", headers=bearer(a.key)
    )
    assert after.json() == before.json()
    collection = await stack.client.get(f"/v1/collections/{a.collection}", headers=bearer(a.key))
    assert collection.json()["chunking"]["max_tokens"] == 64
    assert collection.json()["status"] == "active"
    listing = await stack.client.get("/v1/collections", headers=bearer(b_key))
    assert listing.json()["data"] == []
    usage = await stack.client.get(f"/v1/admin/tenants/{b_tenant}/usage", headers=ADMIN)
    assert usage.json() == {
        "collections": 0,
        "documents": 0,
        "chunks": 0,
        "vectors": 0,
        "stored_bytes": 0,
    }


async def test_revoked_rotated_and_deleted_keys(stack: Stack) -> None:
    a = await _seed(stack, "keys")
    rotated = await stack.client.post(
        f"/v1/admin/tenants/{a.tenant}/keys/{a.key_id}/rotate", headers=ADMIN
    )
    assert rotated.status_code == 201
    new_key = rotated.json()["key"]
    for key in (a.key, new_key):
        ok = await stack.client.get("/v1/collections", headers=bearer(key))
        assert ok.status_code == 200
    revoked = await stack.client.delete(
        f"/v1/admin/tenants/{a.tenant}/keys/{a.key_id}", headers=ADMIN
    )
    assert revoked.status_code == 204
    gone = await stack.client.get("/v1/collections", headers=bearer(a.key))
    assert (gone.status_code, gone.json()["error"]["code"]) == (401, "unauthenticated")
    keys = await stack.client.get(f"/v1/admin/tenants/{a.tenant}/keys", headers=ADMIN)
    revoked_flags = {k["id"]: k["revoked"] for k in keys.json()["data"]}
    assert revoked_flags[a.key_id] is True
    assert all("key" not in k for k in keys.json()["data"])
    deleted = await stack.client.delete(f"/v1/admin/tenants/{a.tenant}", headers=ADMIN)
    assert deleted.status_code == 202
    job = await stack.jobs.get(
        TenantId(uuid.UUID(a.tenant)), JobId(uuid.UUID(deleted.json()["job_id"]))
    )
    assert job.kind == "tenant_delete"
    after = await stack.client.get("/v1/collections", headers=bearer(new_key))
    assert after.status_code == 401
    tenant = await stack.client.get(f"/v1/admin/tenants/{a.tenant}", headers=ADMIN)
    assert tenant.json()["status"] == "deleting"


async def test_tenant_key_cannot_use_admin_api_and_export_is_scoped(stack: Stack) -> None:
    a = await _seed(stack, "export-a")
    b = await _seed(stack, "export-b")
    denied = await stack.client.get(f"/v1/admin/tenants/{a.tenant}/export", headers=bearer(a.key))
    assert denied.status_code == 401
    export = await stack.client.get(
        f"/v1/admin/tenants/{a.tenant}/export?include_originals=true", headers=ADMIN
    )
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/x-ndjson")
    lines = [orjson.loads(line) for line in export.content.splitlines()]
    assert lines[0]["type"] == "tenant"
    assert {line["type"] for line in lines} >= {"collection", "document", "chunk", "version"}
    text = export.text
    assert a.collection in text
    assert b.collection not in text
    assert b.tenant not in text
    missing = await stack.client.get(f"/v1/admin/tenants/{uuid.uuid4()}/export", headers=ADMIN)
    assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")
