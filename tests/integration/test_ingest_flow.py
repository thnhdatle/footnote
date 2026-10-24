# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import itertools
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from footnote.adapters.parsers import SandboxedParser, sniff
from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.search import PgSearchStore
from footnote.adapters.postgres.session import Database, fetch_all, fetch_one
from footnote.adapters.postgres.spaces import HnswParams, PgSpaceAdmin, PgSpaceReader
from footnote.adapters.postgres.tenants import PgTenantAdmin
from footnote.adapters.queue import JobContext, Worker, WorkerConfig
from footnote.app.gc import GcService
from footnote.app.ingest import IngestConfig, IngestDeps, IngestService
from footnote.app.jobs import JobHandlers, JobServices
from footnote.app.reembed import ReembedConfig, ReembedService
from footnote.app.reindex import ReindexService
from footnote.app.search import SearchRequest, SearchService
from footnote.app.webhooks import WebhookNotifier
from footnote.domain.documents import CollectionRecord, DocumentId, NewCollection, NewUpload
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec
from footnote.domain.jobs import JobRecord
from footnote.domain.ports import Embedder, TokenCounter
from footnote.domain.settings import (
    ChunkingSettings,
    ChunkingStrategyName,
    CollectionSettings,
    SearchMode,
)
from footnote.domain.tenancy import Quotas, TenantId
from tests.integration.world import DIMS, SPACE_NAME, hash_vector

pytestmark = pytest.mark.integration

GENERATOR = Path(__file__).resolve().parents[1] / "fixtures" / "make_fixtures.py"
SPACE_B = "it-fake-8-b"
CHUNKING = ChunkingSettings(
    strategy=ChunkingStrategyName.HEADING, max_tokens=64, overlap_tokens=0, min_tokens=0
)


class Clock:
    def now(self) -> float:
        return 1_760_000_000.0

    def monotonic(self) -> float:
        return asyncio.get_running_loop().time()


class Words:
    def count(self, text: str) -> int:
        return len(text.split())


class Tokenizers:
    def for_tokenizer(self, tokenizer_id: str) -> TokenCounter:
        return Words()


@dataclass
class CountingEmbedders:
    texts: list[tuple[int, str]] = field(default_factory=list)

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        return _SpaceEmbedder(space, self)


class _SpaceEmbedder:
    def __init__(self, space: EmbeddingSpace, owner: CountingEmbedders) -> None:
        self._space = space
        self._owner = owner

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        self._owner.texts.extend((self._space.id, t) for t in texts)
        return [hash_vector(f"{self._space.name}:{t}", self._space.dimensions) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return hash_vector(f"{self._space.name}:{text}", self._space.dimensions)


def _fixture(tmp: Path, name: str) -> bytes:
    spec = importlib.util.spec_from_file_location("make_fixtures_ingest", GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures_ingest"] = module
    spec.loader.exec_module(module)
    module.generate(tmp)
    return (tmp / name).read_bytes()


@dataclass
class Flow:
    tenant: TenantId
    collection: CollectionRecord
    space_a: EmbeddingSpace
    space_b: EmbeddingSpace
    documents: PgDocuments
    jobs: PgJobs
    reembed: ReembedService
    search: SearchService
    embedders: CountingEmbedders
    webhooks: list[dict[str, Any]]
    worker: Worker

    def upload(self, data: bytes, filename: str, title: str = "Fixture") -> NewUpload:
        return NewUpload(
            collection_id=self.collection.id,
            external_id=filename,
            title=title,
            metadata={"source": filename},
            media_type=sniff(data, filename),
            source_sha256=hashlib.sha256(data).digest(),
            data=data,
            chunking_hash=CHUNKING.settings_hash(),
        )


@pytest_asyncio.fixture
async def flow(app_db: Database, admin_db: Database) -> AsyncIterator[Flow]:
    admin = PgTenantAdmin(admin_db)
    tenant = (await admin.create_tenant(f"flow-{uuid.uuid4().hex[:8]}", Quotas())).id
    space_admin = PgSpaceAdmin(admin_db, HnswParams())
    space_b = await space_admin.register(SpaceSpec(SPACE_B, "fake", "fake-model-b", DIMS))
    spaces = PgSpaceReader(app_db)
    space_a = await spaces.get(SPACE_NAME)
    collections = PgCollections(app_db)
    collection = await collections.create(
        tenant,
        NewCollection(
            "flow",
            CollectionSettings(chunking=CHUNKING, embedding_space=space_a.name),
            space_a.id,
            webhook_url="https://example.com/hook",
            webhook_secret="s3cret",
        ),
    )
    documents, index, jobs = PgDocuments(app_db), PgIndex(app_db), PgJobs(app_db)
    embedders = CountingEmbedders()
    clock = Clock()
    ingest = IngestService(
        IngestDeps(
            collections,
            documents,
            index,
            spaces,
            SandboxedParser(),
            embedders,
            Tokenizers(),
            WebhookNotifier(jobs, clock),
            clock,
            admin,
        ),
        IngestConfig(gc_grace=timedelta(minutes=10), batch_size=4),
    )
    reembed = ReembedService(
        collections, index, spaces, embedders, jobs, config=ReembedConfig(batch_size=2)
    )
    services = JobServices(
        jobs,
        collections,
        ingest,
        ReindexService(collections, documents),
        reembed,
        GcService(index),
        admin,
    )
    webhooks: list[dict[str, Any]] = []

    async def webhook(ctx: JobContext) -> None:
        record = await jobs.get(ctx.job.tenant_id, ctx.job.id)
        webhooks.append(record.args)

    handlers: dict[str, Any] = {**JobHandlers(services).handlers(), "webhook": webhook}
    worker = Worker(
        jobs,
        handlers,
        WorkerConfig(worker_id=f"flow-{uuid.uuid4().hex[:6]}", poll_interval=0.05),
    )
    search = SearchService(PgSearchStore(app_db), embedders, None, clock)
    yield Flow(
        tenant,
        collection,
        space_a,
        space_b,
        documents,
        jobs,
        reembed,
        search,
        embedders,
        webhooks,
        worker,
    )


async def _until_done(flow: Flow, job_id: Any) -> JobRecord:
    stop = asyncio.Event()
    task = asyncio.create_task(flow.worker.run(stop))
    try:
        async with asyncio.timeout(60):
            while True:
                record = await flow.jobs.get(flow.tenant, job_id)
                if record.status in ("done", "failed"):
                    return record
                await asyncio.sleep(0.02)
    finally:
        stop.set()
        await task


async def _current_chunks(db: Database, tenant: TenantId, document_id: DocumentId) -> list[str]:
    async with db.tenant_tx(tenant) as conn:
        rows = await fetch_all(
            conn,
            "SELECT c.embedded_text FROM footnote.documents d "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
            "AND vc.version_id = d.current_version_id "
            "JOIN footnote.chunks c ON c.tenant_id = vc.tenant_id AND c.id = vc.chunk_id "
            "WHERE d.tenant_id = %(t)s AND d.id = %(d)s ORDER BY vc.ordinal",
            {"t": tenant, "d": document_id},
        )
    return [r["embedded_text"] for r in rows]


async def test_ingest_incremental_and_reembed_with_switch(
    flow: Flow, app_db: Database, tmp_path: Path
) -> None:
    pdf = await _ingest_pdf(flow, app_db, tmp_path)
    await _identical_reupload(flow, pdf)
    md2 = await _incremental_markdown(flow, app_db, tmp_path)
    reembed_job = await _reembed_while_searching(flow)
    await _assert_switched(flow, app_db, reembed_job)
    embedded_before = len(flow.embedders.texts)
    md3 = md2 + b"\n\nOne more paragraph after the switch.\n"
    up3 = await flow.documents.upload(flow.tenant, flow.upload(md3, "simple.md"))
    assert (await _until_done(flow, up3.job_id)).status == "done"
    assert {s for s, _ in flow.embedders.texts[embedded_before:]} == {flow.space_b.id}


async def _ingest_pdf(flow: Flow, app_db: Database, tmp_path: Path) -> bytes:
    pdf = _fixture(tmp_path, "long.pdf")
    first = await flow.documents.upload(flow.tenant, flow.upload(pdf, "long.pdf"))
    assert first.job_id is not None
    record = await _until_done(flow, first.job_id)
    assert (record.status, record.last_error) == ("done", None)

    doc = await flow.documents.get(flow.tenant, flow.collection.id, first.document_id)
    assert doc.status.value == "ready"
    assert doc.current_version == 1
    chunks = await _current_chunks(app_db, flow.tenant, first.document_id)
    assert len(chunks) >= 3
    distinct = set(chunks)
    assert sorted(t for _, t in flow.embedders.texts) == sorted(distinct)
    version = await flow.documents.version(flow.tenant, first.version_id)  # type: ignore[arg-type]
    assert version.pages is not None
    assert version.pages > 1
    elements = await flow.documents.elements(flow.tenant, first.version_id)  # type: ignore[arg-type]
    assert elements
    assert [w["event"] for w in flow.webhooks] == ["document.ready"]
    assert flow.webhooks[0]["version"] == 1

    return pdf


async def _identical_reupload(flow: Flow, pdf: bytes) -> None:
    embedded_before = len(flow.embedders.texts)
    same = await flow.documents.upload(flow.tenant, flow.upload(pdf, "long.pdf"))
    assert same.unchanged
    assert same.job_id is None
    assert len(flow.embedders.texts) == embedded_before


async def _incremental_markdown(flow: Flow, app_db: Database, tmp_path: Path) -> bytes:
    md = _fixture(tmp_path / "md", "simple.md")
    up1 = await flow.documents.upload(flow.tenant, flow.upload(md, "simple.md"))
    assert (await _until_done(flow, up1.job_id)).status == "done"
    old_chunks = set(await _current_chunks(app_db, flow.tenant, up1.document_id))
    embedded_before = len(flow.embedders.texts)
    md2 = md + b"\n\n## Added section\n\nA brand new closing paragraph about footnotes.\n"
    up2 = await flow.documents.upload(flow.tenant, flow.upload(md2, "simple.md"))
    assert up2.version == 2
    assert (await _until_done(flow, up2.job_id)).status == "done"
    new_chunks = set(await _current_chunks(app_db, flow.tenant, up2.document_id))
    added = new_chunks - old_chunks
    assert added
    assert sorted(t for _, t in flow.embedders.texts[embedded_before:]) == sorted(added)

    return md2


async def _reembed_while_searching(flow: Flow) -> Any:
    reembed_job = await flow.reembed.change_space(flow.tenant, flow.collection.id, SPACE_B)
    assert reembed_job is not None
    seen: list[int] = []
    stop_search = asyncio.Event()

    async def searching() -> None:
        while not stop_search.is_set():
            result = await flow.search.search(
                flow.tenant,
                flow.collection.id,
                SearchRequest("footnotes section", mode=SearchMode.HYBRID, rerank=False),
            )
            assert result.hits
            seen.append(result.space.id)
            await asyncio.sleep(0)

    searcher = asyncio.create_task(searching())
    try:
        record = await _until_done(flow, reembed_job)
        async with asyncio.timeout(20):
            while not seen or seen[-1] != flow.space_b.id:
                await asyncio.sleep(0.01)
    finally:
        stop_search.set()
        await searcher
    assert (record.status, record.last_error) == ("done", None)
    assert seen
    assert seen[0] == flow.space_a.id
    assert seen[-1] == flow.space_b.id
    flips = sum(1 for a, b in itertools.pairwise(seen) if a != b)
    assert flips == 1
    assert set(seen) <= {flow.space_a.id, flow.space_b.id}
    return reembed_job


async def _assert_switched(flow: Flow, app_db: Database, reembed_job: Any) -> None:
    async with app_db.tenant_tx(flow.tenant) as conn:
        col = await fetch_one(
            conn,
            "SELECT active_space_id, target_space_id, reembed FROM footnote.collections "
            "WHERE tenant_id = %(t)s AND id = %(c)s",
            {"t": flow.tenant, "c": flow.collection.id},
        )
        lacking = await fetch_one(
            conn,
            "SELECT count(*) AS n FROM footnote.documents d "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
            "AND vc.version_id = d.current_version_id "
            "WHERE d.tenant_id = %(t)s AND d.collection_id = %(c)s AND NOT EXISTS ("
            "SELECT 1 FROM footnote.chunk_vectors v WHERE v.tenant_id = vc.tenant_id "
            "AND v.chunk_id = vc.chunk_id AND v.space_id = %(s)s)",
            {"t": flow.tenant, "c": flow.collection.id, "s": flow.space_b.id},
        )
        gc = await fetch_all(
            conn,
            "SELECT args, run_at > now() + interval '5 minutes' AS delayed FROM footnote.jobs "
            "WHERE tenant_id = %(t)s AND kind = 'gc' AND args ? 'space_id'",
            {"t": flow.tenant},
        )
    assert col is not None
    assert (col["active_space_id"], col["target_space_id"]) == (flow.space_b.id, None)
    assert col["reembed"]["state"] == "done"
    assert col["reembed"]["done"] == col["reembed"]["total"]
    assert lacking is not None
    assert lacking["n"] == 0
    assert [(g["args"]["space_id"], g["delayed"]) for g in gc] == [(flow.space_a.id, True)]
    job = await flow.jobs.get(flow.tenant, reembed_job)
    assert job.progress is not None
    assert job.progress["done"] == job.progress["total"]


async def test_parse_failure_marks_document_failed(flow: Flow, app_db: Database) -> None:
    data = b"%PDF-1.7\nnot really a pdf"
    up = await flow.documents.upload(flow.tenant, flow.upload(data, "broken.pdf"))
    record = await _until_done(flow, up.job_id)
    assert record.status == "failed"
    assert record.last_error is not None
    doc = await flow.documents.get(flow.tenant, flow.collection.id, up.document_id)
    assert doc.status.value == "failed"
    assert doc.error is not None
    assert doc.error["code"] in {"parse_failed", "pdf_encrypted"}
    async with app_db.tenant_tx(flow.tenant) as conn:
        hook = await fetch_one(
            conn,
            "SELECT id FROM footnote.jobs WHERE tenant_id = %(t)s AND kind = 'webhook'",
            {"t": flow.tenant},
        )
    assert hook is not None
    assert (await _until_done(flow, hook["id"])).status == "done"
    assert [w["event"] for w in flow.webhooks] == ["document.failed"]
