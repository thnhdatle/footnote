# SPDX-License-Identifier: Apache-2.0
"""FNT-06: Incremental indexing embeds exactly the changed chunks.

1,000 synthetic Markdown documents (seeded generator, 2-6 sections each) on an
``openai_compat`` space served by fakellm. After the initial ingest the request log is
cleared. (a) One paragraph of one document changes: the texts embedded (sum of the
``input_count`` of the embeddings requests of the test's model) must equal
``|H_new \\ H_old|``, computed here with the domain chunker, the parser and the tokenizer
ingest uses; the ``footnote_embedding_texts_total`` delta and the fakellm ``dropped``
counter must agree. (b) An unchanged document is re-uploaded: 0 embeddings requests.
(c) One document is deleted: no chunk, membership, vector, version or blob row of it is
left (admin DSN).
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any

import httpx

from acceptance.harness import (
    compose_instance,
    connect,
    create_collection,
    create_tenant,
    fake_clear_requests,
    fake_requests,
    register_space,
    wait_job,
)
from acceptance.support import thresholds, write_result
from footnote.adapters.parsers import media
from footnote.adapters.parsers.parse import parse_bytes
from footnote.adapters.tokenizer.counter import HfTokenCounter
from footnote.domain.chunking import chunk_elements
from footnote.domain.settings import ChunkingSettings

SPACE = {
    "name": "fnt06-fake-sd",
    "provider": "openai_compat",
    "model": "fake-embed-fnt06",
    "dimensions": 384,
    "send_dimensions": True,
}
METRIC = "footnote_embedding_texts_total"
SEED = 6
IN_FLIGHT = 16
JOB_TIMEOUT = 900.0
TOKENIZER = Path.home() / ".cache/rayld/footnote/models/bge-small-en-v1.5/tokenizer.json"
CHANGED_DOC = 500
UNCHANGED_DOC = 250
DELETED_DOC = 750

_WORDS = [
    "amber",
    "basalt",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lagoon",
    "meadow",
    "nickel",
    "obsidian",
    "prairie",
    "quartz",
    "river",
    "saffron",
    "tundra",
    "umber",
    "valley",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "anchor",
    "beacon",
    "canyon",
    "dune",
    "estuary",
    "falcon",
    "glacier",
    "heron",
    "island",
]


def _doc_sections(d: int) -> list[list[str]]:
    """Paragraphs per section of document ``d``; every paragraph carries unique tokens."""
    rng = random.Random(SEED * 100_003 + d)  # noqa: S311 - deterministic test content
    sections: list[list[str]] = []
    for s in range(rng.randint(2, 6)):
        paragraphs = []
        for p in range(rng.randint(1, 3)):
            words = " ".join(rng.choices(_WORDS, k=rng.randint(30, 60)))
            paragraphs.append(f"Note d{d}s{s}p{p} {words}.")
        sections.append(paragraphs)
    return sections


def _render(d: int, sections: list[list[str]]) -> bytes:
    parts = [f"# Document {d}\n"]
    for s, paragraphs in enumerate(sections):
        parts.append(f"## Section {s} of {d}\n\n" + "\n\n".join(paragraphs) + "\n")
    return "\n".join(parts).encode()


def _hashes(content: bytes, title: str, counter: HfTokenCounter) -> set[str]:
    """Chunk hashes exactly as ingest computes them for the collection's chunking."""
    elements = parse_bytes(media.MARKDOWN, content)
    drafts = chunk_elements(elements, title, ChunkingSettings(), counter)
    return {draft.content_hash for draft in drafts}


async def _post_and_wait(
    http: httpx.AsyncClient, cid: str, d: int, content: bytes, sem: asyncio.Semaphore
) -> dict[str, Any]:
    async with sem:
        resp = await http.post(
            f"/v1/collections/{cid}/documents",
            files={"file": (f"doc-{d}.md", content)},
            data={"external_id": f"doc-{d}", "title": f"Document {d}"},
        )
        if resp.status_code != 202:
            msg = f"upload doc {d}: {resp.status_code} {resp.text[:300]}"
            raise RuntimeError(msg)
        out: dict[str, Any] = resp.json()
        deadline = time.monotonic() + JOB_TIMEOUT
        while time.monotonic() < deadline:
            job = (await http.get(f"/v1/jobs/{out['job_id']}")).json()
            if job["status"] == "done":
                return out
            if job["status"] == "failed":
                msg = f"ingest of doc {d} failed: {job.get('last_error')}"
                raise RuntimeError(msg)
            await asyncio.sleep(0.2)
        msg = f"ingest of doc {d} not done after {JOB_TIMEOUT}s"
        raise TimeoutError(msg)


async def _ingest_all(url: str, key: str, cid: str, docs: dict[int, bytes]) -> dict[int, str]:
    limits = httpx.Limits(max_connections=IN_FLIGHT)
    headers = {"Authorization": f"Bearer {key}"}
    sem = asyncio.Semaphore(IN_FLIGHT)
    async with httpx.AsyncClient(
        base_url=url, headers=headers, timeout=120.0, limits=limits
    ) as http:
        outs = await asyncio.gather(
            *(_post_and_wait(http, cid, d, c, sem) for d, c in docs.items())
        )
    return {d: str(o["document_id"]) for d, o in zip(docs, outs, strict=True)}


def _model_embeddings(model: str) -> tuple[int, int, int]:
    """(requests, texts, dropped) of the embeddings log for ``model``."""
    log = fake_requests()
    reqs = [r for r in log["requests"] if r.get("model") == model]
    return len(reqs), sum(int(r["input_count"]) for r in reqs), int(log["dropped"])


def _version_hashes(doc_id: str, version: int) -> set[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT c.content_hash FROM footnote.document_versions v "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = v.tenant_id "
            "AND vc.version_id = v.id "
            "JOIN footnote.chunks c ON c.tenant_id = vc.tenant_id AND c.id = vc.chunk_id "
            "WHERE v.document_id = %s AND v.version = %s",
            (doc_id, version),
        ).fetchall()
    return {str(r[0]) for r in rows}


def _remaining_ids(doc_id: str) -> tuple[list[str], list[str]]:
    """(chunk ids, version ids) of the document before its deletion."""
    with connect() as conn:
        chunks = conn.execute(
            "SELECT id::text FROM footnote.chunks WHERE document_id = %s", (doc_id,)
        ).fetchall()
        versions = conn.execute(
            "SELECT id::text FROM footnote.document_versions WHERE document_id = %s", (doc_id,)
        ).fetchall()
    return [r[0] for r in chunks], [r[0] for r in versions]


def _rows_left(doc_id: str, chunk_ids: list[str], version_ids: list[str]) -> dict[str, int]:
    queries = {
        "chunks": ("document_id = %s", doc_id),
        "version_chunks": ("version_id = ANY(%s::uuid[])", version_ids),
        "chunk_vectors": ("chunk_id = ANY(%s::uuid[])", chunk_ids),
        "document_versions": ("document_id = %s", doc_id),
        "document_blobs": ("version_id = ANY(%s::uuid[])", version_ids),
    }
    out: dict[str, int] = {}
    with connect() as conn:
        for table, (where, arg) in queries.items():
            row = conn.execute(
                f"SELECT count(*) FROM footnote.{table} WHERE {where}",  # noqa: S608 - fixed names
                (arg,),
            ).fetchone()
            assert row is not None
            out[table] = int(row[0])
    return out


def test_acc_fnt06_incremental() -> None:
    """Embedded texts equal the changed chunks; identical upload embeds 0; delete leaves 0."""
    expected = thresholds("FNT-06")
    n_docs = int(expected["documents"])
    t0 = time.monotonic()
    counter = HfTokenCounter.from_file(TOKENIZER)
    sections = {d: _doc_sections(d) for d in range(n_docs)}
    docs = {d: _render(d, sections[d]) for d in range(n_docs)}

    inst = compose_instance()
    register_space(inst, SPACE)
    tenant = create_tenant(inst)
    with inst.tenant(tenant.key) as client:
        coll = create_collection(client, "fnt06", embedding_space=SPACE["name"])
        cid = coll["id"]
        doc_ids = asyncio.run(_ingest_all(inst.data_url, tenant.key, cid, docs))
        ingest_seconds = time.monotonic() - t0

        # (a) change one paragraph of a middle section of one document
        fake_clear_requests()
        metric_before = inst.metric_sum(METRIC)
        old_hashes = _hashes(docs[CHANGED_DOC], f"Document {CHANGED_DOC}", counter)
        new_sections = [list(s) for s in sections[CHANGED_DOC]]
        mid = len(new_sections) // 2
        new_sections[mid][0] += " Revised after review with amended figures."
        new_content = _render(CHANGED_DOC, new_sections)
        new_hashes = _hashes(new_content, f"Document {CHANGED_DOC}", counter)
        changed_expected = len(new_hashes - old_hashes)
        resp = client.post(
            f"/v1/collections/{cid}/documents",
            files={"file": (f"doc-{CHANGED_DOC}.md", new_content)},
            data={"external_id": f"doc-{CHANGED_DOC}", "title": f"Document {CHANGED_DOC}"},
        )
        assert resp.status_code == 202, resp.text
        wait_job(client, resp.json()["job_id"], timeout=JOB_TIMEOUT)
        version = int(resp.json()["version"])
        _, embedded_log, dropped = _model_embeddings(SPACE["model"])
        embedded_metric = inst.metric_sum(METRIC) - metric_before
        db_old = _version_hashes(doc_ids[CHANGED_DOC], version - 1)
        db_new = _version_hashes(doc_ids[CHANGED_DOC], version)
        db_new_hashes = len(db_new - db_old)

        # (b) byte-identical re-upload of an unchanged document
        fake_clear_requests()
        resp = client.post(
            f"/v1/collections/{cid}/documents",
            files={"file": (f"doc-{UNCHANGED_DOC}.md", docs[UNCHANGED_DOC])},
            data={"external_id": f"doc-{UNCHANGED_DOC}", "title": f"Document {UNCHANGED_DOC}"},
        )
        unchanged_status = (resp.status_code, resp.json().get("unchanged"))
        identical_requests, identical_embeddings, _ = _model_embeddings(SPACE["model"])

        # (c) delete one document
        chunk_ids, version_ids = _remaining_ids(doc_ids[DELETED_DOC])
        before = _rows_left(doc_ids[DELETED_DOC], chunk_ids, version_ids)
        resp = client.delete(f"/v1/collections/{cid}/documents/{doc_ids[DELETED_DOC]}")
        delete_status = resp.status_code
        after = _rows_left(doc_ids[DELETED_DOC], chunk_ids, version_ids)

    left = sum(after.values())
    measured = {
        "documents": len(docs),
        "changed_chunks_expected": changed_expected,
        "db_new_hashes": db_new_hashes,
        "embedded_texts_log": embedded_log,
        "embedded_texts_metric": embedded_metric,
        "embedded_minus_changed": embedded_log - changed_expected,
        "identical_reupload_embeddings": identical_embeddings,
        "rows_left_after_delete": left,
        "rows_left_per_table": after,
        "dropped": dropped,
        "ingest_seconds": round(ingest_seconds, 1),
        "total_seconds": round(time.monotonic() - t0, 1),
    }
    passed = (
        measured["documents"] >= expected["documents"]
        and changed_expected > 0
        and measured["embedded_minus_changed"] == expected["embedded_minus_changed"]
        and embedded_metric == embedded_log
        and db_new_hashes == changed_expected
        and identical_embeddings <= expected["identical_reupload_embeddings"]
        and identical_requests == 0
        and unchanged_status == (200, True)
        and delete_status == 204
        and sum(before.values()) > 0
        and left <= expected["rows_left_after_delete"]
        and dropped <= expected["dropped"]
    )
    details = (
        f"changed doc {CHANGED_DOC} (section {mid}), version {version}; "
        f"old_hashes={len(old_hashes)} new_hashes={len(new_hashes)} db_old={len(db_old)} "
        f"db_new={len(db_new)}; unchanged upload (status, unchanged)={unchanged_status}; "
        f"delete status={delete_status}; rows before delete={before}"
    )
    write_result("FNT-06", measured, passed, details)
    assert passed, f"FNT-06 failed: {measured}; {details}"
