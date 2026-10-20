# SPDX-License-Identifier: Apache-2.0
"""FNT-07: Hybrid + rerank retrieval latency on 100k TREC-COVID chunks (query embedding included).

100,000 chunks (corpus file order, shipped chunker) in one `english` collection,
bge-small vectors from the evaluation cache, bulk-loaded with binary COPY over the compose
superuser connection (not timed; see `acceptance/fnt07_data.py`). Footnote runs natively
(`footnote serve` on loopback ephemeral ports, shipped defaults). Queries: the 50 TREC-COVID
`text` questions cycled; 50 warm-up then 1,000 sequential `POST .../search` (hybrid, rerank,
k 8) over one keep-alive connection, client wall time. 3 runs; the gate is the median p95.
"""

from __future__ import annotations

import statistics
import time
import uuid
from typing import Any

import httpx

from acceptance import fnt07_data
from acceptance.harness import SUPERUSER_DSN, connect, create_collection, create_tenant
from acceptance.support import thresholds, write_result
from footnote.app.evaluation.metrics import percentile

WARMUP = 50
TENANT_PREFIX = "acc-fnt07-"


def check_hnsw_path(client: httpx.Client, path: str, query: str) -> dict[str, Any]:
    """One debug search: reranked, no exact-scan fallback (the HNSW path is in use)."""
    resp = client.post(
        path, json={"query": query, "mode": "hybrid", "rerank": True, "k": 8, "debug": True}
    )
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    debug = body["debug"]
    assert body["hits"], "search returned no hits"
    assert debug["reranked"] is True, debug
    assert debug["exact_scan_reason"] is None, f"exact scan used: {debug}"
    stages: dict[str, Any] = debug["stage_seconds"]
    return stages


def test_acc_fnt07_latency() -> None:
    """Median p95 over 3 runs of 1,000 hybrid+rerank searches stays below the threshold."""
    expected = thresholds("FNT-07")
    queries = fnt07_data.queries()
    n_chunks = int(expected["chunks"])
    n_queries = int(expected["queries"])
    n_runs = int(expected["runs"])

    t0 = time.monotonic()
    docs = fnt07_data.prepare_documents(n_chunks)
    prep_seconds = time.monotonic() - t0
    vectors = fnt07_data.embed_chunks(docs)
    embed_seconds = time.monotonic() - t0 - prep_seconds

    fnt07_data.drop_stale_tenants(TENANT_PREFIX)
    tenant_id = ""
    runs: list[dict[str, float]] = []
    load: dict[str, Any] = {}
    stages: dict[str, Any] = {}
    try:
        with fnt07_data.native_server() as inst:
            tenant = create_tenant(inst, f"{TENANT_PREFIX}{uuid.uuid4().hex[:8]}")
            tenant_id = tenant.id
            with inst.tenant(tenant.key, timeout=120.0) as client:
                collection = create_collection(client, "fnt07", language="english")
                cid = collection["id"]
                load = fnt07_data.bulk_load(SUPERUSER_DSN, tenant.id, cid, docs, vectors)
                del docs, vectors
                with connect() as conn:
                    row = conn.execute(
                        "SELECT count(*) FROM footnote.chunks c JOIN footnote.documents d "
                        "ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
                        "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
                        "AND vc.version_id = d.current_version_id AND vc.chunk_id = c.id "
                        "WHERE c.collection_id = %s",
                        (cid,),
                    ).fetchone()
                assert row is not None
                searchable = int(row[0])
                path = f"/v1/collections/{cid}/search"
                stages = check_hnsw_path(client, path, queries[0])
                for _ in range(n_runs):
                    fnt07_data.timed_run(client, path, queries, WARMUP)
                    ms = fnt07_data.timed_run(client, path, queries, n_queries)
                    runs.append(
                        {
                            "p50_ms": percentile(ms, 50),
                            "p95_ms": percentile(ms, 95),
                            "p99_ms": percentile(ms, 99),
                            "max_ms": max(ms),
                        }
                    )
    finally:
        if tenant_id:
            with connect(SUPERUSER_DSN) as conn:
                conn.execute("DELETE FROM footnote.tenants WHERE id = %s", (tenant_id,))

    p95s = [r["p95_ms"] for r in runs]
    median_p95 = statistics.median(p95s)
    measured: dict[str, Any] = {
        "chunks": searchable,
        "queries_per_run": n_queries,
        "runs": len(runs),
        "p95_ms_median": round(median_p95, 1),
        "p95_ms_runs": [round(v, 1) for v in p95s],
        "p50_ms_median": round(statistics.median(r["p50_ms"] for r in runs), 1),
        "p99_ms_median": round(statistics.median(r["p99_ms"] for r in runs), 1),
        "debug_stage_seconds": stages,
        "load": load,
        "prepare_seconds": round(prep_seconds, 1),
        "embed_seconds": round(embed_seconds, 1),
    }
    passed = (
        searchable == n_chunks
        and len(runs) == n_runs
        and median_p95 < float(expected["p95_ms_max"])
    )
    details = (
        f"median p95 {median_p95:.1f} ms vs max {expected['p95_ms_max']} ms over {len(runs)} "
        f"runs of {n_queries} queries, {searchable} chunks"
    )
    write_result("FNT-07", measured, passed, details)
    assert passed, f"FNT-07 failed: {details}; {measured}"
