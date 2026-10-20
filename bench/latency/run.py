# SPDX-License-Identifier: Apache-2.0
"""FNT-07 published (not gated) latency numbers on 100k TREC-COVID chunks.

Loads the FNT-07 data once (acceptance/fnt07_data.py) and reuses one tenant for every
sub-scenario: per-stage p95 from debug searches, ``rerank_n`` 50, the fp32 reranker and a
k6 open-model run. Writes ``out/bench/latency.json`` (gitignored; override with ``BENCH_OUT``).

Run: ``uv run --frozen python -m bench.latency.run``. Size overrides for smoke checks:
``BENCH_CHUNKS``, ``BENCH_QUERIES``, ``BENCH_WARMUP``, ``BENCH_RUNS``, ``BENCH_K6_RATE``,
``BENCH_K6_DURATION``, ``BENCH_SKIP`` (comma list of ``stages,rerank50,k6,fp32``).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from acceptance import fnt07_data
from acceptance.harness import SUPERUSER_DSN, connect, create_collection, create_tenant
from acceptance.support import thresholds

from footnote.app.evaluation.metrics import percentile

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(os.environ.get("BENCH_OUT", ROOT / "out" / "bench" / "latency.json"))
K6_SCRIPT = Path(__file__).resolve().parent / "k6.js"
TENANT_PREFIX = "bench-latency-"
FULL_CHUNKS = 100_000
SHIPPED_RERANK_N = 30
RERANK_N_ALT = 50
SKIP = {s for s in os.environ.get("BENCH_SKIP", "").split(",") if s}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _summary(ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(percentile(ms, 50), 1),
        "p95_ms": round(percentile(ms, 95), 1),
        "p99_ms": round(percentile(ms, 99), 1),
    }


def _debug_run(
    client: httpx.Client, path: str, texts: list[str], count: int
) -> tuple[list[float], dict[str, list[float]], set[str]]:
    """`count` sequential debug searches: client ms, per-stage ms, exact-scan reasons seen."""
    exact_scans: set[str] = set()
    body: dict[str, Any] = {"mode": "hybrid", "rerank": True, "k": 8, "debug": True}
    client_ms: list[float] = []
    stages: dict[str, list[float]] = {}
    for i in range(count):
        start = time.perf_counter()
        resp = client.post(path, json={**body, "query": texts[i % len(texts)]})
        client_ms.append((time.perf_counter() - start) * 1000.0)
        assert resp.status_code == 200, f"search failed: {resp.status_code} {resp.text[:300]}"
        debug = resp.json()["debug"]
        assert debug["reranked"] is True, debug
        exact_scans.add(str(debug["exact_scan_reason"]))
        for name, seconds in debug["stage_seconds"].items():
            stages.setdefault(name, []).append(seconds * 1000.0)
    return client_ms, stages, exact_scans


def _repeat(
    client: httpx.Client, path: str, texts: list[str], sizes: tuple[int, int, int]
) -> list[dict[str, float]]:
    """`runs` x (warm-up, then `count` timed searches); sizes = (warmup, count, runs)."""
    warmup, count, runs = sizes
    out = []
    for _ in range(runs):
        fnt07_data.timed_run(client, path, texts, warmup)
        out.append(_summary(fnt07_data.timed_run(client, path, texts, count)))
    return out


def _set_rerank_n(client: httpx.Client, cid: str, value: int) -> None:
    resp = client.patch(f"/v1/collections/{cid}", json={"retrieval": {"rerank_n": value}})
    assert resp.status_code == 200, f"patch failed: {resp.status_code} {resp.text[:300]}"


def _metric(summary: dict[str, Any], name: str) -> dict[str, Any]:
    entry = summary.get("metrics", {}).get(name, {})
    values: dict[str, Any] = entry.get("values", entry)
    return values


def run_k6(base_url: str, key: str, cid: str, texts: list[str]) -> dict[str, Any]:
    """Constant-arrival-rate k6 run; returns latency percentiles, drops and error rate."""
    rate = _env_int("BENCH_K6_RATE", 2)
    duration = os.environ.get("BENCH_K6_DURATION", "5m")
    with tempfile.TemporaryDirectory() as tmp:
        queries_file = Path(tmp) / "queries.json"
        queries_file.write_text(json.dumps(texts))
        summary_file = Path(tmp) / "summary.json"
        env = {
            **os.environ,
            "BASE_URL": base_url,
            "API_KEY": key,
            "COLLECTION_ID": cid,
            "QUERIES_FILE": str(queries_file),
            "RATE": str(rate),
            "DURATION": duration,
        }
        proc = subprocess.run(
            ["k6", "run", "--quiet", f"--summary-export={summary_file}", str(K6_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if not summary_file.exists():
            msg = f"k6 failed ({proc.returncode}): {proc.stderr[-500:]}"
            raise RuntimeError(msg)
        summary = json.loads(summary_file.read_text())
    latency = _metric(summary, "http_req_duration")
    failed = _metric(summary, "http_req_failed")
    dropped = _metric(summary, "dropped_iterations")
    requests = _metric(summary, "http_reqs")
    error_rate = failed.get("value", failed.get("rate", 0.0))
    return {
        "rate_per_s": rate,
        "duration": duration,
        "requests": int(requests.get("count", 0)),
        "p50_ms": round(float(latency["med"]), 1),
        "p95_ms": round(float(latency["p(95)"]), 1),
        "p99_ms": round(float(latency["p(99)"]), 1),
        "dropped_iterations": int(dropped.get("count", 0)),
        "error_rate": round(float(error_rate), 4),
        "k6_exit_code": proc.returncode,
    }


def main() -> None:
    """Load the data once, then run every sub-scenario against the same tenant."""
    expected = thresholds("FNT-07")
    n_chunks = _env_int("BENCH_CHUNKS", int(expected["chunks"]))
    n_queries = _env_int("BENCH_QUERIES", int(expected["queries"]))
    n_runs = _env_int("BENCH_RUNS", int(expected["runs"]))
    warmup = _env_int("BENCH_WARMUP", 50)
    texts = fnt07_data.queries()

    docs = fnt07_data.prepare_documents(n_chunks)
    vectors = fnt07_data.embed_chunks(docs)
    fnt07_data.drop_stale_tenants(TENANT_PREFIX)
    results: dict[str, Any] = {
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "version": os.environ.get("RELEASE", "dev"),
        "chunks": n_chunks,
        "queries_per_run": n_queries,
        "warmup": warmup,
        "runs": n_runs,
    }
    tenant_id = ""
    try:
        with fnt07_data.native_server(log_name="bench-latency-server.log") as inst:
            tenant = create_tenant(inst, f"{TENANT_PREFIX}{uuid.uuid4().hex[:8]}")
            tenant_id = tenant.id
            with inst.tenant(tenant.key, timeout=120.0) as client:
                cid = create_collection(client, "bench-latency", language="english")["id"]
                fnt07_data.bulk_load(SUPERUSER_DSN, tenant.id, cid, docs, vectors)
                del docs, vectors
                path = f"/v1/collections/{cid}/search"
                if "stages" not in SKIP:
                    fnt07_data.timed_run(client, path, texts, warmup)
                    client_ms, stages, exact_scans = _debug_run(client, path, texts, n_queries)
                    results["stage_p95_ms"] = {
                        name: round(percentile(ms, 95), 1) for name, ms in stages.items()
                    }
                    results["exact_scan_reasons"] = sorted(exact_scans)
                    assert exact_scans == {"None"} or n_chunks < FULL_CHUNKS, exact_scans
                    results["client_debug_ms"] = _summary(client_ms)
                if "rerank50" in SKIP:
                    pass
                else:
                    _set_rerank_n(client, cid, RERANK_N_ALT)
                    try:
                        results["rerank_n_50"] = _repeat(
                            client, path, texts, (warmup, n_queries, n_runs)
                        )
                    finally:
                        _set_rerank_n(client, cid, SHIPPED_RERANK_N)
                if "k6" not in SKIP:
                    results["k6"] = run_k6(inst.data_url, tenant.key, cid, texts)
                key = tenant.key
        if "fp32" not in SKIP:
            with (
                fnt07_data.native_server(
                    {"FOOTNOTE_RERANK_VARIANT": "fp32"}, "bench-latency-fp32-server.log"
                ) as inst,
                inst.tenant(key, timeout=120.0) as client,
            ):
                path = f"/v1/collections/{cid}/search"
                results["fp32"] = _repeat(client, path, texts, (warmup, n_queries, n_runs))
    finally:
        if tenant_id:
            with connect(SUPERUSER_DSN) as conn:
                conn.execute("DELETE FROM footnote.tenants WHERE id = %s", (tenant_id,))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
