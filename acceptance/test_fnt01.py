# SPDX-License-Identifier: Apache-2.0
"""FNT-01: Retrieval quality on BEIR SciFact test with shipped defaults.

`footnote eval retrieval --dataset scifact` runs in-process (SciFact corpus
ingested through Footnote's pipeline, 300 test queries, document-level metrics).
The gate reads the shipped row of each mode from `retrieval.csv`; R@100, MRR@10 and the
`bm25s` row are reported, not gated.
"""

from __future__ import annotations

from typing import Any

from acceptance.evalrun import run_eval_retrieval
from acceptance.harness import write_log
from acceptance.support import repo_path, thresholds, write_result

# (mode, metric, threshold key)
GATES = (
    ("vector", "ndcg_at_10", "vector_ndcg10_min"),
    ("lexical", "ndcg_at_10", "lexical_ndcg10_min"),
    ("hybrid+rerank", "ndcg_at_10", "hybrid_rerank_ndcg10_min"),
    ("hybrid+rerank", "recall_at_10", "hybrid_rerank_recall10_min"),
)
REPORTED = ("recall_at_10", "recall_at_100", "ndcg_at_10", "mrr_at_10")


def _shipped(rows: list[dict[str, str]], mode: str) -> dict[str, str]:
    found = [r for r in rows if r["mode"] == mode and r["split"] == "test" and r["shipped"]]
    assert len(found) == 1, f"expected one shipped test row for mode {mode}, found {len(found)}"
    return found[0]


def test_acc_fnt01_retrieval_quality() -> None:
    """Shipped vector, lexical and hybrid+rerank retrieval meet the SciFact floors."""
    expected = thresholds("FNT-01")
    run = run_eval_retrieval("fnt01", "--dataset", "scifact")
    write_log("FNT-01.log", run.output)

    measured: dict[str, Any] = {"run_dir": repo_path(run.directory)}
    failures: list[str] = []
    queries = {mode: int(_shipped(run.rows, mode)["queries"]) for mode in {g[0] for g in GATES}}
    measured["queries"] = min(queries.values())
    if queries and set(queries.values()) != {expected["queries"]}:
        failures.append(f"queries {queries}, expected {expected['queries']} per mode")
    for mode in ("vector", "lexical", "hybrid", "hybrid+rerank"):
        row = _shipped(run.rows, mode)
        for metric in REPORTED:
            measured[f"{mode}_{metric}"] = float(row[metric])
        measured[f"{mode}_p95_ms"] = float(row["p95_ms"])
    for mode, metric, key in GATES:
        value = float(_shipped(run.rows, mode)[metric])
        if value < float(expected[key]):
            failures.append(f"{mode} {metric} {value:.4f} < {expected[key]}")
    bm25 = [r for r in run.rows if r["mode"] == "bm25s" and r["split"] == "test"]
    for row in bm25:
        for metric in REPORTED:
            measured[f"bm25s_{metric}"] = float(row[metric])

    passed = not failures
    details = "; ".join(failures) or "all floors met"
    write_result("FNT-01", measured, passed, details)
    assert passed, f"FNT-01 failed: {details}; measured {measured}"
