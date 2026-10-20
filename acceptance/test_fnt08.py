# SPDX-License-Identifier: Apache-2.0
"""FNT-08: Abstention gate on SciFact answerable vs unanswerable claims.

`footnote eval calibrate --dataset scifact --split train` recomputes the
reranker-gate threshold on the abstention index (corpus minus the unanswerable group's gold
documents); it must lie within `calibration_drift_max` of the committed manifest value.
`footnote eval abstention --dataset scifact` then scores the 100 answerable and 100
unanswerable test claims with the app-layer gate (no LLM) at that committed threshold.
Both run in-process; the curves and the out-of-corpus curve are published in the run
directories under `acceptance/logs/eval-fnt08-*`.
"""

from __future__ import annotations

import json
from typing import Any

from acceptance.evalrun import run_eval
from acceptance.harness import write_log
from acceptance.support import repo_path, thresholds, write_result


def test_acc_fnt08_abstention() -> None:
    """Calibration is stable and the frozen threshold separates the two groups."""
    expected = thresholds("FNT-08")
    cal_dir, cal_out = run_eval(
        "fnt08-calibrate",
        "calibrate",
        "--dataset",
        "scifact",
        "--split",
        "train",
        "--reranker-files",
        "configured",
    )
    abs_dir, abs_out = run_eval("fnt08-abstention", "abstention", "--dataset", "scifact")
    write_log("FNT-08.log", cal_out + "\n" + abs_out)
    calibration = json.loads((cal_dir / "calibration.json").read_text(encoding="utf-8"))
    summary = json.loads((abs_dir / "abstention.json").read_text(encoding="utf-8"))

    failures: list[str] = []
    rerank = [r for r in calibration["rerank"] if r["key"] == summary["reranker_file"]]
    assert len(rerank) == 1, f"no calibration for {summary['reranker_file']}: {calibration}"
    committed = float(summary["manifest_threshold"])
    drift = abs(float(rerank[0]["threshold"]) - committed)
    measured: dict[str, Any] = {
        "auc": summary["auc"],
        "threshold": summary["threshold"],
        "committed_threshold": committed,
        "recalibrated_threshold": rerank[0]["threshold"],
        "calibration_drift": drift,
        "calibration_claims": calibration["claims"],
        "answer_rate": summary["answer_rate"],
        "abstain_rate": summary["abstain_rate"],
        "per_group": summary["per_group"],
        "best_balanced": summary["best_balanced"],
        "vector_gate": summary["vector"],
        "out_of_corpus": summary["out_of_corpus"],
        "curve": repo_path(abs_dir / "abstention_curve.csv"),
    }
    per_group = int(expected["per_group"])
    if set(summary["per_group"].values()) != {per_group}:
        failures.append(f"groups {summary['per_group']}, expected {per_group} each")
    if summary["threshold"] != committed:
        failures.append(f"gate used {summary['threshold']}, committed value is {committed}")
    if drift > float(expected["calibration_drift_max"]):
        failures.append(f"calibration drift {drift:.4f} > {expected['calibration_drift_max']}")
    if summary["auc"] < float(expected["auc_min"]):
        failures.append(f"AUC {summary['auc']:.4f} < {expected['auc_min']}")
    if summary["answer_rate"] < float(expected["answer_rate_min"]):
        failures.append(f"answer rate {summary['answer_rate']:.2%} < {expected['answer_rate_min']}")
    if summary["abstain_rate"] < float(expected["abstain_rate_min"]):
        failures.append(
            f"abstain rate {summary['abstain_rate']:.2%} < {expected['abstain_rate_min']}"
        )

    passed = not failures
    details = "; ".join(failures) or "all thresholds met"
    write_result("FNT-08", measured, passed, details)
    assert passed, f"FNT-08 failed: {details}; measured {measured}"
