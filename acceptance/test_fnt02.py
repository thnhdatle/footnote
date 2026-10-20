# SPDX-License-Identifier: Apache-2.0
"""FNT-02: answer faithfulness and citation precision (LM Studio generator and judge).

The first 100 distinct SciFact test query ids are answered through the app-layer
`/answers` code path with the model named by `FOOTNOTE_ACC_LLM_MODEL` (default
`gemma-4-26b-a4b-it-qat-mlx`) on the LM Studio endpoint `FOOTNOTE_ACC_LMSTUDIO_URL` (default
`http://localhost:1234/v1`), then every answered item is judged by the same model (stated in
the report: same-model judging inflates scores).
The runner executes this criterion last, with the compose stack reduced to Postgres.
"""

from __future__ import annotations

import os
from typing import Any

from acceptance.evalrun import parse_table, run_eval
from acceptance.harness import write_log
from acceptance.support import thresholds, write_result

LMSTUDIO_URL = os.environ.get("FOOTNOTE_ACC_LMSTUDIO_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("FOOTNOTE_ACC_LLM_MODEL", "gemma-4-26b-a4b-it-qat-mlx")


def test_acc_fnt02_faithfulness() -> None:
    """Faithfulness and citation precision over 100 items meet the floors."""
    expected = thresholds("FNT-02")
    items = int(expected["items"])
    env = {"FOOTNOTE_LLM_BASE_URL": LMSTUDIO_URL, "FOOTNOTE_LLM_MODEL": MODEL}
    directory, output = run_eval(
        "fnt02", "answers", "--dataset", "scifact", "--limit", str(items), "--judge-model", MODEL,
        env=env,
    )  # fmt: skip
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    write_log("FNT-02.log", output + "\n" + summary)
    table = {row["Metric"]: row["Value"] for row in parse_table(summary)}

    def num(name: str) -> float:
        return float(table[name])

    measured: dict[str, Any] = {
        "faithfulness": num("faithfulness"),
        "citation_precision": num("citation precision"),
        "answer_rate": num("answer rate"),
        "items": int(num("items")),
        "answered": int(num("answered")),
        "abstained": int(num("abstained")),
        "unsupported": int(num("unsupported")),
        "judge_parse_failures": int(num("judge parse failures")),
        "generator_model": MODEL,
        "judge_model": MODEL,
        "same_model_judging": True,
    }
    failures: list[str] = []
    if measured["items"] != items:
        failures.append(f"{measured['items']} items, expected {items}")
    if measured["faithfulness"] < float(expected["faithfulness_min"]):
        failures.append(
            f"faithfulness {measured['faithfulness']:.4f} < {expected['faithfulness_min']}"
        )
    if measured["citation_precision"] < float(expected["citation_precision_min"]):
        failures.append(
            f"citation precision {measured['citation_precision']:.4f} "
            f"< {expected['citation_precision_min']}"
        )
    passed = not failures
    details = "; ".join(failures) or "all thresholds met (same model generates and judges)"
    write_result("FNT-02", measured, passed, details)
    assert passed, f"FNT-02 failed: {details}; measured {measured}"
