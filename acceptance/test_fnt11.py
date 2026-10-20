# SPDX-License-Identifier: Apache-2.0
"""FNT-11: Offline quickstart completes in under 5 minutes with a valid citation.

The test runs ``examples/quickstart/run.sh``, the README's offline smoke test: compose
stack with the fakellm profile and the quickstart config, tenant and key, collection
with the shipped thresholds, the sample PDF, the ingest job and one question.
The image is built before the timer starts (``make docker``, done by the runner). The
models volume is already filled by the runner's earlier ``compose up``, so the model
download is not part of the measured time on a warm machine.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from acceptance.harness import COMPOSE, ROOT, wait_ready, write_log
from acceptance.support import thresholds, write_result

RUN_SCRIPT = ROOT / "examples" / "quickstart" / "run.sh"
ANSWER_MARKER = "== ask:"


def _answer_from(stdout: str) -> dict[str, Any]:
    """The JSON object ``run.sh`` prints after its ``== ask:`` line."""
    _, sep, tail = stdout.partition(ANSWER_MARKER)
    if not sep:
        return {}
    body = tail.partition("\n")[2]
    try:
        data: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return data


def test_acc_fnt11_quickstart() -> None:
    """The README offline quickstart script succeeds in < 5 minutes with >= 1 valid citation."""
    expected = thresholds("FNT-11")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [str(RUN_SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(expected["seconds_max"]) + 60.0,
        )
    finally:
        seconds = time.monotonic() - started
        # run.sh recreates the compose service with the quickstart config; restore it.
        subprocess.run(
            [*COMPOSE, "up", "-d", "--wait", "footnote"], cwd=ROOT, check=False, capture_output=True
        )
        wait_ready("http://127.0.0.1:9094")
    write_log("FNT-11-run.txt", proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    answer = _answer_from(proc.stdout)
    citations = answer.get("citations") or []
    abstained = bool(answer.get("abstained", True))
    valid = 0 if abstained else len(citations)
    measured = {
        "seconds": round(seconds, 1),
        "valid_citations": valid,
        "exit_code": proc.returncode,
    }
    passed = (
        proc.returncode == 0
        and not abstained
        and seconds < expected["seconds_max"]
        and valid >= expected["valid_citations_min"]
    )
    details = (
        f"abstained={abstained} evidence_score={answer.get('evidence_score')} "
        f"answer={answer.get('answer')!r} stderr={proc.stderr[-300:]!r}"
    )
    write_result("FNT-11", measured, passed, details)
    assert passed, f"FNT-11 failed: {measured}; {details}"
