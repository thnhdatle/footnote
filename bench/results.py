# SPDX-License-Identifier: Apache-2.0
"""Render bench/RESULTS.md from acceptance logs and out/bench/latency.json.

Run: ``uv run --frozen python -m bench.results``. Single-run metrics show min = max = median.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from acceptance.support import CRITERIA_FILE, LOG_DIR, ROOT, load_criteria

RESULTS_JSON = ROOT / "out" / "bench" / "latency.json"
OUT = ROOT / "bench" / "RESULTS.md"
HEADER = "| Metric | Median | Min | Max | Threshold | Pass |\n|---|---|---|---|---|---|"
NA = "n/a"

Row = tuple[str, list[float], str, str]  # metric, samples, threshold text, pass text


def _sh(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.strip()


def machine_line(version: str) -> str:
    """Hardware from the host, Docker VM from criteria.yaml `machine`."""
    chip = _sh(["sysctl", "-n", "machdep.cpu.brand_string"]).removeprefix("Apple ")
    mem = _sh(["sysctl", "-n", "hw.memsize"])
    gib = round(int(mem) / 2**30) if mem.isdigit() else 0
    macos = platform.mac_ver()[0] or platform.release()
    declared = str(re.search(r'machine:\s*"?([^"\n]+)', CRITERIA_FILE.read_text()).group(1))  # type: ignore[union-attr]
    vm = re.search(r"Docker VM [^,]+", declared)
    docker = vm.group(0) if vm else "Docker VM unknown"
    hw = f"MacBook Pro {chip} {gib} GB" if chip else declared.split(", Docker", maxsplit=1)[0]
    python = platform.python_version()
    return f"{hw}, macOS {macos}, {docker}, Python {python}, Footnote {version}"


def _fmt(value: float) -> str:
    return f"{value:.4g}" if abs(value) < 10 else f"{value:.1f}"


def render(title: str, date: str, machine: str, rows: list[Row]) -> str:
    """One RESULTS.md section."""
    lines = [f"## {title} — {date}", f"Machine: {machine}", HEADER]
    for metric, samples, threshold, passed in rows:
        median = statistics.median(samples)
        lines.append(
            f"| {metric} | {_fmt(median)} | {_fmt(min(samples))} | {_fmt(max(samples))} "
            f"| {threshold} | {passed} |"
        )
    return "\n".join(lines) + "\n"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _gate(value: float, minimum: float | None = None, maximum: float | None = None) -> str:
    ok = (minimum is None or value >= minimum) and (maximum is None or value < maximum)
    return "PASS" if ok else "FAIL"


def _threshold(criterion: str) -> dict[str, Any]:
    for item in load_criteria():
        if item["id"] == criterion:
            cfg = item["pass"]["mac"]
            return cfg if isinstance(cfg, dict) else {}
    return {}


def fnt07_section(machine: str, date: str) -> str | None:
    """Gated FNT-07 latency: median of 3 runs of p95 (p50 and p99 published)."""
    log = _load(LOG_DIR / "FNT-07.json")
    if log is None:
        return None
    measured = log["measured"]
    limit = float(_threshold("FNT-07")["p95_ms_max"])
    p95 = [float(v) for v in measured["p95_ms_runs"]]
    rows: list[Row] = [
        ("p95 ms (gated)", p95, f"< {limit:g}", _gate(statistics.median(p95), maximum=limit)),
        ("p50 ms (median of runs)", [float(measured["p50_ms_median"])], "published", NA),
        ("p99 ms (median of runs)", [float(measured["p99_ms_median"])], "published", NA),
    ]
    stage = measured.get("debug_stage_seconds", {})
    rows += [
        (
            f"stage {name} ms (first cold debug query; see p95 below)",
            [float(sec) * 1000.0],
            "published",
            NA,
        )
        for name, sec in stage.items()
    ]
    title = f"FNT-07 hybrid+rerank latency, {measured['chunks']} chunks"
    return render(title, date, machine, rows)


def fnt01_section(machine: str, date: str) -> str | None:
    """FNT-01 quality floors and in-process p95 per mode (single run)."""
    log = _load(LOG_DIR / "FNT-01.json")
    if log is None:
        return None
    m = log["measured"]
    t = _threshold("FNT-01")
    rows: list[Row] = []
    floors = {
        "vector": "vector_ndcg10_min",
        "lexical": "lexical_ndcg10_min",
        "hybrid+rerank": "hybrid_rerank_ndcg10_min",
    }
    for mode in ("vector", "lexical", "hybrid", "hybrid+rerank"):
        ndcg = float(m[f"{mode}_ndcg_at_10"])
        floor = t.get(floors.get(mode, ""))
        rows.append(
            (
                f"{mode} nDCG@10",
                [ndcg],
                f">= {floor}" if floor is not None else "published",
                _gate(ndcg, minimum=float(floor)) if floor is not None else NA,
            )
        )
    recall = float(m["hybrid+rerank_recall_at_10"])
    floor = float(t["hybrid_rerank_recall10_min"])
    rows.append(("hybrid+rerank recall@10", [recall], f">= {floor}", _gate(recall, minimum=floor)))
    rows += [
        (f"{mode} in-process p95 ms", [float(m[f"{mode}_p95_ms"])], "published", NA)
        for mode in ("vector", "lexical", "hybrid", "hybrid+rerank")
    ]
    return render(f"FNT-01 SciFact retrieval, {m['queries']} queries", date, machine, rows)


def fnt08_section(machine: str, date: str) -> str | None:
    """FNT-08 gate quality (single run)."""
    log = _load(LOG_DIR / "FNT-08.json")
    if log is None:
        return None
    m = log["measured"]
    t = _threshold("FNT-08")
    rows: list[Row] = []
    for metric, key, limit_key in (
        ("AUC", "auc", "auc_min"),
        ("answer rate", "answer_rate", "answer_rate_min"),
        ("abstain rate", "abstain_rate", "abstain_rate_min"),
    ):
        value = float(m[key])
        rows.append((metric, [value], f">= {t[limit_key]}", _gate(value, minimum=t[limit_key])))
    drift = float(m["calibration_drift"])
    limit = float(t["calibration_drift_max"])
    rows.append(("calibration drift", [drift], f"<= {limit}", "PASS" if drift <= limit else "FAIL"))
    return render("FNT-08 abstention gate, 100+100 SciFact claims", date, machine, rows)


def _runs_rows(prefix: str, runs: list[dict[str, float]]) -> list[Row]:
    return [
        (f"{prefix} {name} ms", [float(r[f"{name}_ms"]) for r in runs], "published", NA)
        for name in ("p50", "p95", "p99")
    ]


def latency_section(machine: str, date: str) -> str | None:
    """bench/latency published numbers (published, not gated)."""
    data = _load(RESULTS_JSON)
    if data is None:
        return None
    rows: list[Row] = []
    for name, ms in data.get("stage_p95_ms", {}).items():
        rows.append((f"stage {name} p95 ms (debug)", [float(ms)], "published", NA))
    if "client_debug_ms" in data:
        rows += _runs_rows("client (debug)", [data["client_debug_ms"]])
    if "rerank_n_50" in data:
        rows += _runs_rows("rerank_n 50", data["rerank_n_50"])
    if "fp32" in data:
        rows += _runs_rows("fp32 reranker", data["fp32"])
    k6 = data.get("k6")
    if k6:
        label = f"k6 {k6['rate_per_s']} req/s x {k6['duration']}"
        for name in ("p50", "p95", "p99"):
            rows.append((f"{label} {name} ms", [float(k6[f"{name}_ms"])], "published", NA))
        rows.append((f"{label} dropped iterations", [float(k6["dropped_iterations"])], "0", NA))
        rows.append((f"{label} error rate", [float(k6["error_rate"])], "0", NA))
    if not rows:
        return None
    title = f"latency (published, not gated), {data['chunks']} chunks, {data['runs']} runs"
    return render(title, data.get("date", date), machine, rows)


def main() -> int:
    """Write bench/RESULTS.md; fail when nothing was available to render."""
    machine = machine_line(os.environ.get("RELEASE", "dev"))
    date = dt.datetime.now(dt.UTC).date().isoformat()
    sections = [
        build(machine, date)
        for build in (fnt07_section, fnt01_section, fnt08_section, latency_section)
    ]
    body = [s for s in sections if s]
    if not body:
        print("bench.results: no acceptance logs or latency results found")
        return 1
    OUT.write_text(
        "# Benchmark results\n\nGenerated by `make bench`. Quality and abstention metrics are "
        "deterministic single runs (min = max = median).\n\n" + "\n".join(body)
    )
    print(f"wrote {OUT.relative_to(ROOT)} ({len(body)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
