# SPDX-License-Identifier: Apache-2.0
"""Retrieval evaluation outputs: `retrieval.csv` and `ablation.md`.

One `Row` per configuration and split. Reference rows (the BEIR published numbers)
carry only nDCG@10 and appear in `ablation.md`, not in the CSV.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_100", "ndcg_at_10", "mrr_at_10")
CSV_COLUMNS = (
    "name",
    "split",
    "mode",
    "chunking",
    "settings",
    "shipped",
    *METRICS,
    "p50_ms",
    "p95_ms",
    "queries",
)
_HEADERS = {
    "recall_at_1": "R@1",
    "recall_at_5": "R@5",
    "recall_at_10": "R@10",
    "recall_at_100": "R@100",
    "ndcg_at_10": "nDCG@10",
    "mrr_at_10": "MRR@10",
}

# BEIR (Thakur et al., 2021) published nDCG@10 per dataset: BM25 and BM25 + cross-encoder.
BEIR_REFERENCES: Mapping[str, Mapping[str, float]] = {
    "scifact": {"BM25": 0.665, "BM25+CE": 0.688},
}


@dataclass(frozen=True)
class Row:
    """Metrics of one configuration on one split; `group` orders the ablation sections."""

    name: str
    split: str
    mode: str
    chunking: str
    settings: str
    metrics: Mapping[str, float]
    p50_ms: float
    p95_ms: float
    queries: int
    shipped: bool = False
    group: str = ""
    note: str = ""


@dataclass(frozen=True)
class ReportMeta:
    """Provenance printed under the table."""

    run_id: str
    dataset: str
    commit: str
    lines: Sequence[str] = field(default_factory=tuple)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _ms(value: float) -> str:
    return f"{value:.1f}"


def csv_text(rows: Sequence[Row]) -> str:
    """The CSV document (header + one line per row, metrics with 4 decimals)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.name,
                row.split,
                row.mode,
                row.chunking,
                row.settings,
                "yes" if row.shipped else "",
                *(_fmt(row.metrics[m]) for m in METRICS),
                _ms(row.p50_ms),
                _ms(row.p95_ms),
                row.queries,
            ]
        )
    return out.getvalue()


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def ablation_markdown(rows: Sequence[Row], meta: ReportMeta) -> str:
    """`ablation.md`: the row table (shipped rows marked), BEIR references, provenance."""
    header = ["Configuration", "Split", "Settings", *(_HEADERS[m] for m in METRICS)]
    header += ["p50 ms", "p95 ms", "Queries"]
    lines = [
        f"# Retrieval evaluation: {meta.dataset}",
        "",
        f"Run `{meta.run_id}` at commit `{meta.commit}`. Rows marked **shipped** use the "
        "shipped defaults for every setting the row varies. Document-level metrics: "
        "chunks ranked to depth 100 and pooled to documents by first occurrence.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        name = f"{row.name} **shipped**" if row.shipped else row.name
        if row.note:
            name = f"{name} ({row.note})"
        cells = [_cell(name), row.split, _cell(row.settings)]
        cells += [_fmt(row.metrics[m]) for m in METRICS]
        cells += [_ms(row.p50_ms), _ms(row.p95_ms), str(row.queries)]
        lines.append("| " + " | ".join(cells) + " |")
    references = BEIR_REFERENCES.get(meta.dataset)
    lines += ["", "## BEIR published references (nDCG@10, test)", ""]
    if references:
        lines += ["| Reference | nDCG@10 |", "|---|---|"]
        lines += [f"| BEIR {name} | {value:.3f} |" for name, value in references.items()]
    else:
        lines.append(f"No published reference recorded for {meta.dataset}.")
    if meta.lines:
        lines += ["", "## Run", ""]
        lines += [f"- {line}" for line in meta.lines]
    return "\n".join(lines) + "\n"


def write_reports(directory: Path, rows: Sequence[Row], meta: ReportMeta) -> tuple[Path, Path]:
    """Write `retrieval.csv` and `ablation.md` into `directory` (created)."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "retrieval.csv"
    md_path = directory / "ablation.md"
    csv_path.write_text(csv_text(rows), encoding="utf-8")
    md_path.write_text(ablation_markdown(rows, meta), encoding="utf-8")
    return csv_path, md_path
