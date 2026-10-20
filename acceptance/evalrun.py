# SPDX-License-Identifier: Apache-2.0
"""Run `footnote eval` commands in-process for the quality and performance criteria."""

from __future__ import annotations

import contextlib
import csv
import os
import re
import shutil
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from acceptance.harness import ADMIN_DSN, ADMIN_KEY, APP_DSN
from acceptance.support import LOG_DIR, ROOT

EVAL_ENV = {
    "FOOTNOTE_DATABASE_URL": APP_DSN,
    "FOOTNOTE_ADMIN_DATABASE_URL": ADMIN_DSN,
    "FOOTNOTE_AUTH_ADMIN_KEY": ADMIN_KEY,
    "FOOTNOTE_LOG_LEVEL": "warn",
}
_CELL = re.compile(r"(?<!\\)\|")


@dataclass(frozen=True)
class EvalRun:
    """Output of one `eval retrieval` invocation."""

    directory: Path
    output: str
    rows: list[dict[str, str]]
    table: list[dict[str, str]]

    @property
    def ablation_md(self) -> Path:
        """The generated `ablation.md`."""
        return self.directory / "ablation.md"


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_table(markdown: str) -> list[dict[str, str]]:
    """The first pipe table of `ablation.md` as header -> cell dicts."""
    lines = [ln for ln in markdown.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in _CELL.split(lines[0])[1:-1]]
    rows = []
    for line in lines[2:]:
        if not line.startswith("| "):
            break
        cells = [c.strip().replace("\\|", "|") for c in _CELL.split(line)[1:-1]]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells, strict=True)))
    return rows


def run_eval(tag: str, *args: str, env: dict[str, str] | None = None) -> tuple[Path, str]:
    """`footnote eval <args>` through the typer app, in this process.

    Results go to a fresh `acceptance/logs/eval-<tag>/<run_id>/`; a non-zero exit raises.
    Returns the run directory and the command output.
    """
    from footnote.__main__ import app  # noqa: PLC0415 - heavy import

    root = LOG_DIR / f"eval-{tag}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    env = {
        **EVAL_ENV,
        "FOOTNOTE_MODELS_MANIFEST": str(ROOT / "models" / "manifest.yaml"),
        **(env or {}),
    }
    argv = ["eval", *args, "--output-dir", str(root)]
    with _chdir(ROOT):
        result = CliRunner().invoke(app, argv, env=env)
    output = result.output
    if result.exit_code != 0:
        trace = "".join(traceback.format_exception(result.exception)) if result.exception else ""
        msg = f"footnote {' '.join(argv)} exited {result.exit_code}:\n{output[-3000:]}\n{trace}"
        raise AssertionError(msg)
    runs = [p for p in root.iterdir() if p.is_dir()]
    if len(runs) != 1:
        msg = f"expected one run directory under {root}, found {runs}"
        raise AssertionError(msg)
    return runs[0], output


def run_eval_retrieval(tag: str, *args: str) -> EvalRun:
    """`footnote eval retrieval <args>` in this process, with its CSV and table parsed."""
    directory, output = run_eval(tag, "retrieval", *args)
    with (directory / "retrieval.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    table = parse_table((directory / "ablation.md").read_text(encoding="utf-8"))
    return EvalRun(directory, output, rows, table)
