# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by acceptance tests: thresholds from criteria.yaml and result logs."""

import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CRITERIA_FILE = ROOT / "acceptance" / "criteria.yaml"
LOG_DIR = ROOT / "acceptance" / "logs"


def environment() -> str:
    """Return ``ci`` when the CI variable is set, otherwise ``mac``."""
    return "ci" if os.environ.get("CI") else "mac"


def repo_path(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it (for the report)."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def load_criteria() -> list[dict[str, Any]]:
    """Return the criteria list from criteria.yaml."""
    data = yaml.safe_load(CRITERIA_FILE.read_text())
    criteria: list[dict[str, Any]] = data["criteria"]
    return criteria


def thresholds(criterion_id: str) -> dict[str, Any]:
    """Return the pass thresholds of a criterion for the current environment."""
    for criterion in load_criteria():
        if criterion["id"] == criterion_id:
            pass_cfg = criterion["pass"][environment()]
            if not isinstance(pass_cfg, dict):
                msg = f"{criterion_id} has no {environment()} thresholds"
                raise LookupError(msg)
            return pass_cfg
    msg = f"unknown criterion {criterion_id}"
    raise LookupError(msg)


def write_result(
    criterion_id: str, measured: dict[str, Any], passed: bool, details: str = ""
) -> None:
    """Write acceptance/logs/<ID>.json (read by the runner)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"id": criterion_id, "measured": measured, "pass": passed, "details": details}
    (LOG_DIR / f"{criterion_id}.json").write_text(json.dumps(payload, indent=2, default=str))
