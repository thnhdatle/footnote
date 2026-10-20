# SPDX-License-Identifier: Apache-2.0
"""Acceptance runner (`make acceptance`): probes requirements, runs criteria, writes report.md."""

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acceptance.support import LOG_DIR, environment, load_criteria  # noqa: E402

COMPOSE = ["docker", "compose", "-f", "deploy/docker-compose.yml", "-p", "footnote"]
# Skip constructs are not used in acceptance sources: a criterion whose environment is missing
# is reported as Skipped by the runner, never by the test itself.
FORBIDDEN = re.compile(
    r"t\.Skip|Skipf|SkipNow|testing\.Short|pytest\.mark\.skip|pytest\.skip|xfail|pytest\.importorskip"
)
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
DATASET_HOST = "https://public.ukp.informatik.tu-darmstadt.de/"
CACHE_DIR = Path.home() / ".cache" / "rayld" / "footnote"
MIN_FREE_GB_LMSTUDIO = 18.0
# Kinds that measure quality or latency: `--bench` runs them, and they run against a native
# `footnote serve` (the compose `footnote` service is stopped meanwhile).
MEASURED_KINDS = {"quality", "performance"}
RELEASED_FAKELLM = "ghcr.io/thnhdatle/fakellm:v0.1.0"
LOCAL_FAKELLM = "thnhdatle/fakellm:dev"


@dataclass
class Row:
    """One report row."""

    id: str
    result: str
    measured: str = ""
    threshold: str = ""
    note: str = ""


@dataclass
class State:
    """Mutable run state."""

    rows: list[Row] = field(default_factory=list)
    sibling_notes: list[str] = field(default_factory=list)
    fakellm_image: str = ""
    services: list[str] = field(default_factory=list)


def sh(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output."""
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, **kwargs)


def http_ok(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Return whether a GET succeeds and a probe message."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 400, f"HTTP {resp.status}"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return False, f"{url}: {exc}"


def free_memory_gb() -> float:
    """Free plus inactive plus speculative memory in GiB (macOS vm_stat)."""
    out = sh(["vm_stat"]).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size" in out else 16384  # type: ignore[union-attr]
    total = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        total += int(m.group(1)) if m else 0
    return total * page / 2**30


def resolve_fakellm(state: State) -> bool:
    """Pick the fakellm image and export RAYLD_FAKELLM_IMAGE.

    Order: `RAYLD_FAKELLM_IMAGE`, the released image, a local build. Only images already
    present are used; CI pulls the released image before running the suite.
    """
    candidates = [os.environ.get("RAYLD_FAKELLM_IMAGE", ""), RELEASED_FAKELLM, LOCAL_FAKELLM]
    for image in filter(None, candidates):
        if sh(["docker", "image", "inspect", image]).returncode == 0:
            os.environ["RAYLD_FAKELLM_IMAGE"] = image
            kind = "local build" if image.endswith(":dev") else "released"
            state.sibling_notes.append(f"- fakellm: {image} ({kind})")
            state.fakellm_image = image
            return True
    return False


def probe(requirement: str, state: State) -> str | None:  # noqa: PLR0911
    """Return None when satisfied, else the probe failure message."""
    if requirement == "nightly":
        if not os.environ.get("CI") or os.environ.get("FOOTNOTE_NIGHTLY") == "1":
            return None
        cache = sh(
            [
                "uv",
                "run",
                "--frozen",
                "footnote",
                "eval",
                "cache",
                "status",
                "--require",
                "scifact,trec-covid",
            ]
        )
        return None if cache.returncode == 0 else "CI without FOOTNOTE_NIGHTLY and cold eval cache"
    if requirement == "network":
        if (CACHE_DIR / "datasets").exists() and any((CACHE_DIR / "datasets").iterdir()):
            return None
        ok, msg = http_ok(DATASET_HOST)
        return None if ok else msg
    if requirement == "lmstudio":
        ok, msg = http_ok(f"{LMSTUDIO_URL}/models")
        return None if ok else msg
    if requirement == "docker":
        return None if sh(["docker", "info"]).returncode == 0 else "docker info failed"
    if requirement == "mac":
        return (
            None
            if platform.system() == "Darwin" and platform.machine() == "arm64"
            else "not Apple Silicon"
        )
    if requirement.startswith("sibling:"):
        return (
            None
            if state.fakellm_image
            else f"no fakellm image ({RELEASED_FAKELLM} or {LOCAL_FAKELLM})"
        )
    return f"unknown requirement {requirement}"


def compose_up(state: State) -> str | None:
    """Bring the compose stack up; return an error message on failure."""
    services = ["postgres"]
    if state.fakellm_image:
        services.append("fakellm")
    if (ROOT / "Dockerfile").exists():
        build = sh(["make", "docker"])
        if build.returncode != 0:
            return "make docker failed: " + build.stdout[-400:] + build.stderr[-400:]
        services.append("footnote")
    state.services = services
    up = sh([*COMPOSE, "up", "-d", "--wait", *services])
    return None if up.returncode == 0 else (up.stderr or up.stdout)[-600:]


def run_criterion(criterion: dict[str, Any], state: State) -> Row:
    """Execute one criterion's test and build its row."""
    cid = criterion["id"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{cid}.json").unlink(missing_ok=True)
    report_file = LOG_DIR / f"{cid}.pytest.json"
    proc = sh(
        [
            "uv",
            "run",
            "--frozen",
            "pytest",
            "acceptance",
            "-m",
            "acceptance",
            "-k",
            criterion["test"],
            "-p",
            "no:cacheprovider",
            "--json-report",
            f"--json-report-file={report_file}",
            "-x",
            "-q",
        ]
    )
    (LOG_DIR / f"{cid}.txt").write_text(proc.stdout + proc.stderr)
    measured, passed = "", proc.returncode == 0
    result_file = LOG_DIR / f"{cid}.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        measured = json.dumps(data.get("measured", {}), sort_keys=True)
        passed = passed and bool(data.get("pass"))
    else:
        passed = False
    env = environment()
    threshold = json.dumps(criterion["pass"][env], sort_keys=True)
    return Row(cid, "PASS" if passed else "FAIL", measured, threshold)


def check_criteria(criteria: list[dict[str, Any]]) -> list[str]:
    """Every id has exactly one test, and no acceptance source skips itself."""
    problems = []
    sources = {
        p: p.read_text() for p in (ROOT / "acceptance").glob("*.py") if p.name != "runner.py"
    }
    for criterion in criteria:
        count = sum(
            len(re.findall(rf"def {criterion['test']}\(", text)) for text in sources.values()
        )
        if count != 1:
            problems.append(
                f"{criterion['id']}: expected exactly one test {criterion['test']}, found {count}"
            )
    for path, text in sources.items():
        if FORBIDDEN.search(text):
            problems.append(f"{path.name}: uses a skip construct (skips are decided by the runner)")
    return problems


def write_report(state: State, only: list[str] | None, problems: list[str], version: str) -> Path:
    """Generate acceptance/report.md (format described in acceptance/README.md)."""
    lines = [
        f"# Acceptance report — footnote {version} — {dt.date.today().isoformat()}",
        f"Environment: {environment()}",
        "",
        "| ID | Result | Measured | Threshold |",
        "|---|---|---|---|",
    ]
    for r in state.rows:
        lines.append(f"| {r.id} | {r.result} | {r.measured} | {r.threshold} |")
    sections = {
        "Not met": [
            f"- {r.id}: {r.note or 'test failed'}" for r in state.rows if r.result == "FAIL"
        ],
        "Skipped (environment)": [
            f"- {r.id}: {r.note}" for r in state.rows if r.result.startswith("SKIPPED")
        ],
        "Blocked (environment)": [
            f"- {r.id}: {r.note}" for r in state.rows if r.result.startswith("BLOCKED")
        ],
        "Sibling images used": state.sibling_notes,
        "Criteria check": [f"- {p}" for p in problems] or ["- ok"],
    }
    for title, items in sections.items():
        lines += ["", f"## {title}", *(items or ["- none"])]
    out = ROOT / "acceptance" / ("report.md" if not only else "report.partial.md")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    """Run the acceptance suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--version", default="dev", help="version shown in the report heading")
    args = parser.parse_args()
    only = [x for x in args.only.split(",") if x] or None
    criteria = load_criteria()
    if args.bench:
        criteria = [c for c in criteria if c["kind"] in MEASURED_KINDS]
    if only:
        criteria = [c for c in criteria if c["id"] in only]
    state = State()
    problems = check_criteria(load_criteria())
    resolve_fakellm(state)
    compose_error = compose_up(state)
    if compose_error:
        print(f"compose up failed: {compose_error}")
    env = environment()
    ordered = [c for c in criteria if "lmstudio" not in c["requires"]]
    ordered += [c for c in criteria if "lmstudio" in c["requires"]]
    for criterion in ordered:
        cid = criterion["id"]
        if criterion["pass"][env] == "skip":
            state.rows.append(Row(cid, "SKIPPED (environment)", note="ci: skip"))
            continue
        failed = [(req, msg) for req in criterion["requires"] if (msg := probe(req, state))]
        if "lmstudio" in criterion["requires"] and not failed:
            sh([*COMPOSE, "stop", "fakellm", "footnote"])
            if free_memory_gb() < MIN_FREE_GB_LMSTUDIO:
                failed.append(
                    ("lmstudio", f"free memory {free_memory_gb():.1f} GiB < {MIN_FREE_GB_LMSTUDIO}")
                )
        sibling_missing = [r for r, _ in failed if r.startswith("sibling:")]
        if failed and sibling_missing and os.environ.get("CI"):
            # In CI the sibling images are pulled before the suite runs; a missing one is a
            # broken pipeline, not a missing capability, so the criterion fails.
            note = "; ".join(f"requires {r} — probe: {m}" for r, m in failed)
            state.rows.append(Row(cid, "FAIL", note=note))
            continue
        if failed:
            kind = (
                "BLOCKED"
                if failed[0][0] in {"docker", "mac", "lmstudio"} and env == "mac"
                else "SKIPPED"
            )
            note = "; ".join(f"requires {r} — probe: {m}" for r, m in failed)
            state.rows.append(Row(cid, f"{kind} (environment)", note=note))
            continue
        if criterion["kind"] in MEASURED_KINDS:
            sh([*COMPOSE, "stop", "footnote"])
        elif "footnote" in state.services:
            # a measured criterion before this one stopped the compose service
            sh([*COMPOSE, "up", "-d", "--wait", "footnote"])
        state.rows.append(run_criterion(criterion, state))
    sh([*COMPOSE, "down", "-v"]) if os.environ.get("FOOTNOTE_KEEP_STACK") != "1" else None
    out = write_report(state, only, problems, args.version)
    print(out.read_text())
    failures = sum(r.result == "FAIL" for r in state.rows) + len(problems)
    print(f"{'FAIL' if failures else 'PASS'}: acceptance ({failures} failing rows)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
