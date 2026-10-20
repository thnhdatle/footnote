# SPDX-License-Identifier: Apache-2.0
"""Black-box harness for the functional acceptance tests: Footnote as built for compose.

The runner brings up ``deploy/docker-compose.yml`` (Postgres, fakellm, Footnote).
Tests talk to Footnote over HTTP, to fakellm over its ``/_fake`` API and to Postgres
over the admin or superuser DSN for assertions. A test that needs a different
process configuration (fault flags, ``gc_grace``) starts its own container from the
same image with :func:`custom_instance`; the compose ``footnote`` service is stopped
meanwhile so only one worker with one configuration consumes the job queue.

Development knobs (never set by the runner): ``FOOTNOTE_ACC_KEEP_COMPOSE=1`` keeps the
compose service running beside a custom instance; ``FOOTNOTE_ACC_FAKELLM_INTERNAL``
points custom instances at another fakellm on the compose network.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psycopg

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "acceptance" / "logs"
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy/docker-compose.yml"), "-p", "footnote"]
NETWORK = "footnote_default"
MODELS_VOLUME = "footnote_models"
IMAGE = os.environ.get("RAYLD_FOOTNOTE_IMAGE", "thnhdatle/footnote:dev")

PG_PORT = int(os.environ.get("FOOTNOTE_PG_PORT", "55432"))
SUPERUSER_DSN = f"postgresql://postgres:postgres@127.0.0.1:{PG_PORT}/footnote"
ADMIN_DSN = f"postgresql://footnote_admin:footnote_admin@127.0.0.1:{PG_PORT}/footnote"
APP_DSN = f"postgresql://footnote_app:footnote_app@127.0.0.1:{PG_PORT}/footnote"
ADMIN_KEY = "dev-admin-key-change-me"
FAKELLM_URL = f"http://127.0.0.1:{os.environ.get('FOOTNOTE_FAKELLM_PORT', '8090')}"
COMPOSE_DATA_URL = "http://127.0.0.1:8000"
COMPOSE_ADMIN_URL = "http://127.0.0.1:9094"
FAKELLM_INTERNAL = os.environ.get("FOOTNOTE_ACC_FAKELLM_INTERNAL", "http://fakellm:8090/v1")
KEEP_COMPOSE = os.environ.get("FOOTNOTE_ACC_KEEP_COMPOSE") == "1"

# Environment of the compose `footnote` service, as seen from inside the network.
BASE_ENV: dict[str, str] = {
    "FOOTNOTE_SERVER_LISTEN": "0.0.0.0:8000",
    "FOOTNOTE_ADMIN_LISTEN": "0.0.0.0:9094",
    "FOOTNOTE_DATABASE_URL": "postgresql://footnote_app:footnote_app@postgres:5432/footnote",
    "FOOTNOTE_ADMIN_DATABASE_URL": (
        "postgresql://footnote_admin:footnote_admin@postgres:5432/footnote"
    ),
    "FOOTNOTE_AUTH_ADMIN_KEY": ADMIN_KEY,
    "FOOTNOTE_LLM_BASE_URL": FAKELLM_INTERNAL,
    "FOOTNOTE_LLM_MODEL": "fake-default",
    "FOOTNOTE_EMBEDDING_BASE_URL": FAKELLM_INTERNAL,
}


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        msg = f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr[-800:]}"
        raise RuntimeError(msg)
    return proc


def wait_ready(admin_url: str, timeout: float = 180.0) -> None:
    """Wait until ``/readyz`` on the admin listener answers 200."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{admin_url}/readyz", timeout=5.0)
            if resp.status_code == 200:
                return
            last = f"{resp.status_code} {resp.text[:200]}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(0.5)
    msg = f"footnote not ready at {admin_url}: {last}"
    raise TimeoutError(msg)


@dataclass(frozen=True)
class Instance:
    """One running Footnote process (data plane and admin listener URLs)."""

    data_url: str
    admin_url: str
    container: str

    def admin(self, timeout: float = 60.0) -> httpx.Client:
        """Client authenticated with the admin key."""
        return httpx.Client(
            base_url=self.data_url,
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            timeout=timeout,
        )

    def tenant(self, key: str, timeout: float = 60.0) -> httpx.Client:
        """Client authenticated with a tenant key."""
        return httpx.Client(
            base_url=self.data_url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout
        )

    def metrics(self) -> str:
        """Prometheus exposition from the admin listener."""
        resp = httpx.get(f"{self.admin_url}/metrics", timeout=10.0)
        resp.raise_for_status()
        return resp.text

    def metric_sum(self, name: str, **labels: str) -> float:
        """Sum of every sample of ``name`` whose labels include ``labels``."""
        total = 0.0
        for line in self.metrics().splitlines():
            if line.startswith("#") or not line.startswith(name):
                continue
            head, _, value = line.rpartition(" ")
            metric, _, rest = head.partition("{")
            if metric != name:
                continue
            if all(f'{k}="{v}"' in rest for k, v in labels.items()):
                total += float(value)
        return total


def compose_instance() -> Instance:
    """The compose ``footnote`` service (started by the runner)."""
    wait_ready(COMPOSE_ADMIN_URL)
    return Instance(COMPOSE_DATA_URL, COMPOSE_ADMIN_URL, "footnote-footnote-1")


def _host_port(container: str, port: int) -> int:
    out = _run(["docker", "port", container, f"{port}/tcp"]).stdout.strip().splitlines()[0]
    return int(out.rsplit(":", 1)[1])


@contextlib.contextmanager
def custom_instance(
    name: str,
    env: Mapping[str, str] | None = None,
    *,
    wait: bool = True,
    stop_compose: bool = not KEEP_COMPOSE,
) -> Iterator[Instance]:
    """Run the image with extra environment on the compose network.

    The compose ``footnote`` service is stopped for the duration (one queue consumer)
    and started again afterwards. Container logs go to ``acceptance/logs/<name>.log``.
    """
    container = f"fnacc-{name}-{uuid.uuid4().hex[:6]}"
    merged = {**BASE_ENV, **(env or {})}
    if stop_compose:
        _run([*COMPOSE, "stop", "footnote"])
    cmd = [
        "docker", "run", "-d", "--name", container, "--network", NETWORK,
        "-p", "127.0.0.1::8000", "-p", "127.0.0.1::9094",
        "-v", f"{MODELS_VOLUME}:/models", "--read-only", "--tmpfs", "/tmp",  # noqa: S108 - container path
        "--memory", "3g",
    ]  # fmt: skip
    for key, value in merged.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += [IMAGE, "serve", "--with-worker"]
    try:
        _run(cmd)
        inst = Instance(
            f"http://127.0.0.1:{_host_port(container, 8000)}",
            f"http://127.0.0.1:{_host_port(container, 9094)}",
            container,
        )
        if wait:
            wait_ready(inst.admin_url)
        yield inst
    finally:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logs = _run(["docker", "logs", container], check=False)
        (LOG_DIR / f"{name}.log").write_text(logs.stdout + logs.stderr)
        _run(["docker", "rm", "-f", container], check=False)
        if stop_compose:
            _run([*COMPOSE, "start", "footnote"])
            wait_ready(COMPOSE_ADMIN_URL)


# --- tenants, collections, documents ------------------------------------------------


@dataclass(frozen=True)
class Tenant:
    """A tenant created through the admin API with one key."""

    id: str
    name: str
    key: str


def create_tenant(inst: Instance, name: str | None = None) -> Tenant:
    """Create a tenant and a key."""
    name = name or f"acc-{uuid.uuid4().hex[:8]}"
    with inst.admin() as admin:
        resp = admin.post("/v1/admin/tenants", json={"name": name})
        resp.raise_for_status()
        tid = resp.json()["id"]
        resp = admin.post(f"/v1/admin/tenants/{tid}/keys", json={"name": "acceptance"})
        resp.raise_for_status()
        return Tenant(tid, name, resp.json()["key"])


def register_space(inst: Instance, space: Mapping[str, Any]) -> dict[str, Any]:
    """Register an embedding space (idempotent by name)."""
    with inst.admin() as admin:
        resp = admin.post("/v1/admin/embedding-spaces", json=dict(space))
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def create_collection(client: httpx.Client, name: str, **settings: Any) -> dict[str, Any]:
    """Create a collection with optional settings groups."""
    resp = client.post("/v1/collections", json={"name": name, **settings})
    if resp.status_code != 201:
        msg = f"create collection: {resp.status_code} {resp.text[:400]}"
        raise RuntimeError(msg)
    data: dict[str, Any] = resp.json()
    return data


def upload(
    client: httpx.Client,
    collection_id: str,
    filename: str,
    content: bytes,
    **fields: Any,
) -> httpx.Response:
    """Upload one document (multipart); ``fields``: external_id, title, metadata (dict)."""
    data = {
        k: json.dumps(v) if k == "metadata" else str(v) for k, v in fields.items() if v is not None
    }
    return client.post(
        f"/v1/collections/{collection_id}/documents",
        files={"file": (filename, content)},
        data=data,
    )


def wait_job(client: httpx.Client, job_id: str, timeout: float = 600.0) -> dict[str, Any]:
    """Poll a job until done; raise when it fails or times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        job: dict[str, Any] = resp.json()
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            msg = f"job {job_id} ({job['kind']}) failed: {job.get('last_error')}"
            raise RuntimeError(msg)
        time.sleep(0.2)
    msg = f"job {job_id} not done after {timeout}s"
    raise TimeoutError(msg)


def wait_jobs(client: httpx.Client, job_ids: list[str], timeout: float = 1800.0) -> None:
    """Wait for every job (sequential polling; jobs run concurrently in the worker)."""
    deadline = time.monotonic() + timeout
    for job_id in job_ids:
        wait_job(client, job_id, timeout=max(1.0, deadline - time.monotonic()))


# --- fakellm ------------------------------------------------------------------------


def fake_script(rules: list[dict[str, Any]]) -> None:
    """``PUT /_fake/script`` (fakellm scripting API)."""
    httpx.put(f"{FAKELLM_URL}/_fake/script", json=rules, timeout=10.0).raise_for_status()


def fake_clear_script() -> None:
    """``DELETE /_fake/script``; also resets the ``{n}`` counters."""
    httpx.delete(f"{FAKELLM_URL}/_fake/script", timeout=10.0).raise_for_status()


def fake_clear_requests() -> None:
    """``DELETE /_fake/requests``."""
    httpx.delete(f"{FAKELLM_URL}/_fake/requests", timeout=10.0).raise_for_status()


def fake_requests() -> dict[str, Any]:
    """``GET /_fake/requests``: ``{"dropped": n, "requests": [...]}``."""
    resp = httpx.get(f"{FAKELLM_URL}/_fake/requests", timeout=30.0)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


# --- database -----------------------------------------------------------------------


def connect(dsn: str = ADMIN_DSN) -> psycopg.Connection[Any]:
    """Autocommit connection for assertions."""
    return psycopg.connect(dsn, autocommit=True)


def write_log(name: str, text: str) -> Path:
    """Write a text artefact under acceptance/logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / name
    path.write_text(text)
    return path
