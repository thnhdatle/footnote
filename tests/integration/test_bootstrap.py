# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
import pytest_asyncio
import yaml
from fastapi import FastAPI, Request
from tokenizers import Tokenizer, models, pre_tokenizers
from typer.testing import CliRunner

from footnote.__main__ import app as cli
from footnote.api.health import readiness
from footnote.bootstrap import (
    ServeOptions,
    _Server,
    bind,
    bound_address,
    open_runtime,
    run_process,
)
from footnote.config import Settings, load_settings
from tests.integration.conftest import PgInfo

pytestmark = pytest.mark.integration

ADMIN_KEY = "boot-admin-key"
DIMS = 16
SPACE = "it-boot-bow"
EMBED_MODEL = "fake-bow"
CHAT_MODEL = "fake-chat"
ANSWER = "Apple trees bloom in spring [1]."
QUESTION = "apple trees bloom"
DOCUMENT = (
    b"# Orchards\n\nApple trees bloom in spring and the orchard smells of blossom.\n\n"
    b"## Harvest\n\nPears ripen in autumn when the nights get colder.\n"
)
WORDS = re.compile(r"[a-z0-9]+")


def bow_vector(text: str) -> list[float]:
    vector = [0.0] * DIMS
    for word in WORDS.findall(text.lower()):
        vector[hashlib.sha256(word.encode()).digest()[0] % DIMS] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


@dataclass
class FakeProvider:
    base_url: str = ""
    embedded: list[str] = field(default_factory=list)
    chats: list[dict[str, Any]] = field(default_factory=list)
    model_probes: int = 0


def fake_provider_app(state: FakeProvider) -> FastAPI:
    api = FastAPI()

    @api.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        state.model_probes += 1
        return {"object": "list", "data": [{"id": EMBED_MODEL}, {"id": CHAT_MODEL}]}

    @api.post("/v1/embeddings")
    async def embeddings(request: Request) -> dict[str, Any]:
        body = await request.json()
        texts = list(body["input"])
        state.embedded.extend(texts)
        return {
            "object": "list",
            "model": body["model"],
            "data": [
                {"object": "embedding", "index": i, "embedding": bow_vector(t)}
                for i, t in enumerate(texts)
            ],
        }

    @api.post("/v1/chat/completions", response_model=None)
    async def chat(request: Request) -> Any:
        from fastapi.responses import StreamingResponse  # noqa: PLC0415

        body = await request.json()
        state.chats.append(body)
        usage = {"prompt_tokens": 12, "completion_tokens": 7}
        if not body.get("stream"):
            return {
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ANSWER}}],
                "usage": usage,
            }

        async def events() -> AsyncIterator[bytes]:
            for piece in (ANSWER[:10], ANSWER[10:]):
                delta = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                yield b"data: " + orjson.dumps(delta) + b"\n\n"
            yield b"data: " + orjson.dumps({"choices": [], "usage": usage}) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return api


@pytest_asyncio.fixture
async def provider() -> AsyncIterator[FakeProvider]:
    import uvicorn  # noqa: PLC0415

    state = FakeProvider()
    sock = bind("127.0.0.1:0")
    state.base_url = f"http://{bound_address(sock)}/v1"
    server = _Server(uvicorn.Config(fake_provider_app(state), log_config=None, lifespan="off"))
    task = asyncio.create_task(server.serve([sock]))
    try:
        yield state
    finally:
        server.should_exit = True
        await task


def write_models(root: Path) -> tuple[Path, Path]:
    models_dir = root / "models"
    target = models_dir / "it-words" / "tokenizer.json"
    target.parent.mkdir(parents=True)
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(["apple", "trees", "pears"])}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(target))
    data = target.read_bytes()
    manifest = root / "manifest.yaml"
    entry = {
        "id": "it-words",
        "kind": "tokenizer",
        "repo": "local/it-words",
        "revision": "0",
        "files": {
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        },
    }
    manifest.write_text(yaml.safe_dump({"models": [entry]}), encoding="utf-8")
    return models_dir, manifest


def boot_env(pg: PgInfo, provider: FakeProvider, root: Path) -> dict[str, str]:
    models_dir, manifest = write_models(root)
    space = {
        "name": SPACE,
        "provider": "openai_compat",
        "model": EMBED_MODEL,
        "dimensions": DIMS,
        "calibrated_threshold": 0.0,
    }
    return {
        "FOOTNOTE_SERVER_LISTEN": "127.0.0.1:0",
        "FOOTNOTE_ADMIN_LISTEN": "127.0.0.1:0",
        "FOOTNOTE_SHUTDOWN_TIMEOUT": "10s",
        "FOOTNOTE_LOG_LEVEL": "warning",
        "FOOTNOTE_DATABASE_URL": pg.app,
        "FOOTNOTE_ADMIN_DATABASE_URL": pg.admin,
        "FOOTNOTE_AUTH_ADMIN_KEY": ADMIN_KEY,
        "FOOTNOTE_LLM_BASE_URL": provider.base_url,
        "FOOTNOTE_LLM_MODEL": CHAT_MODEL,
        "FOOTNOTE_EMBEDDING_BASE_URL": provider.base_url,
        "FOOTNOTE_EMBEDDING_SPACES": orjson.dumps([space]).decode(),
        "FOOTNOTE_EMBEDDING_DEFAULT_SPACE": SPACE,
        "FOOTNOTE_EMBEDDING_MAX_SPACES": "64",
        "FOOTNOTE_RERANK_PROVIDER": "noop",
        "FOOTNOTE_MODELS_DIR": str(models_dir),
        "FOOTNOTE_MODELS_MANIFEST": str(manifest),
        "FOOTNOTE_CHUNKING_TOKENIZER": "it-words",
        "FOOTNOTE_INGEST_POLL_INTERVAL": "100ms",
        "FOOTNOTE_INGEST_GC_GRACE": "1h",
    }


def metric(exposition: str, series: str) -> float:
    for line in exposition.splitlines():
        name, _, value = line.rpartition(" ")
        if name == series:
            return float(value)
    return 0.0


@dataclass
class Booted:
    data: str
    admin: str
    stop: asyncio.Event
    task: asyncio.Task[None]


async def boot(settings: Settings, env: Mapping[str, str]) -> Booted:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[dict[str, str]] = loop.create_future()
    stop = asyncio.Event()

    def on_ready(addresses: Mapping[str, str]) -> None:
        ready.set_result(dict(addresses))

    options = ServeOptions(data_plane=True, worker=True, on_ready=on_ready)
    task = asyncio.create_task(run_process(settings, options, stop, env))
    done, _ = await asyncio.wait({ready, task}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        task.result()
    assert ready in done, "process did not start listening within 60s"
    addresses = ready.result()
    return Booted(f"http://{addresses['data']}", f"http://{addresses['admin']}", stop, task)


async def _until(predicate: Any, timeout: float = 60.0) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        value = await predicate()
        if value:
            return value
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.1)


async def _tenant_key(client: httpx.AsyncClient) -> dict[str, str]:
    admin = {"Authorization": f"Bearer {ADMIN_KEY}"}
    tenant = await client.post(
        "/v1/admin/tenants", headers=admin, json={"name": f"boot-{uuid.uuid4().hex[:6]}"}
    )
    assert tenant.status_code == 201, tenant.text
    key = await client.post(
        f"/v1/admin/tenants/{tenant.json()['id']}/keys", headers=admin, json={"name": "it"}
    )
    assert key.status_code == 201, key.text
    return {"Authorization": f"Bearer {key.json()['key']}"}


async def _upload_and_index(client: httpx.AsyncClient, auth: dict[str, str]) -> str:
    col = await client.post("/v1/collections", headers=auth, json={"name": "orchard"})
    assert col.status_code == 201, col.text
    cid = col.json()["id"]
    up = await client.post(
        f"/v1/collections/{cid}/documents",
        headers=auth,
        files={"file": ("orchard.md", DOCUMENT)},
        data={"external_id": "orchard"},
    )
    assert up.status_code == 202, up.text

    async def job_done() -> dict[str, Any] | None:
        job = await client.get(f"/v1/jobs/{up.json()['job_id']}", headers=auth)
        assert job.status_code == 200, job.text
        body: dict[str, Any] = job.json()
        assert body["status"] != "failed", body
        return body if body["status"] == "done" else None

    await _until(job_done)
    return str(cid)


async def _search_and_answer(client: httpx.AsyncClient, auth: dict[str, str], cid: str) -> None:
    hits = await client.post(
        f"/v1/collections/{cid}/search", headers=auth, json={"query": QUESTION}
    )
    assert hits.status_code == 200, hits.text
    hit_ids = [h["chunk_id"] for h in hits.json()["hits"]]
    assert "Apple trees bloom" in hits.json()["hits"][0]["text"], hits.text
    answer = await client.post(
        f"/v1/collections/{cid}/answers", headers=auth, json={"question": QUESTION}
    )
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["abstained"], body["unsupported"]) == (False, False), body
    assert body["answer"] == ANSWER
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["chunk_id"] == hit_ids[0]
    streamed = await client.post(
        f"/v1/collections/{cid}/answers",
        headers=auth,
        json={"question": QUESTION, "stream": True},
    )
    assert streamed.status_code == 200, streamed.text
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: citations" in streamed.text
    assert "event: done" in streamed.text


async def test_serve_with_worker_upload_search_answer(
    pg: PgInfo, provider: FakeProvider, tmp_path: Path
) -> None:
    env = boot_env(pg, provider, tmp_path)
    booted = await boot(load_settings(env=env), env)
    try:
        async with (
            httpx.AsyncClient(base_url=booted.data, timeout=30) as client,
            httpx.AsyncClient(base_url=booted.admin, timeout=30) as ops,
        ):
            ready = await ops.get("/readyz")
            assert ready.status_code == 200, ready.text
            assert set(ready.json()["checks"]) >= {
                "database",
                "admin_database",
                "migrations",
                "postgres_versions",
                "app_role",
                "models",
                "llm",
                "embedding",
            }
            assert (await client.get("/v1/version")).json()["version"]
            auth = await _tenant_key(client)
            cid = await _upload_and_index(client, auth)
            await _search_and_answer(client, auth, cid)
            metrics = (await ops.get("/metrics")).text
    finally:
        booted.stop.set()
        await asyncio.wait_for(booted.task, timeout=60)
    succeeded = metric(metrics, 'footnote_jobs_total{kind="ingest",status="succeeded"}')
    assert succeeded >= 1
    assert metric(metrics, 'footnote_ingest_jobs_total{status="succeeded"}') == succeeded
    assert metric(metrics, 'footnote_ingest_duration_seconds_count{stage="embed"}') >= 1
    assert 'footnote_embedding_texts_total{provider="openai_compat"}' in metrics
    assert 'footnote_answers_total{outcome="answered"} 2.0' in metrics
    assert 'footnote_jobs_queue_depth{kind="ingest"}' in metrics
    assert any("Apple trees bloom" in t for t in provider.embedded)
    assert len(provider.chats) == 2
    assert provider.model_probes >= 2
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient(base_url=booted.data) as client:
            await client.get("/healthz")


async def test_readyz_rejects_a_superuser_app_role(
    pg: PgInfo, provider: FakeProvider, tmp_path: Path
) -> None:
    env = {**boot_env(pg, provider, tmp_path), "FOOTNOTE_DATABASE_URL": pg.superuser}
    async with open_runtime(load_settings(env=env), env=env) as rt:
        checks = await readiness(rt.api)
    assert "app role is superuser" in checks["app_role"]
    assert "app role has BYPASSRLS" in checks["app_role"]
    assert checks["migrations"] == []
    assert checks["models"] == []


def test_cli_migrate_and_admin_groups(pg: PgInfo, tmp_path: Path) -> None:
    runner = CliRunner()
    env = {"FOOTNOTE_ADMIN_DATABASE_URL": pg.admin, "FOOTNOTE_EMBEDDING_MAX_SPACES": "64"}
    migrated = runner.invoke(cli, ["migrate"], env=env)
    assert migrated.exit_code == 0, migrated.output
    created = runner.invoke(cli, ["tenants", "create", "cli-tenant"], env=env)
    assert created.exit_code == 0, created.output
    tenant = orjson.loads(created.stdout)
    issued = runner.invoke(cli, ["keys", "create", tenant["id"], "--name", "cli"], env=env)
    assert issued.exit_code == 0, issued.output
    key = orjson.loads(issued.stdout)
    assert key["key"].startswith("fn_")
    listed = runner.invoke(cli, ["keys", "list", tenant["id"]], env=env)
    assert [orjson.loads(line)["id"] for line in listed.stdout.splitlines()] == [key["id"]]
    rotated = runner.invoke(cli, ["keys", "rotate", tenant["id"], key["id"]], env=env)
    assert rotated.exit_code == 0, rotated.output
    revoked = runner.invoke(cli, ["keys", "revoke", tenant["id"], key["id"]], env=env)
    assert revoked.exit_code == 0, revoked.output
    out = tmp_path / "export.ndjson"
    exported = runner.invoke(cli, ["tenants", "export", tenant["id"], "-o", str(out)], env=env)
    assert exported.exit_code == 0, exported.output
    registered = runner.invoke(
        cli,
        ["spaces", "register", "cli-space", "--provider", "openai_compat", "--model", "m",
         "--dimensions", "12"],
        env=env,
    )  # fmt: skip
    assert registered.exit_code == 0, registered.output
    spaces = runner.invoke(cli, ["spaces", "list"], env=env)
    assert "cli-space" in spaces.stdout
    tenants = runner.invoke(cli, ["tenants", "list", "--limit", "500"], env=env)
    assert tenant["id"] in tenants.stdout
    deleted = runner.invoke(cli, ["tenants", "delete", tenant["id"]], env=env)
    assert deleted.exit_code == 0, deleted.output
    assert orjson.loads(deleted.stdout)["job_id"]
    missing = runner.invoke(cli, ["keys", "list", str(uuid.uuid4())], env=env)
    assert missing.exit_code == 1
    assert "not_found" in missing.output
