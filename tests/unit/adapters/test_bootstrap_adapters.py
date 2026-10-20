# SPDX-License-Identifier: Apache-2.0
"""Adapter pieces the composition root relies on: tokenizers, lazy reranker, readiness,
pulls, embedding metrics, worker outcomes, webhook attempts, probes and migrations."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

import footnote.adapters.models.readiness as readiness_module
from footnote.adapters.embedders import (
    DefaultEmbedderFactory,
    OnnxEmbedder,
    OpenAICompatDefaults,
    OpenAICompatEmbedder,
)
from footnote.adapters.llm.probe import models_endpoint_problems
from footnote.adapters.models import (
    FileStatus,
    Manifest,
    ModelEntry,
    ModelFile,
    ModelNeed,
    ModelReadiness,
    pull_files,
)
from footnote.adapters.postgres import health
from footnote.adapters.postgres.migrate import (
    ADMIN_URL_ATTRIBUTE,
    alembic_config,
    find_migrations,
    head_revision,
)
from footnote.adapters.queue import JobContext, Worker, WorkerConfig
from footnote.adapters.rerankers import LazyOnnxReranker, OnnxReranker
from footnote.adapters.tokenizer import ManifestTokenizers, tokenizer_ids
from footnote.adapters.webhooks import WebhookConfig, WebhookJobHandler, WebhookSender
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import UnavailableError
from footnote.domain.jobs import PermanentJobError

from .conftest import build_tokenizer
from .queue.test_worker import FakeStore, eventually
from .webhooks.test_handler import JOB, Collections, Fakes, Jobs

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = b"weights"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _file(path: str = "tokenizer.json", payload: bytes = PAYLOAD, **kw: Any) -> ModelFile:
    return ModelFile(path, hashlib.sha256(payload).hexdigest(), len(payload), **kw)


def _entry(**kw: Any) -> ModelEntry:
    base: dict[str, Any] = {
        "id": "m",
        "kind": "embedder",
        "repo": "o/m",
        "revision": "r1",
        "files": {"tokenizer": _file(), "model": _file("onnx/model.onnx")},
    }
    return ModelEntry(**{**base, **kw})


# --- tokenizers -------------------------------------------------------------------------


def _tokenizer_models(tmp_path: Path) -> tuple[Manifest, Path]:
    models = tmp_path / "models"
    (models / "tok").mkdir(parents=True)
    build_tokenizer().save(str(models / "tok" / "tokenizer.json"))
    manifest = Manifest(
        {
            "tok": _entry(id="tok", files={"tokenizer": _file()}),
            "absent": _entry(id="absent", files={"tokenizer": _file()}),
            "plain": _entry(id="plain", files={"model": _file("m.onnx")}),
        }
    )
    return manifest, models


def test_tokenizer_ids_lists_entries_with_a_tokenizer_file(tmp_path: Path) -> None:
    manifest, _ = _tokenizer_models(tmp_path)
    assert tokenizer_ids(manifest) == frozenset({"tok", "absent"})


def test_manifest_tokenizers_counts_and_caches(tmp_path: Path) -> None:
    manifest, models = _tokenizer_models(tmp_path)
    tokenizers = ManifestTokenizers(manifest, models)
    counter = tokenizers.for_tokenizer("tok")
    assert counter.count("a b c") == 3
    assert tokenizers.for_tokenizer("tok") is counter


@pytest.mark.parametrize("tokenizer_id", ["nothing", "plain", "absent"])
def test_manifest_tokenizers_unavailable(tmp_path: Path, tokenizer_id: str) -> None:
    manifest, models = _tokenizer_models(tmp_path)
    with pytest.raises(UnavailableError):
        ManifestTokenizers(manifest, models).for_tokenizer(tokenizer_id)


def test_manifest_tokenizers_load_a_file_that_appears_later(tmp_path: Path) -> None:
    manifest, models = _tokenizer_models(tmp_path)
    tokenizers = ManifestTokenizers(manifest, models)
    with pytest.raises(UnavailableError):
        tokenizers.for_tokenizer("absent")
    (models / "absent").mkdir()
    build_tokenizer().save(str(models / "absent" / "tokenizer.json"))
    assert tokenizers.for_tokenizer("absent").count("a") == 1


# --- LazyOnnxReranker -------------------------------------------------------------------


def _reranker_entry() -> ModelEntry:
    return _entry(
        id="rr",
        kind="reranker",
        files={"tokenizer": _file()},
        variants={
            "fp32": _file("f.onnx", calibrated_threshold=0.1),
            "int8-arm64": _file("a.onnx", b"arm", calibrated_threshold=0.2),
            "int8-amd64": _file("x.onnx", b"amd", calibrated_threshold=0.3),
        },
    )


def test_lazy_reranker_reads_key_and_threshold_from_the_manifest(tmp_path: Path) -> None:
    entry = _reranker_entry()
    rr = LazyOnnxReranker(tmp_path, entry, variant="fp32")
    assert rr.model_key == entry.variants["fp32"].sha256
    assert rr.calibrated_threshold == 0.1  # nothing was loaded: the directory is empty


def test_lazy_reranker_selects_the_int8_file_of_another_arch(tmp_path: Path) -> None:
    entry = _reranker_entry()
    amd = LazyOnnxReranker(tmp_path, entry, variant="int8", arch="amd64")
    arm = LazyOnnxReranker(tmp_path, entry, variant="int8", arch="arm64")
    assert amd.model_key == entry.variants["int8-amd64"].sha256
    assert amd.calibrated_threshold == 0.3
    assert arm.model_key == entry.variants["int8-arm64"].sha256


async def test_lazy_reranker_empty_input_does_not_load(tmp_path: Path) -> None:
    rr = LazyOnnxReranker(tmp_path, _reranker_entry(), variant="fp32")
    assert await rr.score("q", []) == []


async def test_lazy_reranker_unavailable_while_files_are_missing(tmp_path: Path) -> None:
    rr = LazyOnnxReranker(tmp_path, _reranker_entry(), variant="fp32")
    with pytest.raises(UnavailableError):
        await rr.score("q", ["passage"])


async def test_lazy_reranker_loads_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loads: list[dict[str, Any]] = []

    class Fake:
        async def score(self, query: str, passages: Sequence[str]) -> list[float]:
            return [float(len(p)) for p in passages]

    def fake_from_manifest(models_dir: Path, entry: ModelEntry, **kw: Any) -> Fake:
        loads.append({"dir": models_dir, "id": entry.id, **kw})
        return Fake()

    monkeypatch.setattr(OnnxReranker, "from_manifest", staticmethod(fake_from_manifest))
    rr = LazyOnnxReranker(
        tmp_path, _reranker_entry(), variant="fp32", max_tokens=64, batch_size=4, intra_op_threads=2
    )
    assert await rr.score("q", ["ab", "abc"]) == [2.0, 3.0]
    assert await rr.score("q", ["a"]) == [1.0]
    assert loads == [
        {
            "dir": tmp_path,
            "id": "rr",
            "variant": "fp32",
            "arch": None,
            "max_tokens": 64,
            "batch_size": 4,
            "intra_op_threads": 2,
        }
    ]


# --- ModelReadiness / ModelNeed ---------------------------------------------------------


def test_model_need_files_selects_keys_or_every_plain_file() -> None:
    entry = _reranker_entry()
    assert set(ModelNeed(entry).files()) == {"tokenizer"}
    assert set(ModelNeed(entry, ("fp32", "tokenizer")).files()) == {"fp32", "tokenizer"}
    with pytest.raises(KeyError):
        ModelNeed(entry, ("int8-riscv",)).files()


def test_readiness_turns_ready_once_the_file_appears(tmp_path: Path) -> None:
    entry = _entry(files={"tokenizer": _file()})
    probe = ModelReadiness(tmp_path, [ModelNeed(entry)])
    (problem,) = probe.check()
    assert "m tokenizer: missing" in problem
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "tokenizer.json").write_bytes(PAYLOAD)
    assert probe.check() == []


def test_readiness_reports_a_mismatch(tmp_path: Path) -> None:
    entry = _entry(files={"tokenizer": _file()})
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "tokenizer.json").write_bytes(b"x" * len(PAYLOAD))
    (problem,) = ModelReadiness(tmp_path, [ModelNeed(entry)]).check()
    assert "mismatch" in problem


async def test_readiness_does_not_hash_a_verified_file_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []

    def counting(path: Path, file: ModelFile) -> FileStatus:
        calls.append(path)
        return FileStatus.OK if path.name == "tokenizer.json" else FileStatus.MISSING

    monkeypatch.setattr(readiness_module, "check_file", counting)
    probe = ModelReadiness(tmp_path, [ModelNeed(_entry(), ("model", "tokenizer"))])
    assert len(await probe.problems()) == 1
    assert len(await probe.problems()) == 1
    names = [p.name for p in calls]
    assert names.count("tokenizer.json") == 1
    assert names.count("model.onnx") == 2


# --- pull_files -------------------------------------------------------------------------


def test_pull_files_downloads_only_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _entry()
    present = tmp_path / "m" / "tokenizer.json"
    present.parent.mkdir(parents=True)
    present.write_bytes(PAYLOAD)
    downloads: list[tuple[str, Path]] = []

    def fake_download(client: httpx.Client, url: str, file: ModelFile, target: Path) -> None:
        downloads.append((url, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PAYLOAD)

    monkeypatch.setattr("footnote.adapters.models.pull._download", fake_download)
    paths = pull_files(tmp_path, entry, entry.files)
    assert paths == [present, tmp_path / "m" / "onnx" / "model.onnx"]
    assert downloads == [
        (
            "https://huggingface.co/o/m/resolve/r1/onnx/model.onnx",
            tmp_path / "m" / "onnx" / "model.onnx",
        )
    ]


def test_pull_files_uses_the_given_client_and_leaves_it_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients: list[httpx.Client] = []

    def fake_download(client: httpx.Client, url: str, file: ModelFile, target: Path) -> None:
        clients.append(client)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PAYLOAD)

    monkeypatch.setattr("footnote.adapters.models.pull._download", fake_download)
    entry = _entry(files={"tokenizer": _file()})
    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500))) as client:
        pull_files(tmp_path, entry, entry.files, client=client)
        assert clients == [client]
        assert not client.is_closed


# --- embedder factory metrics -----------------------------------------------------------


class _Embedder:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _factory(
    tmp_path: Path, observed: list[tuple[str, int]] | None = None
) -> DefaultEmbedderFactory:
    manifest = Manifest({"emb": _entry(id="emb", dimensions=4)})
    return DefaultEmbedderFactory(
        manifest,
        tmp_path,
        OpenAICompatDefaults(base_url="http://embed.invalid/v1"),
        on_embed=(lambda provider, n: observed.append((provider, n)))
        if observed is not None
        else None,
    )


async def test_on_embed_counts_texts_per_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(OnnxEmbedder, "from_manifest", staticmethod(lambda *_a, **_k: _Embedder()))
    observed: list[tuple[str, int]] = []
    embedder = _factory(tmp_path, observed).for_space(
        EmbeddingSpace(1, "s", "onnx_local", "emb", 4)
    )
    assert len(await embedder.embed_documents(["a", "b", "c"])) == 3
    await embedder.embed_query("q")
    await embedder.embed_documents([])
    assert observed == [("onnx_local", 3), ("onnx_local", 1)]


async def test_aclose_closes_openai_compat_clients(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    embedder = factory.for_space(EmbeddingSpace(2, "r", "openai_compat", "remote", 4))
    assert isinstance(embedder, OpenAICompatEmbedder)
    client = embedder._client
    assert not client.is_closed
    await factory.aclose()
    assert client.is_closed
    await factory.aclose()  # idempotent


# --- worker outcomes --------------------------------------------------------------------


async def _outcomes(handler: Any, **cfg: Any) -> list[tuple[str, str]]:
    store = FakeStore()
    store.add()
    seen: list[tuple[str, str]] = []
    config = WorkerConfig(worker_id="w", poll_interval=0.01, shutdown_timeout=1.0, **cfg)
    worker = Worker(
        store, {"k": handler}, config, rand=lambda: 1.0, on_outcome=lambda k, s: seen.append((k, s))
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        await eventually(lambda: bool(seen))
    finally:
        stop.set()
        await asyncio.wait_for(task, 5)
    return seen


async def test_worker_reports_success() -> None:
    async def handler(ctx: JobContext) -> None:
        return None

    assert await _outcomes(handler) == [("k", "succeeded")]


async def test_worker_reports_a_retry() -> None:
    async def handler(ctx: JobContext) -> None:
        raise ValueError("boom")

    seen = await _outcomes(handler, backoff_base=3600.0)
    assert seen == [("k", "retried")]


async def test_worker_reports_a_permanent_failure() -> None:
    async def handler(ctx: JobContext) -> None:
        raise PermanentJobError("gone")

    assert await _outcomes(handler) == [("k", "failed")]


async def test_worker_reports_exhausted_attempts_as_failed() -> None:
    async def handler(ctx: JobContext) -> None:
        raise ValueError("again")

    seen = await _outcomes(handler, max_attempts=1)
    assert seen == [("k", "failed")]


# --- webhook on_attempt -----------------------------------------------------------------


async def _attempt_statuses(respond: httpx.Response, attempts_made: int) -> list[str]:
    fakes = Fakes(attempts_made=attempts_made)
    statuses: list[str] = []

    async def resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    sender = WebhookSender(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: respond)),
        WebhookConfig(max_attempts=3),
        resolve,
    )
    handler = WebhookJobHandler(
        Jobs(), Collections(fakes), fakes, sender, on_attempt=statuses.append
    )

    class NoStore:
        pass

    with contextlib.suppress(RuntimeError, PermanentJobError):
        await handler(JobContext(JOB, NoStore(), "w"))  # type: ignore[arg-type]
    return statuses


@pytest.mark.parametrize(
    ("status_code", "attempts_made", "expected"),
    [(200, 0, "delivered"), (500, 1, "pending"), (500, 2, "failed")],
    ids=["delivered", "pending", "failed"],
)
async def test_webhook_handler_reports_each_attempt(
    status_code: int, attempts_made: int, expected: str
) -> None:
    assert await _attempt_statuses(httpx.Response(status_code), attempts_made) == [expected]


# --- models endpoint probe --------------------------------------------------------------


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_probe_ok_sends_key_and_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    async with _client(handler) as client:
        problems = await models_endpoint_problems(
            client, "http://llm.invalid/v1/", api_key="k", headers={"X-Org": "o"}
        )
    assert problems == []
    assert str(seen[0].url) == "http://llm.invalid/v1/models"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert seen[0].headers["x-org"] == "o"


async def test_probe_without_key_sends_no_authorization() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    async with _client(handler) as client:
        assert await models_endpoint_problems(client, "http://llm.invalid/v1") == []
    assert "authorization" not in seen[0].headers


async def test_probe_reports_a_server_error() -> None:
    async with _client(lambda _r: httpx.Response(500)) as client:
        (problem,) = await models_endpoint_problems(client, "http://llm.invalid/v1")
    assert "returned 500" in problem


async def test_probe_reports_a_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with _client(handler) as client:
        (problem,) = await models_endpoint_problems(client, "http://llm.invalid/v1")
    assert "ConnectError" in problem


# --- migrations -------------------------------------------------------------------------


def test_find_migrations_in_the_repository() -> None:
    path = find_migrations()
    assert path is not None
    assert path == ROOT / "migrations"
    revision = head_revision(path)
    assert isinstance(revision, str)
    assert revision


def test_find_migrations_skips_candidates_without_scripts(tmp_path: Path) -> None:
    (tmp_path / "migrations").mkdir()
    assert find_migrations([tmp_path]) is None
    (tmp_path / "migrations" / "env.py").write_text("", encoding="utf-8")
    assert find_migrations([tmp_path]) is None  # no versions/ yet
    (tmp_path / "migrations" / "versions").mkdir()
    assert find_migrations([tmp_path / "elsewhere", tmp_path]) == tmp_path / "migrations"


def test_head_revision_without_scripts_is_none() -> None:
    assert head_revision(None) is None


def test_alembic_config_carries_the_dsn_in_memory() -> None:
    cfg = alembic_config("postgresql://admin@db/x", ROOT / "migrations")
    assert cfg.attributes[ADMIN_URL_ATTRIBUTE] == "postgresql://admin@db/x"
    assert cfg.get_main_option("script_location") == str(ROOT / "migrations")
    assert "postgresql" not in str(cfg.cmd_opts or "")


# --- postgres health helpers (no database) ----------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.8.0", (0, 8, 0)),
        ("0.8.1", (0, 8, 1)),
        ("0.10.2", (0, 10, 2)),
        ("1.0", (1, 0)),
        ("0.8.0-dev", (0, 8, 0)),
        ("0.8.x", (0, 8, 0)),
    ],
)
def test_version_tuple(text: str, expected: tuple[int, ...]) -> None:
    assert health._version_tuple(text) == expected


def test_version_tuple_orders_numerically() -> None:
    assert health._version_tuple("0.10.0") > health.MIN_PGVECTOR
    assert health._version_tuple("0.7.4") < health.MIN_PGVECTOR


async def test_migration_problems_without_scripts_needs_no_database() -> None:
    (problem,) = await health.migration_problems(object(), None)  # type: ignore[arg-type]
    assert "migration scripts not found" in problem
