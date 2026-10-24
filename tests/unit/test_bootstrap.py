# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from footnote import bootstrap
from footnote.adapters.models import Manifest, host_arch, load_manifest
from footnote.app.jobs import JobServices
from footnote.bootstrap import (
    BuildInfo,
    Runtime,
    SystemClock,
    bind,
    bound_address,
    build_info,
    job_handlers,
    load_model_manifest,
    manifest_problems,
    report_queue_depth,
    required_dsn,
    required_models,
    space_completer,
    space_specs,
    split_listen,
)
from footnote.config import ConfigError, Settings, load_settings
from footnote.domain.embedding import SpaceSpec
from footnote.domain.errors import ValidationError
from footnote.domain.jobs import JobKind
from footnote.observability import Metrics

from .app.rig import build

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "models" / "manifest.yaml"
BASE_ENV = {"FOOTNOTE_MODELS_MANIFEST": str(MANIFEST)}


def _settings(**env: str) -> Settings:
    return load_settings(env={**BASE_ENV, **env})


def _manifest() -> Manifest:
    return load_manifest(MANIFEST)


def _spaces_env(*spaces: dict[str, Any], default: str | None = None) -> dict[str, str]:
    return {
        "FOOTNOTE_EMBEDDING_SPACES": json.dumps(list(spaces)),
        "FOOTNOTE_EMBEDDING_DEFAULT_SPACE": default or spaces[0]["name"],
    }


ONNX_SPACE = {"name": "bge", "provider": "onnx_local", "model": "bge-small-en-v1.5"}
REMOTE_SPACE = {"name": "remote", "provider": "openai_compat", "model": "m", "dimensions": 8}


def test_build_info_reads_commit_and_date() -> None:
    info = build_info({"FOOTNOTE_BUILD_COMMIT": "abc123", "FOOTNOTE_BUILD_DATE": "2026-01-01"})
    assert (info.commit, info.build_date) == ("abc123", "2026-01-01")
    assert info.version.count(".") == 2


@pytest.mark.parametrize(
    "env", [{}, {"FOOTNOTE_BUILD_COMMIT": "", "FOOTNOTE_BUILD_DATE": ""}], ids=["absent", "empty"]
)
def test_build_info_defaults_to_unknown(env: dict[str, str]) -> None:
    info = build_info(env)
    assert (info.commit, info.build_date) == ("unknown", "unknown")


def test_build_info_uses_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTNOTE_BUILD_COMMIT", "fromenv")
    assert build_info().commit == "fromenv"


def test_system_clock_reports_wall_and_monotonic_time() -> None:
    clock = SystemClock()
    assert abs(clock.now() - time.time()) < 5
    first = clock.monotonic()
    assert clock.monotonic() >= first


@pytest.mark.parametrize(
    ("admin", "name", "env"),
    [
        (False, "database.url", "FOOTNOTE_DATABASE_URL"),
        (True, "admin_database.url", "FOOTNOTE_ADMIN_DATABASE_URL"),
    ],
    ids=["app", "admin"],
)
def test_required_dsn(admin: bool, name: str, env: str) -> None:
    with pytest.raises(ConfigError) as err:
        required_dsn(_settings(), admin=admin)
    assert name in err.value.errors[0]
    assert env in err.value.errors[0]
    assert required_dsn(_settings(**{env: "postgresql://u@h/db"}), admin=admin) == (
        "postgresql://u@h/db"
    )


def test_load_model_manifest_reports_a_bad_path(tmp_path: Path) -> None:
    settings = load_settings(env={"FOOTNOTE_MODELS_MANIFEST": str(tmp_path / "absent.yaml")})
    with pytest.raises(ConfigError, match=r"models\.manifest"):
        load_model_manifest(settings)


def test_load_model_manifest_reads_the_shipped_file() -> None:
    assert "bge-small-en-v1.5" in load_model_manifest(_settings()).models


def test_shipped_defaults_have_no_manifest_problems() -> None:
    assert manifest_problems(_settings(), _manifest()) == []


def test_manifest_problems_reports_each_reference() -> None:
    settings = load_settings(
        env={
            **BASE_ENV,
            **_spaces_env(
                {"name": "a", "provider": "onnx_local", "model": "nope"},
                {"name": "b", "provider": "onnx_local", "model": "bge-small-en-v1.5",
                 "dimensions": 99},
                {"name": "c", "provider": "onnx_local", "model": "ms-marco-MiniLM-L6-v2"},
                default="a",
            ),
            "FOOTNOTE_RERANK_MODEL": "unknown-reranker",
            "FOOTNOTE_CHUNKING_TOKENIZER": "nothing",
        }
    )  # fmt: skip
    problems = manifest_problems(settings, _manifest())
    assert len(problems) == 5
    assert "embedding.spaces.0.model" in problems[0]
    assert "embedding.spaces.1.dimensions" in problems[1]
    assert "has 384" in problems[1]
    assert "embedding.spaces.2.model" in problems[2]
    assert "rerank.model" in problems[3]
    assert "chunking.tokenizer" in problems[4]


def test_manifest_problems_reranker_without_variant_or_tokenizer() -> None:
    manifest = _manifest()
    reranker = manifest.models["ms-marco-MiniLM-L6-v2"]
    stripped = replace(reranker, variants={})
    problems = manifest_problems(_settings(), Manifest({**manifest.models, stripped.id: stripped}))
    assert len(problems) == 1
    assert "rerank.variant" in problems[0]


def test_manifest_problems_skips_remote_spaces_and_noop_rerank() -> None:
    settings = _settings(
        **_spaces_env(REMOTE_SPACE),
        FOOTNOTE_EMBEDDING_BASE_URL="http://localhost:1/v1",
        FOOTNOTE_RERANK_PROVIDER="noop",
        FOOTNOTE_RERANK_MODEL="whatever",
    )
    assert manifest_problems(settings, _manifest()) == []


def test_required_models_for_the_shipped_defaults() -> None:
    needs = {n.entry.id: n for n in required_models(_settings(), _manifest())}
    assert set(needs) == {"bge-small-en-v1.5", "ms-marco-MiniLM-L6-v2"}
    assert needs["bge-small-en-v1.5"].keys == ("model", "tokenizer")
    assert needs["ms-marco-MiniLM-L6-v2"].keys == (f"int8-{host_arch()}", "tokenizer")
    assert set(needs["ms-marco-MiniLM-L6-v2"].files()) == {f"int8-{host_arch()}", "tokenizer"}


def test_required_models_with_remote_space_and_noop_rerank_is_only_the_tokenizer() -> None:
    settings = _settings(
        **_spaces_env(REMOTE_SPACE),
        FOOTNOTE_EMBEDDING_BASE_URL="http://localhost:1/v1",
        FOOTNOTE_RERANK_PROVIDER="noop",
    )
    (need,) = required_models(settings, _manifest())
    assert (need.entry.id, need.keys) == ("bge-small-en-v1.5", ("tokenizer",))


def test_required_models_ignores_unknown_ids() -> None:
    settings = _settings(FOOTNOTE_CHUNKING_TOKENIZER="nothing", FOOTNOTE_RERANK_MODEL="nothing")
    assert [n.entry.id for n in required_models(settings, _manifest())] == ["bge-small-en-v1.5"]


def _spec(**kw: Any) -> SpaceSpec:
    base: dict[str, Any] = {
        "name": "s",
        "provider": "onnx_local",
        "model": "bge-small-en-v1.5",
        "dimensions": 0,
    }
    return SpaceSpec(**{**base, **kw})


def test_completer_fills_dimensions_and_manifest_threshold() -> None:
    done = space_completer(_manifest())(_spec())
    assert (done.dimensions, done.calibrated_threshold, done.normalise) == (384, 0.7698, True)


def test_completer_keeps_an_explicit_threshold_and_matching_dimensions() -> None:
    done = space_completer(_manifest())(_spec(dimensions=384, calibrated_threshold=0.6))
    assert (done.dimensions, done.calibrated_threshold) == (384, 0.6)


def test_completer_passes_a_remote_space_through() -> None:
    spec = _spec(provider="openai_compat", model="m", dimensions=8)
    assert space_completer(_manifest())(spec) == spec


@pytest.mark.parametrize(
    ("spec", "field_name"),
    [
        (_spec(dimensions=10), "dimensions"),
        (_spec(model="nope"), "model"),
        (_spec(model="ms-marco-MiniLM-L6-v2"), "model"),
        (_spec(provider="openai_compat", model="m", dimensions=0), "dimensions"),
        (_spec(provider="mystery"), "provider"),
    ],
    ids=["wrong-dims", "unknown-model", "not-embedder", "remote-no-dims", "unknown-provider"],
)
def test_completer_rejects(spec: SpaceSpec, field_name: str) -> None:
    with pytest.raises(ValidationError) as err:
        space_completer(_manifest())(spec)
    assert err.value.field == field_name


def test_space_specs_maps_configured_spaces() -> None:
    settings = _settings(
        **_spaces_env(
            {**ONNX_SPACE, "calibrated_threshold": 0.5, "normalise": False},
            REMOTE_SPACE,
        ),
        FOOTNOTE_EMBEDDING_BASE_URL="http://localhost:1/v1",
    )
    onnx, remote = space_specs(settings)
    assert (onnx.name, onnx.dimensions, onnx.calibrated_threshold, onnx.normalise) == (
        "bge",
        0,
        0.5,
        False,
    )
    assert (remote.provider, remote.dimensions) == ("openai_compat", 8)


def test_job_handlers_cover_every_kind() -> None:
    rig = build()
    services = JobServices(
        rig.jobs,
        rig.collections,
        rig.ingest,
        rig.reindex,
        rig.reembed,
        rig.gc,
        rig.tenants,  # type: ignore[arg-type]
    )

    async def webhook(ctx: Any) -> None:
        return None

    handlers = job_handlers(services, webhook)
    assert set(handlers) == {k.value for k in JobKind}
    assert handlers["webhook"] is webhook
    assert JobKind.WEBHOOK.value == "webhook"


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("127.0.0.1:8000", ("127.0.0.1", 8000)),
        (":8000", ("0.0.0.0", 8000)),  # noqa: S104
        ("[::1]:9", ("::1", 9)),
        ("localhost:0", ("localhost", 0)),
    ],
    ids=["ipv4", "any", "ipv6", "name"],
)
def test_split_listen(address: str, expected: tuple[str, int]) -> None:
    assert split_listen(address) == expected


def test_bind_port_zero_gets_a_real_port() -> None:
    sock = bind("127.0.0.1:0")
    try:
        host, _, port = bound_address(sock).rpartition(":")
        assert host == "127.0.0.1"
        assert int(port) > 0
        with socket.create_connection(("127.0.0.1", int(port)), timeout=2):
            pass
    finally:
        sock.close()


def test_bind_empty_host_listens_on_every_interface() -> None:
    sock = bind(":0")
    try:
        assert bound_address(sock).startswith("0.0.0.0:")
    finally:
        sock.close()


def test_bind_ipv6_loopback() -> None:
    try:
        sock = bind("[::1]:0")
    except OSError:
        return
    try:
        assert bound_address(sock).startswith("[::1]:")
    finally:
        sock.close()


def test_bind_a_taken_port_raises_and_releases_the_socket() -> None:
    first = bind("127.0.0.1:0")
    try:
        port = first.getsockname()[1]
        with pytest.raises(OSError):  # noqa: PT011
            bind(f"127.0.0.1:{port}")
    finally:
        first.close()


@dataclass
class _Pools:
    admin: object = field(default_factory=object)


class _Log:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **kw: Any) -> None:
        self.warnings.append((event, kw))


def _runtime(log: _Log) -> Runtime:
    rt = Runtime.__new__(Runtime)
    rt.metrics = Metrics(version="0", commit="c")
    rt.pools = _Pools()  # type: ignore[assignment]
    rt.log = log  # type: ignore[assignment]
    return rt


def _gauge(rt: Runtime, kind: str) -> float:
    body = rt.metrics.render()[0].decode()
    prefix = f'footnote_jobs_queue_depth{{kind="{kind}"}} '
    (line,) = (ln for ln in body.splitlines() if ln.startswith(prefix))
    return float(line.removeprefix(prefix))


async def _eventually(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.005)


async def test_report_queue_depth_sets_gauges_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    async def depth(admin: object) -> dict[str, int]:
        calls.append(admin)
        return {"ingest": 3, "webhook": 1}

    monkeypatch.setattr(bootstrap.pg_health, "queue_depth", depth)
    rt = _runtime(_Log())
    stop = asyncio.Event()
    task = asyncio.create_task(report_queue_depth(rt, stop, 0.01))
    await _eventually(lambda: len(calls) >= 2)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert calls[0] is rt.pools.admin  # type: ignore[attr-defined]
    assert _gauge(rt, "ingest") == 3
    assert _gauge(rt, "webhook") == 1
    assert _gauge(rt, "reindex") == 0


async def test_report_queue_depth_logs_errors_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def depth(admin: object) -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("db down")
        return {"ingest": 7}

    monkeypatch.setattr(bootstrap.pg_health, "queue_depth", depth)
    log = _Log()
    rt = _runtime(log)
    stop = asyncio.Event()
    task = asyncio.create_task(report_queue_depth(rt, stop, 0.01))
    await _eventually(lambda: calls >= 2)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert log.warnings[0] == ("queue depth query failed", {"error": "db down"})
    assert _gauge(rt, "ingest") == 7


async def test_report_queue_depth_returns_at_once_when_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def depth(admin: object) -> dict[str, int]:
        raise AssertionError("must not be queried")

    monkeypatch.setattr(bootstrap.pg_health, "queue_depth", depth)
    stop = asyncio.Event()
    stop.set()
    await asyncio.wait_for(report_queue_depth(_runtime(_Log()), stop, 60.0), 5)


class _FakeServer:
    def __init__(self, name: str, order: list[str]) -> None:
        self.name = name
        self.order = order
        self.should_exit = False


class _App:
    class state:  # noqa: N801
        draining = False


async def test_shutdown_stops_data_plane_and_worker_before_admin() -> None:
    order: list[str] = []
    data = _FakeServer("data", order)
    admin = _FakeServer("admin", order)
    worker_stop = asyncio.Event()

    async def run(server: _FakeServer) -> None:
        while not server.should_exit:
            await asyncio.sleep(0.005)
        order.append(f"{server.name} stopped")

    async def worker() -> None:
        await worker_stop.wait()
        order.append("worker stopped")
        assert not admin.should_exit

    apps = [_App(), _App()]
    data_task = asyncio.create_task(run(data))
    admin_task = asyncio.create_task(run(admin))
    worker_task = asyncio.create_task(worker())
    await bootstrap._shutdown(
        apps,  # type: ignore[arg-type]
        [(data, data_task)],  # type: ignore[list-item]
        worker_task,
        worker_stop,
        (admin, admin_task),  # type: ignore[arg-type]
    )
    assert all(a.state.draining for a in apps)
    assert order.index("admin stopped") == len(order) - 1
    assert {"data stopped", "worker stopped"} <= set(order)


async def test_shutdown_reraises_a_failed_server() -> None:
    admin = _FakeServer("admin", [])

    async def boom() -> None:
        raise RuntimeError("serve failed")

    async def ok() -> None:
        return None

    task = asyncio.create_task(boom())
    with pytest.raises(RuntimeError, match="serve failed"):
        await bootstrap._shutdown(
            [],
            [(admin, task)],
            None,
            asyncio.Event(),
            (admin, asyncio.create_task(ok())),  # type: ignore[list-item,arg-type]
        )


def test_is_domain_error_and_build_info_defaults() -> None:
    assert bootstrap.is_domain_error(ValidationError("f", "m"))
    assert bootstrap.is_domain_error(OSError("port in use"))
    assert not bootstrap.is_domain_error(RuntimeError("bug"))
    assert BuildInfo("1").commit == "unknown"


def test_worker_id_is_unique() -> None:
    assert bootstrap.worker_id() != bootstrap.worker_id()


def test_wrapped_embedders_wrap_each_space_once() -> None:
    from footnote.domain.embedding import EmbeddingSpace  # noqa: PLC0415

    from .app.fakes import FakeEmbedderFactory  # noqa: PLC0415

    inner = FakeEmbedderFactory()
    wrapped: list[tuple[str, object]] = []

    def wrap(space: EmbeddingSpace, embedder: Any) -> Any:
        wrapped.append((space.name, embedder))
        return ("wrapped", embedder)

    factory = bootstrap.WrappedEmbedders(inner, wrap)
    a = EmbeddingSpace(1, "a", "fake", "m", 4)
    b = EmbeddingSpace(2, "b", "fake", "m", 4)
    first = factory.for_space(a)
    assert factory.for_space(a) is first
    assert first == ("wrapped", inner.embedder)
    factory.for_space(b)
    assert [name for name, _ in wrapped] == ["a", "b"]


def test_make_reranker_builds_the_requested_variant() -> None:
    settings = _settings()
    manifest = _manifest()
    configured = bootstrap.make_reranker(settings, manifest)
    fp32 = bootstrap.make_reranker(settings, manifest, "fp32")
    assert configured is not None
    assert fp32 is not None
    entry = manifest.models[settings.rerank.model]
    assert fp32.model_key == entry.variants["fp32"].sha256
    assert configured.model_key != fp32.model_key
    assert bootstrap.make_reranker(_settings(FOOTNOTE_RERANK_PROVIDER="noop"), manifest) is None
