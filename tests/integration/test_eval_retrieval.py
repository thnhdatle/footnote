# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import orjson
import psycopg
import pytest
import uvicorn
from typer.testing import CliRunner

from footnote.__main__ import app as cli
from footnote.app.evaluation.cache import is_warm
from footnote.app.evaluation.datasets import load_beir
from footnote.app.evaluation.retrieval import EvalOptions, plan_ablation, run_retrieval_eval
from footnote.app.evaluation.runtime import open_eval_runtime
from footnote.bootstrap import _Server, bind, bound_address
from footnote.config import load_settings
from tests.integration.conftest import PgInfo
from tests.integration.test_bootstrap import FakeProvider, boot_env, fake_provider_app

pytestmark = pytest.mark.integration

TOPICS = {
    "apple": "apple orchard blossom tree fruit harvest",
    "ocean": "ocean wave tide salt coral reef",
    "rocket": "rocket engine fuel launch orbit thrust",
    "violin": "violin string bow music concert melody",
    "glacier": "glacier ice snow mountain cold melt",
    "bread": "bread flour yeast oven bake crust",
}


@pytest.fixture
def provider() -> Iterator[FakeProvider]:
    state = FakeProvider()
    sock = bind("127.0.0.1:0")
    state.base_url = f"http://{bound_address(sock)}/v1"
    server = _Server(uvicorn.Config(fake_provider_app(state), log_config=None, lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def write_dataset(root: Path) -> Path:
    path = root / "data" / "tiny"
    (path / "qrels").mkdir(parents=True)
    corpus, queries = [], []
    for n, (topic, words) in enumerate(TOPICS.items()):
        corpus.append({"_id": f"{topic}-gold", "title": topic.title(), "text": f"{words}. " * 3})
        corpus.append({"_id": f"{topic}-side", "title": "Notes", "text": f"{topic} misc page {n}"})
        queries.append({"_id": f"q-{topic}", "text": " ".join(words.split()[:3])})
    (path / "corpus.jsonl").write_bytes(b"\n".join(orjson.dumps(d) for d in corpus) + b"\n")
    (path / "queries.jsonl").write_bytes(b"\n".join(orjson.dumps(q) for q in queries) + b"\n")
    topics = list(TOPICS)
    for split, chosen in (("test", topics[:4]), ("train", topics[2:])):
        lines = ["query-id\tcorpus-id\tscore"] + [f"q-{t}\t{t}-gold\t1" for t in chosen]
        (path / "qrels" / f"{split}.tsv").write_text("\n".join(lines) + "\n")
    return path


def eval_env(pg: PgInfo, provider: FakeProvider, root: Path) -> dict[str, str]:
    env = boot_env(pg, provider, root)
    env["FOOTNOTE_EVAL_DATA_DIR"] = str(root / "data")
    env["FOOTNOTE_EVAL_CACHE_DIR"] = str(root / "cache")
    env["FOOTNOTE_CHUNKING_MIN_TOKENS"] = "0"
    write_dataset(root)
    return env


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eval_tenants(pg: PgInfo) -> int:
    with psycopg.connect(pg.admin) as conn:
        row = conn.execute(
            "SELECT count(*) FROM footnote.tenants WHERE name LIKE 'eval-tiny-%'"
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_cli_eval_retrieval_end_to_end_and_cached_rerun(
    pg: PgInfo, provider: FakeProvider, tmp_path: Path
) -> None:
    env = eval_env(pg, provider, tmp_path)
    runner = CliRunner()
    args = [
        "eval",
        "retrieval",
        "--dataset",
        "tiny",
        "--modes",
        "vector,lexical,hybrid",
        "--chunking",
        "heading,recursive",
        "--output-dir",
        str(tmp_path / "results"),
    ]
    first = runner.invoke(cli, args, env=env)
    assert first.exit_code == 0, first.output
    runs = sorted((tmp_path / "results").iterdir())
    assert len(runs) == 1
    rows = read_csv(runs[0] / "retrieval.csv")
    names = [r["name"] for r in rows]
    assert names == [
        "vector [heading]",
        "lexical any+ts_rank [heading]",
        "hybrid lexical weight 0.2 [heading]",
        "vector [recursive]",
        "lexical any+ts_rank [recursive]",
        "hybrid lexical weight 0.2 [recursive]",
        "bm25s",
        "bm25s + vector RRF",
    ]
    assert {r["queries"] for r in rows} == {"4"}
    assert [r["shipped"] for r in rows[:3]] == ["yes"] * 3
    by_name = {r["name"]: r for r in rows}
    assert float(by_name["lexical any+ts_rank [heading]"]["recall_at_10"]) == 1.0
    assert float(by_name["bm25s"]["ndcg_at_10"]) == 1.0
    assert float(by_name["vector [heading]"]["recall_at_100"]) == 1.0
    assert float(by_name["vector [heading]"]["p95_ms"]) > 0.0
    markdown = (runs[0] / "ablation.md").read_text()
    assert "vector [heading] **shipped**" in markdown
    assert "FTS config `english`" in markdown
    assert eval_tenants(pg) == 0
    assert is_warm(tmp_path / "cache", "tiny")
    documents_embedded = [t for t in provider.embedded if "\n" in t]
    assert documents_embedded

    provider.embedded.clear()
    second = runner.invoke(cli, args, env=env)
    assert second.exit_code == 0, second.output
    assert [t for t in provider.embedded if "\n" in t] == []
    assert len(provider.embedded) == 4 * 4
    rerun = read_csv(sorted((tmp_path / "results").iterdir())[-1] / "retrieval.csv")
    metrics = ("recall_at_1", "recall_at_10", "recall_at_100", "ndcg_at_10", "mrr_at_10")
    assert [[r[m] for m in metrics] for r in rerun] == [[r[m] for m in metrics] for r in rows]


class OverlapReranker:
    model_key = "it-overlap"
    calibrated_threshold = None

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        words = set(query.lower().split())
        return [float(len(words & set(p.lower().split()))) for p in passages]


async def test_ablation_runs_every_m3_row_and_keeps_the_tenant_on_request(
    pg: PgInfo, provider: FakeProvider, tmp_path: Path
) -> None:
    env = eval_env(pg, provider, tmp_path)
    settings = load_settings(env=env)
    dataset = load_beir(tmp_path / "data" / "tiny")
    int8, fp32 = OverlapReranker(), OverlapReranker()
    async with open_eval_runtime(settings) as rt:
        services = replace(rt.services, rerankers={"int8": int8, "fp32": fp32})
        options = EvalOptions(plan_ablation(rt.shipped), rt.shipped, "it-ablation", keep=True)
        outcome = await run_retrieval_eval(services, dataset, options)
        assert eval_tenants(pg) == 1
        await services.delete_tenant(outcome.tenant)
    assert eval_tenants(pg) == 0
    names = [(r.name, r.split) for r in outcome.rows]
    assert len(names) == 18 + 3
    assert ("hybrid lexical weight 0.2 [heading]", "train") in names
    assert ("hybrid+rerank fp32 rerank_n 50 [heading]", "test") in names
    assert ("hybrid+rerank int8 rerank_n 30 [recursive]", "test") in names
    assert ("lexical websearch+ts_rank [heading]", "test") in names
    assert ("bm25s", "train") in names
    assert ("bm25s + vector RRF", "test") in names
    assert int8.calls > 0
    assert fp32.calls > 0
    assert set(outcome.corpora) == {"heading", "recursive"}
    reranked = [r for r in outcome.rows if r.mode == "hybrid+rerank"]
    assert all(r.metrics["recall_at_10"] == 1.0 for r in reranked)
    assert sum(r.shipped for r in outcome.rows) == 5
