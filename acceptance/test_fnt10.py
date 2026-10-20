# SPDX-License-Identifier: Apache-2.0
"""FNT-10: Consistency during version swaps and a space change.

A collection of 200 documents whose every chunk of version v of document d
carries the marker ``fnmark-<d>-<v>``. Eight concurrent clients share a budget of hybrid
searches (``rerank`` false, ``debug`` true) while the driver performs version swaps
(re-uploads with the same ``external_id``, round-robin over a hot set) and, after swap 10,
switches the collection from space ``fake-a`` (1024-d) to ``fake-b`` (768-d, dimensions
sent). The process runs with ``ingest.gc_grace`` 0s, so superseded chunks and old-space
vectors are collected while queries run.

Checks per response: no document appears with two versions (mixed-version), each hit's
version was current at some point during the request (lower bound: the newest version
whose ingest job was observed done before the request; upper bound: the newest version
uploaded before the response), and the marker's document matches the hit's
``document_id``. Per client the ``debug.space`` sequence must be ``fake-a…fake-b…``
(exactly one change). After the switch and GC no ``fake-a`` vector of the collection is
left.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx

from acceptance.harness import (
    connect,
    create_collection,
    create_tenant,
    custom_instance,
    register_space,
    write_log,
)
from acceptance.support import thresholds, write_result

DOCS = 200
HOT_DOCS = 10  # swaps go round-robin over these, so each sees several versions
SECTIONS = 3
CLIENTS = 8
SPACE_CHANGE_AFTER = 10
SPACE_A = {
    "name": "fake-a",
    "provider": "openai_compat",
    "model": "fake-embed-a",
    "dimensions": 1024,
}
SPACE_B = {
    "name": "fake-b",
    "provider": "openai_compat",
    "model": "fake-embed-b",
    "dimensions": 768,
    "send_dimensions": True,
}
MARKER = re.compile(r"fnmark-(\d+)-(\d+)")
JOB_TIMEOUT = 600.0
MAX_EXAMPLES = 10

_WORDS = (
    "amber",
    "basalt",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lagoon",
    "meadow",
    "nickel",
    "obsidian",
    "prairie",
    "quartz",
    "river",
    "saffron",
    "tundra",
    "umber",
    "valley",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "anchor",
    "beacon",
    "canyon",
    "dune",
    "estuary",
    "falcon",
    "glacier",
    "heron",
    "island",
    "jasper",
    "kestrel",
    "lantern",
    "mesa",
    "nebula",
    "orchard",
    "pebble",
    "quiver",
    "ridge",
    "summit",
    "timber",
    "upland",
    "vortex",
    "walnut",
    "yonder",
    "zinc",
    "acorn",
    "birch",
    "cedar",
    "dahlia",
    "elm",
    "fern",
    "grove",
    "hazel",
    "iris",
)


def _doc_words(d: int, section: int) -> list[str]:
    rng = random.Random(d * 101 + section)  # noqa: S311 - deterministic test content
    return rng.sample(_WORDS, 6)


def _content(d: int, v: int) -> bytes:
    """Markdown with SECTIONS sections; every paragraph carries the version marker."""
    mark = f"fnmark-{d}-{v}"
    parts = [f"# Report {d}\n\n{mark} opening note for report {d}, revision {v}.\n"]
    for s in range(SECTIONS):
        words = _doc_words(d, s)
        body = " ".join(
            f"The {w} survey of report {d} lists {mark} findings in section {s}." for w in words
        )
        parts.append(f"## Section {s}\n\n{mark} {body}\n")
    return "\n".join(parts).encode()


@dataclass
class _Timeline:
    """Per document: version -> upload sent time / ingest job observed done time."""

    sent: dict[int, dict[int, float]] = field(default_factory=dict)
    done: dict[int, dict[int, float]] = field(default_factory=dict)

    def bounds(self, d: int, start: float, end: float) -> tuple[int, int]:
        """(oldest, newest) version that may be current between start and end."""
        lo = max((v for v, t in self.done.get(d, {}).items() if t < start), default=0)
        hi = max((v for v, t in self.sent.get(d, {}).items() if t < end), default=0)
        return lo, hi


@dataclass
class _State:
    budget: int
    total_steps: int
    steps_done: int = 0
    driver_done: bool = False
    claimed: int = 0
    current_doc: int = 0
    timeline: _Timeline = field(default_factory=_Timeline)

    def allowance(self) -> int:
        if self.driver_done:
            return self.budget
        return self.budget * (self.steps_done + 1) // (self.total_steps + 1)


@dataclass
class _Stats:
    responses: int = 0
    failed: int = 0
    mixed: int = 0
    implausible: int = 0
    doc_mismatch: int = 0
    unmarked_hits: int = 0
    hits: int = 0
    multi_chunk_responses: int = 0
    spaces: dict[int, list[str]] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(msg)


def _query_text(state: _State, rng: random.Random) -> tuple[str, int]:
    """Half the queries aim at the document being swapped, half at a random one."""
    d = state.current_doc if rng.random() < 0.5 else rng.randrange(DOCS)
    words = _doc_words(d, rng.randrange(SECTIONS))
    return f"report {d} {words[0]} {words[1]} survey", d


def _check_response(  # noqa: PLR0913 - one response's context
    body: dict[str, Any],
    *,
    doc_ids: dict[int, str],
    timeline: _Timeline,
    start: float,
    end: float,
    stats: _Stats,
) -> None:
    versions: dict[int, set[int]] = {}
    per_doc: Counter[int] = Counter()
    for hit in body["hits"]:
        stats.hits += 1
        marks = {(int(a), int(b)) for a, b in MARKER.findall(hit["text"])}
        if len(marks) != 1:
            stats.unmarked_hits += 1
            stats.note(f"hit {hit['chunk_id']} has markers {sorted(marks)}")
            continue
        (d, v) = next(iter(marks))
        if doc_ids.get(d) != hit["document_id"]:
            stats.doc_mismatch += 1
            stats.note(f"marker doc {d} on document {hit['document_id']}")
        lo, hi = timeline.bounds(d, start, end)
        if not lo <= v <= hi:
            stats.implausible += 1
            stats.note(f"doc {d} version {v} outside [{lo}, {hi}]")
        versions.setdefault(d, set()).add(v)
        per_doc[d] += 1
    if any(n > 1 for n in per_doc.values()):
        stats.multi_chunk_responses += 1
    mixed = {d: sorted(vs) for d, vs in versions.items() if len(vs) > 1}
    if mixed:
        stats.mixed += 1
        stats.note(f"mixed-version response: {mixed}")


async def _client(  # noqa: PLR0913 - one client's context
    n: int,
    *,
    url: str,
    key: str,
    cid: str,
    doc_ids: dict[int, str],
    state: _State,
    stats: _Stats,
) -> None:
    rng = random.Random(1000 + n)  # noqa: S311 - deterministic query mix
    seq: list[str] = stats.spaces.setdefault(n, [])
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=60.0) as http:
        while state.claimed < state.budget:
            if state.claimed >= state.allowance():
                await asyncio.sleep(0.01)
                continue
            state.claimed += 1
            query, _ = _query_text(state, rng)
            payload = {"query": query, "mode": "hybrid", "rerank": False, "debug": True}
            start = time.monotonic()
            try:
                resp = await http.post(f"/v1/collections/{cid}/search", json=payload)
            except httpx.HTTPError as exc:
                stats.failed += 1
                stats.note(f"client {n}: transport error {exc!r}")
                continue
            end = time.monotonic()
            if not resp.is_success:
                stats.failed += 1
                stats.note(f"client {n}: {resp.status_code} {resp.text[:200]}")
                continue
            body = resp.json()
            stats.responses += 1
            seq.append(body["debug"]["space"])
            _check_response(
                body, doc_ids=doc_ids, timeline=state.timeline, start=start, end=end, stats=stats
            )


async def _wait_job(http: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        resp = await http.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        job: dict[str, Any] = resp.json()
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            msg = f"job {job_id} ({job['kind']}) failed: {job.get('last_error')}"
            raise RuntimeError(msg)
        await asyncio.sleep(0.1)
    msg = f"job {job_id} not done after {JOB_TIMEOUT}s"
    raise TimeoutError(msg)


async def _upload(
    http: httpx.AsyncClient, cid: str, d: int, v: int, timeline: _Timeline
) -> dict[str, Any]:
    """Upload version v of document d and wait for its ingest job."""
    timeline.sent.setdefault(d, {})[v] = time.monotonic()
    resp = await http.post(
        f"/v1/collections/{cid}/documents",
        files={"file": (f"report-{d}.md", _content(d, v))},
        data={"external_id": f"report-{d}", "title": f"Report {d}"},
    )
    if resp.status_code != 202:
        msg = f"upload d={d} v={v}: {resp.status_code} {resp.text[:300]}"
        raise RuntimeError(msg)
    out: dict[str, Any] = resp.json()
    if out["version"] != v:
        msg = f"upload d={d}: expected version {v}, got {out['version']}"
        raise RuntimeError(msg)
    await _wait_job(http, out["job_id"])
    timeline.done.setdefault(d, {})[v] = time.monotonic()
    return out


def _due_jobs(tenant_id: str) -> int:
    """Jobs of the tenant running or due now (gc_grace 0s makes every gc job due)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT count(*) FROM footnote.jobs WHERE tenant_id = %s "
            "AND (status = 'running' OR (status = 'queued' AND run_at <= now()))",
            (tenant_id,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _vector_counts(cid: str) -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.name, count(*) FROM footnote.chunk_vectors v "
            "JOIN footnote.embedding_spaces s ON s.id = v.space_id "
            "WHERE v.collection_id = %s GROUP BY s.name",
            (cid,),
        ).fetchall()
    return {str(name): int(n) for name, n in rows}


def _current_chunks(cid: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT count(DISTINCT vc.chunk_id) FROM footnote.documents d "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
            "AND vc.version_id = d.current_version_id WHERE d.collection_id = %s",
            (cid,),
        ).fetchone()
    assert row is not None
    return int(row[0])


async def _driver(  # noqa: PLR0913 - the scenario's context
    http: httpx.AsyncClient,
    *,
    cid: str,
    tenant_id: str,
    swaps: int,
    state: _State,
    info: dict[str, Any],
) -> None:
    versions = dict.fromkeys(range(DOCS), 1)
    reembed: asyncio.Task[dict[str, Any]] | None = None
    started = time.monotonic()
    for i in range(swaps):
        d = i % HOT_DOCS
        state.current_doc = d
        versions[d] += 1
        await _upload(http, cid, d, versions[d], state.timeline)
        info["swaps_done"] = i + 1
        state.steps_done += 1
        if i + 1 == SPACE_CHANGE_AFTER:
            resp = await http.patch(f"/v1/collections/{cid}", json={"embedding_space": "fake-b"})
            if resp.status_code != 202:
                msg = f"space change: {resp.status_code} {resp.text[:300]}"
                raise RuntimeError(msg)
            info["space_change_at"] = time.monotonic() - started
            reembed = asyncio.create_task(_wait_job(http, resp.json()["job_id"]))
            reembed.add_done_callback(
                lambda _: info.__setitem__("reembed_done_at", time.monotonic() - started)
            )
    info["swaps_done_at"] = time.monotonic() - started
    assert reembed is not None
    job = await reembed
    info["reembed_progress"] = job.get("progress")
    state.steps_done += 1
    deadline = time.monotonic() + JOB_TIMEOUT
    while await asyncio.to_thread(_due_jobs, tenant_id):
        if time.monotonic() > deadline:
            msg = "gc jobs did not finish"
            raise TimeoutError(msg)
        await asyncio.sleep(0.2)
    info["gc_done_at"] = time.monotonic() - started
    info["vectors_after_gc"] = await asyncio.to_thread(_vector_counts, cid)
    state.steps_done += 1
    state.driver_done = True


async def _scenario(url: str, key: str, tenant_id: str, cid: str, expected: dict[str, Any]) -> Any:
    state = _State(budget=int(expected["queries"]), total_steps=int(expected["swaps"]) + 2)
    stats = _Stats()
    info: dict[str, Any] = {}
    headers = {"Authorization": f"Bearer {key}"}
    limits = httpx.Limits(max_connections=16)
    async with httpx.AsyncClient(
        base_url=url, headers=headers, timeout=120.0, limits=limits
    ) as http:
        sem = asyncio.Semaphore(16)

        async def first(d: int) -> dict[str, Any]:
            async with sem:
                return await _upload(http, cid, d, 1, state.timeline)

        uploaded = await asyncio.gather(*(first(d) for d in range(DOCS)))
        doc_ids = {d: str(u["document_id"]) for d, u in enumerate(uploaded)}
        info["initial_vectors"] = await asyncio.to_thread(_vector_counts, cid)
        info["initial_chunks"] = await asyncio.to_thread(_current_chunks, cid)
        started = time.monotonic()
        driver = asyncio.create_task(
            _driver(
                http,
                cid=cid,
                tenant_id=tenant_id,
                swaps=int(expected["swaps"]),
                state=state,
                info=info,
            )
        )
        clients = [
            asyncio.create_task(
                _client(n, url=url, key=key, cid=cid, doc_ids=doc_ids, state=state, stats=stats)
            )
            for n in range(CLIENTS)
        ]
        try:
            await driver
        except BaseException:
            state.claimed = state.budget  # stop the clients
            await asyncio.gather(*clients, return_exceptions=True)
            raise
        await asyncio.gather(*clients)
        info["query_seconds"] = time.monotonic() - started
    info["final_chunks"] = _current_chunks(cid)
    return state, stats, info


def _changes(seq: list[str]) -> int:
    return sum(1 for a, b in itertools.pairwise(seq) if a != b)


def test_acc_fnt10_swap_consistency() -> None:
    """0 mixed-version results, 0 failed queries, one space change per client, GC clean."""
    expected = thresholds("FNT-10")
    t0 = time.monotonic()
    with custom_instance("fnt10", {"FOOTNOTE_INGEST_GC_GRACE": "0s"}) as inst:
        space_a = register_space(inst, SPACE_A)
        space_b = register_space(inst, SPACE_B)
        assert space_a["dimensions"] == SPACE_A["dimensions"], space_a
        assert space_b["dimensions"] == SPACE_B["dimensions"], space_b
        tenant = create_tenant(inst)
        with inst.tenant(tenant.key) as client:
            coll = create_collection(
                client,
                "fnt10",
                embedding_space="fake-a",
                chunking={"max_tokens": 128, "min_tokens": 16, "overlap_tokens": 0},
                retrieval={"k": 20},
            )
        state, stats, info = asyncio.run(
            _scenario(inst.data_url, tenant.key, tenant.id, coll["id"], expected)
        )
        with inst.tenant(tenant.key) as client:
            final = client.get(f"/v1/collections/{coll['id']}").json()

    per_client = [_changes(stats.spaces.get(n, [])) for n in range(CLIENTS)]
    patterns_ok = all(
        seq and seq[0] == "fake-a" and seq[-1] == "fake-b" and set(seq) == {"fake-a", "fake-b"}
        for seq in (stats.spaces.get(n, []) for n in range(CLIENTS))
    )
    vectors = info.get("vectors_after_gc", {})
    old_left = int(vectors.get("fake-a", 0))
    changes_measured: Any = per_client[0] if len(set(per_client)) == 1 else per_client
    measured = {
        "queries": state.claimed,
        "swaps": info.get("swaps_done", 0),
        "mixed_version_results": stats.mixed,
        "failed_queries": stats.failed,
        "space_changes_observed_per_client": changes_measured,
        "old_space_vectors_left": old_left,
    }
    passed = (
        measured["queries"] >= expected["queries"]
        and measured["swaps"] >= expected["swaps"]
        and stats.mixed <= expected["mixed_version_results"]
        and stats.failed <= expected["failed_queries"]
        and all(c == expected["space_changes_observed_per_client"] for c in per_client)
        and patterns_ok
        and old_left <= expected["old_space_vectors_left"]
        and stats.implausible == 0
        and stats.doc_mismatch == 0
        and stats.unmarked_hits == 0
        and final["active_space"] == "fake-b"
        and int(vectors.get("fake-b", 0)) >= info["final_chunks"]
    )
    queries_per_client = {n: len(s) for n, s in stats.spaces.items()}
    b_per_client = {n: s.count("fake-b") for n, s in stats.spaces.items()}
    details = (
        f"docs={DOCS} hot={HOT_DOCS} clients={CLIENTS}; responses={stats.responses} "
        f"hits={stats.hits} multi_chunk_responses={stats.multi_chunk_responses}; "
        f"implausible_versions={stats.implausible} doc_mismatch={stats.doc_mismatch} "
        f"unmarked_hits={stats.unmarked_hits}; per_client_changes={per_client} "
        f"patterns_ok={patterns_ok} queries_per_client={queries_per_client} "
        f"fake_b_per_client={b_per_client}; initial_chunks={info.get('initial_chunks')} "
        f"final_chunks={info.get('final_chunks')} initial_vectors={info.get('initial_vectors')} "
        f"vectors_after_gc={vectors} active_space={final['active_space']}; "
        f"space_change_at={info.get('space_change_at', 0):.1f}s "
        f"reembed_done_at={info.get('reembed_done_at', 0):.1f}s "
        f"swaps_done_at={info.get('swaps_done_at', 0):.1f}s "
        f"gc_done_at={info.get('gc_done_at', 0):.1f}s "
        f"query_seconds={info.get('query_seconds', 0):.1f}s "
        f"runtime={time.monotonic() - t0:.1f}s; examples={stats.examples}"
    )
    write_log(
        "FNT-10-spaces.json",
        json.dumps({str(n): Counter(s) for n, s in stats.spaces.items()}, indent=2),
    )
    write_result("FNT-10", measured, passed, details)
    assert passed, f"FNT-10 failed: {measured}; {details}"
