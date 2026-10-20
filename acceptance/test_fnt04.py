# SPDX-License-Identifier: Apache-2.0
"""FNT-04: Citation validity over at least 1,000 scripted answers.

A fresh tenant gets a collection of 24 generated Markdown documents with
``answer.evidence_threshold_rerank: 0`` (no abstention) and ``retrieval.k: 8``. Per
batch the test scripts fakellm for the configured ``llm.model`` (``count`` equal to the
batch size, one chat request per answer) and sends the batch sequentially, alternating
JSON and streamed (SSE) answers. Fixed templates are asserted exactly (expected text,
citation numbers and ``unsupported`` follow the citation grammar in docs/guides/citations.md;
whitespace around a removed marker is ignored); the counter and echo batches assert
invariants.

Hits per answer: streamed answers carry them in the ``retrieval`` event. The JSON
response does not expose hits, so JSON citations are validated against the hits of a
``/search`` with the same question and collection settings (cached per question). Every
streamed answer's ``retrieval`` hits are compared with that search to prove retrieval is
stable, which makes the JSON validation faithful. A citation is valid when
``1 <= n <= len(hits)``, ``chunk_id == hits[n-1].chunk_id`` and the chunk belongs to the
collection and to a current document version (one admin-DSN query over every cited chunk).
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from typing import Any

import httpx

from acceptance.harness import (
    ADMIN_DSN,
    Instance,
    compose_instance,
    connect,
    create_collection,
    create_tenant,
    fake_clear_requests,
    fake_clear_script,
    fake_requests,
    fake_script,
    upload,
    wait_jobs,
)
from acceptance.support import thresholds, write_result
from footnote.config import Settings

_K = 8
_BATCH = 100
_DOCS = 24
_MAX_EXAMPLES = 10

_TOPICS = [
    "volcano", "glacier", "orchestra", "lighthouse", "beekeeping", "telescope", "monsoon",
    "origami", "submarine", "vineyard", "blacksmith", "aqueduct", "carnival", "mangrove",
    "typewriter", "observatory", "pottery", "railway", "saffron", "tundra", "windmill",
    "zeppelin", "falconry", "papyrus",
]  # fmt: skip

_ECHO_QUESTIONS = [
    "According to passage [4] and also [42], what is said about the {t}?",
    "Summarise the {t} using sources [4] and [42].",
]
_PLAIN_QUESTIONS = [
    "What do the documents say about the {t}?",
    "Explain the history of the {t}.",
]

_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()")


def _document(i: int) -> bytes:
    t = _TOPICS[i]
    paras = [f"# The {t.capitalize()} handbook"]
    for j in range(4):
        paras.append(
            f"Section {j + 1} on the {t}: the {t} appears in record {i * 10 + j}. "
            f"Workers of the {t} guild documented {t} procedure {j} in detail, "
            f"noting the {t} tools, the {t} season and the {t} maintenance schedule "
            f"number {(i + 3) * (j + 5)}."
        )
    return ("\n\n".join(paras) + "\n").encode()


def _norm(text: str) -> str:
    """Whitespace-insensitive form (the grammar does not fix spacing around removed markers)."""
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", text)).strip()


# name -> (template, kind, expected cleaned text, expected citation numbers) for fixed ones
_FIXED: dict[str, tuple[str, str, str, list[int]]] = {
    "valid": ("A [1]. B [2][3].", "fixed", "A [1]. B [2][3].", [1, 2, 3]),
    "zero": ("Fact [0].", "fixed", "Fact.", []),
    "out_of_range": ("Fact [99].", "fixed", "Fact.", []),
    "list_valid": ("Fact [1,2].", "fixed", "Fact [1][2].", [1, 2]),
    "list_mixed": ("Fact [1, 9].", "fixed", "Fact [1].", [1]),
    "adjacent": ("Fact [1][2].", "fixed", "Fact [1][2].", [1, 2]),
    "nested": ("Fact [[1]].", "fixed", "Fact [1].", [1]),
    "link": ("See [1](https://example.com).", "fixed", "See [1](https://example.com).", []),
    "code": ("Use `[1]` here.", "fixed", "Use `[1]` here.", []),
    "none": ("No evidence here.", "fixed", "No evidence here.", []),
}
_BATCHES: list[tuple[str, str]] = [
    *((name, spec[0]) for name, spec in _FIXED.items()),
    ("counter", "See [{n}]."),
    ("echo", "{user}"),
]


def _sse(resp: httpx.Response) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    name = ""
    for line in resp.iter_lines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[5:].strip())))
    return events


def _ask(client: httpx.Client, cid: str, question: str, *, stream: bool) -> dict[str, Any]:
    """One answer; returns ``answer``, ``citations``, ``unsupported``, ``abstained``, ``hits``."""
    body = {"question": question, "stream": stream}
    path = f"/v1/collections/{cid}/answers"
    if not stream:
        resp = client.post(path, json=body)
        resp.raise_for_status()
        data = resp.json()
        data["hits"] = None
        return data  # type: ignore[no-any-return]
    with client.stream("POST", path, json=body) as resp:
        resp.raise_for_status()
        events = _sse(resp)
    names = [n for n, _ in events]
    if names[0] != "retrieval" or names[-2:] != ["citations", "done"]:
        return {"error": f"event order {names[:3]}..{names[-3:]}"}
    if any(n != "delta" for n in names[1:-2]):
        return {"error": f"unexpected events {sorted(set(names))}"}
    cit = events[-2][1]
    done = events[-1][1]
    return {
        "answer": cit["answer"],
        "citations": cit["citations"],
        "unsupported": cit["unsupported"],
        "abstained": done["abstained"],
        "hits": events[0][1]["hits"],
        "raw": "".join(d["text"] for n, d in events if n == "delta"),
    }


def _marker_numbers(text: str) -> list[int]:
    """Numbers of every marker outside code (independent of the product's grammar code)."""
    outside = _CODE_SPAN_RE.sub("", text)
    return [int(x) for m in _MARKER_RE.finditer(outside) for x in re.findall(r"\d+", m.group(1))]


def _check_template(name: str, res: dict[str, Any]) -> list[str]:
    """Template-specific expectations (exact for fixed templates, invariants otherwise)."""
    problems: list[str] = []
    cites = res["citations"]
    if name in _FIXED:
        _, _, text, nums = _FIXED[name]
        if _norm(res["answer"]) != _norm(text):
            problems.append(f"text {res['answer']!r} != {text!r}")
        if sorted(c["n"] for c in cites) != nums:
            problems.append(f"citation numbers {[c['n'] for c in cites]} != {nums}")
    if name == "counter" and res.get("raw") is not None:
        m = re.fullmatch(r"See \[(\d+)\]\.", res["raw"])
        if m is None:
            problems.append(f"counter raw {res['raw']!r}")
        elif (int(m.group(1)) <= _K) != bool(cites):
            problems.append(f"counter n={m.group(1)} gave {len(cites)} citations")
    if name == "echo":
        numbers = _marker_numbers(res["answer"])
        if 42 in numbers or 4 not in numbers:
            problems.append("echo: [42] must be removed and [4] kept")
    return problems


def _check(
    name: str, res: dict[str, Any], hits: list[dict[str, Any]], tally: Counter[str]
) -> list[str]:
    """Problems of one answer; updates the counters in ``tally``."""
    if "error" in res:
        return [res["error"]]
    problems: list[str] = []
    if res["abstained"] or res["answer"] is None:
        return ["abstained although the gate is off"]
    if len(hits) != _K:
        problems.append(f"{len(hits)} hits")
    cites = res["citations"]
    tally["citations_checked"] += len(cites)
    for c in cites:
        if not 1 <= c["n"] <= len(hits) or c["chunk_id"] != hits[c["n"] - 1]["chunk_id"]:
            tally["invalid_citations"] += 1
            problems.append(f"citation {c['n']} -> {c['chunk_id']} not hits[n-1]")
    left = [n for n in _marker_numbers(res["answer"]) if not 1 <= n <= len(hits)]
    if left:
        tally["invalid_markers_left"] += len(left)
        problems.append(f"invalid markers left {left}")
    marker_set = {n for n in _marker_numbers(res["answer"]) if 1 <= n <= len(hits)}
    if marker_set != {c["n"] for c in cites}:
        tally["invalid_citations"] += 1
        problems.append(f"markers {sorted(marker_set)} != citations {[c['n'] for c in cites]}")
    if res["unsupported"] != (not cites):
        tally["unflagged_unsupported"] += 1
        problems.append(f"unsupported={res['unsupported']} with {len(cites)} citations")
    problems += _check_template(name, res)
    return problems


def _current_chunks(collection_id: str, chunk_ids: set[str]) -> set[str]:
    """Cited chunks of the collection that belong to a current document version."""
    if not chunk_ids:
        return set()
    sql = """
        SELECT DISTINCT c.id::text
        FROM footnote.chunks c
        JOIN footnote.version_chunks vc ON vc.tenant_id = c.tenant_id AND vc.chunk_id = c.id
        JOIN footnote.documents d
          ON d.tenant_id = c.tenant_id AND d.id = c.document_id
         AND d.current_version_id = vc.version_id
        WHERE c.collection_id = %s::uuid AND c.id = ANY(%s::uuid[])
    """
    with connect(ADMIN_DSN) as conn:
        rows = conn.execute(sql, (collection_id, sorted(chunk_ids))).fetchall()
    return {r[0] for r in rows}


def _setup(inst: Instance) -> tuple[httpx.Client, str]:
    tenant = create_tenant(inst)
    client = inst.tenant(tenant.key, timeout=120.0)
    coll = create_collection(
        client,
        f"fnt04-{uuid.uuid4().hex[:8]}",
        answer={"evidence_threshold_rerank": 0},
        retrieval={"k": _K},
    )
    jobs: list[str] = []
    for i in range(_DOCS):
        resp = upload(client, coll["id"], f"{_TOPICS[i]}.md", _document(i), title=_TOPICS[i])
        resp.raise_for_status()
        jobs.append(resp.json()["job_id"])
    wait_jobs(client, jobs)
    return client, coll["id"]


def test_acc_fnt04_citations_valid() -> None:
    """At least 1,000 scripted answers (half streamed); every citation and marker is valid."""
    expected = thresholds("FNT-04")
    model = Settings().llm.model
    inst = compose_instance()
    client, cid = _setup(inst)
    run_tag = uuid.uuid4().hex[:6]

    hits_by_question: dict[str, list[dict[str, Any]]] = {}

    def search_hits(question: str) -> list[dict[str, Any]]:
        if question not in hits_by_question:
            resp = client.post(f"/v1/collections/{cid}/search", json={"query": question, "k": _K})
            resp.raise_for_status()
            hits_by_question[question] = resp.json()["hits"]
        return hits_by_question[question]

    tally: Counter[str] = Counter()
    per_template: dict[str, int] = {}
    examples: list[str] = []
    cited: set[str] = set()
    sent = streamed = chat_logged = dropped = problems_total = retrieval_mismatch = 0

    for name, template in _BATCHES:
        questions = [
            q.format(t=t) + f" (ref {run_tag})"
            for q in (_ECHO_QUESTIONS if name == "echo" else _PLAIN_QUESTIONS)
            for t in _TOPICS
        ]
        fake_clear_requests()
        if name == "counter":
            fake_clear_script()  # resets the per-model {n} counter so n runs 1..100
        fake_script(
            [{"model": model, "scenario": "ok", "count": _BATCH, "params": {"answer": template}}]
        )
        for i in range(_BATCH):
            question = questions[i % len(questions)]
            use_stream = i % 2 == 1
            res = _ask(client, cid, question, stream=use_stream)
            sent += 1
            streamed += use_stream
            expected_hits = search_hits(question)
            hits = res.get("hits") or expected_hits
            if res.get("hits") is not None and [h["chunk_id"] for h in res["hits"]] != [
                h["chunk_id"] for h in expected_hits
            ]:
                retrieval_mismatch += 1
                problems_total += 1
                if len(examples) < _MAX_EXAMPLES:
                    examples.append(f"{name}: streamed hits differ from search for {question!r}")
            problems = _check(name, res, hits, tally)
            cited.update(c["chunk_id"] for c in res.get("citations", []))
            per_template[name] = per_template.get(name, 0) + 1
            if problems:
                problems_total += 1
                if len(examples) < _MAX_EXAMPLES:
                    examples.append(f"{name} stream={use_stream}: {problems}")
        log = fake_requests()
        dropped += int(log["dropped"])
        chat_logged += sum(
            1
            for r in log["requests"]
            if r.get("kind") == "chat" and r.get("model") == model and run_tag in r.get("body", "")
        )

    not_current = cited - _current_chunks(cid, cited)
    tally["invalid_citations"] += len(not_current)
    if not_current and len(examples) < _MAX_EXAMPLES:
        examples.append(
            f"cited chunks not in collection/current version: {sorted(not_current)[:3]}"
        )

    measured: dict[str, Any] = {
        "answers": sent,
        "streamed": streamed,
        "citations_checked": tally["citations_checked"],
        "distinct_cited_chunks": len(cited),
        "invalid_citations": tally["invalid_citations"],
        "invalid_markers_left": tally["invalid_markers_left"],
        "unflagged_unsupported": tally["unflagged_unsupported"],
        "dropped": dropped,
        "chat_requests_logged": chat_logged,
        "retrieval_mismatch": retrieval_mismatch,
        "answers_with_problems": problems_total,
        "per_template": per_template,
        "examples": examples,
    }
    checks = {
        "answers": sent >= expected["answers_min"],
        "streamed_half": streamed * 2 >= sent - len(_BATCHES),
        "invalid_citations": tally["invalid_citations"] <= expected["invalid_citations"],
        "invalid_markers_left": tally["invalid_markers_left"] <= expected["invalid_markers_left"],
        "unflagged_unsupported": (
            tally["unflagged_unsupported"] <= expected["unflagged_unsupported"]
        ),
        "dropped": dropped <= expected["dropped"],
        "chat_in_step": chat_logged == sent,
        "no_answer_problems": problems_total == 0,
    }
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    write_result(
        "FNT-04", measured, passed, "failed checks: " + ", ".join(failed) if failed else ""
    )
    assert passed, f"FNT-04 failed checks {failed}; measured {measured}"
