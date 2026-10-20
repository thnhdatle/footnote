# SPDX-License-Identifier: Apache-2.0
"""FNT-05: Cross-tenant isolation under 10,000 calls plus RLS-only run.

Tenants A and B (3 collections, 50 documents each; every string carries a per-tenant
marker) are exercised on a dedicated instance with a seeded mix of normal traffic and
cross-tenant probes (B addressing A's collections, documents, version originals, chunks,
jobs, deliveries, real and guessed ids, filters with A's ids, uploads into A's
collections, admin endpoints with tenant keys). A leak is any 2xx for a resource that is
not the caller's, any response (body or headers) holding the other tenant's marker,
ids, key or webhook secret, any id in a response owned by another tenant (checked over
the admin DSN), or a change to the other tenant's rows. The mix is replayed on an
instance with `testing.disable_app_tenant_predicates` and `FOOTNOTE_ENABLE_TEST_FAULTS=1`
(RLS alone). Then raw SQL, role and catalog assertions run as `footnote_app`, and an
instance started with a BYPASSRLS role must answer `/readyz` 503.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg
from psycopg import errors, sql
from ulid import ULID

from acceptance import harness as h
from acceptance.support import thresholds, write_result

SEED = 20261020
CONCURRENCY = 16
N_COLLECTIONS = 3
N_DOCS = 50
N_REVISED = 3  # documents per tenant that get a second version
FAULT_WARNING = "testing.disable_app_tenant_predicates is active"
SPACE = "bge-small-en-v1.5-onnx"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# `Database.tenant_filter` renders `"<alias>".tenant_id = $n`; hand-written predicates
# elsewhere are unquoted, so this pattern identifies the removable app predicates.
APP_PREDICATE_RE = re.compile(r'"\w+"\.tenant_id = \$\d+')
FAULT_PREDICATE_RE = re.compile(r"\b(?:WHERE|AND) TRUE\b")
GLOBAL_READ_ONLY = {"embedding_spaces", "alembic_version"}
NO_ACCESS = {"tenants", "api_keys"}
NO_WHERE = sql.SQL("")
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
INSTANCE_ENV = {
    "FOOTNOTE_LLM_MODEL": "fake-fnt05",
    # fakellm echoes the prompt: any foreign chunk in an answer's context shows up.
    "FOOTNOTE_LLM_EXTRA_HEADERS": json.dumps({"X-Fake-Scenario": "echo"}),
    "FOOTNOTE_WEBHOOKS_ALLOW_HTTP": "true",
    "FOOTNOTE_WEBHOOKS_ALLOW_PRIVATE": "true",
}
FAULT_ENV = {
    **INSTANCE_ENV,
    "FOOTNOTE_TESTING_DISABLE_APP_TENANT_PREDICATES": "true",
    "FOOTNOTE_ENABLE_TEST_FAULTS": "1",
}
QUESTIONS = (
    "How many vacation days do employees get?",
    "What is the parental leave policy?",
    "How are travel expenses reimbursed?",
    "Which equipment does the company provide?",
)


# --- tenant worlds ------------------------------------------------------------------


@dataclass
class Doc:
    """One uploaded document and the exact upload that reproduces its current version."""

    id: str
    collection_id: str
    external_id: str
    title: str
    metadata: dict[str, str]
    content: bytes
    versions: int = 1


@dataclass
class World:
    """Everything one tenant owns, for probing and for leak scanning."""

    label: str
    mark: str
    tenant_id: str = ""
    key: str = ""
    key_id: str = ""
    collections: list[str] = field(default_factory=list)
    docs: list[Doc] = field(default_factory=list)
    chunks: list[tuple[str, str]] = field(
        default_factory=list
    )  # (collection_id, chunk_id), current
    all_chunks: list[tuple[str, str]] = field(default_factory=list)  # incl. superseded
    jobs: list[str] = field(default_factory=list)
    deliveries: list[str] = field(default_factory=list)
    version_ids: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)

    @property
    def ids(self) -> set[str]:
        """Every id of this tenant known to the test."""
        out = {self.tenant_id, self.key_id, *self.collections, *self.jobs, *self.deliveries}
        out.update(d.id for d in self.docs)
        out.update(c for _, c in self.all_chunks)
        out.update(self.version_ids)
        return out

    @property
    def strings(self) -> list[str]:
        """Marker and secret strings that must never reach the other tenant."""
        return [self.mark, self.key, *self.secrets]


def _doc_body(mark: str, j: int, revision: int) -> bytes:
    paragraphs = [
        f"Employees receive {20 + j % 7} vacation days per year {mark}d{j}p1r{revision}.",
        f"Parental leave lasts {10 + j % 5} weeks for every parent {mark}d{j}p2r{revision}.",
        f"Travel expenses are reimbursed within {j % 9 + 3} days {mark}d{j}p3r{revision}.",
    ]
    sections = "\n\n".join(f"## Section {i + 1} {mark}\n\n{p}" for i, p in enumerate(paragraphs))
    return f"# Handbook {j} {mark}\n\n{sections}\n".encode()


def _create_tenant(inst: h.Instance, world: World) -> None:
    with inst.admin() as admin:
        resp = admin.post("/v1/admin/tenants", json={"name": f"{world.mark}-tenant"})
        resp.raise_for_status()
        world.tenant_id = resp.json()["id"]
        resp = admin.post(f"/v1/admin/tenants/{world.tenant_id}/keys", json={"name": world.mark})
        resp.raise_for_status()
        world.key = resp.json()["key"]
        world.key_id = resp.json()["id"]


def _seed_world(inst: h.Instance, world: World) -> None:
    """Tenant, key, collections (one with a webhook), documents, second versions."""
    _create_tenant(inst, world)
    with inst.tenant(world.key) as client:
        for i in range(N_COLLECTIONS):
            settings: dict[str, Any] = {
                "answer": {"evidence_threshold_rerank": 0, "evidence_threshold_vector": 0}
            }
            if i == 0:  # unreachable private target: deliveries exist and fail
                settings["webhook"] = {"url": f"http://127.0.0.1:9/{world.mark}/hook"}
            col = h.create_collection(client, f"{world.mark}-c{i}", **settings)
            world.collections.append(col["id"])
            if col.get("webhook_secret"):
                world.secrets.append(col["webhook_secret"])
        for j in range(N_DOCS):
            cid = world.collections[j % N_COLLECTIONS]
            doc = Doc(
                "",
                cid,
                f"{world.mark}-e{j}",
                f"{world.mark} Handbook {j}",
                {"marker": f"{world.mark}-m{j}", "team": "hr"},
                _doc_body(world.mark, j, 1),
            )
            doc.id = _upload(client, doc, world)
            world.docs.append(doc)
        h.wait_jobs(client, list(world.jobs))
        revised = [world.docs[j] for j in range(N_REVISED)]
        for j, doc in enumerate(revised):
            doc.content = _doc_body(world.mark, j, 2)
            doc.versions = 2
            _upload(client, doc, world)
        h.wait_jobs(client, list(world.jobs))


def _upload(client: httpx.Client, doc: Doc, world: World) -> str:
    resp = h.upload(
        client,
        doc.collection_id,
        f"{doc.external_id}.md",
        doc.content,
        external_id=doc.external_id,
        title=doc.title,
        metadata=doc.metadata,
    )
    if resp.status_code != 202:
        msg = f"upload {doc.external_id}: {resp.status_code} {resp.text[:300]}"
        raise RuntimeError(msg)
    world.jobs.append(resp.json()["job_id"])
    doc_id: str = resp.json()["document_id"]
    return doc_id


def _load_db_ids(world: World) -> None:
    """Version, chunk, job and delivery ids of the tenant, read over the admin DSN."""
    tid = world.tenant_id
    with h.connect() as conn:
        world.version_ids = [
            str(r[0])
            for r in conn.execute(
                "SELECT id FROM footnote.document_versions WHERE tenant_id = %s", (tid,)
            )
        ]
        world.all_chunks = [
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT collection_id, id FROM footnote.chunks WHERE tenant_id = %s", (tid,)
            )
        ]
        world.chunks = [
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT DISTINCT c.collection_id, c.id FROM footnote.documents d"
                " JOIN footnote.version_chunks m"
                "   ON m.tenant_id = d.tenant_id AND m.version_id = d.current_version_id"
                " JOIN footnote.chunks c ON c.tenant_id = m.tenant_id AND c.id = m.chunk_id"
                " WHERE d.tenant_id = %s",
                (tid,),
            )
        ]
        world.jobs = sorted(
            {*world.jobs}
            | {
                str(r[0])
                for r in conn.execute("SELECT id FROM footnote.jobs WHERE tenant_id = %s", (tid,))
            }
        )
        world.deliveries = [
            str(r[0])
            for r in conn.execute(
                "SELECT id FROM footnote.webhook_deliveries WHERE tenant_id = %s", (tid,)
            )
        ]


def _wait_deliveries(world: World, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _load_db_ids(world)
        if world.deliveries:
            return
        time.sleep(1.0)
    msg = f"tenant {world.label}: no webhook deliveries after {timeout}s"
    raise TimeoutError(msg)


def _snapshot(world: World) -> list[Any]:
    """The tenant's collection and document rows (without timestamps) for tamper checks."""
    with h.connect() as conn:
        cols = conn.execute(
            "SELECT id, name, settings, status, webhook_url, chunk_count FROM footnote.collections"
            " WHERE tenant_id = %s ORDER BY id",
            (world.tenant_id,),
        ).fetchall()
        docs = conn.execute(
            "SELECT id, collection_id, external_id, title, metadata, status, current_version_id"
            " FROM footnote.documents WHERE tenant_id = %s ORDER BY id",
            (world.tenant_id,),
        ).fetchall()
        counts = conn.execute(
            "SELECT (SELECT count(*) FROM footnote.chunks WHERE tenant_id = %(t)s),"
            " (SELECT count(*) FROM footnote.document_versions WHERE tenant_id = %(t)s),"
            " (SELECT count(*) FROM footnote.chunk_vectors WHERE tenant_id = %(t)s)",
            {"t": world.tenant_id},
        ).fetchone()
    return [cols, docs, counts]


# --- traffic ------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One planned request; `kind` decides what a 2xx means."""

    actor: str  # "A" or "B"
    kind: str  # own | filter | cross | admin | guess
    label: str
    method: str
    path: str
    expect: frozenset[int]
    json_body: Any = None
    upload: Doc | None = None


OK = frozenset({200})
NOT_FOUND = frozenset({404})
UNAUTH = frozenset({401})


class Planner:
    """Seeded mix of normal traffic and cross-tenant probes."""

    def __init__(self, a: World, b: World, seed: int) -> None:
        self.rng = random.Random(seed)  # noqa: S311 - seeded traffic mix, not crypto
        self.worlds = {"A": a, "B": b}
        self.table: list[tuple[int, Callable[[World, World], Call]]] = [
            (14, self._own_read),
            (8, self._own_search),
            (2, self._own_answer),
            (2, self._own_upload_unchanged),
            (8, self._filter_probe),
            (30, self._cross_read),
            (6, self._cross_write),
            (6, self._cross_search),
            (6, self._admin_probe),
            (8, self._guess_probe),
        ]
        self.weights = [w for w, _ in self.table]

    def plan(self, n: int) -> list[Call]:
        """`n` calls; the attacker is B in 85% of the probes."""
        calls: list[Call] = []
        for _ in range(n):
            gen = self.rng.choices([g for _, g in self.table], weights=self.weights)[0]
            attacker = "B" if self.rng.random() < 0.85 else "A"
            me = self.worlds[attacker]
            other = self.worlds["A" if attacker == "B" else "B"]
            calls.append(gen(me, other))
        return calls

    # helpers

    def _doc(self, w: World) -> Doc:
        return self.rng.choice(w.docs)

    def _col(self, w: World) -> str:
        return self.rng.choice(w.collections)

    def _version(self, d: Doc) -> int:
        return self.rng.randint(1, d.versions)

    def _search_body(self, filters: dict[str, Any] | None = None, query: str = "") -> Any:
        body: dict[str, Any] = {
            "query": query or self.rng.choice(QUESTIONS),
            "mode": self.rng.choice(["hybrid", "hybrid", "vector", "lexical"]),
            "rerank": self.rng.random() < 0.15,
            "k": self.rng.choice([3, 8, 20]),
        }
        if self.rng.random() < 0.2:
            body["debug"] = True
        if filters:
            body["filters"] = filters
        return body

    # normal traffic

    def _own_read(self, me: World, _: World) -> Call:
        d = self._doc(me)
        cid, chunk = self.rng.choice(me.chunks)
        options = [
            ("list_collections", "/v1/collections?limit=10"),
            ("get_collection", f"/v1/collections/{self._col(me)}"),
            ("list_documents", f"/v1/collections/{self._col(me)}/documents?limit=20"),
            ("get_document", f"/v1/collections/{d.collection_id}/documents/{d.id}"),
            ("get_text", f"/v1/collections/{d.collection_id}/documents/{d.id}/text"),
            (
                "get_elements",
                f"/v1/collections/{d.collection_id}/documents/{d.id}/text?format=elements",
            ),
            ("get_original", f"/v1/collections/{d.collection_id}/documents/{d.id}/original"),
            (
                "get_version_original",
                f"/v1/collections/{d.collection_id}/documents/{d.id}/versions/{self._version(d)}/original",
            ),
            ("get_chunk", f"/v1/collections/{cid}/chunks/{chunk}"),
            ("get_job", f"/v1/jobs/{self.rng.choice(me.jobs)}"),
            ("list_deliveries", f"/v1/collections/{me.collections[0]}/webhook-deliveries"),
        ]
        label, path = self.rng.choice(options)
        return Call(me.label, "own", f"own_{label}", "GET", path, OK)

    def _own_search(self, me: World, _: World) -> Call:
        filters = None
        if self.rng.random() < 0.3:
            filters = {"document_ids": [d.id for d in self.rng.sample(me.docs, 5)]}
        path = f"/v1/collections/{self._col(me)}/search"
        return Call(me.label, "own", "own_search", "POST", path, OK, self._search_body(filters))

    def _own_answer(self, me: World, _: World) -> Call:
        body = {
            "question": self.rng.choice(QUESTIONS),
            "rerank": False,
            "stream": self.rng.random() < 0.3,
        }
        path = f"/v1/collections/{self._col(me)}/answers"
        return Call(me.label, "own", "own_answer", "POST", path, OK, body)

    def _own_upload_unchanged(self, me: World, _: World) -> Call:
        d = self._doc(me)
        path = f"/v1/collections/{d.collection_id}/documents"
        return Call(me.label, "own", "own_upload_unchanged", "POST", path, OK, upload=d)

    def _filter_probe(self, me: World, other: World) -> Call:
        """Own collection, the other tenant's ids, metadata or text in the request."""
        foreign = [d.id for d in self.rng.sample(other.docs, self.rng.choice([1, 5, 20]))]
        if self.rng.random() < 0.5:
            foreign += [d.id for d in self.rng.sample(me.docs, 2)]
        od = self._doc(other)
        col = self._col(me)
        choice = self.rng.randrange(5)
        if choice == 0:
            body = self._search_body({"document_ids": foreign})
            return Call(
                me.label,
                "filter",
                "filter_search_doc_ids",
                "POST",
                f"/v1/collections/{col}/search",
                OK,
                body,
            )
        if choice == 1:
            body = self._search_body({"metadata": {"marker": od.metadata["marker"]}})
            return Call(
                me.label,
                "filter",
                "filter_search_metadata",
                "POST",
                f"/v1/collections/{col}/search",
                OK,
                body,
            )
        if choice == 2:  # the other tenant's text marker as the query (lexical leg)
            body = self._search_body(query=f"{other.mark}d{self.rng.randrange(N_DOCS)}p1r1")
            body["mode"] = self.rng.choice(["lexical", "hybrid"])
            return Call(
                me.label,
                "filter",
                "filter_search_foreign_text",
                "POST",
                f"/v1/collections/{col}/search",
                OK,
                body,
            )
        body = {
            "question": self.rng.choice(QUESTIONS),
            "rerank": False,
            "filters": {"document_ids": foreign},
            "stream": choice == 4,
        }
        return Call(
            me.label,
            "filter",
            "filter_answer_doc_ids",
            "POST",
            f"/v1/collections/{col}/answers",
            OK,
            body,
        )

    # cross-tenant probes

    def _cross_read(self, me: World, other: World) -> Call:
        d = self._doc(other)
        mine = self._col(me)
        base = self.rng.choice([d.collection_id, mine])  # foreign path or own collection path
        ocid, ochunk = self.rng.choice(other.all_chunks)
        options = [
            ("get_collection", f"/v1/collections/{self._col(other)}"),
            ("list_documents", f"/v1/collections/{self._col(other)}/documents"),
            ("get_document", f"/v1/collections/{base}/documents/{d.id}"),
            ("get_text", f"/v1/collections/{base}/documents/{d.id}/text"),
            ("get_elements", f"/v1/collections/{base}/documents/{d.id}/text?format=elements"),
            ("get_original", f"/v1/collections/{base}/documents/{d.id}/original"),
            (
                "get_version_original",
                f"/v1/collections/{base}/documents/{d.id}/versions/{self._version(d)}/original",
            ),
            ("get_chunk", f"/v1/collections/{self.rng.choice([ocid, mine])}/chunks/{ochunk}"),
            ("get_job", f"/v1/jobs/{self.rng.choice(other.jobs)}"),
            ("get_job_delivery_id", f"/v1/jobs/{self.rng.choice(other.deliveries)}"),
            (
                "get_chunk_delivery_id",
                f"/v1/collections/{mine}/chunks/{self.rng.choice(other.deliveries)}",
            ),
            (
                "get_document_version_id",
                f"/v1/collections/{mine}/documents/{self.rng.choice(other.version_ids)}",
            ),
            ("list_deliveries", f"/v1/collections/{other.collections[0]}/webhook-deliveries"),
            ("get_job_tenant_id", f"/v1/jobs/{other.tenant_id}"),
        ]
        label, path = self.rng.choice(options)
        return Call(me.label, "cross", f"cross_{label}", "GET", path, NOT_FOUND)

    def _cross_write(self, me: World, other: World) -> Call:
        d = self._doc(other)
        base = self.rng.choice([d.collection_id, self._col(me)])
        choice = self.rng.randrange(6)
        if choice == 0:
            return Call(
                me.label,
                "cross",
                "cross_patch_collection",
                "PATCH",
                f"/v1/collections/{self._col(other)}",
                NOT_FOUND,
                {"retrieval": {"k": 3}},
            )
        if choice == 1:
            return Call(
                me.label,
                "cross",
                "cross_delete_collection",
                "DELETE",
                f"/v1/collections/{self._col(other)}",
                NOT_FOUND,
            )
        if choice == 2:
            return Call(
                me.label,
                "cross",
                "cross_patch_document",
                "PATCH",
                f"/v1/collections/{base}/documents/{d.id}",
                NOT_FOUND,
                {"title": "fnt05-overwritten", "metadata": {"x": "y"}},
            )
        if choice == 3:
            return Call(
                me.label,
                "cross",
                "cross_delete_document",
                "DELETE",
                f"/v1/collections/{base}/documents/{d.id}",
                NOT_FOUND,
            )
        # upload into the other tenant's collection (own content, so nothing echoes)
        mine = Doc(
            "",
            self._col(other),
            f"fnt05-intruder-{self.rng.randrange(10**9)}",
            "intruder",
            {"x": "y"},
            b"# intruder\n\nnot yours\n",
        )
        return Call(
            me.label,
            "cross",
            "cross_upload",
            "POST",
            f"/v1/collections/{mine.collection_id}/documents",
            NOT_FOUND,
            upload=mine,
        )

    def _cross_search(self, me: World, other: World) -> Call:
        col = self._col(other)
        if self.rng.random() < 0.6:
            body = self._search_body({"document_ids": [self._doc(other).id]})
            return Call(
                me.label,
                "cross",
                "cross_search",
                "POST",
                f"/v1/collections/{col}/search",
                NOT_FOUND,
                body,
            )
        body = {
            "question": self.rng.choice(QUESTIONS),
            "rerank": False,
            "stream": self.rng.random() < 0.5,
        }
        return Call(
            me.label,
            "cross",
            "cross_answer",
            "POST",
            f"/v1/collections/{col}/answers",
            NOT_FOUND,
            body,
        )

    def _admin_probe(self, me: World, other: World) -> Call:
        tid = self.rng.choice([other.tenant_id, me.tenant_id])
        kid = other.key_id if tid == other.tenant_id else me.key_id
        options: list[tuple[str, str, str, Any]] = [
            ("list_tenants", "GET", "/v1/admin/tenants", None),
            ("create_tenant", "POST", "/v1/admin/tenants", {"name": "fnt05-intruder"}),
            ("get_tenant", "GET", f"/v1/admin/tenants/{tid}", None),
            (
                "patch_tenant",
                "PATCH",
                f"/v1/admin/tenants/{tid}",
                {"quotas": {"max_documents": 1}},
            ),
            ("delete_tenant", "DELETE", f"/v1/admin/tenants/{tid}", None),
            ("usage", "GET", f"/v1/admin/tenants/{tid}/usage", None),
            ("export", "GET", f"/v1/admin/tenants/{tid}/export?include_originals=true", None),
            ("list_keys", "GET", f"/v1/admin/tenants/{tid}/keys", None),
            ("create_key", "POST", f"/v1/admin/tenants/{tid}/keys", {"name": "intruder"}),
            ("revoke_key", "DELETE", f"/v1/admin/tenants/{tid}/keys/{kid}", None),
            ("rotate_key", "POST", f"/v1/admin/tenants/{tid}/keys/{kid}/rotate", None),
            ("list_spaces", "GET", "/v1/admin/embedding-spaces", None),
            (
                "register_space",
                "POST",
                "/v1/admin/embedding-spaces",
                {"name": SPACE, "provider": "onnx_local", "model": "bge-small-en-v1.5"},
            ),
            ("delete_space", "DELETE", "/v1/admin/embedding-spaces/fnt05-no-such-space", None),
        ]
        label, method, path, body = self.rng.choice(options)
        return Call(me.label, "admin", f"admin_{label}", method, path, UNAUTH, body)

    def _guess(self, other: World) -> str:
        choice = self.rng.randrange(3)
        if choice == 0:
            return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        if choice == 1:  # a fresh ULID, the format the service issues
            return str(ULID().to_uuid())
        real = self.rng.choice(sorted(other.ids))  # neighbour of a real id
        flipped = format(int(real[-1], 16) ^ (1 + self.rng.randrange(15)), "x")
        return real[:-1] + flipped

    def _guess_probe(self, me: World, other: World) -> Call:
        g = self._guess(other)
        mine = self._col(me)
        options = [
            ("job", f"/v1/jobs/{g}"),
            ("collection", f"/v1/collections/{g}"),
            ("document", f"/v1/collections/{mine}/documents/{g}"),
            ("document_text", f"/v1/collections/{self._col(other)}/documents/{g}/text"),
            ("chunk", f"/v1/collections/{mine}/chunks/{g}"),
            ("deliveries", f"/v1/collections/{g}/webhook-deliveries"),
            ("version_original", f"/v1/collections/{g}/documents/{g}/versions/1/original"),
        ]
        label, path = self.rng.choice(options)
        return Call(me.label, "guess", f"guess_{label}", "GET", path, NOT_FOUND)


@dataclass
class Stats:
    """Outcome of one traffic run."""

    calls: int = 0
    leaks: int = 0
    unexpected: int = 0
    transport_errors: int = 0
    own_marker_hits: int = 0  # positive control: own responses do carry own markers
    by_label: Counter[str] = field(default_factory=Counter)
    leaks_by_label: Counter[str] = field(default_factory=Counter)
    samples: list[str] = field(default_factory=list)
    uuids: dict[str, set[str]] = field(default_factory=lambda: {"A": set(), "B": set()})

    def note(self, text: str) -> None:
        if len(self.samples) < 20:
            self.samples.append(text[:400])


class Scanner:
    """Finds the other tenant's strings and ids in a response."""

    def __init__(self, worlds: dict[str, World]) -> None:
        self.worlds = worlds

    def foreign(self, actor: str, text: str) -> list[str]:
        other = self.worlds["A" if actor == "B" else "B"]
        hits = [s for s in other.strings if s and s in text]
        hits += sorted(set(UUID_RE.findall(text)) & other.ids)
        return hits


async def _send(client: httpx.AsyncClient, call: Call) -> httpx.Response:
    if call.upload is not None:
        d = call.upload
        data = {"external_id": d.external_id, "title": d.title, "metadata": json.dumps(d.metadata)}
        files = {"file": (f"{d.external_id}.md", d.content)}
        return await client.request(call.method, call.path, data=data, files=files)
    if call.json_body is not None:
        return await client.request(call.method, call.path, json=call.json_body)
    return await client.request(call.method, call.path)


def _judge(call: Call, resp: httpx.Response, scanner: Scanner, stats: Stats) -> None:
    stats.calls += 1
    stats.by_label[call.label] += 1
    text = resp.text + "\n" + "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
    stats.uuids[call.actor].update(UUID_RE.findall(text))
    found = scanner.foreign(call.actor, text)
    if call.kind == "own" and scanner.worlds[call.actor].mark in text:
        stats.own_marker_hits += 1
    foreign_2xx = call.kind in {"cross", "admin", "guess"} and 200 <= resp.status_code < 300
    if found or foreign_2xx:
        stats.leaks += 1
        stats.leaks_by_label[call.label] += 1
        stats.note(f"LEAK {call.actor} {call.method} {call.path} -> {resp.status_code} {found}")
    elif resp.status_code not in call.expect:
        stats.unexpected += 1
        stats.note(
            f"UNEXPECTED {call.actor} {call.method} {call.path} -> "
            f"{resp.status_code} {resp.text[:200]}"
        )


async def _run_traffic(inst: h.Instance, worlds: dict[str, World], calls: list[Call]) -> Stats:
    stats = Stats()
    scanner = Scanner(worlds)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    clients = {
        label: httpx.AsyncClient(
            base_url=inst.data_url,
            headers={"Authorization": f"Bearer {w.key}"},
            timeout=120.0,
            limits=limits,
        )
        for label, w in worlds.items()
    }
    queue: asyncio.Queue[Call] = asyncio.Queue()
    for call in calls:
        queue.put_nowait(call)

    async def worker() -> None:
        while True:
            try:
                call = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                resp = await _send(clients[call.actor], call)
            except httpx.HTTPError as exc:
                stats.transport_errors += 1
                stats.note(f"TRANSPORT {call.method} {call.path}: {exc!r}")
                continue
            _judge(call, resp, scanner, stats)

    try:
        await asyncio.gather(*(worker() for _ in range(CONCURRENCY)))
    finally:
        for client in clients.values():
            await client.aclose()
    return stats


def _foreign_owned_ids(worlds: dict[str, World], stats: list[Stats]) -> list[str]:
    """Ids seen in responses to one tenant that the database says another tenant owns."""
    problems: list[str] = []
    owners = (
        "SELECT id, tenant_id FROM footnote.collections UNION ALL "
        "SELECT id, tenant_id FROM footnote.documents UNION ALL "
        "SELECT id, tenant_id FROM footnote.document_versions UNION ALL "
        "SELECT id, tenant_id FROM footnote.chunks UNION ALL "
        "SELECT id, tenant_id FROM footnote.jobs UNION ALL "
        "SELECT id, tenant_id FROM footnote.webhook_deliveries UNION ALL "
        "SELECT id, tenant_id FROM footnote.api_keys UNION ALL "
        "SELECT id, id FROM footnote.tenants"
    )
    with h.connect() as conn:
        for label, world in worlds.items():
            seen = sorted(set().union(*(s.uuids[label] for s in stats)))
            rows = conn.execute(
                f"SELECT o.id::text, o.tenant_id::text FROM ({owners}) AS o "  # noqa: S608 - constant text
                "WHERE o.id = ANY(%s::uuid[])",
                (seen,),
            ).fetchall()
            problems += [
                f"{label} saw {rid} owned by {owner}"
                for rid, owner in rows
                if owner != world.tenant_id
            ]
    return problems


# --- database observation -------------------------------------------------------------


def _container_ip(container: str) -> str:
    fmt = '{{(index .NetworkSettings.Networks "' + h.NETWORK + '").IPAddress}}'
    proc = subprocess.run(
        ["docker", "inspect", "-f", fmt, container], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@contextmanager
def _sample_queries(container: str) -> Iterator[set[str]]:
    """Collect the SQL text of the instance's backends (superuser view) while traffic runs."""
    ip = _container_ip(container)
    seen: set[str] = set()
    stop = threading.Event()

    def loop() -> None:
        with psycopg.connect(h.SUPERUSER_DSN, autocommit=True) as conn:
            while not stop.is_set():
                rows = conn.execute(
                    "SELECT query FROM pg_stat_activity WHERE client_addr = %s::inet"
                    " AND usename = 'footnote_app' AND query LIKE '%%footnote.%%'",
                    (ip,),
                ).fetchall()
                seen.update(r[0] for r in rows)
                time.sleep(0.002)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield seen
    finally:
        stop.set()
        thread.join(timeout=10)


def _service_session_users(container: str) -> set[str]:
    with psycopg.connect(h.SUPERUSER_DSN, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT usename FROM pg_stat_activity WHERE client_addr = %s::inet",
            (_container_ip(container),),
        ).fetchall()
    return {r[0] for r in rows}


def _readiness(inst: h.Instance) -> tuple[int, dict[str, Any]]:
    resp = httpx.get(f"{inst.admin_url}/readyz", timeout=30.0)
    body: dict[str, Any] = resp.json()
    return resp.status_code, body


def _container_logs(container: str) -> str:
    proc = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True, check=False
    )
    return proc.stdout + proc.stderr


# --- raw SQL, role and catalog assertions -------------------------------------------------


@dataclass
class Checks:
    """Raw-SQL leaks and failed role/catalog assertions."""

    leaks: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    passed: int = 0

    def check(self, ok: bool, what: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(what)


def _tables(conn: psycopg.Connection[Any]) -> list[tuple[str, bool, bool, bool]]:
    """(name, relrowsecurity, relforcerowsecurity, has tenant_id) of every footnote table."""
    rows = conn.execute(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,"
        " EXISTS (SELECT 1 FROM pg_attribute a WHERE a.attrelid = c.oid"
        "         AND a.attname = 'tenant_id' AND NOT a.attisdropped)"
        " FROM pg_class c WHERE c.relnamespace = 'footnote'::regnamespace"
        " AND c.relkind IN ('r', 'p') ORDER BY c.relname"
    ).fetchall()
    return [(str(r[0]), bool(r[1]), bool(r[2]), bool(r[3])) for r in rows]


def _count(
    conn: psycopg.Connection[Any], table: str, where: sql.Composable = NO_WHERE, *params: Any
) -> int:
    query = sql.SQL("SELECT count(*) FROM footnote.{} {}").format(sql.Identifier(table), where)
    row = conn.execute(query, params).fetchone()
    return int(row[0]) if row else 0


def _visibility(a: World, b: World, checks: Checks) -> None:
    """No, empty, random, malformed and the other tenant's setting see no foreign rows."""
    with h.connect(h.APP_DSN) as conn:
        tenant_tables = [t for t, _, _, has in _tables(conn) if has and t not in NO_ACCESS]
    checks.check(len(tenant_tables) >= 9, f"tenant tables found: {tenant_tables}")
    settings: list[tuple[str, str | None]] = [
        ("none", None),
        ("empty", ""),
        ("random", str(uuid.uuid4())),
        ("malformed", "not-a-uuid"),
    ]
    for table in tenant_tables:
        for name, value in settings:
            with h.connect(h.APP_DSN) as conn:
                if value is not None:
                    conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (value,))
                try:
                    n = _count(conn, table)
                except errors.InvalidTextRepresentation as exc:
                    # a malformed setting is an error, never a bypass; others must not fail
                    checks.check(name == "malformed", f"{table} with {name} setting: {exc}")
                    n = 0
                if n != 0:
                    checks.leaks.append(f"{table}: {n} rows visible with {name} setting")
        with h.connect(h.APP_DSN) as conn:
            conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (b.tenant_id,))
            foreign = _count(conn, table, sql.SQL("WHERE tenant_id <> %s"), b.tenant_id)
            of_a = _count(conn, table, sql.SQL("WHERE tenant_id = %s"), a.tenant_id)
            own = _count(conn, table, sql.SQL("WHERE tenant_id = %s"), b.tenant_id)
        if foreign or of_a:
            checks.leaks.append(f"{table}: B's setting sees {foreign} foreign rows, {of_a} of A")
        if table in {"collections", "documents", "document_versions", "chunks", "jobs"}:
            checks.check(own > 0, f"{table}: B's setting sees its own rows (positive control)")


type Probe = tuple[str, sql.Composable, tuple[type[Exception], ...]]


def _probe(checks: Checks, b: World, probe: Probe, params: dict[str, Any]) -> None:
    """Run a write as footnote_app with B's setting; it must raise one of the accepted errors."""
    what, statement, accepted = probe
    with h.connect(h.APP_DSN) as conn:
        try:
            with conn.transaction(force_rollback=True):
                conn.execute("SELECT set_config('footnote.tenant_id', %s, true)", (b.tenant_id,))
                cur = conn.execute(statement, params)
                rowcount = cur.rowcount
        except accepted:
            checks.passed += 1
            return
        except psycopg.Error as exc:
            checks.failed.append(f"{what}: unexpected {type(exc).__name__}: {exc}")
            return
    checks.leaks.append(f"{what}: succeeded ({rowcount} rows)")


def _affects_nothing(
    checks: Checks, b: World, what: str, statement: sql.Composable, params: dict[str, Any]
) -> None:
    with h.connect(h.APP_DSN) as conn, conn.transaction(force_rollback=True):
        conn.execute("SELECT set_config('footnote.tenant_id', %s, true)", (b.tenant_id,))
        rowcount = conn.execute(statement, params).rowcount
    if rowcount != 0:
        checks.leaks.append(f"{what}: affected {rowcount} rows of A")


def _write_probes(a: World, b: World, checks: Checks) -> None:
    """Cross-tenant insert/update/delete as footnote_app with tenant B's setting."""
    rls = (errors.InsufficientPrivilege,)
    rls_or_fk = (errors.InsufficientPrivilege, errors.ForeignKeyViolation)
    p: dict[str, Any] = {
        "a": a.tenant_id,
        "b": b.tenant_id,
        "id": str(uuid.uuid4()),
        "acol": a.collections[0],
        "adoc": a.docs[0].id,
        "h": "0" * 64,
    }
    q = sql.SQL
    space = q("(SELECT id FROM footnote.embedding_spaces ORDER BY id LIMIT 1)")
    probes: list[Probe] = [
        (
            "insert collection as A",
            q(
                "INSERT INTO footnote.collections (tenant_id, id, name, active_space_id)"
                " VALUES (%(a)s, %(id)s, 'fnt05-probe', {})"
            ).format(space),
            rls,
        ),
        (
            "insert document as A",
            q(
                "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id)"
                " VALUES (%(a)s, %(id)s, %(acol)s, 'fnt05-probe')"
            ),
            rls,
        ),
        (
            "insert document of B into A's collection",
            q(
                "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id)"
                " VALUES (%(b)s, %(id)s, %(acol)s, 'fnt05-probe')"
            ),
            rls_or_fk,
        ),
        (
            "insert chunk of B referencing A's document",
            q(
                "INSERT INTO footnote.chunks (tenant_id, id, collection_id, document_id,"
                " content_hash, text, embedded_text, token_count, fts_config)"
                " VALUES (%(b)s, %(id)s, %(acol)s, %(adoc)s, %(h)s, 'x', 'x', 1, 'simple')"
            ),
            rls_or_fk,
        ),
        (
            "insert job as A",
            q("INSERT INTO footnote.jobs (tenant_id, id, kind) VALUES (%(a)s, %(id)s, 'gc')"),
            rls,
        ),
        (
            "insert delivery as A",
            q(
                "INSERT INTO footnote.webhook_deliveries (tenant_id, id, collection_id, event, url)"
                " VALUES (%(a)s, %(id)s, %(acol)s, 'document.ready', 'http://x')"
            ),
            rls,
        ),
        (
            "insert usage as A",
            q("INSERT INTO footnote.tenant_usage (tenant_id, day) VALUES (%(a)s, '2000-01-01')"),
            rls,
        ),
        (
            "move B's collections to A",
            q("UPDATE footnote.collections SET tenant_id = %(a)s WHERE tenant_id = %(b)s"),
            rls_or_fk,
        ),
        (
            "insert api key",
            q(
                "INSERT INTO footnote.api_keys (tenant_id, id, key_hash, prefix)"
                " VALUES (%(b)s, %(id)s, decode(repeat('00', 32), 'hex'), 'fn_x')"
            ),
            rls,
        ),
        (
            "insert tenant",
            q("INSERT INTO footnote.tenants (id, name) VALUES (%(id)s, 'fnt05-probe')"),
            rls,
        ),
    ]
    for probe in probes:
        _probe(checks, b, probe, p)
    for table in ("collections", "documents", "chunks", "jobs", "webhook_deliveries"):
        upd = q("UPDATE footnote.{} SET tenant_id = tenant_id WHERE tenant_id = %(a)s")
        dele = q("DELETE FROM footnote.{} WHERE tenant_id = %(a)s")
        _affects_nothing(checks, b, f"update {table} of A", upd.format(sql.Identifier(table)), p)
        _affects_nothing(checks, b, f"delete {table} of A", dele.format(sql.Identifier(table)), p)


def _role_and_catalog(checks: Checks) -> None:
    """Runtime role assertions over the service's app DSN (same role and URL)."""
    with h.connect(h.APP_DSN) as conn:
        row = conn.execute(
            "SELECT current_user::text, r.rolsuper, r.rolbypassrls,"
            " pg_has_role(current_user, 'footnote_admin', 'MEMBER'),"
            " EXISTS (SELECT 1 FROM pg_class c WHERE c.relnamespace = 'footnote'::regnamespace"
            "         AND c.relowner = r.oid),"
            " EXISTS (SELECT 1 FROM pg_proc p WHERE p.pronamespace = 'footnote'::regnamespace"
            "         AND p.proowner = r.oid),"
            " EXISTS (SELECT 1 FROM pg_type t WHERE t.typnamespace = 'footnote'::regnamespace"
            "         AND t.typowner = r.oid),"
            " EXISTS (SELECT 1 FROM pg_namespace n WHERE n.nspname = 'footnote'"
            "         AND n.nspowner = r.oid)"
            " FROM pg_roles r WHERE r.rolname = current_user"
        ).fetchone()
        assert row is not None
        user, superuser, bypass, admin_member, owns_rel, owns_proc, owns_type, owns_schema = row
        checks.check(user == "footnote_app", f"current_user = 'footnote_app' (got {user})")
        checks.check(superuser is False, "rolsuper = f")
        checks.check(bypass is False, "rolbypassrls = f")
        checks.check(
            not (owns_rel or owns_proc or owns_type or owns_schema),
            "owns no object in schema footnote",
        )
        checks.check(admin_member is False, "not a member of footnote_admin")
        tables = _tables(conn)
    checks.check(bool(tables), "schema footnote has tables")
    for name, rls, force, _ in tables:
        if name in NO_ACCESS:
            with h.connect(h.APP_DSN) as conn:
                try:
                    conn.execute(
                        sql.SQL("SELECT 1 FROM footnote.{} LIMIT 1").format(sql.Identifier(name))
                    )
                    denied = False
                except errors.InsufficientPrivilege as exc:
                    denied = "permission denied" in str(exc)
            checks.check(denied, f"{name}: SELECT raises permission denied")
        elif name in GLOBAL_READ_ONLY:
            with h.connect(h.APP_DSN) as conn:
                privs = {
                    p: bool(
                        conn.execute(
                            "SELECT has_table_privilege(current_user, %s::regclass, %s)",
                            (f"footnote.{name}", p),
                        ).fetchone()[0]
                    )  # type: ignore[index]
                    for p in ("SELECT", *WRITE_PRIVILEGES)
                }
            granted = sorted(p for p, ok in privs.items() if ok)
            checks.check(granted == ["SELECT"], f"{name}: SELECT-only privilege (got {granted})")
        else:
            checks.check(rls and force, f"{name}: relrowsecurity and relforcerowsecurity")


# --- BYPASSRLS readiness sub-test -----------------------------------------------------------


def _bypass_readyz(checks: Checks) -> dict[str, Any]:
    """An instance whose app DSN role has BYPASSRLS must answer /readyz 503 (app_role)."""
    role = f"fnt05_bypass_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    ident = sql.Identifier(role)
    with h.connect(h.SUPERUSER_DSN) as su:
        su.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER BYPASSRLS PASSWORD {}").format(
                ident, sql.Literal(password)
            )
        )
        # Enough to pass the other database checks, nothing to claim or touch tenant rows.
        su.execute(sql.SQL("GRANT CONNECT ON DATABASE footnote TO {}").format(ident))
        su.execute(sql.SQL("GRANT USAGE ON SCHEMA footnote TO {}").format(ident))
        su.execute(
            sql.SQL(
                "GRANT SELECT ON footnote.alembic_version, footnote.embedding_spaces TO {}"
            ).format(ident)
        )
    env = {
        **INSTANCE_ENV,
        "FOOTNOTE_DATABASE_URL": f"postgresql://{role}:{password}@postgres:5432/footnote",
    }
    result: dict[str, Any] = {"status": None, "app_role": None}
    try:
        with h.custom_instance("fnt05-bypassrls", env, wait=False) as inst:
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                try:
                    resp = httpx.get(f"{inst.admin_url}/readyz", timeout=10.0)
                    data_plane = httpx.get(f"{inst.data_url}/readyz", timeout=10.0)
                except httpx.HTTPError:
                    time.sleep(1.0)
                    continue
                problems = resp.json().get("checks", {}).get("app_role") or []
                result = {
                    "status": resp.status_code,
                    "data_plane_status": data_plane.status_code,
                    "app_role": problems,
                    "checks": resp.json().get("checks"),
                }
                if problems or resp.status_code == 200:
                    break
                time.sleep(1.0)
    finally:
        with h.connect(h.SUPERUSER_DSN) as su:
            su.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s", (role,)
            )
            su.execute(sql.SQL("DROP OWNED BY {}").format(ident))
            su.execute(sql.SQL("DROP ROLE {}").format(ident))
    checks.check(result["status"] == 503, f"BYPASSRLS role: /readyz 503 (got {result})")
    checks.check(result.get("data_plane_status") == 503, "BYPASSRLS role: data-plane /readyz 503")
    checks.check(
        result["app_role"] == ["app role has BYPASSRLS"],
        f"BYPASSRLS role: app_role check names BYPASSRLS (got {result['app_role']})",
    )
    return result


# --- the criterion ----------------------------------------------------------------------------


def _verify_instance(inst: h.Instance, checks: Checks, name: str) -> None:
    """The service's own connection is footnote_app and its readiness app_role check is clean."""
    status, body = _readiness(inst)
    app_role = (body.get("checks") or {}).get("app_role")
    checks.check(app_role == [], f"{name}: readiness app_role clean (got {app_role}, {status})")
    users = _service_session_users(inst.container)
    checks.check(
        "footnote_app" in users and users <= {"footnote_app", "footnote_admin"},
        f"{name}: service sessions run as footnote_app/footnote_admin (got {users})",
    )


def test_acc_fnt05_tenant_isolation() -> None:
    """Two tenants, 10,000 mixed calls, 1,000 RLS-only calls, raw SQL, roles and catalog."""
    limits = thresholds("FNT-05")
    started = time.monotonic()
    a = World("A", f"fnt05a{uuid.uuid4().hex}")
    b = World("B", f"fnt05b{uuid.uuid4().hex}")
    worlds = {"A": a, "B": b}
    checks = Checks()
    setup_problems: list[str] = []

    with h.custom_instance("fnt05-normal", INSTANCE_ENV) as inst:
        _seed_world(inst, a)
        _seed_world(inst, b)
        _wait_deliveries(a)
        _wait_deliveries(b)
        before = {k: _snapshot(w) for k, w in worlds.items()}
        calls = Planner(a, b, SEED).plan(int(limits["calls"]))
        with _sample_queries(inst.container) as normal_sql:
            normal = asyncio.run(_run_traffic(inst, worlds, calls))
        _verify_instance(inst, checks, "normal")
    seed_seconds_and_normal = time.monotonic() - started

    with h.custom_instance("fnt05-fault", FAULT_ENV) as fault_inst:
        if FAULT_WARNING not in _container_logs(fault_inst.container):
            setup_problems.append("fault instance did not log the predicate warning")
        fault_calls = Planner(a, b, SEED + 1).plan(int(limits["fault_calls"]))
        with _sample_queries(fault_inst.container) as fault_sql:
            fault = asyncio.run(_run_traffic(fault_inst, worlds, fault_calls))
        _verify_instance(fault_inst, checks, "fault")

    # The flag really removes the app predicates: the normal instance's SQL carries
    # `"<alias>".tenant_id = $n`, the fault instance's never does and has `TRUE` instead.
    normal_pred = sum(1 for s in normal_sql if APP_PREDICATE_RE.search(s))
    fault_pred = sum(1 for s in fault_sql if APP_PREDICATE_RE.search(s))
    fault_true = sum(1 for s in fault_sql if FAULT_PREDICATE_RE.search(s))
    if not (normal_pred > 0 and fault_pred == 0 and fault_true > 0):
        setup_problems.append(
            f"predicate removal not observed: normal {normal_pred}/{len(normal_sql)} with app "
            f"predicates; fault {fault_pred}/{len(fault_sql)} with, {fault_true} with TRUE"
        )

    if not (normal.own_marker_hits and fault.own_marker_hits):
        setup_problems.append("scanner positive control: no own marker seen in own responses")
    after = {k: _snapshot(w) for k, w in worlds.items()}
    tampered = [k for k in worlds if before[k] != after[k]]
    owned = _foreign_owned_ids(worlds, [normal, fault])

    _visibility(a, b, checks)
    _write_probes(a, b, checks)
    _role_and_catalog(checks)
    bypass = _bypass_readyz(checks)

    leaks = normal.leaks + fault.leaks + len(owned) + len(tampered) + len(checks.leaks)
    unexpected = normal.unexpected + fault.unexpected
    measured: dict[str, Any] = {
        "calls": normal.calls,
        "fault_calls": fault.calls,
        "leaks": leaks,
        "role_assertions_failed": len(checks.failed),
        "role_assertions_passed": checks.passed,
        "traffic_leaks": normal.leaks,
        "fault_traffic_leaks": fault.leaks,
        "foreign_owned_ids": len(owned),
        "tampered_tenants": tampered,
        "sql_leaks": len(checks.leaks),
        "unexpected_status": unexpected,
        "transport_errors": normal.transport_errors + fault.transport_errors,
        "own_marker_hits": normal.own_marker_hits + fault.own_marker_hits,
        "probe_calls": sum(
            n
            for lbl, n in (normal.by_label + fault.by_label).items()
            if lbl.startswith(("cross_", "admin_", "guess_", "filter_"))
        ),
        "sampled_sql": {
            "normal": len(normal_sql),
            "normal_with_app_predicates": normal_pred,
            "fault": len(fault_sql),
            "fault_with_true": fault_true,
        },
        "bypassrls_readyz": bypass.get("status"),
        "seconds_seed_and_normal": round(seed_seconds_and_normal, 1),
        "seconds_total": round(time.monotonic() - started, 1),
    }
    passed = (
        normal.calls >= limits["calls"]
        and fault.calls >= limits["fault_calls"]
        and leaks <= limits["leaks"]
        and len(checks.failed) <= limits["role_assertions_failed"]
        and unexpected == 0
        and not setup_problems
    )
    details = json.dumps(
        {
            "failed_assertions": checks.failed,
            "sql_leaks": checks.leaks,
            "foreign_ids": owned[:20],
            "setup_problems": setup_problems,
            "normal_samples": normal.samples,
            "fault_samples": fault.samples,
            "leaks_by_label": dict(normal.leaks_by_label + fault.leaks_by_label),
            "calls_by_label": dict(sorted((normal.by_label + fault.by_label).items())),
            "bypassrls": bypass,
        },
        default=str,
    )
    write_result("FNT-05", measured, passed, details)
    assert passed, f"FNT-05 failed: {measured} {details[:3000]}"
