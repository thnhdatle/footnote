# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics registry."""

from __future__ import annotations

import platform
from typing import Final, Literal

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

HTTP_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
STAGE_BUCKETS: Final = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0)
INGEST_BUCKETS: Final = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0, 900.0)

type Outcome = Literal["answered", "abstained", "unsupported", "error"]
type ScanReason = Literal["small_collection", "fallback"]
type JobStatus = Literal["succeeded", "failed", "retried", "cancelled"]


class Metrics:
    """All metrics on one registry, exposed with `render()`."""

    def __init__(self, *, version: str, commit: str = "unknown") -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        r = self.registry
        self.http_requests = Counter(
            "footnote_http_requests_total",
            "HTTP requests.",
            ["route", "method", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "footnote_http_request_duration_seconds",
            "HTTP request duration by route template.",
            ["route"],
            buckets=HTTP_BUCKETS,
            registry=r,
        )
        self.jobs = Counter(
            "footnote_jobs_total", "Jobs by kind and status.", ["kind", "status"], registry=r
        )
        self.ingest_jobs = Counter(
            "footnote_ingest_jobs_total",
            "Ingest jobs by status (legacy series).",
            ["status"],
            registry=r,
        )
        self.jobs_queue_depth = Gauge(
            "footnote_jobs_queue_depth", "Queued jobs by kind.", ["kind"], registry=r
        )
        self.ingest_duration = Histogram(
            "footnote_ingest_duration_seconds",
            "Ingest duration by stage.",
            ["stage"],
            buckets=INGEST_BUCKETS,
            registry=r,
        )
        self.search_duration = Histogram(
            "footnote_search_duration_seconds",
            "Search duration by stage.",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=r,
        )
        self.answers = Counter(
            "footnote_answers_total", "Answers by outcome.", ["outcome"], registry=r
        )
        self.embedding_requests = Counter(
            "footnote_embedding_requests_total", "Embedding requests.", ["provider"], registry=r
        )
        self.embedding_texts = Counter(
            "footnote_embedding_texts_total", "Texts embedded.", ["provider"], registry=r
        )
        self.embedding_truncated = Counter(
            "footnote_embedding_truncated_total", "Texts truncated to the model limit.", registry=r
        )
        self.search_exact_scan = Counter(
            "footnote_search_exact_scan_total",
            "Exact-scan searches by reason.",
            ["reason"],
            registry=r,
        )
        self.webhook_deliveries = Counter(
            "footnote_webhook_deliveries_total",
            "Webhook delivery attempts by status.",
            ["status"],
            registry=r,
        )
        build = Gauge(
            "footnote_build_info",
            "Build information.",
            ["version", "commit", "python_version"],
            registry=r,
        )
        build.labels(version, commit, platform.python_version()).set(1)

    def observe_job(self, kind: str, status: str) -> None:
        self.jobs.labels(kind, status).inc()
        if kind == "ingest":
            self.ingest_jobs.labels(status).inc()

    def observe_embedding(self, provider: str, texts: int) -> None:
        self.embedding_requests.labels(provider).inc()
        self.embedding_texts.labels(provider).inc(texts)

    def observe_answer(self, outcome: Outcome) -> None:
        self.answers.labels(outcome).inc()

    def observe_exact_scan(self, reason: ScanReason) -> None:
        self.search_exact_scan.labels(reason).inc()

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
