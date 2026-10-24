# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from footnote.observability import Metrics

METRIC_NAMES = [
    "footnote_http_requests_total",
    "footnote_http_request_duration_seconds",
    "footnote_jobs_total",
    "footnote_ingest_jobs_total",
    "footnote_jobs_queue_depth",
    "footnote_ingest_duration_seconds",
    "footnote_ingest_images_total",
    "footnote_search_duration_seconds",
    "footnote_answers_total",
    "footnote_embedding_requests_total",
    "footnote_embedding_texts_total",
    "footnote_embedding_truncated_total",
    "footnote_search_exact_scan_total",
    "footnote_webhook_deliveries_total",
    "footnote_build_info",
]


def sample(m: Metrics, name: str, **labels: str) -> float | None:
    return m.registry.get_sample_value(name, labels)


def test_registry_is_private_and_lists_every_f11_metric() -> None:
    a, b = Metrics(version="1"), Metrics(version="1")
    a.answers.labels("answered").inc()
    assert sample(b, "footnote_answers_total", outcome="answered") is None
    body, content_type = a.render()
    text = body.decode()
    assert content_type.startswith("text/plain")
    for name in METRIC_NAMES:
        assert f"# TYPE {name.removesuffix('_total')}" in text or f"# TYPE {name} " in text, name


def test_build_info_labels() -> None:
    m = Metrics(version="0.1.0", commit="abc")
    value = m.registry.get_sample_value(
        "footnote_build_info",
        {
            "version": "0.1.0",
            "commit": "abc",
            "python_version": __import__("platform").python_version(),
        },
    )
    assert value == 1


def test_ingest_jobs_equal_jobs_with_kind_ingest() -> None:
    m = Metrics(version="1")
    m.observe_job("ingest", "succeeded")
    m.observe_job("ingest", "failed")
    m.observe_job("gc", "succeeded")
    assert sample(m, "footnote_ingest_jobs_total", status="succeeded") == 1
    assert sample(m, "footnote_jobs_total", kind="ingest", status="succeeded") == 1
    assert sample(m, "footnote_jobs_total", kind="gc", status="succeeded") == 1
    assert sample(m, "footnote_ingest_jobs_total", status="gc") is None


def test_helpers_and_histograms() -> None:
    m = Metrics(version="1")
    m.observe_embedding("openai_compat", 5)
    m.observe_answer("abstained")
    m.observe_exact_scan("fallback")
    m.observe_image("failed")
    m.embedding_truncated.inc()
    m.http_requests.labels("/v1/search", "POST", "200").inc()
    m.http_duration.labels("/v1/search").observe(0.02)
    m.search_duration.labels("rerank").observe(0.3)
    m.ingest_duration.labels("parse").observe(1.2)
    m.jobs_queue_depth.labels("ingest").set(4)
    m.webhook_deliveries.labels("delivered").inc()
    assert sample(m, "footnote_embedding_requests_total", provider="openai_compat") == 1
    assert sample(m, "footnote_embedding_texts_total", provider="openai_compat") == 5
    assert sample(m, "footnote_answers_total", outcome="abstained") == 1
    assert sample(m, "footnote_search_exact_scan_total", reason="fallback") == 1
    assert sample(m, "footnote_ingest_images_total", outcome="failed") == 1
    assert sample(m, "footnote_embedding_truncated_total") == 1
    assert sample(m, "footnote_jobs_queue_depth", kind="ingest") == 4
    assert sample(m, "footnote_search_duration_seconds_count", stage="rerank") == 1
    assert sample(m, "footnote_http_request_duration_seconds_count", route="/v1/search") == 1
