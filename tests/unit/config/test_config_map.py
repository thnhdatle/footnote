# SPDX-License-Identifier: Apache-2.0
"""Settings to adapter config mappings."""

from __future__ import annotations

from datetime import timedelta

import pytest

from footnote.config import ConfigError, load_settings
from footnote.config_map import (
    admin_pool_settings,
    app_pool_settings,
    embedding_defaults,
    llm_config,
    snapshot_settings,
    webhook_config,
    worker_config,
)


def test_mappings_carry_values_and_secrets() -> None:
    env = {
        "FOOTNOTE_LLM_BASE_URL": "http://llm/v1",
        "FOOTNOTE_LLM_MODEL": "m",
        "FOOTNOTE_LLM_API_KEY": "k",
        "FOOTNOTE_EMBEDDING_API_KEY": "e",
        "FOOTNOTE_DATABASE_URL": "postgresql://app",
        "FOOTNOTE_ADMIN_DATABASE_URL": "postgresql://admin",
        "FOOTNOTE_INGEST_WORKERS": "2",
        "FOOTNOTE_INGEST_JOB_LEASE": "90s",
        "FOOTNOTE_RETRIEVAL_EF_SEARCH": "80",
        "FOOTNOTE_WEBHOOKS_ALLOW_HTTP": "true",
    }
    s = load_settings(None, env)
    llm = llm_config(s)
    assert (llm.base_url, llm.api_key, llm.timeout, llm.max_retries) == (
        "http://llm/v1",
        "k",
        120.0,
        2,
    )
    assert embedding_defaults(s).api_key == "e"
    worker = worker_config(s, "w1")
    assert (worker.worker_id, worker.concurrency, worker.lease) == ("w1", 2, timedelta(seconds=90))
    assert worker.shutdown_timeout == 30.0
    assert webhook_config(s).allow_http is True
    assert snapshot_settings(s).ef_search == 80
    app = app_pool_settings(s)
    assert (app.conninfo, app.max_size, app.prepare_threshold) == ("postgresql://app", 20, 5)
    assert admin_pool_settings(s).conninfo == "postgresql://admin"


def test_pools_need_urls() -> None:
    s = load_settings(None, {})
    with pytest.raises(ConfigError, match="FOOTNOTE_DATABASE_URL"):
        app_pool_settings(s)
    with pytest.raises(ConfigError, match="FOOTNOTE_ADMIN_DATABASE_URL"):
        admin_pool_settings(s)
