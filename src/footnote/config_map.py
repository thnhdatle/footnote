# SPDX-License-Identifier: Apache-2.0
"""Trivial mappings from `Settings` to the config objects the adapters take.

Only the bootstrap and CLI import this module (it depends on adapters).
"""

from __future__ import annotations

from datetime import timedelta

from footnote.adapters.embedders.factory import OpenAICompatDefaults
from footnote.adapters.llm.client import LLMConfig
from footnote.adapters.postgres.search import SnapshotSettings
from footnote.adapters.postgres.session import PoolSettings
from footnote.adapters.queue.worker import WorkerConfig
from footnote.adapters.webhooks.sender import WebhookConfig
from footnote.config import ConfigError, Settings, env_name


def llm_config(settings: Settings) -> LLMConfig:
    """`llm.*` for the chat client."""
    c = settings.llm
    return LLMConfig(
        base_url=c.base_url,
        model=c.model,
        api_key=c.api_key.get_secret_value(),
        extra_headers=dict(c.extra_headers),
        extra_body=dict(c.extra_body),
        temperature=c.temperature,
        max_tokens=c.max_tokens,
        timeout=c.timeout,
        max_retries=c.max_retries,
    )


def embedding_defaults(settings: Settings) -> OpenAICompatDefaults:
    """`embedding.*` connection settings shared by every openai_compat space."""
    c = settings.embedding
    return OpenAICompatDefaults(
        base_url=c.base_url,
        api_key=c.api_key.get_secret_value(),
        extra_headers=dict(c.extra_headers),
        batch_size=c.batch_size,
        max_concurrency=c.max_concurrency,
        timeout=c.timeout,
        max_retries=c.max_retries,
    )


def worker_config(settings: Settings, worker_id: str) -> WorkerConfig:
    """`ingest.*` for the queue worker."""
    c = settings.ingest
    return WorkerConfig(
        worker_id=worker_id,
        concurrency=c.workers,
        lease=timedelta(seconds=c.job_lease),
        max_attempts=c.job_max_attempts,
        poll_interval=c.poll_interval,
        shutdown_timeout=settings.shutdown_timeout,
    )


def webhook_config(settings: Settings) -> WebhookConfig:
    """`webhooks.*` for the sender."""
    c = settings.webhooks
    return WebhookConfig(
        allow_http=c.allow_http,
        allow_private=c.allow_private,
        timeout=c.timeout,
        max_attempts=c.max_attempts,
    )


def snapshot_settings(settings: Settings) -> SnapshotSettings:
    """`retrieval.*` scan options for search transactions."""
    r = settings.retrieval
    return SnapshotSettings(
        ef_search=r.ef_search,
        iterative_scan=r.iterative_scan,
        max_scan_tuples=r.max_scan_tuples,
    )


def app_pool_settings(settings: Settings) -> PoolSettings:
    """`database.*` for the `footnote_app` pool; ConfigError when no URL is configured."""
    c = settings.database
    url = c.url.get_secret_value()
    if not url:
        raise ConfigError(
            [f"database.url: not set (use {env_name(('database', 'url'))} or url_file)"]
        )
    return PoolSettings(
        conninfo=url,
        max_size=c.pool_max,
        prepare_threshold=c.prepare_threshold,
    )


def admin_pool_settings(settings: Settings) -> PoolSettings:
    """`admin_database.*` for the `footnote_admin` pool; ConfigError when no URL is configured."""
    c = settings.admin_database
    url = c.url.get_secret_value()
    if not url:
        raise ConfigError(
            [f"admin_database.url: not set (use {env_name(('admin_database', 'url'))} or url_file)"]
        )
    return PoolSettings(conninfo=url, max_size=c.pool_max)
