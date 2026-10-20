# SPDX-License-Identifier: Apache-2.0
"""Collections use case: create, read, update settings, delete.

Settings arrive as partial patches merged onto the current settings (or the configured
defaults on create) and are validated as a whole before anything is written. A change
that needs background work starts its job in the same transaction as the settings
write: `chunking` → `reindex`, `language` → `fts_reconfig`; `collection_delete` is
enqueued with the `deleting` mark. Changing `embedding_space` goes through
`ReembedService.change_space` (target space + `reembed` job, or cancel). The webhook URL
is checked against the SSRF policy (scheme, then every resolved address) when it is
set; a new URL or `rotate_secret` generates a secret that is returned once.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from footnote.app.prompts_loader import available_templates
from footnote.app.reembed import ReembedService
from footnote.domain.deliveries import DeliveryRecord
from footnote.domain.documents import CollectionId, CollectionRecord, JobId, NewCollection
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import ConflictError, NotFoundError, ValidationError
from footnote.domain.jobs import JobKind, JobRecord, NewJob
from footnote.domain.netpolicy import check_addresses, parse_webhook_url
from footnote.domain.ports import (
    CollectionRepository,
    HostResolver,
    JobQueue,
    SpaceReader,
    TenantReader,
    WebhookDeliveryLog,
)
from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.tenancy import TenantId
from footnote.domain.webhooks import WebhookEvent, WebhookSettings

NAME_MAX_CHARS = 256
COLLECTION_EXISTS = "collection_exists"


def new_webhook_secret() -> str:
    """A random webhook signing secret (`whsec_` + 32 random bytes, URL-safe base64)."""
    return "whsec_" + secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class SettingsPatch:
    """Partial settings; each section maps field names to new values (None = unchanged).

    `retrieval["rrf_weights"]` may name one leg; the other keeps its value.
    """

    chunking: Mapping[str, Any] | None = None
    retrieval: Mapping[str, Any] | None = None
    answer: Mapping[str, Any] | None = None
    language: Language | None = None
    embedding_space: str | None = None


@dataclass(frozen=True, slots=True)
class WebhookPatch:
    """Webhook change: `url` None with `set_url` removes it; `events` None keeps them."""

    set_url: bool = False
    url: str | None = None
    events: tuple[WebhookEvent, ...] | None = None
    rotate_secret: bool = False


@dataclass(frozen=True, slots=True)
class ReembedProgress:
    """`{target_space, done, total, state}` shown on the collection while a re-embed runs."""

    target_space: str
    done: int
    total: int
    state: str


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """A collection as the API shows it."""

    record: CollectionRecord
    active_space: str
    reembed: ReembedProgress | None
    jobs: tuple[JobRecord, ...] | None = None


@dataclass(frozen=True, slots=True)
class CollectionChange:
    """Outcome of create/update: jobs started and a webhook secret shown once."""

    collection: CollectionInfo
    job_ids: tuple[JobId, ...] = ()
    webhook_secret: str | None = None


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Defaults and validation inputs from config (`collections.*`, `chunking.*`, …)."""

    defaults: CollectionSettings = field(default_factory=CollectionSettings)
    tokenizer_ids: frozenset[str] | None = None
    templates: frozenset[str] = field(default_factory=lambda: frozenset(available_templates()))
    allow_http_webhooks: bool = False
    allow_private_webhooks: bool = False


@dataclass(frozen=True, slots=True)
class CollectionDeps:
    """Ports used by the collections use case; `tenants` (admin pool) enables quotas."""

    collections: CollectionRepository
    jobs: JobQueue
    spaces: SpaceReader
    reembed: ReembedService
    resolver: HostResolver
    deliveries: WebhookDeliveryLog
    tenants: TenantReader | None = None
    new_secret: Callable[[], str] = new_webhook_secret


def _section[T](current: T, values: Mapping[str, Any] | None, name: str) -> T:
    if not values:
        return current
    try:
        return replace(current, **dict(values))  # type: ignore[type-var]
    except TypeError as exc:
        raise ValidationError(name, f"unknown field: {exc}") from exc


def merge_settings(current: CollectionSettings, patch: SettingsPatch) -> CollectionSettings:
    """Apply a patch to settings (no validation)."""
    retrieval_values = dict(patch.retrieval or {})
    if "rrf_weights" in retrieval_values:
        weights = retrieval_values["rrf_weights"] or {}
        retrieval_values["rrf_weights"] = {**current.retrieval.rrf_weights, **dict(weights)}
    return replace(
        current,
        chunking=_section(current.chunking, patch.chunking, "chunking"),
        retrieval=_section(current.retrieval, retrieval_values, "retrieval"),
        answer=_section(current.answer, patch.answer, "answer"),
        language=patch.language or current.language,
        embedding_space=patch.embedding_space or current.embedding_space,
    )


def _validate_name(name: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= NAME_MAX_CHARS:
        raise ValidationError("name", f"must be 1..{NAME_MAX_CHARS} characters")
    return name


class CollectionService:
    """Collections of a tenant."""

    def __init__(self, deps: CollectionDeps, config: CollectionConfig | None = None) -> None:
        self._d = deps
        self._cfg = config or CollectionConfig()

    def _validate(self, settings: CollectionSettings) -> None:
        settings.chunking.validate(self._cfg.tokenizer_ids)
        settings.retrieval.validate()
        settings.answer.validate(self._cfg.templates)

    async def _space(self, name: str) -> EmbeddingSpace:
        try:
            return await self._d.spaces.get(name)
        except NotFoundError as exc:
            raise ValidationError("embedding_space", f"unknown space {name!r}") from exc

    async def _check_url(self, url: str) -> None:
        target = parse_webhook_url(url, allow_http=self._cfg.allow_http_webhooks)
        addresses = await self._d.resolver.resolve(target.host, target.port)
        check_addresses(list(addresses), allow_private=self._cfg.allow_private_webhooks)

    async def _info(self, record: CollectionRecord, *, with_jobs: bool = False) -> CollectionInfo:
        active = await self._d.spaces.get_by_id(record.active_space_id)
        progress = None
        if record.target_space_id is not None:
            target = await self._d.spaces.get_by_id(record.target_space_id)
            state = record.reembed or {}
            progress = ReembedProgress(
                target.name,
                int(state.get("done", 0)),
                int(state.get("total", 0)),
                str(state.get("state", "running")),
            )
        jobs = None
        if with_jobs:
            jobs = tuple(await self._d.jobs.active_for_collection(record.tenant_id, record.id))
        return CollectionInfo(record, active.name, progress, jobs)

    async def create(
        self,
        tenant: TenantId,
        name: str,
        patch: SettingsPatch | None = None,
        webhook: WebhookPatch | None = None,
    ) -> CollectionChange:
        """Create a collection; 429 `quota_exceeded` beyond `max_collections`."""
        name = _validate_name(name)
        settings = merge_settings(self._cfg.defaults, patch or SettingsPatch())
        if webhook is not None and webhook.events is not None:
            settings = replace(settings, webhook=WebhookSettings(webhook.events))
        self._validate(settings)
        space = await self._space(settings.embedding_space)
        url = webhook.url if webhook is not None else None
        secret = None
        if url:
            await self._check_url(url)
            secret = self._d.new_secret()
        if self._d.tenants is not None:
            quotas = (await self._d.tenants.get_tenant(tenant)).quotas
            if quotas.max_collections:
                quotas.check("max_collections", await self._d.collections.count(tenant), 1)
        try:
            record = await self._d.collections.create(
                tenant, NewCollection(name, settings, space.id, url or None, secret)
            )
        except ConflictError as exc:
            msg = f"a collection named {name!r} exists"
            raise ConflictError(msg, code=COLLECTION_EXISTS) from exc
        return CollectionChange(await self._info(record), (), secret)

    async def get(self, tenant: TenantId, collection_id: CollectionId) -> CollectionInfo:
        """One collection with its jobs in progress."""
        record = await self._d.collections.get(tenant, collection_id)
        return await self._info(record, with_jobs=True)

    async def list_page(
        self, tenant: TenantId, after: CollectionId | None, limit: int
    ) -> list[CollectionInfo]:
        """Collections ordered by id (jobs are shown by `get` only)."""
        records = await self._d.collections.list_page(tenant, after, limit)
        names = {s.id: s.name for s in await self._d.spaces.list_spaces()}
        out = []
        for r in records:
            progress = None
            if r.target_space_id is not None:
                state = r.reembed or {}
                progress = ReembedProgress(
                    names.get(r.target_space_id, str(r.target_space_id)),
                    int(state.get("done", 0)),
                    int(state.get("total", 0)),
                    str(state.get("state", "running")),
                )
            out.append(
                CollectionInfo(r, names.get(r.active_space_id, str(r.active_space_id)), progress)
            )
        return out

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        patch: SettingsPatch,
        webhook: WebhookPatch | None = None,
    ) -> CollectionChange:
        """Apply a settings patch; job ids are returned for changes that start jobs."""
        current = await self._d.collections.get(tenant, collection_id)
        settings = merge_settings(current.settings, patch)
        if webhook is not None and webhook.events is not None:
            settings = replace(settings, webhook=WebhookSettings(webhook.events))
        self._validate(settings)
        if patch.embedding_space is not None:
            await self._space(patch.embedding_space)
        hook, secret = await self._webhook_change(current, webhook)
        jobs: list[NewJob] = []
        args = {"collection_id": str(collection_id)}
        if settings.chunking.settings_hash() != current.settings.chunking.settings_hash():
            jobs.append(NewJob(JobKind.REINDEX.value, args))
        if settings.language is not current.settings.language:
            jobs.append(
                NewJob(JobKind.FTS_RECONFIG.value, {**args, "language": settings.language.value})
            )
        record, ids = await self._d.collections.update(
            tenant, collection_id, settings, webhook=hook, jobs=jobs
        )
        job_ids = list(ids)
        if patch.embedding_space is not None:
            reembed = await self._d.reembed.change_space(
                tenant, collection_id, patch.embedding_space
            )
            if reembed is not None:
                job_ids.append(reembed)
            record = await self._d.collections.get(tenant, collection_id)
        return CollectionChange(await self._info(record), tuple(job_ids), secret)

    async def _webhook_change(
        self, current: CollectionRecord, webhook: WebhookPatch | None
    ) -> tuple[tuple[str | None, str | None] | None, str | None]:
        """New (url, secret) column values, or None to keep them, and a secret to show."""
        if webhook is None or (not webhook.set_url and not webhook.rotate_secret):
            return None, None
        url = webhook.url if webhook.set_url else current.webhook_url
        if not url:
            if webhook.rotate_secret:
                raise ValidationError("webhook.rotate_secret", "the collection has no webhook url")
            return (None, None), None
        if url != current.webhook_url:
            await self._check_url(url)
        elif not webhook.rotate_secret and current.webhook_secret:
            return None, None
        secret = self._d.new_secret()
        return (url, secret), secret

    async def delete(self, tenant: TenantId, collection_id: CollectionId) -> JobId:
        """Hide the collection now; its rows go with a `collection_delete` job."""
        job = NewJob(JobKind.COLLECTION_DELETE.value, {"collection_id": str(collection_id)})
        ids = await self._d.collections.mark_deleting(tenant, collection_id, (job,))
        return ids[0]

    async def deliveries(
        self, tenant: TenantId, collection_id: CollectionId, limit: int
    ) -> list[DeliveryRecord]:
        """Newest webhook deliveries of a collection."""
        await self._d.collections.get(tenant, collection_id)
        return await self._d.deliveries.list_for_collection(tenant, collection_id, limit)
