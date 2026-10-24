# SPDX-License-Identifier: Apache-2.0
"""Schemas of the collection routes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from footnote.api.schemas.common import Request
from footnote.app.collections import CollectionChange, CollectionInfo, SettingsPatch, WebhookPatch
from footnote.domain.deliveries import DeliveryRecord
from footnote.domain.jobs import JobRecord
from footnote.domain.settings import (
    ChunkingStrategyName,
    Language,
    LexicalMatch,
    LexicalRank,
)
from footnote.domain.webhooks import WebhookEvent


class ChunkingIn(Request):
    """Chunking settings; omitted fields keep their value."""

    strategy: ChunkingStrategyName | None = None
    max_tokens: int | None = Field(default=None, description="32-2048")
    overlap_tokens: int | None = Field(default=None, description="0 to max_tokens/2")
    min_tokens: int | None = Field(default=None, description="0 to max_tokens")
    tokenizer: str | None = Field(default=None, description="Manifest id with a tokenizer file")


class RrfWeightsIn(Request):
    """Per-leg RRF weights, each 0-10, not both 0."""

    lexical: float | None = None
    vector: float | None = None


class RetrievalIn(Request):
    """Retrieval settings; omitted fields keep their value."""

    lexical_match: LexicalMatch | None = None
    lexical_rank: LexicalRank | None = None
    k_lexical: int | None = Field(default=None, description="1-1000")
    k_vector: int | None = Field(default=None, description="1-1000")
    rrf_k: int | None = Field(default=None, description="1-1000")
    rrf_weights: RrfWeightsIn | None = None
    rerank: bool | None = None
    rerank_n: int | None = Field(default=None, description="1-200 and >= k")
    k: int | None = Field(default=None, description="1-100")


class AnswerSettingsIn(Request):
    """Answer settings; a threshold set to null falls back to config/calibration."""

    evidence_threshold_rerank: float | None = Field(default=None, description="0-1 or null")
    evidence_threshold_vector: float | None = Field(default=None, description="0-1 or null")
    prompt_template: str | None = None
    unsupported_as_abstain: bool | None = None


class WebhookIn(Request):
    """Webhook subscription; `url: null` removes it."""

    url: str | None = Field(default=None, max_length=2048)
    events: list[WebhookEvent] | None = None
    rotate_secret: bool = False


class CollectionSettingsIn(Request):
    chunking: ChunkingIn | None = None
    retrieval: RetrievalIn | None = None
    answer: AnswerSettingsIn | None = None
    language: Language | None = None
    embedding_space: str | None = Field(default=None, description="A registered space name")
    webhook: WebhookIn | None = None

    def settings_patch(self) -> SettingsPatch:

        def section(model: BaseModel | None) -> dict[str, Any] | None:
            return None if model is None else model.model_dump(exclude_unset=True)

        return SettingsPatch(
            chunking=section(self.chunking),
            retrieval=section(self.retrieval),
            answer=section(self.answer),
            language=self.language,
            embedding_space=self.embedding_space,
        )

    def webhook_patch(self) -> WebhookPatch | None:
        if self.webhook is None:
            return None
        sent = self.webhook.model_fields_set
        return WebhookPatch(
            set_url="url" in sent,
            url=self.webhook.url,
            events=None if self.webhook.events is None else tuple(self.webhook.events),
            rotate_secret=self.webhook.rotate_secret,
        )


class CollectionCreate(CollectionSettingsIn):
    """`POST /v1/collections`."""

    name: str = Field(min_length=1, max_length=256)


class CollectionPatch(CollectionSettingsIn):
    """`PATCH /v1/collections/{id}`: chunking → reindex job, language → fts_reconfig job,
    embedding_space → reembed job (202); other changes 200."""


class ChunkingOut(BaseModel):
    """Effective chunking settings."""

    strategy: ChunkingStrategyName
    max_tokens: int
    overlap_tokens: int
    min_tokens: int
    tokenizer: str


class RetrievalOut(BaseModel):
    """Effective retrieval settings."""

    lexical_match: LexicalMatch
    lexical_rank: LexicalRank
    k_lexical: int
    k_vector: int
    rrf_k: int
    rrf_weights: dict[str, float]
    rerank: bool
    rerank_n: int
    k: int


class AnswerSettingsOut(BaseModel):
    """Effective answer settings."""

    evidence_threshold_rerank: float | None
    evidence_threshold_vector: float | None
    prompt_template: str
    unsupported_as_abstain: bool


class WebhookOut(BaseModel):
    """Webhook subscription (the secret is never shown again)."""

    url: str | None
    events: list[WebhookEvent]
    has_secret: bool


class ReembedOut(BaseModel):
    """Re-embed progress."""

    target_space: str
    done: int
    total: int
    state: str


class JobSummary(BaseModel):
    """A queued or running job of the collection."""

    id: UUID
    kind: str
    status: str
    progress: dict[str, Any] | None


class CollectionOut(BaseModel):
    """A collection."""

    id: UUID
    name: str
    status: str
    chunk_count: int
    active_space: str
    embedding_space: str = Field(description="Target space while re-embedding, else active")
    language: Language
    reembed: ReembedOut | None
    chunking: ChunkingOut
    retrieval: RetrievalOut
    answer: AnswerSettingsOut
    webhook: WebhookOut
    jobs: list[JobSummary] | None = Field(
        default=None, description="Jobs in progress (single-collection reads only)"
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, info: CollectionInfo) -> CollectionOut:
        r = info.record
        s = r.settings
        return cls(
            id=r.id,
            name=r.name,
            status=r.status,
            chunk_count=r.chunk_count,
            active_space=info.active_space,
            embedding_space=info.reembed.target_space if info.reembed else info.active_space,
            language=s.language,
            reembed=None if info.reembed is None else ReembedOut(**asdict(info.reembed)),
            chunking=ChunkingOut(**asdict(s.chunking)),
            retrieval=RetrievalOut(**asdict(s.retrieval)),
            answer=AnswerSettingsOut(**asdict(s.answer)),
            webhook=WebhookOut(
                url=r.webhook_url,
                events=list(s.webhook.events),
                has_secret=bool(r.webhook_secret),
            ),
            jobs=None if info.jobs is None else [job_summary(j) for j in info.jobs],
            created_at=r.created_at,
            updated_at=r.updated_at,
        )


def job_summary(job: JobRecord) -> JobSummary:
    return JobSummary(id=job.id, kind=job.kind, status=job.status, progress=job.progress)


class CollectionChangeOut(CollectionOut):
    """Create/patch result: started jobs and a webhook secret shown once."""

    job_id: UUID | None = Field(default=None, description="First job started (202)")
    job_ids: list[UUID] = Field(default_factory=list)
    webhook_secret: str | None = Field(default=None, description="Returned once")

    @classmethod
    def of_change(cls, change: CollectionChange) -> CollectionChangeOut:
        base = CollectionOut.of(change.collection).model_dump()
        return cls(
            **base,
            job_id=change.job_ids[0] if change.job_ids else None,
            job_ids=list(change.job_ids),
            webhook_secret=change.webhook_secret,
        )


class DeliveryOut(BaseModel):
    """One webhook delivery."""

    id: UUID
    event: str
    url: str
    status: str
    attempts: int
    response_code: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None

    @classmethod
    def of(cls, d: DeliveryRecord) -> DeliveryOut:
        return cls(
            id=d.id,
            event=d.event,
            url=d.url,
            status=d.status,
            attempts=d.attempts,
            response_code=d.response_code,
            error=d.error,
            created_at=d.created_at,
            updated_at=d.updated_at,
            delivered_at=d.delivered_at,
        )
