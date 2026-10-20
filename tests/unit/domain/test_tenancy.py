# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tenants and quotas."""

from __future__ import annotations

from uuid import UUID

import pytest

from footnote.domain.errors import QuotaExceededError, ValidationError
from footnote.domain.tenancy import Quotas, Tenant, TenantId, TenantStatus


def test_quotas_default_unlimited() -> None:
    quotas = Quotas()
    quotas.validate()
    quotas.check("max_documents", 10**9, 10**9)


@pytest.mark.parametrize(
    "name", ["max_collections", "max_documents", "max_chunks", "max_upload_bytes_per_day"]
)
def test_quotas_validate_rejects_negative(name: str) -> None:
    with pytest.raises(ValidationError) as info:
        Quotas(**{name: -1}).validate()
    assert info.value.field == f"quotas.{name}"


def test_quotas_check_at_limit_ok_and_over_limit_raises() -> None:
    quotas = Quotas(max_documents=5)
    quotas.check("max_documents", 3, 2)
    with pytest.raises(QuotaExceededError) as info:
        quotas.check("max_documents", 3, 3)
    assert info.value.code == "quota_exceeded"
    assert "max_documents" in info.value.message


def test_quotas_check_zero_limit_is_unlimited() -> None:
    Quotas(max_chunks=0).check("max_chunks", 100, 100)


def test_tenant_defaults_to_active() -> None:
    tenant = Tenant(TenantId(UUID(int=1)), "acme", Quotas())
    assert tenant.status is TenantStatus.ACTIVE
    assert TenantStatus.DELETING.value == "deleting"
