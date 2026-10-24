# SPDX-License-Identifier: Apache-2.0
"""Bearer authentication."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from footnote.api.deps import ApiDeps, get_deps
from footnote.domain.errors import UnauthenticatedError
from footnote.domain.tenancy import TenantId

MAX_TOKEN_CHARS = 512

_tenant_scheme = HTTPBearer(
    auto_error=False, scheme_name="ApiKey", description="Tenant API key (`fn_…`)."
)
_admin_scheme = HTTPBearer(
    auto_error=False, scheme_name="AdminKey", description="Admin key (`auth.admin_key`)."
)


@dataclass(frozen=True, slots=True)
class Principal:
    tenant: TenantId
    key_id: UUID


def _token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthenticatedError("missing bearer token")
    token = credentials.credentials.strip()
    if not token or len(token) > MAX_TOKEN_CHARS:
        raise UnauthenticatedError("invalid api key")
    return token


async def tenant_principal(
    request: Request,
    deps: Annotated[ApiDeps, Depends(get_deps)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_tenant_scheme)],
) -> Principal:
    """Resolves the API key of a request to its tenant."""
    tenant, key_id = await deps.keys.resolve(_token(credentials))
    request.state.tenant = str(tenant)
    structlog.contextvars.bind_contextvars(tenant=str(tenant))
    return Principal(tenant, key_id)


async def require_admin(
    deps: Annotated[ApiDeps, Depends(get_deps)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_admin_scheme)],
) -> None:
    """Accepts only the admin key."""
    token = _token(credentials)
    expected = deps.settings.admin_key
    if not expected:
        raise UnauthenticatedError("the admin API is disabled (no auth.admin_key)")
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise UnauthenticatedError("invalid admin key")


Tenant = Annotated[Principal, Depends(tenant_principal)]
Admin = Depends(require_admin)
