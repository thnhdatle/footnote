# SPDX-License-Identifier: Apache-2.0
"""Bearer authentication.

Tenant routes take `Authorization: Bearer <api key>`; the key is resolved through the
`KeyResolver` port (`footnote_resolve_key`, which ignores revoked and expired keys) and
every repository call of the request is made with the resolved tenant id, so each
transaction runs under that tenant's RLS setting. Unknown, revoked or expired keys and
malformed headers are 401 `unauthenticated`; another tenant's resource ids are 404,
never 403, because RLS makes them absent.

Admin routes compare the bearer token with the configured admin key in constant time;
tenant keys (and every token while no admin key is configured) are 401.
"""

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
    """The authenticated tenant of a request."""

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
    """Resolve the request's API key to its tenant (401 `unauthenticated` otherwise)."""
    tenant, key_id = await deps.keys.resolve(_token(credentials))
    request.state.tenant = str(tenant)
    structlog.contextvars.bind_contextvars(tenant=str(tenant))
    return Principal(tenant, key_id)


async def require_admin(
    deps: Annotated[ApiDeps, Depends(get_deps)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_admin_scheme)],
) -> None:
    """Accept only the configured admin key (constant-time comparison)."""
    token = _token(credentials)
    expected = deps.settings.admin_key
    if not expected:
        raise UnauthenticatedError("the admin API is disabled (no auth.admin_key)")
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise UnauthenticatedError("invalid admin key")


Tenant = Annotated[Principal, Depends(tenant_principal)]
Admin = Depends(require_admin)
