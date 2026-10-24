# SPDX-License-Identifier: Apache-2.0
"""Admin commands for tenants, keys and embedding spaces."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import IO, TYPE_CHECKING, Annotated, Any
from uuid import UUID

import orjson
import typer
from pydantic import BaseModel

from footnote.cli_common import EXIT_FAILURE, fail, settings_from
from footnote.config import EXIT_CODE_INVALID, ConfigError
from footnote.domain.errors import DomainError
from footnote.domain.tenancy import TenantId

if TYPE_CHECKING:
    from footnote.app.admin import AdminService
    from footnote.domain.embedding import EmbeddingSpace
    from footnote.domain.tenancy import ApiKeyRecord

tenants_app = typer.Typer(help="Manage tenants (admin pool).", no_args_is_help=True)
keys_app = typer.Typer(help="Manage tenant API keys (admin pool).", no_args_is_help=True)
spaces_app = typer.Typer(help="Manage embedding spaces (admin pool).", no_args_is_help=True)

TenantArg = Annotated[UUID, typer.Argument(help="Tenant id.")]
_PAGE = 500
KeyArg = Annotated[UUID, typer.Argument(help="Key id.")]


def _emit(value: BaseModel | dict[str, Any]) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    typer.echo(orjson.dumps(data).decode())


def run_admin[T](ctx: typer.Context, action: Callable[[AdminService], Awaitable[T]]) -> T:
    from footnote.bootstrap import open_admin  # noqa: PLC0415 - heavy import

    settings = settings_from(ctx)

    async def main() -> T:
        async with open_admin(settings) as admin:
            return await action(admin)

    try:
        return asyncio.run(main())
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except DomainError as exc:
        raise fail(f"{exc.code}: {exc.message}", EXIT_FAILURE) from exc
    except Exception as exc:
        raise fail(f"{type(exc).__name__}: {exc}") from exc


@tenants_app.command("create")
def tenants_create(
    ctx: typer.Context, name: Annotated[str, typer.Argument(help="Tenant name.")]
) -> None:
    """Create a tenant with `tenancy.default_quotas`."""
    from footnote.api.schemas.admin import TenantOut  # noqa: PLC0415

    tenant = run_admin(ctx, lambda admin: admin.create_tenant(name))
    _emit(TenantOut.of(tenant))


@tenants_app.command("list")
def tenants_list(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option(min=1, max=500)] = 50,
    after: Annotated[UUID | None, typer.Option(help="List tenants after this id.")] = None,
) -> None:
    """List tenants ordered by id (one JSON object per line)."""
    from footnote.api.schemas.admin import TenantOut  # noqa: PLC0415

    start = TenantId(after) if after else None
    for tenant in run_admin(ctx, lambda admin: admin.list_tenants(start, limit)):
        _emit(TenantOut.of(tenant))


@tenants_app.command("delete")
def tenants_delete(ctx: typer.Context, tenant_id: TenantArg) -> None:
    """Revoke the tenant's keys now; a `tenant_delete` job removes its data."""
    job_id = run_admin(ctx, lambda admin: admin.delete_tenant(TenantId(tenant_id)))
    _emit({"tenant_id": str(tenant_id), "job_id": str(job_id)})


@tenants_app.command("export")
def tenants_export(
    ctx: typer.Context,
    tenant_id: TenantArg,
    include_vectors: Annotated[bool, typer.Option("--include-vectors")] = False,
    include_originals: Annotated[bool, typer.Option("--include-originals")] = False,
    output: Annotated[
        typer.FileBinaryWrite | None, typer.Option("--output", "-o", help="File (default stdout).")
    ] = None,
) -> None:
    """Write the tenant's NDJSON export (same records as the admin API)."""
    out: IO[bytes] = output if output is not None else sys.stdout.buffer

    async def action(admin: AdminService) -> int:
        records = await admin.export(
            TenantId(tenant_id),
            include_vectors=include_vectors,
            include_originals=include_originals,
        )
        count = 0
        async for record in records:
            out.write(orjson.dumps({"type": record.type, "data": record.data}) + b"\n")
            count += 1
        out.flush()
        return count

    run_admin(ctx, action)


@keys_app.command("create")
def keys_create(
    ctx: typer.Context,
    tenant_id: TenantArg,
    name: Annotated[str, typer.Option(help="Key name.")] = "default",
    expires_at: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%dT%H:%M:%S%z"], help="Expiry with time zone."),
    ] = None,
) -> None:
    """Issue a key; the plaintext key is printed once."""
    from footnote.api.schemas.admin import IssuedKeyOut  # noqa: PLC0415

    issued = run_admin(ctx, lambda admin: admin.create_key(TenantId(tenant_id), name, expires_at))
    _emit(IssuedKeyOut.of_issued(issued))


@keys_app.command("list")
def keys_list(ctx: typer.Context, tenant_id: TenantArg) -> None:
    """List the tenant's keys (no secrets)."""
    from footnote.api.schemas.admin import KeyOut  # noqa: PLC0415

    async def pages(admin: AdminService) -> list[ApiKeyRecord]:
        found: list[ApiKeyRecord] = []
        while batch := await admin.list_keys(
            TenantId(tenant_id), found[-1].id if found else None, _PAGE
        ):
            found += batch
        return found

    for key in run_admin(ctx, pages):
        _emit(KeyOut.of(key))


@keys_app.command("revoke")
def keys_revoke(ctx: typer.Context, tenant_id: TenantArg, key_id: KeyArg) -> None:
    """Revoke a key now."""
    run_admin(ctx, lambda admin: admin.revoke_key(TenantId(tenant_id), key_id))
    _emit({"key_id": str(key_id), "revoked": True})


@keys_app.command("rotate")
def keys_rotate(ctx: typer.Context, tenant_id: TenantArg, key_id: KeyArg) -> None:
    """Issue a replacement; the old key stays valid for `auth.key_rotation_grace`."""
    from footnote.api.schemas.admin import IssuedKeyOut  # noqa: PLC0415

    issued = run_admin(ctx, lambda admin: admin.rotate_key(TenantId(tenant_id), key_id))
    _emit(IssuedKeyOut.of_issued(issued))


@spaces_app.command("register")
def spaces_register(  # noqa: PLR0913, PLR0917 - one option per space field
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Space name.")],
    provider: Annotated[str, typer.Option(help="onnx_local or openai_compat.")],
    model: Annotated[str, typer.Option(help="Manifest id or provider model.")],
    dimensions: Annotated[int | None, typer.Option(min=1)] = None,
    send_dimensions: Annotated[bool, typer.Option("--send-dimensions")] = False,
    calibrated_threshold: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
) -> None:
    """Register an embedding space (idempotent by name)."""
    from footnote.api.schemas.admin import SpaceOut  # noqa: PLC0415
    from footnote.domain.embedding import SpaceSpec  # noqa: PLC0415

    spec = SpaceSpec(
        name=name,
        provider=provider,
        model=model,
        dimensions=dimensions or 0,
        send_dimensions=send_dimensions,
        calibrated_threshold=calibrated_threshold,
    )
    space = run_admin(ctx, lambda admin: admin.register_space(spec))
    _emit(SpaceOut.of(space))


@spaces_app.command("list")
def spaces_list(ctx: typer.Context) -> None:
    """List registered spaces."""
    from footnote.api.schemas.admin import SpaceOut  # noqa: PLC0415

    async def pages(admin: AdminService) -> list[EmbeddingSpace]:
        found: list[EmbeddingSpace] = []
        while batch := await admin.list_spaces(found[-1].id if found else None, _PAGE):
            found += batch
        return found

    for space in run_admin(ctx, pages):
        _emit(SpaceOut.of(space))
