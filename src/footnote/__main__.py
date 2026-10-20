# SPDX-License-Identifier: Apache-2.0
"""Command line entry point (typer): `serve`, `worker`, `migrate`, `healthcheck`, `config`,
`models`, `version`, the admin groups `tenants`, `keys`, `spaces` and `eval`.

Exit codes: 0 success, 1 failure (not ready, unreachable dependency, failed check),
2 invalid configuration or usage. Heavy modules (the composition root,
ONNX Runtime, FastAPI) are imported inside the commands that need them, so
`footnote healthcheck` (the Docker HEALTHCHECK, no shell in the image) starts quickly.
"""

from __future__ import annotations

import asyncio
import signal
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml

from footnote.cli_common import fail, settings_from
from footnote.config import EXIT_CODE_INVALID, ConfigError, Settings

if TYPE_CHECKING:
    from footnote.adapters.models import Manifest, ModelNeed
    from footnote.bootstrap import ServeOptions

HEALTHCHECK_TIMEOUT = 4.0
_HTTP_OK = 200
_LOOPBACK = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}  # noqa: S104 - mapping, not a bind

app = typer.Typer(
    help="Footnote: RAG on Postgres with verifiable citations.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Validate or print the configuration.", no_args_is_help=True)
models_app = typer.Typer(help="Download or verify the pinned local models.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (default: $FOOTNOTE_CONFIG)."),
]


@app.callback()
def root(ctx: typer.Context, config: ConfigOption = None) -> None:
    """Footnote command line."""
    ctx.obj = {"config": config}


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(pkg_version("rayld-footnote"))


# --- serve / worker ---------------------------------------------------------------------


def run_until_signalled(settings: Settings, options: ServeOptions) -> None:
    """Run the process until SIGTERM/SIGINT, then shut down gracefully; exit 1 on startup
    failures (database unreachable, port in use, ...)."""
    from footnote.bootstrap import run_process  # noqa: PLC0415 - heavy import

    async def main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await run_process(settings, options, stop)

    try:
        asyncio.run(main())
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except Exception as exc:
        raise fail(f"{type(exc).__name__}: {exc}") from exc


@app.command()
def serve(
    ctx: typer.Context,
    with_worker: Annotated[
        bool, typer.Option("--with-worker", help="Also run the job worker in this process.")
    ] = False,
    listen: Annotated[
        str | None, typer.Option(help="Data plane address (overrides server.listen).")
    ] = None,
    admin_listen: Annotated[
        str | None, typer.Option(help="Admin listener address (overrides admin.listen).")
    ] = None,
) -> None:
    """Serve the data plane and the admin listener (optionally with the worker)."""
    from footnote.bootstrap import ServeOptions  # noqa: PLC0415

    overrides = _listen_overrides(listen, admin_listen)
    run_until_signalled(
        settings_from(ctx, overrides), ServeOptions(data_plane=True, worker=with_worker)
    )


@app.command()
def worker(
    ctx: typer.Context,
    admin_listen: Annotated[
        str | None, typer.Option(help="Admin listener address (overrides admin.listen).")
    ] = None,
) -> None:
    """Run the job worker (with the admin listener for health and metrics)."""
    from footnote.bootstrap import ServeOptions  # noqa: PLC0415

    overrides = _listen_overrides(None, admin_listen)
    run_until_signalled(settings_from(ctx, overrides), ServeOptions(data_plane=False, worker=True))


def _listen_overrides(listen: str | None, admin_listen: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if listen is not None:
        overrides["FOOTNOTE_SERVER_LISTEN"] = listen
    if admin_listen is not None:
        overrides["FOOTNOTE_ADMIN_LISTEN"] = admin_listen
    return overrides


# --- migrate / healthcheck --------------------------------------------------------------


@app.command()
def migrate(
    ctx: typer.Context,
    revision: Annotated[str, typer.Option(help="Target revision.")] = "head",
) -> None:
    """Apply schema migrations as footnote_admin (FOOTNOTE_ADMIN_DATABASE_URL)."""
    from footnote.adapters.postgres.migrate import upgrade  # noqa: PLC0415
    from footnote.bootstrap import required_dsn  # noqa: PLC0415

    settings = settings_from(ctx)
    try:
        dsn = required_dsn(settings, admin=True)
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    try:
        upgrade(dsn, revision=revision)
    except Exception as exc:
        raise fail(f"migration failed: {type(exc).__name__}: {exc}") from exc
    typer.echo(f"migrated to {revision}")


def readyz_url(listen: str) -> str:
    """The admin listener's `/readyz` URL reachable from this host (wildcards → loopback)."""
    host, _, port = listen.rpartition(":")
    host = host.strip("[]")
    host = _LOOPBACK.get(host, host)
    if port == "0":
        msg = "admin.listen uses an ephemeral port (0); nothing to probe"
        raise ValueError(msg)
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/readyz"


@app.command()
def healthcheck(
    ctx: typer.Context,
    timeout: Annotated[float, typer.Option(help="Seconds before giving up.")] = HEALTHCHECK_TIMEOUT,
) -> None:
    """Exit 0 when the admin listener's /readyz returns 200 (Docker HEALTHCHECK)."""
    import httpx  # noqa: PLC0415

    settings = settings_from(ctx)
    try:
        url = readyz_url(settings.admin.listen)
    except ValueError as exc:
        raise fail(str(exc)) from exc
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        raise fail(f"{url}: {type(exc).__name__}") from exc
    if response.status_code != _HTTP_OK:
        raise fail(f"not ready ({response.status_code}): {response.text[:2000]}")
    typer.echo("ready")


# --- config -----------------------------------------------------------------------------


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """Validate the configuration and its model-manifest references (exit 2 on errors)."""
    from footnote.bootstrap import load_model_manifest, manifest_problems  # noqa: PLC0415

    settings = settings_from(ctx)
    try:
        problems = manifest_problems(settings, load_model_manifest(settings))
    except ConfigError as exc:
        problems = exc.errors
    if problems:
        raise fail(str(ConfigError(problems)), EXIT_CODE_INVALID)
    typer.echo("configuration is valid")


@config_app.command("print")
def config_print(
    ctx: typer.Context,
    redact: Annotated[
        bool, typer.Option("--redact/--no-redact", help="Show secrets as *** (default).")
    ] = True,
) -> None:
    """Print the effective configuration as YAML."""
    settings = settings_from(ctx)
    data = settings.redacted() if redact else _revealed(settings)
    typer.echo(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), nl=False)


def _revealed(settings: Settings) -> dict[str, Any]:
    dump: dict[str, Any] = settings.model_dump(mode="json")
    from footnote.config import SECRET_PATHS  # noqa: PLC0415

    for path in SECRET_PATHS:
        node: Any = settings
        for part in path:
            node = getattr(node, part)
        target = dump
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = node.get_secret_value()
    return dump


# --- models -----------------------------------------------------------------------------

AllOption = Annotated[
    bool, typer.Option("--all", help="Every manifest model and variant, not only the configured.")
]


def _needs(settings: Settings, every: bool) -> tuple[Manifest, list[ModelNeed]]:
    from footnote.adapters.models import ModelNeed  # noqa: PLC0415
    from footnote.bootstrap import load_model_manifest, required_models  # noqa: PLC0415

    try:
        manifest = load_model_manifest(settings)
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    if every:
        return manifest, [ModelNeed(m, (*m.files, *m.variants)) for m in manifest.models.values()]
    return manifest, required_models(settings, manifest)


@models_app.command("pull")
def models_pull(ctx: typer.Context, every: AllOption = False) -> None:
    """Download missing or corrupt model files into models.dir (SHA-256 checked)."""
    from footnote.adapters.models import pull_files  # noqa: PLC0415
    from footnote.domain.errors import DomainError  # noqa: PLC0415

    settings = settings_from(ctx)
    _, needs = _needs(settings, every)
    for need in needs:
        try:
            paths = pull_files(settings.models.dir, need.entry, need.files())
        except DomainError as exc:
            raise fail(exc.message) from exc
        for path in paths:
            typer.echo(f"ok\t{need.entry.id}\t{path}")


@models_app.command("verify")
def models_verify(ctx: typer.Context, every: AllOption = False) -> None:
    """Check size and SHA-256 of the model files; exit 1 when any is missing or wrong."""
    from footnote.adapters.models import FileStatus, check_file, local_path  # noqa: PLC0415

    settings = settings_from(ctx)
    _, needs = _needs(settings, every)
    bad = 0
    for need in needs:
        for name, file in need.files().items():
            path = local_path(settings.models.dir, need.entry, file)
            status = check_file(path, file)
            bad += status is not FileStatus.OK
            typer.echo(f"{status.value}\t{need.entry.id}\t{name}\t{path}")
    if bad:
        raise fail(f"{bad} model file(s) missing or mismatched; run `footnote models pull`")


def _register_admin_commands() -> None:
    from footnote.cli_admin import keys_app, spaces_app, tenants_app  # noqa: PLC0415

    app.add_typer(tenants_app, name="tenants")
    app.add_typer(keys_app, name="keys")
    app.add_typer(spaces_app, name="spaces")

    from footnote.cli_eval import eval_app  # noqa: PLC0415

    app.add_typer(eval_app, name="eval")


_register_admin_commands()


def main() -> None:
    """Run the CLI."""
    app(prog_name="footnote")


if __name__ == "__main__":
    main()
