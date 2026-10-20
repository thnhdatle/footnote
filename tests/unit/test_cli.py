# SPDX-License-Identifier: Apache-2.0
"""CLI tests: exit codes, configuration handling and the commands' wiring (no database)."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from footnote import __main__ as cli
from footnote.__main__ import app, readyz_url
from footnote.adapters.models import ModelEntry, ModelFile
from footnote.bootstrap import ServeOptions
from footnote.config import Settings

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "models" / "manifest.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No FOOTNOTE_* variable of the developer's shell leaks into a test."""
    for name in list(os.environ):
        if name.startswith("FOOTNOTE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FOOTNOTE_MODELS_MANIFEST", str(MANIFEST))


def test_cli_version_prints_semver() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip().count(".") == 2


# --- config -----------------------------------------------------------------------------


def test_config_validate_accepts_the_defaults() -> None:
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_config_validate_reports_a_bad_yaml_value(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("server:\n  listen: nonsense\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(path), "config", "validate"])
    assert result.exit_code == 2
    assert "invalid configuration" in result.stderr
    assert "server.listen" in result.stderr


def test_config_validate_reports_a_tokenizer_missing_from_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOOTNOTE_CHUNKING_TOKENIZER", "not-in-manifest")
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 2
    assert "chunking.tokenizer" in result.stderr


def test_config_validate_reports_an_unreadable_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOOTNOTE_MODELS_MANIFEST", str(tmp_path / "absent.yaml"))
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 2
    assert "models.manifest" in result.stderr


def test_config_print_redacts_secrets_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTNOTE_AUTH_ADMIN_KEY", "super-secret-key")
    result = runner.invoke(app, ["config", "print"])
    assert result.exit_code == 0
    assert "super-secret-key" not in result.stdout
    assert "***" in result.stdout


def test_config_print_no_redact_shows_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTNOTE_AUTH_ADMIN_KEY", "super-secret-key")
    result = runner.invoke(app, ["config", "print", "--no-redact"])
    assert result.exit_code == 0
    assert "super-secret-key" in result.stdout


def test_config_option_is_honoured(tmp_path: Path) -> None:
    path = tmp_path / "footnote.yaml"
    path.write_text("server:\n  listen: 127.0.0.1:4242\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(path), "config", "print"])
    assert result.exit_code == 0
    assert "127.0.0.1:4242" in result.stdout


def test_config_environment_variable_selects_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "footnote.yaml"
    path.write_text("admin:\n  listen: 127.0.0.1:4343\n", encoding="utf-8")
    monkeypatch.setenv("FOOTNOTE_CONFIG", str(path))
    assert "127.0.0.1:4343" in runner.invoke(app, ["config", "print"]).stdout


# --- healthcheck ------------------------------------------------------------------------


class _Stub(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


@pytest.fixture
def stub_server() -> Iterator[Any]:
    """Serve `/readyz` on 127.0.0.1:<free port>; `server.status` picks the answer."""

    class Handler(_Stub):
        pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_healthcheck_ok(monkeypatch: pytest.MonkeyPatch, stub_server: Any) -> None:
    server, _ = stub_server
    monkeypatch.setenv("FOOTNOTE_ADMIN_LISTEN", f"127.0.0.1:{server.server_address[1]}")
    result = runner.invoke(app, ["healthcheck"])
    assert result.exit_code == 0
    assert "ready" in result.stdout


def test_healthcheck_not_ready(monkeypatch: pytest.MonkeyPatch, stub_server: Any) -> None:
    server, handler = stub_server
    handler.status = 503
    monkeypatch.setenv("FOOTNOTE_ADMIN_LISTEN", f"127.0.0.1:{server.server_address[1]}")
    result = runner.invoke(app, ["healthcheck"])
    assert result.exit_code == 1
    assert "not ready (503)" in result.stderr


def test_healthcheck_connection_refused(monkeypatch: pytest.MonkeyPatch, stub_server: Any) -> None:
    server, _ = stub_server
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    monkeypatch.setenv("FOOTNOTE_ADMIN_LISTEN", f"127.0.0.1:{port}")
    result = runner.invoke(app, ["healthcheck", "--timeout", "1"])
    assert result.exit_code == 1
    assert f"127.0.0.1:{port}" in result.stderr


def test_healthcheck_cannot_probe_port_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTNOTE_ADMIN_LISTEN", "127.0.0.1:0")
    result = runner.invoke(app, ["healthcheck"])
    assert result.exit_code == 1
    assert "ephemeral" in result.stderr


@pytest.mark.parametrize(
    ("listen", "url"),
    [
        ("0.0.0.0:9094", "http://127.0.0.1:9094/readyz"),
        (":9094", "http://127.0.0.1:9094/readyz"),
        ("[::]:9094", "http://[::1]:9094/readyz"),
        ("10.1.2.3:80", "http://10.1.2.3:80/readyz"),
        ("[::1]:7", "http://[::1]:7/readyz"),
        ("example.org:1", "http://example.org:1/readyz"),
    ],
    ids=["any-v4", "empty", "any-v6", "host", "v6-loopback", "name"],
)
def test_readyz_url(listen: str, url: str) -> None:
    assert readyz_url(listen) == url


# --- models -----------------------------------------------------------------------------

TOKENIZER_BYTES = b'{"tokenizer": true}'


def _model_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sha: str | None = None) -> Path:
    """A one-model manifest, a models dir and a config needing only the tokenizer."""
    digest = sha or hashlib.sha256(TOKENIZER_BYTES).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "models:\n"
        "  - id: tiny\n    kind: embedder\n    repo: o/tiny\n    revision: r1\n"
        "    files:\n"
        f"      tokenizer: {{path: tokenizer.json, sha256: {digest}, "
        f"bytes: {len(TOKENIZER_BYTES)}}}\n"
        "    dimensions: 4\n",
        encoding="utf-8",
    )
    models = tmp_path / "models"
    monkeypatch.setenv("FOOTNOTE_MODELS_MANIFEST", str(manifest))
    monkeypatch.setenv("FOOTNOTE_MODELS_DIR", str(models))
    monkeypatch.setenv("FOOTNOTE_CHUNKING_TOKENIZER", "tiny")
    monkeypatch.setenv("FOOTNOTE_RERANK_PROVIDER", "noop")
    monkeypatch.setenv(
        "FOOTNOTE_EMBEDDING_SPACES",
        json.dumps([{"name": "r", "provider": "openai_compat", "model": "m", "dimensions": 4}]),
    )
    monkeypatch.setenv("FOOTNOTE_EMBEDDING_DEFAULT_SPACE", "r")
    monkeypatch.setenv("FOOTNOTE_EMBEDDING_BASE_URL", "http://localhost:1/v1")
    return models / "tiny" / "tokenizer.json"


def test_models_verify_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _model_env(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(TOKENIZER_BYTES)
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0
    assert result.stdout.startswith("ok\ttiny\ttokenizer\t")


def test_models_verify_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _model_env(monkeypatch, tmp_path)
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 1
    assert "missing\ttiny" in result.stdout
    assert "models pull" in result.stderr


def test_models_verify_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _model_env(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * len(TOKENIZER_BYTES))
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 1
    assert "mismatch\ttiny" in result.stdout


def test_models_verify_all_covers_every_manifest_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOOTNOTE_MODELS_DIR", str(tmp_path / "empty"))
    result = runner.invoke(app, ["models", "verify", "--all"])
    assert result.exit_code == 1
    # bge: model + tokenizer; reranker: tokenizer + fp32 + int8 for both architectures
    assert result.stdout.count("missing\t") == 6


def test_models_verify_bad_manifest_path_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOOTNOTE_MODELS_MANIFEST", str(tmp_path / "absent.yaml"))
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 2


def test_models_pull_downloads_the_needed_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _model_env(monkeypatch, tmp_path)
    pulled: list[tuple[Path, str, list[str]]] = []

    def fake_pull(models_dir: Path, entry: ModelEntry, files: dict[str, ModelFile]) -> list[Path]:
        pulled.append((models_dir, entry.id, sorted(files)))
        return [models_dir / entry.id / "tokenizer.json"]

    monkeypatch.setattr("footnote.adapters.models.pull_files", fake_pull)
    result = runner.invoke(app, ["models", "pull"])
    assert result.exit_code == 0
    assert pulled == [(tmp_path / "models", "tiny", ["tokenizer"])]
    assert result.stdout.startswith("ok\ttiny\t")


def test_models_pull_failure_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from footnote.domain.errors import UpstreamError  # noqa: PLC0415

    _model_env(monkeypatch, tmp_path)

    def fake_pull(*args: Any, **kwargs: Any) -> list[Path]:
        raise UpstreamError("download of tokenizer.json failed")

    monkeypatch.setattr("footnote.adapters.models.pull_files", fake_pull)
    result = runner.invoke(app, ["models", "pull"])
    assert result.exit_code == 1
    assert "download of tokenizer.json failed" in result.stderr


def test_models_pull_through_the_real_function_uses_the_download_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _model_env(monkeypatch, tmp_path)
    urls: list[str] = []

    def fake_download(client: Any, url: str, file: ModelFile, path: Path) -> None:
        urls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(TOKENIZER_BYTES)

    monkeypatch.setattr("footnote.adapters.models.pull._download", fake_download)
    result = runner.invoke(app, ["models", "pull"])
    assert result.exit_code == 0
    assert urls == ["https://huggingface.co/o/tiny/resolve/r1/tokenizer.json"]
    assert target.read_bytes() == TOKENIZER_BYTES


# --- migrate ----------------------------------------------------------------------------


def test_migrate_needs_the_admin_dsn() -> None:
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 2
    assert "FOOTNOTE_ADMIN_DATABASE_URL" in result.stderr


def test_migrate_calls_upgrade_with_dsn_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("FOOTNOTE_ADMIN_DATABASE_URL", "postgresql://admin@db/footnote")
    monkeypatch.setattr(
        "footnote.adapters.postgres.migrate.upgrade",
        lambda dsn, revision="head": calls.append((dsn, revision)),
    )
    result = runner.invoke(app, ["migrate", "--revision", "abc123"])
    assert result.exit_code == 0
    assert calls == [("postgresql://admin@db/footnote", "abc123")]
    assert "migrated to abc123" in result.stdout


def test_migrate_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(dsn: str, revision: str = "head") -> None:
        raise RuntimeError("no connection")

    monkeypatch.setenv("FOOTNOTE_ADMIN_DATABASE_URL", "postgresql://admin@db/footnote")
    monkeypatch.setattr("footnote.adapters.postgres.migrate.upgrade", boom)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 1
    assert "migration failed: RuntimeError: no connection" in result.stderr


# --- serve / worker ---------------------------------------------------------------------


@pytest.mark.parametrize("command", [["serve"], ["worker"]])
def test_serve_and_worker_reject_invalid_config(
    monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    monkeypatch.setenv("FOOTNOTE_SERVER_LISTEN", "nonsense")
    monkeypatch.setenv("FOOTNOTE_ADMIN_LISTEN", "nonsense")
    result = runner.invoke(app, command)
    assert result.exit_code == 2
    assert "invalid configuration" in result.stderr


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Settings, ServeOptions]]:
    seen: list[tuple[Settings, ServeOptions]] = []
    monkeypatch.setattr(cli, "run_until_signalled", lambda s, o: seen.append((s, o)))
    return seen


def test_serve_passes_listen_overrides_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    result = runner.invoke(
        app,
        ["serve", "--listen", "127.0.0.1:0", "--admin-listen", "127.0.0.1:0", "--with-worker"],
    )
    assert result.exit_code == 0
    ((settings, options),) = seen
    assert (settings.server.listen, settings.admin.listen) == ("127.0.0.1:0", "127.0.0.1:0")
    assert (options.data_plane, options.worker) == (True, True)


def test_serve_defaults_to_the_data_plane_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    assert runner.invoke(app, ["serve"]).exit_code == 0
    ((settings, options),) = seen
    assert settings.server.listen == "127.0.0.1:8000"
    assert (options.data_plane, options.worker) == (True, False)


def test_flag_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    monkeypatch.setenv("FOOTNOTE_SERVER_LISTEN", "127.0.0.1:1111")
    runner.invoke(app, ["serve", "--listen", "127.0.0.1:2222"])
    assert seen[0][0].server.listen == "127.0.0.1:2222"


def test_worker_runs_the_worker_without_a_data_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    result = runner.invoke(app, ["worker", "--admin-listen", "127.0.0.1:0"])
    assert result.exit_code == 0
    ((settings, options),) = seen
    assert settings.admin.listen == "127.0.0.1:0"
    assert (options.data_plane, options.worker) == (False, True)


# --- admin groups -----------------------------------------------------------------------

TENANT = "00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize(
    "args",
    [
        ["tenants", "create", "acme"],
        ["tenants", "list"],
        ["tenants", "delete", TENANT],
        ["tenants", "export", TENANT],
        ["keys", "create", TENANT],
        ["keys", "list", TENANT],
        ["keys", "revoke", TENANT, TENANT],
        ["keys", "rotate", TENANT, TENANT],
        ["spaces", "list"],
    ],
    ids=lambda a: " ".join(a[:2]),
)
def test_admin_commands_need_the_admin_dsn(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "FOOTNOTE_ADMIN_DATABASE_URL" in result.stderr


def test_spaces_register_needs_the_admin_dsn() -> None:
    result = runner.invoke(
        app, ["spaces", "register", "x", "--provider", "onnx_local", "--model", "bge-small-en-v1.5"]
    )
    assert result.exit_code == 2
    assert "FOOTNOTE_ADMIN_DATABASE_URL" in result.stderr
