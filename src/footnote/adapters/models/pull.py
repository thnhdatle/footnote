# SPDX-License-Identifier: Apache-2.0
"""Download of pinned model files from Hugging Face."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import httpx

from footnote.adapters.models.manifest import ModelEntry, ModelFile, local_path
from footnote.adapters.models.verify import FileStatus, check_file, selected_files
from footnote.domain.errors import UpstreamError

HF_BASE = "https://huggingface.co"
_TIMEOUT = httpx.Timeout(30.0, read=120.0)


def file_url(entry: ModelEntry, file: ModelFile, base: str = HF_BASE) -> str:
    return f"{base}/{entry.repo}/resolve/{entry.revision}/{file.path}"


def _download(client: httpx.Client, url: str, file: ModelFile, target: Path) -> None:
    tmp = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as out:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    out.write(chunk)
        if digest.hexdigest() != file.sha256:
            raise UpstreamError(f"download of {file.path} does not match the pinned sha256")
        os.replace(tmp, target)
    except httpx.HTTPError as exc:
        raise UpstreamError(f"download of {file.path} failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def pull_model(
    models_dir: Path,
    entry: ModelEntry,
    variants: list[str] | None = None,
    *,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Downloads the missing or corrupt files of a model entry."""
    return pull_files(models_dir, entry, selected_files(entry, variants), client=client)


def pull_files(
    models_dir: Path,
    entry: ModelEntry,
    files: Mapping[str, ModelFile],
    *,
    client: httpx.Client | None = None,
) -> list[Path]:
    owned = client is None
    http = client or httpx.Client(timeout=_TIMEOUT)
    paths: list[Path] = []
    try:
        for file in files.values():
            target = local_path(models_dir, entry, file)
            if check_file(target, file) is not FileStatus.OK:
                _download(http, file_url(entry, file), file, target)
            paths.append(target)
    finally:
        if owned:
            http.close()
    return paths
