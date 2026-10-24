# SPDX-License-Identifier: Apache-2.0
"""Model manifest, download and SHA-256 verification."""

from footnote.adapters.models.manifest import (
    Manifest,
    ModelEntry,
    ModelFile,
    host_arch,
    load_manifest,
    local_path,
    reranker_variant_key,
)
from footnote.adapters.models.pull import file_url, pull_files, pull_model
from footnote.adapters.models.readiness import ModelNeed, ModelReadiness
from footnote.adapters.models.verify import (
    FileCheck,
    FileStatus,
    check_file,
    sha256_file,
    verify_model,
)

__all__ = [
    "FileCheck",
    "FileStatus",
    "Manifest",
    "ModelEntry",
    "ModelFile",
    "ModelNeed",
    "ModelReadiness",
    "check_file",
    "file_url",
    "host_arch",
    "load_manifest",
    "local_path",
    "pull_files",
    "pull_model",
    "reranker_variant_key",
    "sha256_file",
    "verify_model",
]
