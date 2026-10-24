# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from footnote.adapters.models import (
    FileStatus,
    ModelEntry,
    ModelFile,
    file_url,
    host_arch,
    load_manifest,
    local_path,
    pull_model,
    reranker_variant_key,
    sha256_file,
    verify_model,
)
from footnote.domain.errors import NotFoundError, UpstreamError, ValidationError

ROOT = Path(__file__).resolve().parents[4]
PAYLOAD = b"weights"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _entry() -> ModelEntry:
    return ModelEntry(
        id="m",
        kind="reranker",
        repo="org/m",
        revision="rev1",
        files={"tokenizer": ModelFile("tokenizer.json", SHA, len(PAYLOAD))},
        variants={"fp32": ModelFile("onnx/model.onnx", SHA, len(PAYLOAD), 0.5)},
    )


def test_shipped_manifest_loads() -> None:
    manifest = load_manifest(ROOT / "models" / "manifest.yaml")
    bge = manifest.get("bge-small-en-v1.5")
    assert (bge.kind, bge.dimensions, bge.pooling, bge.max_tokens) == ("embedder", 384, "cls", 512)
    assert bge.files["model"].sha256 == (
        "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
    )
    rr = manifest.get("ms-marco-MiniLM-L6-v2")
    assert set(rr.variants) == {"fp32", "int8-arm64", "int8-amd64"}
    assert rr.variants[reranker_variant_key("int8", "arm64")].calibrated_threshold == 0.2844
    assert rr.max_tokens == 256


def test_unknown_model_and_bad_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(ROOT / "models" / "manifest.yaml")
    with pytest.raises(NotFoundError):
        manifest.get("nope")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [{id: x}]", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_manifest(bad)
    with pytest.raises(ValidationError):
        load_manifest(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("variant", "arch", "expected"),
    [("fp32", "arm64", "fp32"), ("int8", "arm64", "int8-arm64"), ("int8", "amd64", "int8-amd64")],
)
def test_variant_key(variant: str, arch: str, expected: str) -> None:
    assert reranker_variant_key(variant, arch) == expected


def test_variant_key_defaults_and_errors() -> None:
    assert reranker_variant_key("int8") == f"int8-{host_arch()}"
    with pytest.raises(ValidationError):
        reranker_variant_key("fp16")


def test_verify_reports_missing_mismatch_ok(tmp_path: Path) -> None:
    entry = _entry()
    statuses = lambda: {c.name: c.status for c in verify_model(tmp_path, entry, ["fp32"])}  # noqa: E731
    assert statuses() == {"tokenizer": FileStatus.MISSING, "fp32": FileStatus.MISSING}
    tok = local_path(tmp_path, entry, entry.files["tokenizer"])
    tok.parent.mkdir(parents=True)
    tok.write_bytes(PAYLOAD)
    model = local_path(tmp_path, entry, entry.variants["fp32"])
    model.parent.mkdir(parents=True)
    model.write_bytes(b"corrupt")
    assert statuses() == {"tokenizer": FileStatus.OK, "fp32": FileStatus.MISMATCH}
    model.write_bytes(b"x" * len(PAYLOAD))
    assert statuses()["fp32"] is FileStatus.MISMATCH
    assert sha256_file(tok) == SHA


def test_pull_downloads_verifies_and_skips_good_files(tmp_path: Path) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, content=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    entry = _entry()
    paths = pull_model(tmp_path, entry, ["fp32"], client=client)
    assert [p.name for p in paths] == ["tokenizer.json", "model.onnx"]
    assert urls[1] == "https://huggingface.co/org/m/resolve/rev1/onnx/model.onnx"
    assert urls[1] == file_url(entry, entry.variants["fp32"])
    assert all(c.status is FileStatus.OK for c in verify_model(tmp_path, entry, ["fp32"]))
    pull_model(tmp_path, entry, ["fp32"], client=client)
    assert len(urls) == 2


def test_pull_rejects_hash_mismatch_and_leaves_nothing(tmp_path: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"evil"))
    )
    with pytest.raises(UpstreamError):
        pull_model(tmp_path, _entry(), client=client)
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_pull_http_error(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(404)))
    with pytest.raises(UpstreamError):
        pull_model(tmp_path, _entry(), client=client)
