# SPDX-License-Identifier: Apache-2.0
"""Loading, env overrides, secrets and validation of the configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from footnote.config import (
    SECRET_PATHS,
    ConfigError,
    Settings,
    config_leaves,
    env_name,
    load_settings,
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_file_or_env() -> None:
    assert load_settings(None, {}) == Settings()


def test_file_values_and_units(tmp_path: Path) -> None:
    path = write(tmp_path, "shutdown_timeout: 1m\ningest: {max_file_bytes: 1MiB, gc_grace: 1d}\n")
    s = load_settings(path, {})
    assert s.shutdown_timeout == 60.0
    assert s.ingest.max_file_bytes == 1024**2
    assert s.ingest.gc_grace == 86400.0


def test_gc_grace_accepts_zero_but_other_durations_do_not() -> None:
    assert load_settings(None, {"FOOTNOTE_INGEST_GC_GRACE": "0s"}).ingest.gc_grace == 0.0
    with pytest.raises(ConfigError):
        load_settings(None, {"FOOTNOTE_INGEST_JOB_LEASE": "0s"})


def test_env_overrides_file_and_path_comes_from_env(tmp_path: Path) -> None:
    path = write(tmp_path, "database: {pool_max: 5}\nllm: {model: from-file}\n")
    env = {
        "FOOTNOTE_CONFIG": str(path),
        "FOOTNOTE_DATABASE_POOL_MAX": "7",
        "FOOTNOTE_LOG_ACCESS_LOG": "true",
        "FOOTNOTE_LLM_EXTRA_HEADERS": '{"X-Fake-Scenario": "echo"}',
        "FOOTNOTE_DATABASE_PREPARE_THRESHOLD": "null",
        "FOOTNOTE_EMBEDDING_HNSW_M": "8",
    }
    s = load_settings(None, env)
    assert s.database.pool_max == 7
    assert s.llm.model == "from-file"
    assert s.log.access_log is True
    assert s.llm.extra_headers == {"X-Fake-Scenario": "echo"}
    assert s.database.prepare_threshold is None
    assert s.embedding.hnsw.m == 8


def test_secrets_from_env_and_files_never_in_yaml(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("from-file\n", encoding="utf-8")
    path = write(tmp_path, f"llm: {{api_key_file: {key}}}\nauth: {{admin_key_file: {key}}}\n")
    env = {"FOOTNOTE_DATABASE_URL": "postgresql://a", "FOOTNOTE_AUTH_ADMIN_KEY": "env-wins"}
    s = load_settings(path, env)
    assert s.llm.api_key.get_secret_value() == "from-file"
    assert s.auth.admin_key.get_secret_value() == "env-wins"
    assert s.database.url.get_secret_value() == "postgresql://a"
    assert "from-file" not in repr(s)
    assert "postgresql://a" not in repr(s)


def test_secret_in_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="FOOTNOTE_LLM_API_KEY"):
        load_settings(write(tmp_path, "llm: {api_key: sk-123}\n"), {})


def test_redacted_hides_secrets() -> None:
    s = load_settings(None, {"FOOTNOTE_LLM_API_KEY": "sk-123"})
    dump = s.redacted()
    assert dump["llm"]["api_key"] == "***"
    assert dump["database"]["url"] == ""
    assert "sk-123" not in str(dump)


def test_all_errors_reported_at_once(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "log: {level: loud}\ningest: {parse_timeout: soon}\nbogus: 1\n"
        "llm: {api_key_file: /nonexistent/key}\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_settings(path, {})
    assert any("api_key_file" in e for e in exc.value.errors)
    path.write_text("log: {level: loud}\ningest: {parse_timeout: soon}\nbogus: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_settings(path, {})
    text = "\n".join(exc.value.errors)
    for needle in ("log.level", "parse_timeout", "bogus"):
        assert needle in text, needle
    assert len(exc.value.errors) == 3


def test_domain_bound_violations_reported_together(tmp_path: Path) -> None:
    path = write(tmp_path, "retrieval: {k: 500}\nchunking: {max_tokens: 10}\n")
    with pytest.raises(ConfigError) as exc:
        load_settings(path, {})
    text = "\n".join(exc.value.errors)
    assert "chunking.max_tokens" in text
    assert "retrieval.k" in text


@pytest.mark.parametrize(
    ("yaml_text", "needle"),
    [
        ("retrieval: {rerank_n: 4, k: 8}", "rerank_n"),
        ("retrieval: {rrf_weights: {lexical: 0, vector: 0}}", "rrf_weights"),
        ("retrieval: {rrf_weights: {lexical: 11}}", "rrf_weights.lexical"),
        ("retrieval: {k_vector: 5000}", "k_vector"),
        ("chunking: {max_tokens: 400, overlap_tokens: 300}", "overlap_tokens"),
        ("answer: {evidence_threshold_vector: 1.5}", "evidence_threshold_vector"),
        ("server: {listen: 'nonsense'}", "listen"),
        ("server: {listen: '127.0.0.1:99999'}", "port"),
        ("llm: {temperature: 3}", "temperature"),
        ("llm: {base_url: 'http://h/v1'}", "model is required"),
        ("database: {pool_max: 0}", "pool_max"),
        ("embedding: {hnsw: {m: 16, ef_construction: 16}}", "ef_construction"),
        ("embedding: {default_space: nope}", "default_space"),
        (
            "embedding: {max_spaces: 1, spaces: [{name: a, provider: onnx_local, model: m},"
            " {name: b, provider: onnx_local, model: m}], default_space: a}",
            "max_spaces",
        ),
        (
            "embedding: {spaces: [{name: bge-small-en-v1.5-onnx, provider: onnx_local, model: m},"
            " {name: bge-small-en-v1.5-onnx, provider: onnx_local, model: m}]}",
            "unique",
        ),
        (
            "embedding: {spaces: [{name: x, provider: openai_compat, model: m}], default_space: x,"
            " base_url: http://h}",
            "dimensions",
        ),
        (
            "embedding: {spaces: [{name: x, provider: openai_compat, model: m, dimensions: 8}],"
            " default_space: x}",
            "base_url",
        ),
        (
            "embedding: {spaces: [{name: BAD NAME, provider: onnx_local, model: m}],"
            " default_space: 'BAD NAME'}",
            "name",
        ),
        ("rerank: {variant: int4}", "variant"),
        ("eval: {calibration_target_answer_rate: 0}", "calibration_target_answer_rate"),
    ],
)
def test_validation_rejects(tmp_path: Path, yaml_text: str, needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        load_settings(write(tmp_path, yaml_text + "\n"), {})


@pytest.mark.parametrize("text", ["- a\n- b\n", "a: [unclosed\n"])
def test_bad_yaml_shapes(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="config file"):
        load_settings(write(tmp_path, text), {})


def test_missing_file_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file"):
        load_settings(tmp_path / "missing.yaml", {})
    assert load_settings(write(tmp_path, ""), {}) == Settings()


def test_bad_env_json_reported() -> None:
    with pytest.raises(ConfigError, match="FOOTNOTE_LLM_EXTRA_BODY"):
        load_settings(None, {"FOOTNOTE_LLM_EXTRA_BODY": "{not json"})


def test_env_names_are_unique_and_secrets_known() -> None:
    names = [env_name(path) for path, _ in config_leaves()]
    assert len(names) == len(set(names))
    assert {env_name(p) for p in SECRET_PATHS} == {
        "FOOTNOTE_DATABASE_URL",
        "FOOTNOTE_ADMIN_DATABASE_URL",
        "FOOTNOTE_AUTH_ADMIN_KEY",
        "FOOTNOTE_LLM_API_KEY",
        "FOOTNOTE_EMBEDDING_API_KEY",
    }


def test_test_fault_flag_needs_env(tmp_path: Path) -> None:
    s = load_settings(write(tmp_path, "testing: {disable_app_tenant_predicates: true}\n"), {})
    assert s.tenant_predicates_disabled({}) is False
    assert s.tenant_predicates_disabled({"FOOTNOTE_ENABLE_TEST_FAULTS": "1"}) is True
    assert Settings().tenant_predicates_disabled({"FOOTNOTE_ENABLE_TEST_FAULTS": "1"}) is False


def test_collection_defaults_follow_config(tmp_path: Path) -> None:
    s = load_settings(
        write(
            tmp_path, "retrieval: {k: 3, rerank_n: 10}\ncollections: {default_language: english}\n"
        ),
        {},
    )
    d = s.collection_defaults()
    assert d.retrieval.k == 3
    assert d.language.value == "english"
    assert d.embedding_space == "bge-small-en-v1.5-onnx"
    assert d.chunking.max_tokens == 400
    assert d.answer.prompt_template == "answer_v1"
