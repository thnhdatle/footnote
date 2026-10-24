# SPDX-License-Identifier: Apache-2.0
"""Settings loader: YAML file and `FOOTNOTE_*` environment overrides with fail-fast validation."""

from __future__ import annotations

import json
import os
import re
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from footnote.domain import errors as domain_errors
from footnote.domain.settings import (
    AnswerSettings,
    ChunkingSettings,
    ChunkingStrategyName,
    CollectionSettings,
    Language,
    LexicalMatch,
    LexicalRank,
    RetrievalSettings,
    SearchMode,
)
from footnote.units import parse_duration, parse_size

ENV_PREFIX = "FOOTNOTE_"
CONFIG_ENV = "FOOTNOTE_CONFIG"
TEST_FAULTS_ENV = "FOOTNOTE_ENABLE_TEST_FAULTS"
EXIT_CODE_INVALID = 2

Duration = Annotated[float, BeforeValidator(parse_duration), Field(gt=0)]
NonNegativeDuration = Annotated[float, BeforeValidator(parse_duration), Field(ge=0)]
Size = Annotated[int, BeforeValidator(parse_size), Field(gt=0)]
Listen = Annotated[str, Field(pattern=r"^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]*):[0-9]{1,5}$")]

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_PORT_MAX = 65535
_NULLS = {"null", "none", "~"}


class ConfigError(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
        self.errors = errors


def secret_field() -> Any:
    return Field(default=SecretStr(""), json_schema_extra={"secret": True})


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_listen(value: str) -> str:
    port = int(value.rsplit(":", 1)[1])
    if not 0 <= port <= _PORT_MAX:
        raise ValueError(f"port {port} out of range")
    return value


class ServerConfig(_Section):
    listen: Listen = "127.0.0.1:8000"

    _port = field_validator("listen")(_check_listen)


class AdminConfig(_Section):
    listen: Listen = "127.0.0.1:9094"

    _port = field_validator("listen")(_check_listen)


class LogConfig(_Section):
    format: Literal["json", "text"] = "json"
    level: Literal["debug", "info", "warn", "warning", "error"] = "info"
    capture_content: bool = False
    access_log: bool = False


class DatabaseConfig(_Section):
    url: SecretStr = secret_field()
    url_file: str = ""
    pool_max: int = Field(default=20, ge=1, le=1000)
    prepare_threshold: int | None = Field(default=5, ge=0)
    statement_timeout: Duration = 30.0


class AdminDatabaseConfig(_Section):
    url: SecretStr = secret_field()
    url_file: str = ""
    pool_max: int = Field(default=4, ge=1, le=100)


class AuthConfig(_Section):
    admin_key: SecretStr = secret_field()
    admin_key_file: str = ""
    key_rotation_grace: Duration = 86400.0


class QuotaConfig(_Section):
    max_collections: int = Field(default=0, ge=0)
    max_documents: int = Field(default=0, ge=0)
    max_chunks: int = Field(default=0, ge=0)
    max_upload_bytes_per_day: int = Field(default=0, ge=0)


class TenancyConfig(_Section):
    default_quotas: QuotaConfig = Field(default_factory=QuotaConfig)


class CollectionsConfig(_Section):
    default_language: Language = Language.SIMPLE


class LLMSection(_Section):
    base_url: str = ""
    model: str = ""
    api_key: SecretStr = secret_field()
    api_key_file: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=131072)
    timeout: Duration = 120.0
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def _model_with_base_url(self) -> LLMSection:
        if self.base_url.strip() and not self.model.strip():
            raise ValueError("model is required when base_url is set")
        return self


class VisionSection(_Section):
    base_url: str = ""
    model: str = ""
    api_key: SecretStr = secret_field()
    api_key_file: str = ""
    timeout: Duration = 60.0
    max_tokens: int = Field(default=512, ge=1, le=131072)
    min_image_px: int = Field(default=64, ge=1)
    max_images: int = Field(default=50, ge=0)
    max_images_per_page: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _model_with_base_url(self) -> VisionSection:
        if self.base_url.strip() and not self.model.strip():
            raise ValueError("model is required when base_url is set")
        return self


class SpaceConfig(_Section):
    name: str
    provider: Literal["onnx_local", "openai_compat"]
    model: str
    dimensions: int | None = Field(default=None, ge=1, le=16000)
    send_dimensions: bool = False
    normalise: bool = True
    calibrated_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def _name_ok(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("must match [a-z0-9][a-z0-9._-]{0,62}")
        return value

    @model_validator(mode="after")
    def _remote_needs_dimensions(self) -> SpaceConfig:
        if self.provider == "openai_compat" and self.dimensions is None:
            raise ValueError("dimensions is required for provider openai_compat")
        return self


class HnswConfig(_Section):
    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=64, ge=4, le=1000)

    @model_validator(mode="after")
    def _ef_vs_m(self) -> HnswConfig:
        if self.ef_construction < 2 * self.m:
            raise ValueError("ef_construction must be >= 2 * m")
        return self


def _default_spaces() -> list[SpaceConfig]:
    return [
        SpaceConfig(name="bge-small-en-v1.5-onnx", provider="onnx_local", model="bge-small-en-v1.5")
    ]


class EmbeddingConfig(_Section):
    default_space: str = "bge-small-en-v1.5-onnx"
    spaces: list[SpaceConfig] = Field(default_factory=_default_spaces)
    base_url: str = ""
    api_key: SecretStr = secret_field()
    api_key_file: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)
    batch_size: int = Field(default=64, ge=1, le=2048)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    timeout: Duration = 30.0
    max_retries: int = Field(default=3, ge=0, le=10)
    max_spaces: int = Field(default=8, ge=1, le=64)
    hnsw: HnswConfig = Field(default_factory=HnswConfig)

    @model_validator(mode="after")
    def _spaces_consistent(self) -> EmbeddingConfig:
        names = [s.name for s in self.spaces]
        if len(set(names)) != len(names):
            raise ValueError("spaces: names must be unique")
        if len(names) > self.max_spaces:
            raise ValueError(f"spaces: at most max_spaces ({self.max_spaces}) allowed")
        if self.default_space not in names:
            raise ValueError("default_space must name a configured space")
        if not self.base_url and any(s.provider == "openai_compat" for s in self.spaces):
            raise ValueError("base_url is required when an openai_compat space is configured")
        return self


def _expand(value: object) -> object:
    return Path(value).expanduser() if isinstance(value, str | Path) else value


class ModelsConfig(_Section):
    dir: Annotated[Path, BeforeValidator(_expand)] = Field(
        default_factory=lambda: Path("~/.cache/rayld/footnote/models").expanduser()
    )
    manifest: Path = Path("models/manifest.yaml")
    intra_op_threads: int = Field(default=0, ge=0, le=256)

    @field_validator("dir", mode="after")
    @classmethod
    def _expanded(cls, value: Path) -> Path:
        return value.expanduser()


class RerankConfig(_Section):
    provider: Literal["onnx_cross_encoder", "noop"] = "onnx_cross_encoder"
    model: str = "ms-marco-MiniLM-L6-v2"
    variant: Literal["int8", "fp32"] = "int8"
    max_tokens: int = Field(default=256, ge=8, le=2048)
    batch_size: int = Field(default=32, ge=1, le=512)


class ChunkingConfig(_Section):
    strategy: ChunkingStrategyName = ChunkingStrategyName.HEADING
    tokenizer: str = "bge-small-en-v1.5"
    max_tokens: int = 400
    overlap_tokens: int = 40
    min_tokens: int = 50


class RrfWeights(_Section):
    lexical: float = 0.2
    vector: float = 1.0


class RetrievalConfig(_Section):
    default_mode: SearchMode = SearchMode.HYBRID
    lexical_match: LexicalMatch = LexicalMatch.ANY
    lexical_rank: LexicalRank = LexicalRank.TS_RANK
    k_lexical: int = 50
    k_vector: int = 50
    rrf_k: int = 60
    rrf_weights: RrfWeights = Field(default_factory=RrfWeights)
    rerank: bool = True
    rerank_n: int = 30
    k: int = 8
    ef_search: int = Field(default=64, ge=1, le=1000)
    iterative_scan: Literal["off", "relaxed_order", "strict_order"] = "relaxed_order"
    max_scan_tuples: int = Field(default=20000, ge=1)
    exact_scan_max: int = Field(default=20000, ge=0)


class AnswerConfig(_Section):
    evidence_threshold_rerank: float | None = None
    evidence_threshold_vector: float | None = None
    unsupported_as_abstain: bool = False
    prompt_template: str = "answer_v1"


class IngestConfig(_Section):
    workers: int = Field(default=4, ge=1, le=256)
    max_file_bytes: Size = 50 * 1024**2
    max_pages: int = Field(default=2000, ge=1)
    max_uncompressed_bytes: Size = 200 * 1024**2
    max_archive_members: int = Field(default=10000, ge=1)
    parse_timeout: Duration = 300.0
    retain_originals: bool = True
    gc_grace: NonNegativeDuration = 600.0
    job_lease: Duration = 60.0
    job_max_attempts: int = Field(default=5, ge=1, le=100)
    poll_interval: Duration = 1.0


class WebhooksConfig(_Section):
    allow_http: bool = False
    allow_private: bool = False
    timeout: Duration = 5.0
    max_attempts: int = Field(default=5, ge=1, le=50)


class EvalConfig(_Section):
    cache_dir: Annotated[Path, BeforeValidator(_expand)] = Field(
        default_factory=lambda: Path("~/.cache/rayld/footnote/eval").expanduser()
    )
    calibration_target_answer_rate: float = Field(default=0.85, gt=0.0, le=1.0)

    @field_validator("cache_dir", mode="after")
    @classmethod
    def _expanded(cls, value: Path) -> Path:
        return value.expanduser()


class TestingConfig(_Section):
    __test__ = False

    disable_app_tenant_predicates: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", frozen=True, env_prefix=ENV_PREFIX)

    server: ServerConfig = Field(default_factory=ServerConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    shutdown_timeout: Duration = 30.0
    log: LogConfig = Field(default_factory=LogConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    admin_database: AdminDatabaseConfig = Field(default_factory=AdminDatabaseConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)
    collections: CollectionsConfig = Field(default_factory=CollectionsConfig)
    llm: LLMSection = Field(default_factory=LLMSection)
    vision: VisionSection = Field(default_factory=VisionSection)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    webhooks: WebhooksConfig = Field(default_factory=WebhooksConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003 - signature fixed by pydantic-settings
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)

    @model_validator(mode="after")
    def _domain_bounds(self) -> Settings:
        problems = self.domain_errors()
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def chunking_settings(self) -> ChunkingSettings:
        c = self.chunking
        return ChunkingSettings(
            strategy=c.strategy,
            max_tokens=c.max_tokens,
            overlap_tokens=c.overlap_tokens,
            min_tokens=c.min_tokens,
            tokenizer=c.tokenizer,
        )

    def retrieval_settings(self) -> RetrievalSettings:
        r = self.retrieval
        return RetrievalSettings(
            lexical_match=r.lexical_match,
            lexical_rank=r.lexical_rank,
            k_lexical=r.k_lexical,
            k_vector=r.k_vector,
            rrf_k=r.rrf_k,
            rrf_weights={"lexical": r.rrf_weights.lexical, "vector": r.rrf_weights.vector},
            rerank=r.rerank,
            rerank_n=r.rerank_n,
            k=r.k,
        )

    def answer_settings(self) -> AnswerSettings:
        a = self.answer
        return AnswerSettings(
            evidence_threshold_rerank=a.evidence_threshold_rerank,
            evidence_threshold_vector=a.evidence_threshold_vector,
            prompt_template=a.prompt_template,
            unsupported_as_abstain=a.unsupported_as_abstain,
        )

    def collection_defaults(self) -> CollectionSettings:
        return CollectionSettings(
            chunking=self.chunking_settings(),
            retrieval=self.retrieval_settings(),
            answer=self.answer_settings(),
            language=self.collections.default_language,
            embedding_space=self.embedding.default_space,
        )

    def domain_errors(self) -> list[str]:
        problems: list[str] = []
        for check in (
            self.chunking_settings().validate,
            self.retrieval_settings().validate,
            self.answer_settings().validate,
        ):
            try:
                check()
            except domain_errors.ValidationError as exc:
                problems.append(exc.message)
        return problems

    def tenant_predicates_disabled(self, env: Mapping[str, str] | None = None) -> bool:
        source = os.environ if env is None else env
        return self.testing.disable_app_tenant_predicates and source.get(TEST_FAULTS_ENV) == "1"

    def redacted(self) -> dict[str, Any]:
        dump: dict[str, Any] = self.model_dump(mode="json")
        for path in SECRET_PATHS:
            node = dump
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = "***" if node[path[-1]] else ""
        return dump


type Path_ = tuple[str, ...]


def _unwrap(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _is_section(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def config_leaves(
    model: type[BaseModel] = Settings, prefix: Path_ = ()
) -> Iterator[tuple[Path_, FieldInfo]]:
    for name, info in model.model_fields.items():
        annotation = _unwrap(info.annotation)
        if _is_section(annotation):
            yield from config_leaves(annotation, (*prefix, name))
        else:
            yield (*prefix, name), info


def env_name(path: Path_) -> str:
    return ENV_PREFIX + "_".join(path).upper()


def _is_secret(info: FieldInfo) -> bool:
    extra = info.json_schema_extra
    return isinstance(extra, dict) and bool(extra.get("secret"))


SECRET_PATHS: tuple[Path_, ...] = tuple(p for p, i in config_leaves() if _is_secret(i))


def _coerce_env(info: FieldInfo, raw: str) -> Any:
    annotation = info.annotation
    if raw.strip().lower() in _NULLS and _unwrap(annotation) is not annotation:
        return None
    if get_origin(_unwrap(annotation)) in (dict, list):
        return json.loads(raw)
    return raw


def _get(data: Mapping[str, Any], path: Path_) -> Any:
    node: Any = data
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _put(data: dict[str, Any], path: Path_, value: Any) -> None:
    node = data
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"config file {path}: {exc}")
        return {}
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        errors.append(f"config file {path}: top level must be a mapping")
        return {}
    return loaded


def _overlay_env(data: dict[str, Any], env: Mapping[str, str], errors: list[str]) -> None:
    for path, info in config_leaves():
        raw = env.get(env_name(path))
        if raw is None:
            continue
        try:
            _put(data, path, _coerce_env(info, raw))
        except ValueError as exc:
            errors.append(f"{env_name(path)}: {exc}")


def _resolve_secret_files(data: dict[str, Any], errors: list[str]) -> None:
    for path in SECRET_PATHS:
        if _get(data, path):
            continue
        file_path = _get(data, (*path[:-1], f"{path[-1]}_file"))
        if not file_path:
            continue
        try:
            _put(data, path, Path(str(file_path)).expanduser().read_text("utf-8").strip())
        except OSError as exc:
            errors.append(f"{'.'.join(path)}_file: cannot read {file_path}: {exc.strerror}")


def _format_validation(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "config"
        message = err["msg"].removeprefix("Value error, ")
        out.append(f"{where}: {message}")
    return out


def load_settings(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """Load the settings from the YAML file and the environment; raise ConfigError when invalid."""
    source = os.environ if env is None else env
    errors: list[str] = []
    data: dict[str, Any] = {}
    file_path = path if path else source.get(CONFIG_ENV)
    if file_path:
        data = _read_yaml(Path(file_path).expanduser(), errors)
        for secret in SECRET_PATHS:
            if _get(data, secret) is not None:
                errors.append(
                    f"{'.'.join(secret)}: secrets are not allowed in the config file; "
                    f"use {env_name(secret)} or {secret[-1]}_file"
                )
    _overlay_env(data, source, errors)
    _resolve_secret_files(data, errors)
    if errors:
        raise ConfigError(errors)
    try:
        return Settings(**data)
    except ValidationError as exc:
        raise ConfigError(_format_validation(exc)) from exc
