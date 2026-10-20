# SPDX-License-Identifier: Apache-2.0
"""Embedding spaces and the vector-index expression rules."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from footnote.domain.errors import ValidationError

VECTOR_MAX_DIMS = 2000
HALFVEC_MAX_DIMS = 4000
_SPACE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


class IndexType(StrEnum):
    """pgvector type used in the per-space expression index."""

    VECTOR = "vector"
    HALFVEC = "halfvec"


def index_type_for(dimensions: int) -> IndexType:
    """Pick the index type for a dimension count; > 4000 is unsupported."""
    if dimensions < 1:
        raise ValidationError("dimensions", "must be >= 1")
    if dimensions <= VECTOR_MAX_DIMS:
        return IndexType.VECTOR
    if dimensions <= HALFVEC_MAX_DIMS:
        return IndexType.HALFVEC
    raise ValidationError(
        "dimensions",
        f"{dimensions} exceeds {HALFVEC_MAX_DIMS}",
        code="embedding_dimension_unsupported",
    )


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Registered (provider, model, dimensions, normalisation) with an integer id."""

    id: int
    name: str
    provider: str
    model: str
    dimensions: int
    send_dimensions: bool = False
    normalise: bool = True
    calibrated_threshold: float | None = None

    @property
    def index_type(self) -> IndexType:
        """Index type derived from the dimensions."""
        return index_type_for(self.dimensions)

    def cast(self) -> str:
        """SQL cast used on both sides of `<=>` so the partial index matches."""
        return f"{self.index_type.value}({int(self.dimensions)})"

    def opclass(self) -> str:
        """HNSW operator class for this space."""
        return f"{self.index_type.value}_cosine_ops"

    def index_name(self) -> str:
        """Name of the partial HNSW index of this space."""
        return f"chunk_vectors_hnsw_s{int(self.id)}"


@dataclass(frozen=True, slots=True)
class SpaceSpec:
    """A space to register (config `embedding.spaces[]` or the admin API)."""

    name: str
    provider: str
    model: str
    dimensions: int
    send_dimensions: bool = False
    normalise: bool = True
    calibrated_threshold: float | None = None


def validate_space_name(name: str) -> None:
    """Space names are lowercase identifiers safe for logs and config."""
    if not _SPACE_NAME.fullmatch(name):
        raise ValidationError("name", "must match ^[a-z0-9][a-z0-9._-]{0,62}$")


def l2_normalise(vector: Sequence[float]) -> list[float]:
    """Return the unit-length copy of `vector`; the zero vector is returned unchanged."""
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0.0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 when either is zero)."""
    if len(a) != len(b):
        raise ValueError("cosine_similarity: dimension mismatch")
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
