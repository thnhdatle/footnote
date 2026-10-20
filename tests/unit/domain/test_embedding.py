# SPDX-License-Identifier: Apache-2.0
"""Unit tests for embedding spaces, index rules and vector helpers."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from footnote.domain.embedding import (
    EmbeddingSpace,
    IndexType,
    cosine_similarity,
    index_type_for,
    l2_normalise,
    validate_space_name,
)
from footnote.domain.errors import ValidationError


@pytest.mark.parametrize(
    ("dims", "expected"),
    [
        (1, IndexType.VECTOR),
        (2000, IndexType.VECTOR),
        (2001, IndexType.HALFVEC),
        (4000, IndexType.HALFVEC),
    ],
)
def test_index_type_for_bounds(dims: int, expected: IndexType) -> None:
    assert index_type_for(dims) is expected


def test_index_type_for_above_4000_unsupported() -> None:
    with pytest.raises(ValidationError) as info:
        index_type_for(4001)
    assert info.value.code == "embedding_dimension_unsupported"


@pytest.mark.parametrize("dims", [0, -5])
def test_index_type_for_below_one_rejected(dims: int) -> None:
    with pytest.raises(ValidationError) as info:
        index_type_for(dims)
    assert info.value.code == "validation_failed"


def test_space_cast_opclass_index_name_vector() -> None:
    space = EmbeddingSpace(7, "s", "onnx", "m", 384)
    assert space.index_type is IndexType.VECTOR
    assert space.cast() == "vector(384)"
    assert space.opclass() == "vector_cosine_ops"
    assert space.index_name() == "chunk_vectors_hnsw_s7"


def test_space_cast_opclass_halfvec() -> None:
    space = EmbeddingSpace(1, "s", "openai", "m", 3072)
    assert space.cast() == "halfvec(3072)"
    assert space.opclass() == "halfvec_cosine_ops"


@pytest.mark.parametrize("name", ["a", "bge-small.v1_x", "0abc", "a" * 63])
def test_validate_space_name_accepts(name: str) -> None:
    validate_space_name(name)


@pytest.mark.parametrize("name", ["", "A", "-a", ".a", "a b", "a" * 64, "a/b"])
def test_validate_space_name_rejects(name: str) -> None:
    with pytest.raises(ValidationError):
        validate_space_name(name)


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=64))
def test_l2_normalise_unit_norm(vector: list[float]) -> None:
    assume(any(abs(x) > 1e-100 for x in vector))
    out = l2_normalise(vector)
    assert math.sqrt(sum(x * x for x in out)) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalise_zero_vector_unchanged() -> None:
    assert l2_normalise([0, 0, 0]) == [0.0, 0.0, 0.0]


def test_l2_normalise_known_value() -> None:
    assert l2_normalise([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_cosine_similarity_values() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-2, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 1], [0, 0]) == 0.0


def test_cosine_similarity_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_validate_space_name_rejects_trailing_newline() -> None:
    with pytest.raises(ValidationError):
        validate_space_name("a\n")
