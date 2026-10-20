# SPDX-License-Identifier: Apache-2.0
"""Collection setting bounds enforced through `POST /v1/collections` (accepted at the bound,
rejected beyond it, with the setting named in `details[0].field`)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.api.conftest import ApiRig, error_of

CASES: list[Any] = [
    # retrieval.k (rerank_n must stay >= k)
    pytest.param({"retrieval": {"k": 1}}, None, id="k-1"),
    pytest.param({"retrieval": {"k": 100, "rerank_n": 200}}, None, id="k-100"),
    pytest.param({"retrieval": {"k": 0}}, "retrieval.k", id="k-0"),
    pytest.param({"retrieval": {"k": 101, "rerank_n": 200}}, "retrieval.k", id="k-101"),
    # legs and RRF constant
    pytest.param({"retrieval": {"k_lexical": 1}}, None, id="k_lexical-1"),
    pytest.param({"retrieval": {"k_lexical": 1000}}, None, id="k_lexical-1000"),
    pytest.param({"retrieval": {"k_lexical": 0}}, "retrieval.k_lexical", id="k_lexical-0"),
    pytest.param({"retrieval": {"k_lexical": 1001}}, "retrieval.k_lexical", id="k_lexical-1001"),
    pytest.param({"retrieval": {"k_vector": 1}}, None, id="k_vector-1"),
    pytest.param({"retrieval": {"k_vector": 1000}}, None, id="k_vector-1000"),
    pytest.param({"retrieval": {"k_vector": 0}}, "retrieval.k_vector", id="k_vector-0"),
    pytest.param({"retrieval": {"k_vector": 1001}}, "retrieval.k_vector", id="k_vector-1001"),
    pytest.param({"retrieval": {"rrf_k": 1}}, None, id="rrf_k-1"),
    pytest.param({"retrieval": {"rrf_k": 1000}}, None, id="rrf_k-1000"),
    pytest.param({"retrieval": {"rrf_k": 0}}, "retrieval.rrf_k", id="rrf_k-0"),
    pytest.param({"retrieval": {"rrf_k": 1001}}, "retrieval.rrf_k", id="rrf_k-1001"),
    # RRF weights
    pytest.param({"retrieval": {"rrf_weights": {"lexical": 0}}}, None, id="weight-lexical-0"),
    pytest.param({"retrieval": {"rrf_weights": {"vector": 0}}}, None, id="weight-vector-0"),
    pytest.param(
        {"retrieval": {"rrf_weights": {"lexical": 10, "vector": 10}}}, None, id="weights-10"
    ),
    pytest.param(
        {"retrieval": {"rrf_weights": {"lexical": 11}}},
        "retrieval.rrf_weights.lexical",
        id="weight-lexical-11",
    ),
    pytest.param(
        {"retrieval": {"rrf_weights": {"vector": 11}}},
        "retrieval.rrf_weights.vector",
        id="weight-vector-11",
    ),
    pytest.param(
        {"retrieval": {"rrf_weights": {"lexical": -1}}},
        "retrieval.rrf_weights.lexical",
        id="weight-negative",
    ),
    pytest.param(
        {"retrieval": {"rrf_weights": {"lexical": 0, "vector": 0}}},
        "retrieval.rrf_weights",
        id="weights-both-0",
    ),
    # rerank_n
    pytest.param({"retrieval": {"rerank_n": 1, "k": 1}}, None, id="rerank_n-1"),
    pytest.param({"retrieval": {"rerank_n": 200}}, None, id="rerank_n-200"),
    pytest.param({"retrieval": {"rerank_n": 8}}, None, id="rerank_n-equals-k"),
    pytest.param({"retrieval": {"rerank_n": 0}}, "retrieval.rerank_n", id="rerank_n-0"),
    pytest.param({"retrieval": {"rerank_n": 201}}, "retrieval.rerank_n", id="rerank_n-201"),
    pytest.param({"retrieval": {"rerank_n": 7}}, "retrieval.rerank_n", id="rerank_n-below-k"),
    # chunking
    pytest.param(
        {"chunking": {"max_tokens": 32, "overlap_tokens": 0, "min_tokens": 0}},
        None,
        id="max_tokens-32",
    ),
    pytest.param({"chunking": {"max_tokens": 2048}}, None, id="max_tokens-2048"),
    pytest.param({"chunking": {"max_tokens": 31}}, "chunking.max_tokens", id="max_tokens-31"),
    pytest.param({"chunking": {"max_tokens": 2049}}, "chunking.max_tokens", id="max_tokens-2049"),
    pytest.param({"chunking": {"max_tokens": 100, "overlap_tokens": 0}}, None, id="overlap-0"),
    pytest.param({"chunking": {"max_tokens": 100, "overlap_tokens": 50}}, None, id="overlap-half"),
    pytest.param(
        {"chunking": {"max_tokens": 100, "overlap_tokens": 51}},
        "chunking.overlap_tokens",
        id="overlap-over-half",
    ),
    pytest.param(
        {"chunking": {"max_tokens": 100, "overlap_tokens": -1}},
        "chunking.overlap_tokens",
        id="overlap-negative",
    ),
    pytest.param({"chunking": {"max_tokens": 100, "min_tokens": 0}}, None, id="min_tokens-0"),
    pytest.param({"chunking": {"max_tokens": 100, "min_tokens": 100}}, None, id="min_tokens-max"),
    pytest.param(
        {"chunking": {"max_tokens": 100, "min_tokens": 101}},
        "chunking.min_tokens",
        id="min_tokens-over-max",
    ),
    pytest.param({"chunking": {"tokenizer": "bge-small-en-v1.5"}}, None, id="tokenizer-known"),
    pytest.param({"chunking": {"tokenizer": "other"}}, "chunking.tokenizer", id="tokenizer-other"),
    # answer
    pytest.param({"answer": {"evidence_threshold_rerank": 0}}, None, id="rerank-threshold-0"),
    pytest.param({"answer": {"evidence_threshold_rerank": 1}}, None, id="rerank-threshold-1"),
    pytest.param({"answer": {"evidence_threshold_rerank": None}}, None, id="rerank-threshold-null"),
    pytest.param(
        {"answer": {"evidence_threshold_rerank": -0.1}},
        "answer.evidence_threshold_rerank",
        id="rerank-threshold-below",
    ),
    pytest.param(
        {"answer": {"evidence_threshold_rerank": 1.1}},
        "answer.evidence_threshold_rerank",
        id="rerank-threshold-above",
    ),
    pytest.param({"answer": {"evidence_threshold_vector": 0}}, None, id="vector-threshold-0"),
    pytest.param({"answer": {"evidence_threshold_vector": 1}}, None, id="vector-threshold-1"),
    pytest.param({"answer": {"evidence_threshold_vector": None}}, None, id="vector-threshold-null"),
    pytest.param(
        {"answer": {"evidence_threshold_vector": -0.1}},
        "answer.evidence_threshold_vector",
        id="vector-threshold-below",
    ),
    pytest.param(
        {"answer": {"evidence_threshold_vector": 1.1}},
        "answer.evidence_threshold_vector",
        id="vector-threshold-above",
    ),
    pytest.param({"answer": {"prompt_template": "answer_v1"}}, None, id="template-known"),
    pytest.param(
        {"answer": {"prompt_template": "nope"}}, "answer.prompt_template", id="template-unknown"
    ),
]


@pytest.mark.parametrize(("body", "bad_field"), CASES)
async def test_create_enforces_setting_bounds(
    rig: ApiRig, body: dict[str, Any], bad_field: str | None
) -> None:
    r = await rig.client.post(
        "/v1/collections", headers=rig.auth(), json={"name": "bounds", **body}
    )
    if bad_field is None:
        assert r.status_code == 201, r.text
        return
    assert r.status_code == 422, r.text
    err = error_of(r)
    assert err["code"] == "validation_failed"
    assert err["details"][0]["field"] == bad_field


@pytest.mark.parametrize(("body", "bad_field"), [c for c in CASES if c.values[1] is not None])
async def test_patch_enforces_the_same_bounds(
    rig: ApiRig, body: dict[str, Any], bad_field: str
) -> None:
    r = await rig.client.patch(rig.col_url(), headers=rig.auth(), json=body)
    assert r.status_code == 422, r.text
    assert error_of(r)["details"][0]["field"] == bad_field
