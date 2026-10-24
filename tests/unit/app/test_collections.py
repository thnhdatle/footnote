# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from footnote.app.collections import SettingsPatch, merge_settings, new_webhook_secret
from footnote.domain.errors import ValidationError
from footnote.domain.settings import CollectionSettings, Language


def test_merge_settings_keeps_unpatched_sections() -> None:
    base = CollectionSettings()
    merged = merge_settings(base, SettingsPatch(retrieval={"k": 5}, language=Language.ENGLISH))
    assert merged.retrieval.k == 5
    assert merged.retrieval.rrf_weights == base.retrieval.rrf_weights
    assert merged.chunking == base.chunking
    assert merged.language is Language.ENGLISH


def test_merge_settings_merges_one_rrf_weight() -> None:
    merged = merge_settings(
        CollectionSettings(), SettingsPatch(retrieval={"rrf_weights": {"lexical": 0.0}})
    )
    assert merged.retrieval.rrf_weights == {"lexical": 0.0, "vector": 1.0}


def test_merge_settings_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError) as err:
        merge_settings(CollectionSettings(), SettingsPatch(chunking={"bogus": 1}))
    assert err.value.field == "chunking"


def test_webhook_secrets_are_random_and_prefixed() -> None:
    a, b = new_webhook_secret(), new_webhook_secret()
    assert a != b
    assert a.startswith("whsec_")
    assert len(a) > 40
