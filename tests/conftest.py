# SPDX-License-Identifier: Apache-2.0

import os
import threading
from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile(
    "fuzz",
    max_examples=int(os.environ.get("FUZZTIME", "30")) * 20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _no_leftover_threads() -> Iterator[None]:
    before = {t.ident for t in threading.enumerate()}
    yield
    leaked = [t.name for t in threading.enumerate() if t.ident not in before and not t.daemon]
    assert not leaked, f"leaked threads: {leaked}"
