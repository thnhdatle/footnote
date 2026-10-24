# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GENERATOR = Path(__file__).resolve().parents[3] / "fixtures" / "make_fixtures.py"


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    spec = importlib.util.spec_from_file_location("make_fixtures_parsers", _GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures_parsers"] = module
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("corpus")
    module.generate(out)
    return out
