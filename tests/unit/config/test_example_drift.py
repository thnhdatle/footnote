# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from footnote.config import (
    SECRET_PATHS,
    Settings,
    SpaceConfig,
    config_leaves,
    env_name,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "deploy" / "config.example.yaml"
GUIDE = ROOT / "docs" / "guides" / "configuration.md"


def example_data() -> dict[str, Any]:
    loaded = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def lookup(data: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    node: Any = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def test_example_loads_and_equals_defaults() -> None:
    assert load_settings(EXAMPLE, {}) == Settings()


def test_every_option_is_in_the_example_except_secrets() -> None:
    data = example_data()
    missing = [
        ".".join(path)
        for path, _ in config_leaves()
        if path not in SECRET_PATHS and not lookup(data, path)[0]
    ]
    assert missing == []


def test_example_has_no_secret_values() -> None:
    data = example_data()
    assert [p for p in SECRET_PATHS if lookup(data, p)[0]] == []


def test_every_example_key_is_an_option() -> None:
    leaves = {path for path, _ in config_leaves()}
    sections = {path[:i] for path in leaves for i in range(1, len(path))}

    def walk(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in node.items():
            path = (*prefix, key)
            assert path in leaves or path in sections, ".".join(path)
            if path in sections:
                assert isinstance(value, dict)
                walk(value, path)

    walk(example_data(), ())


def test_space_fields_are_all_shown_in_the_example() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    for name in SpaceConfig.model_fields:
        assert f"{name}:" in text, name


def test_every_option_line_has_a_comment() -> None:
    uncommented = []
    for number, line in enumerate(EXAMPLE.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "- ")):
            continue
        is_section = stripped.endswith(":")
        if not is_section and "#" not in line:
            uncommented.append(number)
    assert uncommented == []


def test_guide_documents_every_option_and_env_name() -> None:
    if not GUIDE.exists():
        pytest.fail("docs/guides/configuration.md is missing")
    text = GUIDE.read_text(encoding="utf-8")
    undocumented = [
        ".".join(path) for path, _ in config_leaves() if f"`{'.'.join(path)}`" not in text
    ]
    assert undocumented == []
    for path in SECRET_PATHS:
        assert env_name(path) in text
