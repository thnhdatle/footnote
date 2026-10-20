# SPDX-License-Identifier: Apache-2.0
"""Acceptance pytest configuration: every test in this directory carries the acceptance marker."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every collected acceptance test."""
    for item in items:
        item.add_marker(pytest.mark.acceptance)
