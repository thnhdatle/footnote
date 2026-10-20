# SPDX-License-Identifier: Apache-2.0
"""SystemHostResolver: the HostResolver adapter used when a webhook URL is set."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from footnote.adapters.webhooks import SystemHostResolver
from footnote.adapters.webhooks import sender as sender_module


async def test_resolver_resolves_loopback_name() -> None:
    addresses = await SystemHostResolver().resolve("localhost", 443)
    assert addresses
    assert set(addresses) <= {"127.0.0.1", "::1"}


@pytest.mark.parametrize("error", [OSError("no such host"), UnicodeError("bad idna")])
async def test_resolver_failure_gives_no_address(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def failing(host: str, port: int) -> Sequence[str]:
        raise error

    monkeypatch.setattr(sender_module, "system_resolver", failing)
    assert await SystemHostResolver().resolve("hooks.example.com", 443) == []
