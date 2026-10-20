# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the webhook sender: signing, SSRF policy, redirects, limits."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import httpx
import pytest

from footnote.adapters.webhooks import DeliveryResult, WebhookConfig, WebhookSender
from footnote.domain.webhooks import verify_signature

SECRET = b"s3cret"
BODY = b'{"event":"document.ready"}'
Handler = Callable[[httpx.Request], httpx.Response]


def _resolver(*addresses: str) -> Callable[[str, int], object]:
    async def resolve(host: str, port: int) -> Sequence[str]:
        return list(addresses)

    return resolve


def _sender(
    handler: Handler, *addresses: str, **config: object
) -> tuple[WebhookSender, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    resolver = _resolver(*(addresses or ("93.184.216.34",)))
    return WebhookSender(client, WebhookConfig(**config), resolver), seen  # type: ignore[arg-type]


async def test_delivers_signed_json_to_the_checked_address() -> None:
    sender, seen = _sender(lambda _r: httpx.Response(204), "93.184.216.34", "8.8.8.8")
    result = await sender.send("https://hooks.example.com/in?x=1", SECRET, BODY, 1_700_000_000)
    assert result == DeliveryResult(True, status_code=204)
    request = seen[0]
    assert str(request.url) == "https://93.184.216.34/in?x=1"
    assert request.headers["host"] == "hooks.example.com"
    assert request.extensions["sni_hostname"] == "hooks.example.com"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-footnote-timestamp"] == "1700000000"
    assert verify_signature(
        SECRET, 1_700_000_000, request.content, request.headers["x-footnote-signature"]
    )
    assert request.content == BODY


async def test_non_default_port_in_host_header_and_ipv6_literal() -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200), "2606:4700:4700::1111")
    await sender.send("https://hooks.example.com:8443/h", SECRET, BODY, 1)
    assert str(seen[0].url) == "https://[2606:4700:4700::1111]:8443/h"
    assert seen[0].headers["host"] == "hooks.example.com:8443"


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "100.64.0.1", "169.254.169.254",
     "::1", "fe80::1", "fc00::1", "::ffff:10.0.0.1", "0.0.0.0", "224.0.0.1"],  # noqa: S104 - the unspecified address under test
)  # fmt: skip
async def test_private_addresses_are_rejected_without_connecting(address: str) -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200), address)
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert not result.delivered
    assert result.permanent
    assert result.error is not None
    assert seen == []


async def test_allow_private_connects_to_private_address() -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200), "10.0.0.5", allow_private=True)
    assert (await sender.send("https://hooks.internal/", SECRET, BODY, 1)).delivered
    assert str(seen[0].url) == "https://10.0.0.5/"


async def test_one_private_address_among_public_ones_rejects() -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200), "8.8.8.8", "192.168.0.9")
    assert (await sender.send("https://hooks.example.com/", SECRET, BODY, 1)).permanent
    assert seen == []


async def test_http_requires_allow_http() -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200))
    result = await sender.send("http://hooks.example.com/", SECRET, BODY, 1)
    assert result.permanent
    assert seen == []
    sender, seen = _sender(lambda _r: httpx.Response(200), allow_http=True)
    assert (await sender.send("http://hooks.example.com/", SECRET, BODY, 1)).delivered
    assert seen[0].headers["host"] == "hooks.example.com"


async def test_redirects_are_not_followed() -> None:
    sender, seen = _sender(
        lambda _r: httpx.Response(302, headers={"Location": "https://169.254.169.254/"})
    )
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert result == DeliveryResult(False, status_code=302, error="receiver returned status 302")
    assert len(seen) == 1


@pytest.mark.parametrize("status", [400, 404, 500, 503])
async def test_error_statuses_are_retryable_failures(status: int) -> None:
    sender, _ = _sender(lambda _r: httpx.Response(status))
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert not result.delivered
    assert not result.permanent
    assert result.status_code == status


async def test_response_body_read_is_bounded() -> None:
    chunks_read = 0

    async def body() -> object:
        nonlocal chunks_read
        for _ in range(1000):
            chunks_read += 1
            yield b"x" * 512

    sender, _ = _sender(lambda _r: httpx.Response(200, content=body()))  # type: ignore[arg-type]
    assert (await sender.send("https://hooks.example.com/", SECRET, BODY, 1)).delivered
    assert chunks_read <= 3


async def test_timeout_is_a_retryable_failure() -> None:
    gate = asyncio.Event()

    class Slow(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await gate.wait()
            return httpx.Response(200)

    sender = WebhookSender(
        httpx.AsyncClient(transport=Slow()),
        WebhookConfig(timeout=0.05),
        _resolver("8.8.8.8"),  # type: ignore[arg-type]
    )
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert result == DeliveryResult(False, error="timeout")


async def test_transport_error_and_dns_failure_are_retryable() -> None:
    def boom(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    sender, _ = _sender(boom)
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert result.error == "transport error ConnectError"
    assert not result.permanent

    async def broken(host: str, port: int) -> Sequence[str]:
        raise OSError("nxdomain")

    sender = WebhookSender(httpx.AsyncClient(), WebhookConfig(), broken)
    result = await sender.send("https://hooks.example.com/", SECRET, BODY, 1)
    assert not result.delivered
    assert not result.permanent


@pytest.mark.parametrize(
    "url",
    [
        "ftp://x.example.com/",
        "https://user:pw@x.example.com/",
        "https://x.example.com/#frag",
        "nonsense",
    ],
)
async def test_invalid_urls_are_permanent(url: str) -> None:
    sender, seen = _sender(lambda _r: httpx.Response(200))
    assert (await sender.send(url, SECRET, BODY, 1)).permanent
    assert seen == []
