# SPDX-License-Identifier: Apache-2.0
"""Signed webhook POST with the webhook SSRF policy.

The host is resolved here, every address is vetted by `domain.netpolicy`, and the
request goes to the first checked address (no re-resolution) with Host and TLS SNI taken
from the URL. Redirects are not followed, the whole exchange is bounded by
`WebhookConfig.timeout` and at most 1 KiB of the response body is read.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from footnote.domain.errors import ValidationError
from footnote.domain.netpolicy import WebhookTarget, check_addresses, parse_webhook_url
from footnote.domain.webhooks import sign

_BODY_LIMIT = 1024
_DEFAULT_PORTS = {"https": 443, "http": 80}

type Resolver = Callable[[str, int], Awaitable[Sequence[str]]]


async def system_resolver(host: str, port: int) -> Sequence[str]:
    """Resolve `host` with the system resolver (A and AAAA, order preserved, deduplicated)."""
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


class SystemHostResolver:
    """`HostResolver` port over the system resolver; unresolvable hosts give no address."""

    async def resolve(self, host: str, port: int) -> Sequence[str]:
        """Addresses of `host` (empty when resolution fails)."""
        try:
            return await system_resolver(host, port)
        except (OSError, UnicodeError):
            return []


@dataclass(frozen=True, slots=True)
class WebhookConfig:
    """`webhooks.*` settings used by the sender."""

    allow_http: bool = False
    allow_private: bool = False
    timeout: float = 5.0
    max_attempts: int = 5


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Outcome of one attempt; `permanent` marks failures a retry cannot fix (policy)."""

    delivered: bool
    status_code: int | None = None
    error: str | None = None
    permanent: bool = False


def _host_header(target: WebhookTarget) -> str:
    host = f"[{target.host}]" if ":" in target.host else target.host
    return host if target.port == _DEFAULT_PORTS[target.scheme] else f"{host}:{target.port}"


def _ip_url(target: WebhookTarget, address: str) -> str:
    ip = f"[{address}]" if ":" in address else address
    return f"{target.scheme}://{ip}:{target.port}{target.path_qs}"


class WebhookSender:
    """Delivers one signed payload per `send` call."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        config: WebhookConfig,
        resolver: Resolver = system_resolver,
    ) -> None:
        self._client = client
        self._cfg = config
        self._resolver = resolver

    @property
    def max_attempts(self) -> int:
        """Delivery attempts before a webhook job fails for good."""
        return self._cfg.max_attempts

    async def send(self, url: str, secret: bytes, body: bytes, timestamp: int) -> DeliveryResult:
        """POST `body` to `url`; 2xx is delivered. Never raises for delivery problems."""
        try:
            target = parse_webhook_url(url, allow_http=self._cfg.allow_http)
            address = await self._resolve(target)
        except ValidationError as exc:
            return DeliveryResult(False, error=exc.message, permanent=True)
        except OSError as exc:
            return DeliveryResult(False, error=f"dns resolution failed: {exc.__class__.__name__}")
        try:
            async with asyncio.timeout(self._cfg.timeout):
                return await self._post(target, address, secret, body, timestamp)
        except TimeoutError:
            return DeliveryResult(False, error="timeout")
        except httpx.HTTPError as exc:
            return DeliveryResult(False, error=f"transport error {exc.__class__.__name__}")

    async def _resolve(self, target: WebhookTarget) -> str:
        async with asyncio.timeout(self._cfg.timeout):
            addresses = await self._resolver(target.host, target.port)
        return check_addresses(addresses, allow_private=self._cfg.allow_private)

    async def _post(
        self, target: WebhookTarget, address: str, secret: bytes, body: bytes, timestamp: int
    ) -> DeliveryResult:
        request = self._client.build_request(
            "POST",
            _ip_url(target, address),
            content=body,
            headers={
                "Host": _host_header(target),
                "Content-Type": "application/json",
                "X-Footnote-Timestamp": str(timestamp),
                "X-Footnote-Signature": sign(secret, timestamp, body),
                "User-Agent": "footnote-webhook/1",
                "Accept-Encoding": "identity",  # the 1 KiB read limit applies to wire bytes
            },
            extensions={"sni_hostname": target.host},
        )
        response = await self._client.send(request, stream=True, follow_redirects=False)
        try:
            received = 0
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received >= _BODY_LIMIT:
                    break
        finally:
            await response.aclose()
        code = response.status_code
        if 200 <= code < 300:  # noqa: PLR2004 - HTTP success class
            return DeliveryResult(True, status_code=code)
        return DeliveryResult(False, status_code=code, error=f"receiver returned status {code}")


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_hostresolver() -> ports.HostResolver:
        return SystemHostResolver()
