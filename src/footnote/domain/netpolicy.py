# SPDX-License-Identifier: Apache-2.0
"""Outbound webhook network policy: URL parsing and address screening."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from footnote.domain.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

_FIELD = "webhook.url"
_CODE = "webhook_url_rejected"
_MAX_URL_LENGTH = 2048
_DEFAULT_PORTS = {"https": 443, "http": 80}
_MAX_PORT = 65535
_V4_EXTRA_DENIED = (ipaddress.IPv4Network("192.0.0.0/24"),)
_V6_EMBEDDED_V4_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("::/96"),
)
_V6_6TO4 = ipaddress.IPv6Network("2002::/16")
_V6_MAPPED = ipaddress.IPv6Network("::ffff:0:0/96")


@dataclass(frozen=True)
class WebhookTarget:
    scheme: str
    host: str
    port: int
    path_qs: str


def _reject(message: str) -> ValidationError:
    return ValidationError(_FIELD, message, code=_CODE)


def parse_webhook_url(url: str, *, allow_http: bool) -> WebhookTarget:
    """Parses and vets a webhook URL."""
    if len(url) > _MAX_URL_LENGTH:
        raise _reject(f"url is longer than {_MAX_URL_LENGTH} characters")
    if not url.isprintable() or " " in url:
        raise _reject("url contains whitespace or control characters")
    if "#" in url:
        raise _reject("url must not contain a fragment")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError as exc:
        raise _reject("url is malformed") from exc
    scheme = parts.scheme
    if scheme not in _DEFAULT_PORTS or (scheme == "http" and not allow_http):
        raise _reject("url scheme must be https")
    if "@" in parts.netloc:
        raise _reject("url must not contain credentials")
    if not host:
        raise _reject("url has no host")
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    elif not 0 < port <= _MAX_PORT:
        raise _reject("url port is out of range")
    path_qs = parts.path or "/"
    if parts.query:
        path_qs = f"{path_qs}?{parts.query}"
    return WebhookTarget(scheme=scheme, host=host, port=port, path_qs=path_qs)


def _embedded_v4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr in _V6_6TO4:
        return ipaddress.IPv4Address((int(addr) >> 80) & 0xFFFFFFFF)
    if any(addr in net for net in _V6_EMBEDDED_V4_PREFIXES):
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    return None


def _v4_public(addr: ipaddress.IPv4Address) -> bool:
    return (
        addr.is_global
        and not addr.is_multicast
        and not any(addr in net for net in _V4_EXTRA_DENIED)
    )


def _parse(
    addr: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not isinstance(addr, str):
        return addr
    try:
        return ipaddress.ip_address(addr)
    except ValueError:
        return None


def is_public_address(addr: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    parsed = _parse(addr)
    if parsed is None:
        return False
    if isinstance(parsed, ipaddress.IPv4Address):
        return _v4_public(parsed)
    embedded = _embedded_v4(parsed)
    if embedded is not None and not _v4_public(embedded):
        return False
    if parsed in _V6_MAPPED:
        return True
    return parsed.is_global and not parsed.is_multicast


def check_addresses(addresses: Sequence[str], *, allow_private: bool) -> str:
    """Vets the resolved addresses and returns the one to connect to."""
    if not addresses:
        raise _reject("host did not resolve to any address")
    for address in addresses:
        if _parse(address) is None:
            raise _reject(f"address {address!r} is not a valid IP address")
        if not allow_private and not is_public_address(address):
            raise _reject(f"address {address} is not a public address")
    return addresses[0]
