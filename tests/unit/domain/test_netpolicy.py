# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.errors import ValidationError
from footnote.domain.netpolicy import (
    WebhookTarget,
    check_addresses,
    is_public_address,
    parse_webhook_url,
)

_REJECTED_V4 = [
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
]
_REJECTED_V6 = [
    "::/128",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
    "2001:db8::/32",
    "100::/64",
]


def _in(net: str) -> st.SearchStrategy[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    network = ipaddress.ip_network(net)
    start = int(network.network_address)
    return st.integers(start, start + network.num_addresses - 1).map(
        lambda n: ipaddress.ip_address(n) if network.version == 4 else ipaddress.IPv6Address(n)
    )


def _v4(net: str) -> st.SearchStrategy[ipaddress.IPv4Address]:
    network = ipaddress.IPv4Network(net)
    start = int(network.network_address)
    return st.integers(start, start + network.num_addresses - 1).map(ipaddress.IPv4Address)


@pytest.mark.parametrize("net", _REJECTED_V4)
@given(data=st.data())
def test_is_public_address_rejects_v4_range(net: str, data: st.DataObject) -> None:
    addr = data.draw(_v4(net))
    assert is_public_address(addr) is False
    assert is_public_address(str(addr)) is False
    assert is_public_address(ipaddress.IPv6Address(f"::ffff:{addr}")) is False


@pytest.mark.parametrize("net", _REJECTED_V6)
@given(data=st.data())
def test_is_public_address_rejects_v6_range(net: str, data: st.DataObject) -> None:
    assert is_public_address(data.draw(_in(net))) is False


@pytest.mark.parametrize(
    "addr",
    [
        "255.255.255.255",
        "169.254.169.254",
        "::ffff:169.254.169.254",
        "::ffff:127.0.0.1",
        "::7f00:1",
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "2002:7f00:1::",
        "2002:a00:1::1",
        "2002:c0a8:1::",
        "not-an-ip",
        "",
        "1.2.3",
    ],
)
def test_is_public_address_rejects_special_forms(addr: str) -> None:
    assert is_public_address(addr) is False


@given(_v4("8.0.0.0/8"))
def test_is_public_address_accepts_public_v4(addr: ipaddress.IPv4Address) -> None:
    assert is_public_address(addr) is True
    assert is_public_address(ipaddress.IPv6Address(f"::ffff:{addr}")) is True


@given(_in("2606:4700::/32"))
def test_is_public_address_accepts_public_v6(addr: ipaddress.IPv6Address) -> None:
    assert is_public_address(addr) is True


@pytest.mark.parametrize(
    "addr",
    ["8.8.8.8", "1.1.1.1", "192.0.0.9", "2606:4700::1111", "64:ff9b::808:808"],
)
def test_is_public_address_accepts_known_public(addr: str) -> None:
    expected = addr != "192.0.0.9"
    assert is_public_address(addr) is expected


def test_check_addresses_returns_first_address() -> None:
    assert check_addresses(["8.8.8.8", "1.1.1.1"], allow_private=False) == "8.8.8.8"


def test_check_addresses_rejects_any_private_address() -> None:
    with pytest.raises(ValidationError) as exc:
        check_addresses(["8.8.8.8", "169.254.169.254"], allow_private=False)
    assert exc.value.code == "webhook_url_rejected"
    assert exc.value.field == "webhook.url"
    assert "169.254.169.254" in exc.value.message


def test_check_addresses_rejects_empty_list() -> None:
    with pytest.raises(ValidationError) as exc:
        check_addresses([], allow_private=True)
    assert exc.value.code == "webhook_url_rejected"
    assert exc.value.message == "webhook.url: host did not resolve to any address"


def test_check_addresses_rejects_malformed_even_when_private_allowed() -> None:
    with pytest.raises(ValidationError) as exc:
        check_addresses(["10.0.0.1", "bogus"], allow_private=True)
    assert exc.value.code == "webhook_url_rejected"
    assert "bogus" in exc.value.message


@given(st.lists(st.ip_addresses(), min_size=1, max_size=5))
def test_check_addresses_allow_private_accepts_everything(
    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> None:
    strs = [str(a) for a in addrs]
    assert check_addresses(strs, allow_private=True) == strs[0]


def test_parse_webhook_url_https_defaults() -> None:
    target = parse_webhook_url("https://Example.com/hook", allow_http=False)
    assert target == WebhookTarget("https", "example.com", 443, "/hook")


def test_parse_webhook_url_keeps_query_port_and_default_path() -> None:
    assert parse_webhook_url("https://a.io:8443/x/y?k=v&z=1", allow_http=False) == WebhookTarget(
        "https", "a.io", 8443, "/x/y?k=v&z=1"
    )
    assert parse_webhook_url("https://a.io", allow_http=False).path_qs == "/"
    assert parse_webhook_url("https://a.io?q=1", allow_http=False).path_qs == "/?q=1"


def test_parse_webhook_url_http_requires_flag() -> None:
    assert parse_webhook_url("http://a.io/h", allow_http=True) == WebhookTarget(
        "http", "a.io", 80, "/h"
    )
    with pytest.raises(ValidationError) as exc:
        parse_webhook_url("http://a.io/h", allow_http=False)
    assert exc.value.code == "webhook_url_rejected"
    assert exc.value.field == "webhook.url"


def test_parse_webhook_url_accepts_ip_literals() -> None:
    assert parse_webhook_url("https://10.0.0.1:9/x", allow_http=False) == WebhookTarget(
        "https", "10.0.0.1", 9, "/x"
    )
    assert parse_webhook_url("https://[2606:4700::1]:444/x", allow_http=False) == WebhookTarget(
        "https", "2606:4700::1", 444, "/x"
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://a.io/x",
        "a.io/x",
        "",
        "https:///x",
        "https://user@a.io/x",
        "https://user:pw@a.io/x",
        "https://:pw@a.io/x",
        "https://a.io/x#frag",
        "https://a.io/x#",
        "https://a.io:0/x",
        "https://a.io:99999/x",
        "https://a.io:abc/x",
        "https://[::1/x",
        "https://a.io/x y",
        "https://a.io/x\n",
        "https://a.io/x\x7f",
        "https://a.io/" + "a" * 2048,
    ],
)
def test_parse_webhook_url_rejects_bad_urls(url: str) -> None:
    with pytest.raises(ValidationError) as exc:
        parse_webhook_url(url, allow_http=True)
    assert exc.value.code == "webhook_url_rejected"
    assert exc.value.field == "webhook.url"


def test_parse_webhook_url_accepts_max_length() -> None:
    url = "https://a.io/" + "a" * (2048 - len("https://a.io/"))
    assert len(url) == 2048
    assert parse_webhook_url(url, allow_http=False).path_qs == url[len("https://a.io") :]
