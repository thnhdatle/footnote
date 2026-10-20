# SPDX-License-Identifier: Apache-2.0
"""Outbound webhook delivery: SSRF-checked, signed POSTs and the `webhook` job.

::

    sender = WebhookSender(httpx.AsyncClient(), WebhookConfig())
    result = await sender.send(url, secret, body, timestamp)
"""

from footnote.adapters.webhooks.handler import WebhookJobHandler
from footnote.adapters.webhooks.sender import (
    DeliveryResult,
    SystemHostResolver,
    WebhookConfig,
    WebhookSender,
    system_resolver,
)

__all__ = [
    "DeliveryResult",
    "SystemHostResolver",
    "WebhookConfig",
    "WebhookJobHandler",
    "WebhookSender",
    "system_resolver",
]
