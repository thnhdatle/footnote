# SPDX-License-Identifier: Apache-2.0
"""Outbound webhook delivery."""

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
