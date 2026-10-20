# SPDX-License-Identifier: Apache-2.0
"""Readiness probe of an OpenAI-compatible endpoint: `GET {base_url}/models`.

Never requests a completion or an embedding, so a probe cannot load a model in LM Studio
or consume a fakellm script entry.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

_OK_MAX = 299


async def models_endpoint_problems(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    api_key: str = "",
    headers: Mapping[str, str] | None = None,
) -> list[str]:
    """An empty list when `GET {base_url}/models` answers 2xx, else the reason."""
    request_headers = dict(headers or {})
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/models"
    try:
        response = await client.get(url, headers=request_headers)
    except httpx.HTTPError as exc:
        return [f"GET {url} failed: {type(exc).__name__}"]
    if response.status_code > _OK_MAX:
        return [f"GET {url} returned {response.status_code}"]
    return []
